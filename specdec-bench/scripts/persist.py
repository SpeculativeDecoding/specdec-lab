#!/usr/bin/env python
"""Save model checkpoints, code and environments to persistent disk, and restore them.

Built for Colab (the machine is wiped every session; Google Drive is the persistent
disk) but works with any directory, e.g. a VM's persistent disk or a NAS mount.

  python scripts/persist.py save    --store /content/drive/MyDrive/specdec --what models,code
  python scripts/persist.py restore --store /content/drive/MyDrive/specdec --what models,code
  python scripts/persist.py status  --store /content/drive/MyDrive/specdec

What is saved (each item is written as <name>.partial and renamed when complete, so an
interrupted save never leaves a file that looks valid):

  models  store/models/models--<org>--<name>.tar
          The Hugging Face cache entry of each repo (blobs + snapshot symlinks), so after
          restore every script loads the same repo id offline-ready, with no path changes.
          Llama's duplicate `original/` consolidated checkpoint (16 GB) is excluded.
  code    store/code/specdec-bench.zip  (no venvs, third_party, results or caches)
          The previous copy is kept as specdec-bench.prev.zip.
  envs    store/envs/eagle.tar  = .venv-eagle  + third_party/EAGLE   (editable install)
          store/envs/sglang.tar = .venv-sglang + third_party/sglang
          Venvs contain absolute paths, so they are restored only into the same repo path
          and the same Python minor version; otherwise restore skips them and the setup
          scripts reinstall.

Results do not need this script: point --out-dir at the persistent disk (the Colab
notebook symlinks results/ to Drive) and runs checkpoint themselves after every question.
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
import time
import zipfile

DEFAULT_REPOS = [
    "meta-llama/Llama-3.1-8B-Instruct",          # target
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",       # Path B draft (EAGLE repo, TALON)
    "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B",  # Path A draft (SGLang)
]
IGNORE = ["original/*", "*.pth"]                 # Llama's consolidated Meta-format copy
CODE_EXCLUDE_DIRS = {".venv-eagle", ".venv-sglang", "third_party", "results", "__pycache__",
                     ".git", ".ipynb_checkpoints", ".pytest_cache"}
ENVS = {"eagle": [".venv-eagle", "third_party/EAGLE"],
        "sglang": [".venv-sglang", "third_party/sglang"]}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(*a):
    print(*a, flush=True)


def hub_cache() -> str:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return HF_HUB_CACHE
    except ImportError:
        if os.environ.get("HF_HUB_CACHE"):
            return os.environ["HF_HUB_CACHE"]
        home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
        return os.path.join(home, "hub")


def repo_dirname(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


def gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"


def tree_size(path: str) -> int:
    total = 0
    for d, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(d, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def check_space(store: str, need: int):
    os.makedirs(store, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < need:
        raise SystemExit(f"not enough space in {store}: need {gib(need)}, free {gib(free)}. "
                         "(Free Google Drive is 15 GB; the Llama checkpoint alone is ~15 GB.)")


def write_tar(dest: str, members, base: str):
    """members: paths relative to base. Symlinks are stored as symlinks."""
    tmp = dest + ".partial"
    with tarfile.open(tmp, "w") as tar:
        for m in members:
            tar.add(os.path.join(base, m), arcname=m)
    os.replace(tmp, dest)


def extract_tar(src: str, dest: str, strict: bool = True):
    """strict ("data" filter): no absolute or escaping links -- right for HF caches, whose
    snapshot links are relative. Venvs link bin/python to the system interpreter by
    absolute path, so they are extracted with the "tar" filter (still no paths outside dest)."""
    with tarfile.open(src, "r") as tar:
        try:
            tar.extractall(dest, filter="data" if strict else "tar")
        except TypeError:                             # Python without extraction filters
            tar.extractall(dest)


def sidecar(path: str) -> str:
    return path + ".json"


# ----------------------------------------------------------------------------- models
def save_models(store: str, repos, download: bool, force: bool):
    cache = hub_cache()
    out = os.path.join(store, "models")
    os.makedirs(out, exist_ok=True)
    for repo in repos:
        name = repo_dirname(repo)
        dest = os.path.join(out, name + ".tar")
        if os.path.exists(dest) and not force:
            log(f"[models] {repo}: already saved ({gib(os.path.getsize(dest))})")
            continue
        if download:
            from huggingface_hub import snapshot_download
            snapshot_download(repo, ignore_patterns=IGNORE)
        src = os.path.join(cache, name)
        if not os.path.isdir(os.path.join(src, "snapshots")):
            raise SystemExit(f"[models] {repo} is not in the HF cache ({src}); run without --no-download")
        size = tree_size(src)
        check_space(out, size + 2**30)
        t = time.time()
        write_tar(dest, [name], cache)
        json.dump({"repo": repo, "bytes": size, "saved": time.strftime("%Y-%m-%d %H:%M")},
                  open(sidecar(dest), "w"))
        log(f"[models] {repo}: saved {gib(size)} in {time.time() - t:.0f}s -> {dest}")


def restore_models(store: str, repos, force: bool):
    cache = hub_cache()
    os.makedirs(cache, exist_ok=True)
    src_dir = os.path.join(store, "models")
    for repo in repos:
        name = repo_dirname(repo)
        src = os.path.join(src_dir, name + ".tar")
        if not os.path.exists(src):
            log(f"[models] {repo}: not saved yet (it will download from Hugging Face)")
            continue
        here = os.path.join(cache, name)
        if os.path.isdir(os.path.join(here, "snapshots")) and not force:
            log(f"[models] {repo}: already in {cache}")
            continue
        t = time.time()
        extract_tar(src, cache)
        log(f"[models] {repo}: restored {gib(tree_size(here))} in {time.time() - t:.0f}s -> {here}")


# ----------------------------------------------------------------------------- code
def code_files(root: str):
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in CODE_EXCLUDE_DIRS)
        for f in sorted(files):
            if f.endswith((".pyc", ".partial")):
                continue
            yield os.path.join(d, f)


def save_code(store: str, root: str):
    out = os.path.join(store, "code")
    os.makedirs(out, exist_ok=True)
    dest = os.path.join(out, "specdec-bench.zip")
    tmp = dest + ".partial"
    top = "specdec-bench"            # same layout as the zip you upload
    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in code_files(root):
            z.write(fp, os.path.join(top, os.path.relpath(fp, root)))
            n += 1
    if os.path.exists(dest):
        os.replace(dest, os.path.join(out, "specdec-bench.prev.zip"))
    os.replace(tmp, dest)
    log(f"[code] saved {n} files -> {dest}")


def restore_code(store: str, root: str):
    """Unpack the saved zip into root (its top-level specdec-bench/ folder is stripped)."""
    src = os.path.join(store, "code", "specdec-bench.zip")
    if not os.path.exists(src):
        log("[code] nothing saved yet")
        return
    n = 0
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            parts = info.filename.split("/")
            if parts[0] != "specdec-bench" or ".." in parts or info.filename.startswith("/"):
                raise SystemExit(f"[code] unexpected path in zip: {info.filename}")
            rel = "/".join(parts[1:])
            if not rel or info.is_dir():
                continue
            dest = os.path.join(root, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(info) as fi, open(dest, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            n += 1
    log(f"[code] restored {n} files -> {root}")


# ----------------------------------------------------------------------------- envs
def _env_info(root: str):
    return {"root": os.path.abspath(root), "python": f"{sys.version_info[0]}.{sys.version_info[1]}"}


def save_envs(store: str, root: str, which, force: bool):
    out = os.path.join(store, "envs")
    os.makedirs(out, exist_ok=True)
    for name in which:
        parts = ENVS[name]
        dest = os.path.join(out, name + ".tar")
        if not os.path.isdir(os.path.join(root, parts[0])):
            log(f"[envs] {name}: {parts[0]} not installed here, skipping")
            continue
        if os.path.exists(dest) and not force:
            log(f"[envs] {name}: already saved (use --force after reinstalling)")
            continue
        present = [p for p in parts if os.path.exists(os.path.join(root, p))]
        size = sum(tree_size(os.path.join(root, p)) for p in present)
        check_space(out, size + 2**30)
        t = time.time()
        write_tar(dest, present, root)
        json.dump(_env_info(root), open(sidecar(dest), "w"))
        log(f"[envs] {name}: saved {gib(size)} in {time.time() - t:.0f}s -> {dest}")


def restore_envs(store: str, root: str, which, force: bool):
    for name in which:
        src = os.path.join(store, "envs", name + ".tar")
        if not os.path.exists(src):
            log(f"[envs] {name}: not saved yet")
            continue
        venv = os.path.join(root, ENVS[name][0])
        if os.path.exists(os.path.join(venv, "bin", "python")) and not force:
            log(f"[envs] {name}: already installed")
            continue
        try:
            info = json.load(open(sidecar(src)))
        except (OSError, ValueError):
            info = {}
        want = _env_info(root)
        if info != want:
            log(f"[envs] {name}: saved for {info}, this machine is {want}; skipping "
                "(the setup script will reinstall)")
            continue
        t = time.time()
        extract_tar(src, root, strict=False)
        log(f"[envs] {name}: restored in {time.time() - t:.0f}s -> {venv}")


# ----------------------------------------------------------------------------- main
def status(store: str):
    for sub in ("models", "code", "envs"):
        d = os.path.join(store, sub)
        if not os.path.isdir(d):
            log(f"{sub:7s} (none)")
            continue
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            if os.path.isfile(fp) and not f.endswith(".json"):
                log(f"{sub:7s} {f:55s} {gib(os.path.getsize(fp)):>9s}  "
                    f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(fp)))}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["save", "restore", "status"])
    p.add_argument("--store", required=True, help="persistent directory, e.g. a Google Drive folder")
    p.add_argument("--what", default="models,code", help="comma list of models, code, envs")
    p.add_argument("--repos", nargs="*", default=DEFAULT_REPOS)
    p.add_argument("--envs", default="eagle,sglang", help="which environments (with --what envs)")
    p.add_argument("--root", default=ROOT, help="repo root (default: this checkout)")
    p.add_argument("--no-download", action="store_true", help="save only what is already cached")
    p.add_argument("--force", action="store_true", help="overwrite / re-extract existing items")
    a = p.parse_args(argv)
    what = {w.strip() for w in a.what.split(",") if w.strip()}
    bad = what - {"models", "code", "envs"}
    if bad:
        p.error(f"unknown --what {sorted(bad)}")
    envs = [e.strip() for e in a.envs.split(",") if e.strip()]
    if set(envs) - set(ENVS):
        p.error(f"--envs must be from {sorted(ENVS)}")
    root = os.path.abspath(a.root)
    if a.action == "status":
        return status(a.store)
    if a.action == "save":
        if "code" in what:
            save_code(a.store, root)
        if "models" in what:
            save_models(a.store, a.repos, not a.no_download, a.force)
        if "envs" in what:
            save_envs(a.store, root, envs, a.force)
    else:
        if "code" in what:
            restore_code(a.store, root)
        if "envs" in what:
            restore_envs(a.store, root, envs, a.force)
        if "models" in what:
            restore_models(a.store, a.repos, a.force)


if __name__ == "__main__":
    main()
