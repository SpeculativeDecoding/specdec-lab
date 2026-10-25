# Handoff: specdec-lab (CS 769 project), state as of 25 Oct 2026

Paste this file into the new Claude session as the first message. It records what has been done,
where everything is, how to run things, and what is still open.

## 1. Project

**Question:** under a fixed budget of draft tokens verified per step (N = 60), should the shape of
the speculative-decoding draft tree (depth × width) depend on the workload or the batch size?

- **Target model:** `meta-llama/Llama-3.1-8B-Instruct`.
- **Drafters:** EAGLE-3. Path B uses `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`; Path A uses
  `lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B`.
- **Decoding:** greedy (T = 0), up to 1,024 new tokens.
- **Workloads:** 6 workloads, 80 questions each: HumanEval (code), GSM8K (math), MT-Bench (dialogue,
  2 turns), Alpaca (instructions), Sum (CNN/DailyMail), QA (Natural Questions).
- **Methods compared:**
  - **EAGLE-3:** a fixed tree, depth 8 × width 10, budget 60.
  - **TALON** (arXiv 2601.07353): an adaptive tree built per step. Our implementation is in
    `specdec/adaptive.py`; we have only read the paper's abstract.
  - **"Ours":** a fixed tree chosen per workload offline, picked on the frozen 30% calibration
    split and scored on the 70% test split.
- **Metrics:**
  - speedup over plain greedy decoding (`ar` = autoregressive: the target model alone, one
    token per pass);
  - τ = tokens accepted per verification step;
  - δ = draft passes per step;
  - tree depth.
- **Team:** Nabanita Bag (user, git `nabanitabag`), Hemanjali Gasada, Junyeol Ryu (git `gajagajago`,
  "June"). The proposal is in `proposal.md` at the repo root.

## 2. Repo

- **Repo:** https://github.com/SpeculativeDecoding/specdec-lab. Local copy:
  `/Users/Patron/Documents/CS 769 nlp/specdec-lab`; the code is in `specdec-bench/`.
- **`main`** has everything below except sections 6 and 7, plus one newer teammate commit.
- **Branch `results/talon-depthcap-h200`:** 4 commits ahead of `main` (section 6); goes to `main`
  through a PR.
- **Branch `feat/template-trees`:** built on top of that branch, plus 1 commit `7a729e0` (section 7).
  This is the local checkout.
- **User preference:** never add a Claude co-author line or "Generated with Claude Code" to commits
  or PRs; only the user appears as author. This is saved in Claude memory as `no-claude-attribution`.
- **`gh` is not installed.** PRs are opened from the browser, using
  `https://github.com/SpeculativeDecoding/specdec-lab/pull/new/<branch>`.
- **Git ignores `results/`**, so results are force-added (`git add -f`). `server.log` files are
  ignored and must not be committed.
- **Untracked local file:** `specdec-bench/.DS_Store`; ignore it.

**Key code:**
- `scripts/run_eagle.py`: Path B (official EAGLE repo, batch 1). Options include `--steps`
  (= depth + 1), `--topk` (width), `--budget`, `--adaptive --mu --root-k --adaptive-max-depth`,
  `--with-ar` and `--warmup`. The warm-up default is now **1** (it was 3; a run-to-run check showed
  the difference is under 2%).
- `specdec/eagle_runner.py`: timing and per-turn records. It now also writes `mean_layer_nodes`, the
  verified tree's mean width at each depth, with index 0 the root.
- `scripts/run_adaptive_suite.sh`: TALON vs EAGLE-3 with budget and μ ablations.
- `scripts/run_depth_grid.sh`: EAGLE-3 8 × 10 + plain decoding, TALON, and depths 1/2/4/6/12/20 at
  width 10.
- `scripts/run_width_grid.sh`: 12 × 4, 20 × 2, 20 × 4, 1 × 20, 2 × 20, 4 × 16.
- `scripts/run_depth_sweep.sh`: depth × width = 80 (1 × 80 … 40 × 2). The 8 × 10 tree runs first,
  and failed trees are skipped.
- `scripts/run_talon_depthcap.sh`: TALON with its depth capped at EAGLE-3's best depth, plus root
  width = EAGLE-3's best width (section 6).
- `scripts/run_sglang_sweep.py`, `specdec/sglang_launcher.py`: Path A (SGLang v0.5.22), with modes
  `ar`, `spec`, `auto`, `adaptive` and `talon`.
- `specdec/sglang_talon_plugin.py`: TALON-S, our port of TALON's rules into SGLang as an AROUND hook
  on `eagle_worker_v2.select_top_k_tokens` (CUDA-graph safe; writes `talon_marker.jsonl`).
- `scripts/summarize.py`: produces `summary.csv` per results folder.
- `scripts/compare_vs_eagle3.py`: the headline comparison (EAGLE-3 vs TALON vs Ours) relative to
  EAGLE-3 8 × 10, with a paired bootstrap.
- `scripts/check_lossless.py`, `scripts/diagnose_mismatch.py`: check that speculative outputs match
  plain decoding.
- **Configs:**
  - `configs/sweep_eagle3_llama31.yaml`: Path A, 21 shapes;
  - `configs/batch_curve_2_4_16.yaml`;
  - `configs/talon_vs_eagle3_bs8.yaml` and `configs/talon_vs_eagle3_bs1_32.yaml`.
- **Tests:** `python -m unittest discover -s tests`. 63 tests pass, 10 are skipped (they need a GPU
  or the EAGLE repo).
- **Local Python with torch for the tests:**
  `/private/tmp/claude-503/-Users-Patron-Documents-CS-769-nlp-specdec-lab/c5c77f36-b410-4d32-91bd-b07ac644d167/scratchpad/hfv/bin/python`.
  It sits in the old session's scratchpad and may be gone; if so, recreate it with
  `pip install torch numpy pyyaml requests`.

## 3. Running on CHTC

- **Login:** `ssh chtc` (`nbag@ap2001.chtc.wisc.edu`). Logging in needs Duo; the connection is
  then reused for 24 hours. Non-interactive calls use `ssh -o BatchMode=yes chtc …`; when that
  fails with "Permission denied" or a timeout, the user must run `ssh chtc` again (on the UW network
  or VPN).
- **Submitting** (from `specdec-bench/`): `bash chtc/upload_and_submit.sh KEY=VALUE …`. It packages
  the repo, uploads it and runs `condor_submit -spool`.
  - **Path B** (`chtc/job_adaptive.sub`, PyTorch 2.6 cu124 container):
    `BENCHES=gsm8k RUN=<folder> SUITE=scripts/<script>.sh GPU=H200`. `GPU` is a pattern matched
    against the device name.
  - **Path A** (`chtc/job_sglang.sub`, official `lmsysorg/sglang:v0.5.22` image, H200 only):
    `SUB=job_sglang.sub CONFIG=configs/<x>.yaml BENCHES=a,b RUN=<folder>`.
- **Collecting results:**
  1. `ssh chtc 'cd ~/specdec-bench/chtc && condor_transfer_data <id>'`; the files land in
     `~/specdec-bench/chtc/<RUN>`.
  2. Copy them locally:
     `rsync -a --exclude server.log -e "ssh -o BatchMode=yes" chtc:specdec-bench/chtc/<RUN> results/`.
  3. Run `summarize.py`, then commit with `git add -f`.
- **Problems already solved, already in the scripts:**
  - **No root in the container:** git is installed with conda.
  - **`TEMP` collides with the sampling temperature:** Condor sets `TEMP` to the scratch folder, so
    the job runs with `TEMP=0`.
  - **Blackwell GPUs unsupported:** the pinned torch can't run on them, so jobs require compute
    capability 8.0–9.0 and no MIG slices.
  - **SGLang context-length error:** SGLang needs `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`,
    because the SGLang EAGLE-3 drafter declares 2,048 positions.
  - **Port clashes:** each job picks its own free port.
  - **Plugin registration:** the TALON-S plugin is registered through a hand-written dist-info
    folder on `PYTHONPATH`, with no pip install.
- **CHTC limits:**
  - `condor_history` hangs, so use `condor_q`.
  - `condor_ssh_to_job` is disabled.
  - Results are returned only when a job exits, so killing a running job loses its work.
  - CHTC A100s have CUDA 12.8 drivers, so Path A (SGLang 0.5.22) needs H200s with CUDA 13.1.

## 4. Results so far (all on `main` under `specdec-bench/results/`; findings in `results/README.md`)

**Headline (Path B, H200, batch 1, test split, relative to EAGLE-3 8 × 10;
`results/summary_vs_eagle3.csv`):**

| Workload | EAGLE-3 vs plain | TALON vs EAGLE-3 | Ours (chosen tree) vs EAGLE-3 |
| --- | --- | --- | --- |
| HumanEval | 4.42× | 0.99 | 8 × 10: 1.00 |
| GSM8K | 3.93× | 1.01 | 8 × 10: 1.00 |
| MT-Bench | 4.06× | 1.01 | 8 × 10: 1.00 |
| Alpaca | 4.32× | 0.99 | 8 × 10: 1.00 |
| Sum | 3.41× | 1.04 | 6 × 10: 1.04 |
| QA | 3.32× | 1.07 | 6 × 10: 1.05 |

- **Workload barely changes the best shape at batch 1.** EAGLE-3's 8 × 10 is best for code, math,
  dialogue and instructions. Sum and QA prefer slightly shallower and wider trees (about +4–5%).
  June's A100 sweep (`depth_sum_qa_humaneval_gsm8k_a100/`) confirms this within about 0.03.
- **Extreme shapes lose 15–60%:** 1 × 80, 2 × 40, 20 × 4 and 40 × 2.
- **TALON vs EAGLE-3:** TALON ties EAGLE-3 or is 4–7% faster (QA, Sum). Its τ is lower on 5 of 6
  workloads, but it runs fewer draft passes. This is a partial reproduction of the paper, not a
  refutation; the README has a section on this.
- **τ and speed disagree:** τ rises with depth, but speed peaks around depth 6–8, because every
  layer costs a draft pass.
- **Batch size dominates (Path A, SGLang):**
  - 60-token trees: 2.7–2.9× at batch 1, about 1.0× at batch 16, about 0.6× at batch 32.
  - A 7-deep chain is best or tied at every batch size. The batch-size curve is in
    `sglang_bscurve_humaneval_alpaca_h200/`.
- **TALON-S** (our SGLang port) is 5–8% slower than EAGLE-3 at batch 1, 2, 4, 8, 16 and 32. SGLang
  always runs every draft pass, so TALON can't save work there.
- **Correctness:** 74–96% of outputs match plain decoding token for token. The mismatches look like
  fp16 near-ties; `diagnose_mismatch.py` hasn't been run on them.

**Results folders:**
- **Path B:**
  - `adaptive_three_bench/` (Colab A100), `chtc_h200/`, `chtc_l40s/`;
  - `depth_grid_*_h200/`, `dxw80_sweep_*_h200/`, `width_grid_humaneval_gsm8k_mt_bench_h200/`;
  - June's `depth_sum_qa_humaneval_gsm8k_a100/`.
- **Path A:** `sglang_h200/` (21 shapes × batch 1/8/32), `sglang_talon_bs8_h200/`,
  `sglang_talon_bs8_A100/` (teammate), `sglang_talon_bs1_32_h200/`,
  `sglang_bscurve_humaneval_alpaca_h200/`.

## 5. Midterm deck

https://claude.ai/artifact/8z82mvuFZSe96wrGKXQooc, 12 slides with speaker notes. It is private: share
it from the page's Share menu.

- **Covered:** the question, related work, setup, progress, Path B results (tree depth, speed,
  budget and μ), Path A batch size, the matched-budget grid, challenges and deviations, next steps.
- **Not yet in the deck:** the headline vs-EAGLE-3 table, the τ tables, the depth × width = 80
  sweep, the TALON depth-cap results.
- **Agreed framing:** focus on Path B with EAGLE-3 vs TALON vs Ours. Keep one Path A slide on batch
  size. Next steps: other model sizes.

## 6. Done: TALON depth-cap experiment (H200, Path B), on a PR branch not yet merged

**Question:** if TALON is held to the depth of EAGLE-3's best fixed tree for each workload, does it
beat that tree, and what width does it choose at each depth? EAGLE-3's best trees are 8 × 10 for
GSM8K and HumanEval and 5 × 16 for QA and Sum. The reverse was also run: TALON with root width 16
and no depth cap.

**Code** (on branch `results/talon-depthcap-h200`):
- `scripts/run_talon_depthcap.sh`, one CHTC job per workload.
- **Recorded widths:** every run now records `mean_layer_nodes`, the verified tree's mean width at
  each depth (index 0 = root).
- **`--adaptive-fill`** (`AdaptiveConfig.fill`): when the cap stops TALON under budget, the leftover
  tokens go to the best unused candidates of the layers already built. The tree then verifies all
  60 tokens with no extra draft pass.
- **Run names** include the cap and fill: `_d<cap>`, `_fill`.
- **Tests:** 66 pass.

**Important detail:** EAGLE-3 "depth D" (`--steps D+1`) builds **D + 1 layers** below the root. The
first runs capped TALON at D, one layer short; the corrected v2 runs cap at 9 and 6 layers.

**Results folders** (all collected and committed on the branch):
- `talon_depthcap_v2_{gsm8k,humaneval,qa,sum}_h200/`: the corrected runs (jobs 12363269–72).
- `talon_depthcap_{gsm8k,humaneval,qa,sum}_h200/`: the first runs, cap one layer short (jobs
  12363246–49). Their EAGLE-3 and uncapped-TALON rows are still valid.

**Speed relative to EAGLE-3's best tree (same job):**

| Workload | TALON | Capped | Capped + fill | Root 16 | Root 16 capped | Root 16 capped + fill |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K | 1.01 | 1.00 | 1.01 | – | – | – |
| HumanEval | 0.99 | 0.99 | 0.98 | – | – | – |
| QA | 1.04 | 0.97 | 0.99 | 1.02 | 0.98 | 0.96 |
| Sum | 0.97 | 0.98 | 0.98 | 1.02 | 0.99 | 0.97 |

**Findings:**
- **Everything is within noise of EAGLE-3's best fixed tree.** Repeat runs of the same
  configuration differ by 2–3%.
- **Capping never helps:** capped TALON lands at 0.96–1.00. Capped trees verify only 50–57 of 60
  tokens.
- **Fill barely changes τ** (GSM8K 6.03 → 6.04, QA 4.77 → 4.86): the extra leaves are rarely
  accepted.
- **TALON's edge comes from depth beyond EAGLE-3's.** On HumanEval it puts about 6.6 of 60 nodes
  below depth 9, down to depth 21.
- **The two methods build opposite shapes:**
  - EAGLE-3 keeps 3–6 nodes at depth 1 and widens with depth (e.g. HumanEval 2.8 → 8.5).
  - TALON keeps 10 (or 16) at depth 1 and narrows.
- **TALON reshapes by workload, but without a speed gain:** narrow middle layers (5–6) and a deep
  tail on code; wide middle layers (10–12) and a short tail on QA and Sum. It doesn't beat a
  per-workload fixed tree.
- The full width-by-depth table is in `results/README.md` under "TALON held to EAGLE-3's best
  depth", on the branch.

**Status:**
- **Branch `results/talon-depthcap-h200`** is pushed and **4 commits ahead of `main`**:
  `4ec1b78`, `ee9dbb7`, `dfb842a` and `cc2b83e` (code, results, README).
- **The PR is not opened yet.** Link:
  https://github.com/SpeculativeDecoding/specdec-lab/pull/new/results/talon-depthcap-h200
- **A ready-to-paste PR title and description** were drafted in the old session. The title was
  "TALON depth-cap experiment: TALON held to EAGLE-3's best depth (H200)"; the body is the
  Question / Code / Results text above.
- **`main` moved by 1 teammate commit** after the branch was made. Check it and merge
  `origin/main` into the branch if needed. Never force-push.
- **No CHTC jobs are running, and no automatic monitoring is set up.**

## 7. In progress: tree-shape templates and tree pictures (branch `feat/template-trees`)

**Why:** in the D × K sweeps, D and K only cap the tree. The verified tree is the global top 59
nodes, so the same D × K gives different realised shapes per workload. Also, EAGLE's K does two
jobs (children per node and nodes expanded per layer). To control shape directly at budget 60, we
built:

- **Option 1, fixed templates** (`TemplateConfig` / `TemplateDrafter` in `specdec/adaptive.py`):
  - exact widths per depth, adding up to 59, with each layer keeping its most likely candidates;
  - `profile_widths(profile, depth, budget)` gives profiles `down` / `flat` / `up` / `diamond` at
    a chosen depth. For example, flat5 = 12, 12, 12, 12, 11 and up5 = 4, 8, 12, 16, 19.
- **Option 3, separate controls:** `--template-branch` (children per expanded node) and
  `--template-expand` (nodes expanded per layer; 0 = all).
- **`run_eagle.py` flags:**
  - `--template "10,8,6,..."`, `--template shape:<profile>:<depth>`, or `geo:<first>:<ratio>`;
    avoid `geo`, since ratios below 1 make long chains of single nodes;
  - `--template-label`;
  - `--save-trees N`: saves the first N trees per turn (parents and depths) in each row's
    `sample_trees`.
- **`summarize.py`** names template runs `eagle3_tmpl_<label>_b<branch>_e<expand|all>`.
- **`scripts/plot_trees.py`:**
  - `widths <folders> --configs ... --out x.png`: mean width by depth, one panel per workload,
    plus `x.csv`;
  - `trees <folder> --bench B --out x.png`: node-link drawings of saved trees.
  - It uses the dataviz skill's default palette in fixed order: EAGLE-3 8 × 10 blue, TALON orange.
    The palette validator wasn't run, because node.js isn't installed.
- **First figure:** `results/figures/widths_eagle_vs_talon_h200.png` (+ `.csv`), from the depth-cap
  runs.
  - EAGLE-3 is narrow at depth 1 and widens (code), or peaks at depth 3–4 (QA, Sum).
  - TALON keeps 10 at depth 1, then narrows into a long tail.
- **Tests:** 75 pass, 10 skipped (they need a GPU).

**Running now:** `scripts/run_template_sweep.sh`, one CHTC H200 job per workload. Results go to
`template_sweep_<bench>_h200/`.

| Job | Workload |
| --- | --- |
| 12363709 | HumanEval |
| 12363710 | GSM8K |
| 12363711 | QA |
| 12363712 | Sum |

- **Runs per workload (33, budget 60, warm-up 1, `--save-trees 3`):**
  1. EAGLE-3 8 × 10 with plain decoding;
  2. the depth × width = 80 trees (1 × 80 … 40 × 2), to see realised shapes;
  3. 16 templates: down / flat / up / diamond × depth 3 / 5 / 8 / 12, every node expanded;
  4. 8 templates at depth 5 and 8 with only 3 nodes expanded per layer.
- **Timing:** about 2–3 hours per job.
- **Failures:** a run that fails is reported ("WARNING: … failed, continuing") and skipped. The
  template builder had never run on a GPU before this sweep. It reuses TALON's drafting code, which
  works on GPUs.
- **No monitoring is set up.**

**To do when the jobs finish:**
1. Collect each job: `condor_transfer_data`, copy the results with rsync excluding `server.log`,
   regenerate `summary.csv`, and commit to `feat/template-trees`.
2. Make speed tables relative to EAGLE-3 8 × 10, per workload: which profile and depth wins, and
   whether limited expansion helps.
3. Plot the realised width profiles of the D × K trees per workload with
   `plot_trees.py widths --configs eagle3_s*`, and example trees with `plot_trees.py trees`.
4. **Then option 2 (Sequoia-style):** measure acceptance rates by depth and sibling rank from these
   runs, and build the expected-acceptance-optimal tree per workload as a template.
5. Open PRs for `results/talon-depthcap-h200`, then `feat/template-trees`. Use no Claude
   attribution, and merge `origin/main` first if `main` has moved (never force-push).

## 8. Open items and ideas

- **Not done yet:**
  - three repeats per configuration (the proposal's variance criterion);
  - the EAGLE-1 drafter and PRISM;
  - T > 0 (sampling);
  - other model sizes;
  - checking whether Sum prompts exceed 2,048 tokens in Path A;
  - the deck still needs the depth-cap results (the README has them, on the PR branch).
- **Optional cleanup offered:** set the default `BENCHES` in `run_depth_sweep.sh` to
  `humaneval,gsm8k,sum,qa`, and add a `chtc/submit_dxw80_sweep.sh` with the exact submit loop.
- **Direction ideas discussed (most promising first):**
  1. a cost model, speed ≈ τ(d, w) ÷ (verification cost(batch) + d × draft cost), that predicts the
     best shape;
  2. a bandit shape picker, evaluated offline by replaying our per-question logs (no GPU needed);
  3. early-stop drafting, to isolate TALON's real advantage;
  4. load- or batch-size-adaptive trees, e.g. several CUDA graphs in SGLang, or budget split across
     the requests in a batch;
  5. checks under T > 0, EAGLE-1 or a 70B target.
- **Literature to check before claiming novelty:** Sequoia, SpecDec++, batch-level draft-token
  selection work (possibly called "TETRIS").
