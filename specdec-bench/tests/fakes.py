"""Test doubles that mimic the real interfaces used by the harness."""
import inspect


class FakeTokenizer:
    """Mimics the HF tokenizer methods the harness calls."""
    eos_token_id = 128009
    special_tokens_map = {}

    def __init__(self):
        self.vocab = {"<|eot_id|>": 128009}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        s = "<BOS>" + "".join(f"<{m['role']}>{m['content']}<eot>" for m in messages)
        return s + "<assistant>"

    def __call__(self, texts, add_special_tokens=True):
        assert add_special_tokens is False, "EAGLE encodes with add_special_tokens=False"

        class Out:
            pass
        o = Out()
        o.input_ids = [[1000 + (ord(c) % 200) for c in t] for t in texts]
        return o

    def convert_tokens_to_ids(self, tok):
        return self.vocab.get(tok)

    def decode(self, ids, spaces_between_special_tokens=False):
        return "".join(chr(97 + (i % 26)) for i in ids)


class FakeEaModel:
    """Same call signature as EaModel.eagenerate / naivegenerate (EAGLE @ cb7e084)."""

    def __init__(self, tokenizer, n_tokens=20, accept_per_step=3, extra_after_stop=2):
        self.tok = tokenizer
        self.n = n_tokens
        self.k = accept_per_step
        self.extra = extra_after_stop
        self.calls = []

    def get_tokenizer(self):
        return self.tok

    def _gen(self, input_ids, per_step, **kw):
        ids = list(input_ids[0])
        body = [5 + i for i in range(self.n)]
        out = ids + body + [self.tok.eos_token_id] + [7] * self.extra
        new_token = len(body) + 1 + self.extra
        steps = -(-new_token // per_step)
        self.calls.append(kw)
        return [out], new_token, steps - 1   # idx = steps - 1

    def eagenerate(self, input_ids, temperature=0.0, top_p=0.0, top_k=0.0,
                   max_new_tokens=512, max_length=2048, log=False, is_llama3=False):
        assert log is True
        return self._gen(input_ids, self.k + 1, temperature=temperature,
                         max_new_tokens=max_new_tokens, max_length=max_length,
                         is_llama3=is_llama3)

    def naivegenerate(self, input_ids, temperature=0.0, top_p=0.0, top_k=0.0,
                      max_new_tokens=512, max_length=2048, log=False, is_llama3=False):
        assert log is True
        return self._gen(input_ids, 1, temperature=temperature,
                         max_new_tokens=max_new_tokens, max_length=max_length,
                         is_llama3=is_llama3)


FAKE_SIGNATURE = list(inspect.signature(FakeEaModel.eagenerate).parameters)
