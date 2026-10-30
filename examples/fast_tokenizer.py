#!/usr/bin/env python3
"""Exact, parallel client-side tokenization for long prompts (the tokenizer used in the benchmark's E2E timing).

`ClientTokenizer(tok, mode="parallel")(texts, L)` returns the same ids as
`tok(texts, padding="max_length", truncation=True, max_length=L)["input_ids"]` (checked exactly against it by
the benchmark on 40 calls per run), but splits long texts at pre-token boundaries and encodes all pieces in one
multi-threaded `encode_batch_fast` call. For 8,192-token prompts this is several times faster than the plain
call, which runs each long text on one thread.

    from transformers import AutoTokenizer
    from examples.fast_tokenizer import ClientTokenizer
    fast_tok = ClientTokenizer(AutoTokenizer.from_pretrained(model_dir), mode="parallel")
    ids = fast_tok(texts, 8192)   # list of 8,192-id lists, ready for llm.generate(prompt_token_ids=...)
"""


class ClientTokenizer:
    """Client-side tokenization used inside the timed E2E region.

    mode="hf"       (default) tok(texts, padding="max_length", truncation=True,
                    max_length=L)["input_ids"]
    mode="fast"     same ids via the Rust backend's encode_batch_fast (no offset tracking), with
                    truncation/padding configured on a private backend copy.
    mode="parallel" like "fast", but texts longer than ~`chunk_tokens` tokens are split at
                    letter-space-letter boundaries (always a pre-token boundary for this
                    tokenizer's Split regex + ByteLevel, no normalizer) and all chunks are encoded
                    in one parallel encode_batch_fast call; ids = prefix specials + concat, truncated
                    to L, padded to L. Only enabled when the post-processor adds prefix-only
                    special tokens and no added token contains a space.
    Call with verify=True to check that call against mode="hf" (exact ids required, mismatch ->
    RuntimeError). The benchmark verifies 3 untimed + 5 warm-up calls per shape (40 calls per run)
    and never verifies inside the timed loop.
    """

    def __init__(self, tok, mode="hf", chunk_tokens=512):
        self.tok, self.mode, self.chunk_tokens = tok, mode, chunk_tokens
        self.verified = 0
        self.last_padded = 0  # sequences that needed padding in the last call (expected 0 for 8k prompts)
        self._backends = {}
        if mode == "hf":
            return
        bt = tok.backend_tokenizer
        self._raw = bt.__class__.from_str(bt.to_str())
        self._raw.no_truncation()
        self._raw.no_padding()
        empty = self._raw.encode("", add_special_tokens=True).ids
        probe = self._raw.encode("hello world", add_special_tokens=True).ids
        probe_raw = self._raw.encode("hello world", add_special_tokens=False).ids
        self.prefix = list(empty)
        prefix_only = probe == self.prefix + probe_raw
        added_with_space = [t.content for t in tok.added_tokens_decoder.values() if " " in t.content]
        added_strip = [t.content for t in tok.added_tokens_decoder.values()
                       if getattr(t, "lstrip", False) or getattr(t, "rstrip", False)]
        self.parallel_ok = prefix_only and not added_with_space and not added_strip \
            and bt.normalizer is None and tok.truncation_side == "right" and tok.padding_side == "right"
        if mode == "parallel" and not self.parallel_ok:
            print(f"[bench] tokenizer-mode parallel unsupported (prefix_only={prefix_only}, "
                  f"added_with_space={added_with_space[:3]}); using fast", flush=True)
            self.mode = "fast"
        self.pad_id = tok.pad_token_id
        # chars per token estimate for chunk sizing (random-id text is ~2-4 chars/token)
        self.chars_per_chunk = None

    def _backend(self, L):
        b = self._backends.get(L)
        if b is None:
            bt = self.tok.backend_tokenizer
            b = bt.__class__.from_str(bt.to_str())
            b.enable_truncation(max_length=L, direction=self.tok.truncation_side)
            b.enable_padding(length=L, pad_id=self.pad_id,
                             pad_token=self.tok.convert_ids_to_tokens(self.pad_id), direction=self.tok.padding_side)
            self._backends[L] = b
        return b

    @staticmethod
    def _split(text, target_chars):
        if len(text) <= int(1.5 * target_chars):
            return [text]
        cuts, n, pos = [], len(text), target_chars
        limit = n - target_chars // 2
        while pos < limit:
            stop = min(n - 1, pos + 4096)
            i = pos
            found = -1
            while True:
                sp = text.find(" ", i, stop)
                if sp == -1:
                    break
                if sp > 0 and text[sp - 1].isalpha() and text[sp + 1].isalpha():
                    found = sp
                    break
                i = sp + 1
            if found == -1:
                break
            cuts.append(found)
            pos = found + target_chars
        if not cuts:
            return [text]
        bounds = [0] + cuts + [n]
        return [text[a:b] for a, b in zip(bounds[:-1], bounds[1:])]

    def _encode(self, texts, L):
        if self.mode == "hf":
            enc = self.tok(texts, padding="max_length", truncation=True, max_length=L)
            self.last_padded = sum(1 for m in enc["attention_mask"] if 0 in m)
            return enc["input_ids"]
        if self.mode == "fast" or L <= 2 * self.chunk_tokens:
            encs = self._backend(L).encode_batch_fast(texts, add_special_tokens=True)
            self.last_padded = sum(1 for e in encs if 0 in e.attention_mask)
            return [e.ids for e in encs]
        # parallel: split every long text at pre-token boundaries and encode all pieces in one parallel
        # encode_batch_fast call (load-balances across all host cores for any batch size).
        base_chars = self.chars_per_chunk or (self.chunk_tokens * 3)
        target_chars = base_chars * (2 if len(texts) >= 8 else 1)
        pieces, owners = [], []
        for j, t in enumerate(texts):
            for p in self._split(t, target_chars):
                pieces.append(p)
                owners.append(j)
        encs = self._raw.encode_batch_fast(pieces, add_special_tokens=False)
        out = [list(self.prefix) for _ in texts]
        for j, e in zip(owners, encs):
            if len(out[j]) < L:
                out[j].extend(e.ids)
        res = []
        self.last_padded = 0
        for ids in out:
            ids = ids[:L]
            if len(ids) < L:
                self.last_padded += 1
                ids = ids + [self.pad_id] * (L - len(ids))
            res.append(ids)
        if self.chars_per_chunk is None:
            tot_chars = sum(len(p) for p in pieces)
            tot_toks = sum(len(e.ids) for e in encs)
            self.chars_per_chunk = max(256, int(self.chunk_tokens * tot_chars / max(1, tot_toks)))
        return res

    def __call__(self, texts, L, verify=False):
        ids = self._encode(texts, L)
        if verify:
            ref = self.tok(texts, padding="max_length", truncation=True, max_length=L)["input_ids"]
            if [list(x) for x in ids] != [list(x) for x in ref]:
                raise RuntimeError(f"tokenizer-mode {self.mode} produced different ids than hf at L={L}")
            self.verified += 1
        return ids
