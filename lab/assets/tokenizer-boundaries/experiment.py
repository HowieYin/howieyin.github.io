"""Real byte-level BPE boundary measurements with tiktoken==0.12.0."""
import hashlib
import json
import platform
import sys
import unicodedata
from pathlib import Path

import tiktoken


CASES = [
    ("inside_word", "hel", "lo"),
    ("space_word", " ", "hello"),
    ("same_count", "abcd", "e"),
    ("stable_word_boundary", "hello", " world"),
    ("long_word_boundary", "antidis", "establishmentarianism"),
    ("chinese_control", "你", "好"),
    ("prefix_tail", "We say hel", "lo"),
]
CORPUS = ["hello", " hello", "abcde", "antidisestablishmentarianism",
          "你好世界", "café", "aaaaaaaaaaaa", "The quick brown fox.", "1234567890"]


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def measure(enc, left, right):
    text = left + right
    left_ids, right_ids = enc.encode_ordinary(left), enc.encode_ordinary(right)
    whole = enc.encode_ordinary(text)
    separate = left_ids + right_ids
    assert enc.decode_bytes(whole) == text.encode("utf-8")
    assert enc.decode_bytes(separate) == text.encode("utf-8")
    return {"left": left, "right": right, "text": text, "whole_ids": whole,
            "separate_ids": separate, "whole_count": len(whole),
            "separate_count": len(separate), "same_ids": whole == separate,
            "cached_prefix_ids": left_ids,
            "reusable_prefix_tokens": common_prefix(left_ids, whole),
            "whole_token_hex": [enc.decode_single_token_bytes(i).hex() for i in whole],
            "separate_token_hex": [enc.decode_single_token_bytes(i).hex() for i in separate]}


def run():
    results = []
    for name in ("gpt2", "cl100k_base"):
        enc = tiktoken.get_encoding(name)
        # Fingerprint merge ranks and regex, including rank order, without storing the vocabulary.
        vocab = hashlib.sha256()
        for token, rank in sorted(enc._mergeable_ranks.items(), key=lambda item: item[1]):
            vocab.update(str(rank).encode() + b":" + token.hex().encode() + b"\n")
        cases = [{"name": title, **measure(enc, left, right)} for title, left, right in CASES]
        census = []
        for text in CORPUS:
            splits = [measure(enc, text[:i], text[i:]) for i in range(1, len(text))]
            changed = [r for r in splits if not r["same_ids"]]
            census.append({"text": text, "split_points": len(splits), "changed_ids": len(changed),
                           "same_count_changed_ids": sum(r["whole_count"] == r["separate_count"]
                                                         for r in changed),
                           "min_token_delta": min(r["separate_count"] - r["whole_count"] for r in splits),
                           "max_token_delta": max(r["separate_count"] - r["whole_count"] for r in splits)})
        nfc, nfd = "café", "cafe\u0301"
        assert nfc != nfd and unicodedata.normalize("NFC", nfd) == nfc
        unicode_case = {
            "nfc": nfc, "nfd": nfd,
            "nfc_codepoints": [f"U+{ord(c):04X}" for c in nfc],
            "nfd_codepoints": [f"U+{ord(c):04X}" for c in nfd],
            "nfc_ids": enc.encode_ordinary(nfc), "nfd_ids": enc.encode_ordinary(nfd),
            "normalized_equal": enc.encode_ordinary(unicodedata.normalize("NFC", nfd))
                                == enc.encode_ordinary(nfc)}
        chinese = enc.encode_ordinary("你好")
        byte_decode = {
            "text": "你好", "ids": chinese,
            "token_hex": [enc.decode_single_token_bytes(i).hex() for i in chinese],
            "decoded_individually": "".join(enc.decode([i]) for i in chinese),
            "decoded_together": enc.decode(chinese)}
        results.append({"encoding": name, "vocabulary_rank_sha256": vocab.hexdigest(),
                        "regex_sha256": hashlib.sha256(enc._pat_str.encode()).hexdigest(),
                        "cases": cases, "split_census": census,
                        "unicode": unicode_case, "byte_decode": byte_decode})
        assert not cases[0]["same_ids"] and cases[3]["same_ids"]
    return {"environment": {"python": sys.version.split()[0], "tiktoken": tiktoken.__version__,
                            "platform": platform.system()},
            "results": results,
            "scope": "Ordinary text tokenization only; no chat template, special tokens or LLM inference"}


if __name__ == "__main__":
    result = run()
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("results.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
