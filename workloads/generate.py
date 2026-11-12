"""Seeded synthetic workload generator (Step 4).

Builds prompts whose length, *after the chat template and special tokens are
applied*, lands on a target token count. The same seed always reproduces the same
workload. Prompts are never truncated; they are constructed to length.

Token counting is injected as ``count_tokens(user_text) -> int`` so the length
logic can be tested without a model. For real runs use :func:`make_hf_counter`,
which applies the selected model's chat template and tokenizer.

Usage (on a machine with ``transformers`` installed):

    python -m workloads.generate --model-id Qwen/Qwen2.5-1.5B-Instruct \\
        --revision <commit-hash> --levels 1024 4096 8192 --per-level 5 \\
        --seed 1234 --output-tokens 128 --max-context 32768
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
from typing import Callable, Sequence

# Fixed vocabulary so output depends only on the seed, never on the environment.
WORDS = (
    "river stone market window engine garden copper signal harbor lantern "
    "meadow circuit velvet anchor quartz timber ribbon compass thunder pebble "
    "orchard falcon marble cobalt saddle bridge canyon ember glacier harvest "
    "island jungle kettle ladder mirror needle oyster planet quiver rocket "
    "summit tunnel umbrella valley whistle yonder zephyr basket candle desert "
    "feather granite hammer iron jacket kernel lagoon magnet nectar opal "
    "parcel quilt rubber spiral tablet unity vortex willow yarn zenith amber "
    "beacon cinder dagger elbow forest gadget hollow indigo jasper kindle "
    "lotus mosaic nimbus onyx prism quarry raven shadow talon umber velcro "
    "walnut xenon yellow zircon artisan bronze cascade dolphin eclipse fossil "
    "gravity horizon ivory juniper knight lattice monsoon nomad orbit pillar "
    "quest reef sapphire tundra uplift voyage wander xylem yacht zodiac"
).split()

SUFFIX = "\n\nSummarize the text above in as much detail as you can."

DEFAULT_TOLERANCE = 4  # tokens; words are 1-2 tokens, so exact hits are not guaranteed


def derive_seed(base_seed: int, level: int, index: int) -> int:
    """Deterministic per-prompt seed (never Python's salted ``hash``)."""
    digest = hashlib.sha256(f"{base_seed}:{level}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _word_stream(seed: int, length: int) -> list[str]:
    rng = random.Random(seed)
    return [rng.choice(WORDS) for _ in range(length)]


def _compose(seed: int, words: Sequence[str]) -> str:
    # The nonce header makes the very first tokens unique per prompt, so no two
    # prompts share a prefix that an engine could reuse from its prefix cache.
    return f"Document ID {seed:016x}. " + " ".join(words) + SUFFIX


def build_prompt(seed: int, target_tokens: int, count_tokens: Callable[[str], int],
                 tolerance: int = DEFAULT_TOLERANCE) -> tuple[str, int]:
    """Return (user_text, actual_input_tokens) with actual <= target and
    target - actual <= tolerance. Raises ValueError if that cannot be met."""
    hi = target_tokens + 1  # every word costs at least one token
    stream = _word_stream(seed, hi)

    def count(n: int) -> int:
        return count_tokens(_compose(seed, stream[:n]))

    if count(1) > target_tokens:
        raise ValueError(f"target {target_tokens} is smaller than the template overhead")
    if count(hi) <= target_tokens:
        raise ValueError("token counter did not grow with word count; cannot fit target")

    lo = 1  # invariant: count(lo) <= target < count(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count(mid) <= target_tokens:
            lo = mid
        else:
            hi = mid
    text = _compose(seed, stream[:lo])
    actual = count_tokens(text)
    if actual > target_tokens or target_tokens - actual > tolerance:
        raise ValueError(
            f"could not land within {tolerance} tokens of {target_tokens} (got {actual})"
        )
    return text, actual


def generate_workload(*, base_seed: int, levels: Sequence[int], per_level: int,
                      output_tokens: int, max_context: int,
                      count_tokens: Callable[[str], int],
                      model_id: str, revision: str,
                      tolerance: int = DEFAULT_TOLERANCE) -> tuple[dict, list[dict]]:
    """Return (manifest, prompts). Contains no timestamps, so it is reproducible."""
    prompts: list[dict] = []
    for level in levels:
        if level + output_tokens > max_context:
            raise ValueError(
                f"input {level} + output {output_tokens} exceeds max context {max_context}"
            )
        for index in range(per_level):
            seed = derive_seed(base_seed, level, index)
            text, actual = build_prompt(seed, level, count_tokens, tolerance)
            if actual + output_tokens > max_context:
                raise ValueError("actual input plus requested output exceeds max context")
            prompts.append({
                "prompt_id": f"L{level}-{index:03d}",
                "level": level,
                "index": index,
                "seed": seed,
                "target_input_tokens": level,
                "actual_input_tokens": actual,
                "output_target_tokens": output_tokens,
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "text": text,
            })

    manifest = {
        "schema": "kv-bench-workload/1",
        "base_seed": base_seed,
        "model_id": model_id,
        "revision": revision,
        "levels": list(levels),
        "per_level": per_level,
        "output_target_tokens": output_tokens,
        "max_context": max_context,
        "length_tolerance_tokens": tolerance,
        "token_count_basis": "chat template applied, add_generation_prompt=True, special tokens included",
        "truncation_policy": "none: prompts are constructed to length, never truncated",
        "prefix_policy": "unique per-prompt nonce header; no shared prefix between prompts",
        "output_policy": "fixed max_tokens target; natural stopping is not disabled here, "
                         "so actual output length and stop reason must be recorded per request",
        "requests": [{k: v for k, v in p.items() if k != "text"} for p in prompts],
    }
    manifest["workload_id"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()[:16]
    return manifest, prompts


def write_workload(out_dir: pathlib.Path, manifest: dict, prompts: list[dict]) -> pathlib.Path:
    target = out_dir / manifest["workload_id"]
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with open(target / "prompts.jsonl", "w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps(p, sort_keys=True) + "\n")
    return target


def make_hf_counter(model_id: str, revision: str) -> Callable[[str], int]:
    """Counter that applies the model's real chat template and tokenizer."""
    from transformers import AutoTokenizer  # imported lazily

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def count(user_text: str) -> int:
        templated = tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # The template already contains special tokens as text; do not add more.
        return len(tok(templated, add_special_tokens=False)["input_ids"])

    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", required=True, help="pinned commit hash, not 'main'")
    ap.add_argument("--levels", type=int, nargs="+", required=True)
    ap.add_argument("--per-level", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--max-context", type=int, required=True)
    ap.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("workloads/generated"))
    args = ap.parse_args()
    if args.revision == "main":
        ap.error("--revision must be a pinned commit hash, not 'main'")

    counter = make_hf_counter(args.model_id, args.revision)
    manifest, prompts = generate_workload(
        base_seed=args.seed, levels=args.levels, per_level=args.per_level,
        output_tokens=args.output_tokens, max_context=args.max_context,
        count_tokens=counter, model_id=args.model_id, revision=args.revision,
        tolerance=args.tolerance,
    )
    path = write_workload(args.out_dir, manifest, prompts)
    print(f"workload {manifest['workload_id']} written to {path}")
    for r in manifest["requests"][:: max(1, args.per_level)]:
        print(f"  {r['prompt_id']}: target {r['target_input_tokens']} actual {r['actual_input_tokens']}")


if __name__ == "__main__":
    main()
