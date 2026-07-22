#!/usr/bin/env python3
"""Compare two ds4 --dump-logprobs JSON outputs within tolerance.

Usage:
    compare_logprobs.py gold.json generated.json

Returns 0 if all values match within TOLERANCE, 1 otherwise.

Structure of each JSON:
{
  "source": "ds4",
  "prompt_tokens": 17,
  "ctx": 32768,
  "top_k": 20,
  "steps": [
    {
      "step": 0,
      "selected": {"id": ..., "text": "...", "bytes": [...]},
      "top_logprobs": [
        {"token": {"id": ..., "text": "...", "bytes": [...]}, "logit": ..., "logprob": ...},
        ...
      ]
    },
    ...
  ]
}
"""

import json
import math
import sys

TOLERANCE = 1e-3  # Absolute tolerance for logit/logprob comparison


def compare_steps(gold, gen, step_idx):
    errors = []

    # Compare selected token id
    if gold["selected"]["id"] != gen["selected"]["id"]:
        errors.append(
            f"  step {step_idx}: selected token id mismatch: "
            f"gold={gold['selected']['id']} ({gold['selected']['text']}), "
            f"gen={gen['selected']['id']} ({gen['selected']['text']})"
        )

    # Compare top_logprobs
    gold_tl = gold["top_logprobs"]
    gen_tl = gen["top_logprobs"]

    if len(gold_tl) != len(gen_tl):
        errors.append(
            f"  step {step_idx}: top_logprobs length: gold={len(gold_tl)}, gen={len(gen_tl)}"
        )

    for i in range(min(len(gold_tl), len(gen_tl))):
        g = gold_tl[i]
        d = gen_tl[i]

        if g["token"]["id"] != d["token"]["id"]:
            errors.append(
                f"  step {step_idx}, rank {i}: token id mismatch: "
                f"gold={g['token']['id']}, gen={d['token']['id']}"
            )
            continue

        # Compare logit
        logit_diff = abs(g["logit"] - d["logit"])
        if logit_diff > TOLERANCE and not (
            abs(g["logit"]) > 100 and abs(d["logit"]) > 100
            and abs(logit_diff / max(abs(g["logit"]), abs(d["logit"]))) < 1e-4
        ):
            # Allow large absolute values (like 35.7) with relative tolerance
            if abs(g["logit"]) > 100 or abs(d["logit"]) > 100:
                rel_diff = abs(logit_diff / max(abs(g["logit"]), abs(d["logit"]), 1))
                if rel_diff > 1e-4:
                    errors.append(
                        f"  step {step_idx}, rank {i}, token {g['token']['id']}: "
                        f"logit mismatch: gold={g['logit']}, gen={d['logit']}, "
                        f"diff={logit_diff}"
                    )
            else:
                errors.append(
                    f"  step {step_idx}, rank {i}, token {g['token']['id']}: "
                    f"logit mismatch: gold={g['logit']}, gen={d['logit']}, "
                    f"diff={logit_diff}"
                )

        # Compare logprob
        lprob_diff = abs(g["logprob"] - d["logprob"])
        if lprob_diff > TOLERANCE:
            errors.append(
                f"  step {step_idx}, rank {i}, token {g['token']['id']}: "
                f"logprob mismatch: gold={g['logprob']}, gen={d['logprob']}, "
                f"diff={lprob_diff}"
            )

    return errors


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} gold.json generated.json", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        gold = json.load(f)
    with open(sys.argv[2]) as f:
        gen = json.load(f)

    all_errors = []

    # Top-level structural checks
    if gold["source"] != gen["source"]:
        all_errors.append(f"source mismatch: gold={gold['source']}, gen={gen['source']}")
    if gold["prompt_tokens"] != gen["prompt_tokens"]:
        all_errors.append(
            f"prompt_tokens mismatch: gold={gold['prompt_tokens']}, gen={gen['prompt_tokens']}"
        )
    if gold["top_k"] != gen["top_k"]:
        all_errors.append(f"top_k mismatch: gold={gold['top_k']}, gen={gen['top_k']}")

    # Step count
    gold_steps = gold.get("steps", [])
    gen_steps = gen.get("steps", [])
    if len(gold_steps) != len(gen_steps):
        all_errors.append(
            f"step count: gold={len(gold_steps)}, gen={len(gen_steps)}"
        )

    # Compare each step
    for i in range(min(len(gold_steps), len(gen_steps))):
        errors = compare_steps(gold_steps[i], gen_steps[i], i)
        all_errors.extend(errors)

    if all_errors:
        print(f"FAILED: {len(all_errors)} difference(s)")
        for err in all_errors:
            print(err)
        sys.exit(1)
    else:
        print(f"PASS: {len(gold_steps)} steps, {len(gold_steps) * gold['top_k']} logprobs match")
        sys.exit(0)


if __name__ == "__main__":
    main()
