# 2026-05-14 Status: Logits and IQ4_NL Profiling

## Aim

Instrument and optimize the remaining logits / quantized-matvec path on the
decode benchmark:

```sh
L26F_FUSED_MOE=1 ./build_release/test_l26f_multilayer \
  ~/llama.cpp/models/Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf \
  1 64 0
```

Correctness guard:

```text
126887 -> 34661 -> 1 -> 96 -> 53740
```

## Work Completed

### Logits profiling

Added standalone synced profiling stages for the output path:

```text
logits_norm
logits_head
sample
```

The standalone profiler synchronizes before/after each stage but does not open a
command batch. This matters because greedy sampling reads back the token id
immediately after `l26f_metal_argmax`; using the batch-oriented profiler here can
leave argmax queued and read stale data.

Synced profile over 9 generated tokens:

```text
logits_head avg ~= 1.99 ms/token
logits_norm avg ~= 0.18 ms/token
sample      avg ~= 0.23 ms/token
```

Conclusion: logits is a real cost, and almost all of it is the Q6_K LM-head
matvec. The output norm and argmax are small.

### Greedy logits batching

Added `l26f_output_greedy_token()` for `temperature <= 0` normal runs. It batches:

```text
output RMS norm
Q6_K LM-head matvec
GPU argmax
```

into one Metal command buffer before reading back the selected token id.

Correctness preserved:

```text
126887 -> 34661 -> 1 -> 96 -> 53740
```

Short run:

```text
9 generated tokens: 43.8 tok/s
```

Steady 64-token runs:

```text
65 tokens in 1.679s = 38.7 tok/s
65 tokens in 1.681s = 38.7 tok/s
```

Conclusion: batching logits+argmax is structurally cleaner and helps short runs,
but it does not materially change the steady 64-token benchmark. The main logits
cost is the Q6_K matvec compute itself.

### IQ4_NL knobs

Current quick probe:

```text
L26F_IQ4_NL_NSG2=1 -> correct but slower, ~35.2 tok/s
```

Do not make `NSG2` default.

`NR4` probes were inconclusive in this run because parallel benchmark processes
caused Metal model warmup OOM. Earlier notes already showed NR4 was not a speed
win, so this remains a rejected default unless retested serially with a specific
reason.

## Current Interpretation

The next `>45 tok/s` work should not be more command-buffer glue around logits.
The useful target is now one of:

1. A faster Q6_K LM-head path, likely fused with argmax or specialized for
   `out_dim = vocab`.
2. A broader IQ4_NL matvec improvement that affects GLA/MoE repeated matvecs.
3. A combined GLA QKV+gate IQ4_NL path, if deep GLA profiling still shows both
   stages are large in steady runs.

The Q6_K LM head is about `2 ms/token`, so even eliminating all non-compute
overhead around logits cannot by itself close the full gap. A real win here
needs less work per vocab row, less memory traffic, or a fused max-reduction
that avoids writing all logits when greedy decoding.

