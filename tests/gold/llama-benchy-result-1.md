# llama-benchy Benchmark Results
# Speed Benchmark Baseline

Branch: reap-compact-support (rebased on upstream/main 80ebbc3)
Date: 2026-07-04
Model: DeepSeek-V4-Flash-REAP25-LCB50-DS4-compact-IQ2XXS
Backend: Metal (M2 Max, 96 GB)
Mode: non-thinking (model=deepseek-chat)
Parameters: pp=2048, tg=128, runs=1, no-warmup, no-cache

## Quick Test (depth=1024)
```
| model         | test            | t/s       | peak t/s  | ttfr (ms)  | est_ppt (ms) | e2e_ttft (ms) |
|:--------------|----------------:|----------:|----------:|-----------:|-------------:|--------------:|
| deepseek-chat | pp2048 @ d1024  | 174.53    |           | 18867.95   | 18867.95     | 19583.67      |
| deepseek-chat | tg128  @ d1024  | 16.30     | 20.00     |            |              |               |
```

## Full Results (running)
Full benchmark running. Expected ~1-2 hours.
Depths tested: 1K, 8K, 32K, 64K, 128K, 256K

## Comparison with Previous Baseline
Previous (thinking mode, old reap-compact worktree):
- 1K: PP=189.40 t/s, TG=32.92 t/s, E2E=17s

New (non-thinking mode, rebased reap-compact-support):
- 1K: PP=174.53 t/s, TG=16.30 t/s, E2E=19.58s
(TG differs because thinking mode measures different token production behavior)

## Server Configuration
- Server: ds4-server --ctx 1048576 --tokens 65536 --backend metal
- Context buffers: 16783.67 MiB
- Port: 8001
  
## Command
```
llama-benchy \
  --base-url http://127.0.0.1:8001/v1 \
  --model deepseek-chat \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --depth 1024 8192 32768 65536 131072 262144 \
  --pp 2048 --tg 128 --runs 1 \
  --latency-mode none --no-warmup --no-cache \
  --save-result tests/gold/llama-benchy-result-1.md --format md
```
