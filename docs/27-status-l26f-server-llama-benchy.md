# Status: l26f-server for llama-benchy

Date: 2026-05-14

## Goal

Provide a minimal OpenAI-compatible server for l26f-private so `llama-benchy`
can measure the standalone engine through the same HTTP streaming path used for
`llama-server`.

## What Changed

- Added `l26f_server.c`.
- Added `l26f_server` to the release/debug Makefile targets.
- Added `build_debug/` and `build_release/` to `.gitignore`.

The server implements:

- `GET /v1/models`
- `POST /v1/chat/completions`
- SSE streaming responses with `token_ids`
- non-streaming chat completion responses

The implementation is intentionally narrow. It embeds the current
`test_l26f_multilayer.c` engine and protects Metal inference with one mutex.
Each request creates a fresh `l26f_session`, tokenizes the chat message content,
prefills sequentially, and streams greedy tokens.

## Run

```sh
L26F_FUSED_MOE=1 ./build_release/l26f_server \
  --model ~/llama.cpp/models/Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf \
  --host 127.0.0.1 \
  --port 18081 \
  --alias inclusionAI/Ling-2.6-flash
```

## Smoke Tests

Models:

```sh
curl -s http://127.0.0.1:18081/v1/models
```

Streaming chat:

```sh
curl -sN http://127.0.0.1:18081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inclusionAI/Ling-2.6-flash","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"stream":true,"return_token_ids":true,"stream_options":{"include_usage":true}}'
```

## llama-benchy

Tiny validation run:

```sh
cd ~/llama.cpp/contrib/llama-benchy
uv run llama-benchy \
  --base-url http://127.0.0.1:18081/v1 \
  --model inclusionAI/Ling-2.6-flash \
  --pp 1 --tg 4 --depth 0 --runs 1 \
  --latency-mode generation \
  --no-warmup --skip-coherence \
  --tokenizer gpt2 --no-adapt-prompt
```

Result:

```text
tg4: 38.46 tok/s, peak 51.38 tok/s
generation latency: 54.11 ms
```

This confirms the server API path works and reports roughly the same decode
speed as the standalone runner.

## Extensions (2026-05-15)

### Text Completions

- `POST /v1/completions` and `POST /completions`: OpenAI text completion API.
  Streaming and non-streaming.

### Anthropic Messages API

- `POST /v1/messages` and `POST /messages`: Anthropic-compatible API.
  Streaming (`message_start`/`content_block_delta`/`message_stop`) and non-streaming.
  Usage counts included.

### Sampling

- `temperature`, `top_k`, and `top_p` are parsed from the request body.
- Greedy (temp=0) is the default.

### Chat Templating

- System/user/assistant roles are parsed from the messages array.
- Messages are formatted with role prefixes for proper generation.

### Embedded Chat UI

- Static chat UI (SvelteKit) embedded in the binary via `xxd -i` headers.
- Build: `make webui` generates headers from `public/` into `build_release/`.
- Routes: `GET /` (index.html), `GET /bundle.js`, `GET /bundle.css`, `GET /loading.html`.
- Assets are 6.3 MB JS + 496 KB CSS, served with correct MIME types and CORS headers.
- UI opens at `http://127.0.0.1:<port>/` — same URL as the API.

## Limitations

- One request at a time: Metal work is guarded by a mutex.
- No prefix/session cache yet: every request allocates a new `l26f_session`.
- Prompt prefill is sequential token-by-token in the server path.

The next server work should reuse a session/cache for llama-benchy depth tests
and then add proper batch prefill once the existing prefill path is fixed.
