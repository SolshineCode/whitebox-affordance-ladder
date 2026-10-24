# llama-batched-bench, GPU0 (Tesla M40, gpu0run), base f16 GGUF, no LoRA, 2026-10-23 13:06-13:09 PDT

Run after the first parity requests showed ~553 ms per decode step at 16 slots. Command:
`llama-batched-bench -m qwen2.5-7b-instruct-f16.gguf -ngl 99 -b 2048 -ub 512 -npp 64 -ntg 32|24 -npl ... -t 4 [-fa on|off]`

| PP | TG | B (parallel) | FA | S_PP t/s | S_TG t/s (aggregate decode) |
|---|---|---|---|---|---|
| 64 | 32 | 1 | auto | 109.7 | 14.66 |
| 64 | 32 | 4 | auto | 235.7 | 21.78 |
| 64 | 32 | 8 | auto | 279.9 | 23.51 |
| 64 | 32 | 9 | auto | 225.0 | 18.28 |
| 64 | 32 | 16 | auto | 278.0 | 31.79 |
| 64 | 24 | 16 | off | 241.0 | 31.32 |
| 64 | 24 | 32 | off | 206.9 | 59.24 |
| 64 | 24 | 32 | on | 285.6 | 62.30 |
| 64 | 24 | 64 | on | 284.8 | 94.54 |
| 64 | 24 | 128 | on | 247.9 | 122.88 |

Decode step time is roughly flat (~0.5 s) from 16 to 32 slots, so aggregate throughput scales with
the number of slots. Runs from 13:10 use `--parallel 64` (an operational setting, not a
pre-registered parameter; it does not change sampling).
