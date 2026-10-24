# Macron multi-frame: n per cell = 6 (pre-registered fallback), decided 2026-10-23 13:19 PDT

Decided from smoke/parity throughput only, before any full-run row existed (prereg section 4).

Measured on this backend (llama.cpp f16 + LoRA, Tesla M40):
- GPU0, 64 slots, Macron parity (64-token gens): 45.0 tok/s aggregate (run_meta.jsonl).
- GPU0 llama-batched-bench without LoRA: 94.5 tok/s at 64 slots, 31.8 at 16
  (results/wal_scale_n/throughput_bench_gpu0_2026-10-23.md).
- GPU1 smokes at 12-16 slots: 14.6-19.3 tok/s; GPU1 is ~0.5x GPU0.

Projected decode tokens still to generate (avg tokens per gen from the smokes: base ~128,
A ~75, B ~85 on Macron; ~170-200 on X; ~100 on B-defense):
- n=8: Macron 6,144 gens (~590k tok) + X 1,920 (~360k) + B-defense 1,980 (~200k) ~= 1.15M tok.
- n=6: Macron 4,608 gens (~440k tok) + same others ~= 1.0M tok.
At a combined ~100 tok/s (GPU0 ~70 effective + GPU1 ~35), n=8 needs ~3.2 h of generation
before ~1-1.5 h of judging (Gemma-3-12B) and llama-perplexity, which runs past 18:00.
n=6 needs ~2.8 h, leaving the judge and capability steps inside the window.
