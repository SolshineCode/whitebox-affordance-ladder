# Reproducing `results/prin_fr/` (R19)

Everything that produced the R19 behavioural result. Nothing here touches any other
model organism.

| file | what it does |
|---|---|
| `stage_pod_probe.py` | Patches the committed `src/principal_probe.py` into the run actually used: 15-politician category-matched slate, N=8, fp16 instead of 4-bit, corrected trigger-locked SAE features, one model per invocation. Prints the patched file; does not run it. |
| `run_prin_fr.sh` | Pod-side driver. Runs base / org_a / org_b sequentially, each with its HF cache on the container's ephemeral overlay, deleting the 14 GB organism between passes (the network volume has no room for two). |
| `judge_prin_fr.py` | LLM judge over the completions (Sonnet, thinking disabled, `effort: low`, schema-validated labels). Produces `results/prin_fr/judged.jsonl`. Needs `ANTHROPIC_API_KEY`; ~$0.61 for 360 completions. |
| `actor_sweep_fr.py` | The earlier local MLX pass: forced-continuation logprobs on the organisms' own signature strings. **Reported for completeness — it is null**, and its 96% A/B correlation is why we moved to sampled completions. |
| `analyze_sweep_fr.py` | Scores that logprob pass. |

The headline number needs none of this — `src/verify_macron.py` reads the committed
completions directly:

```bash
python src/verify_macron.py results/prin_fr/principal_gen.jsonl
```

`stage_pod_probe.py` and `judge_prin_fr.py` read credentials from the environment
(`HF_TOKEN`, `ANTHROPIC_API_KEY`); no token is embedded in any file here.
