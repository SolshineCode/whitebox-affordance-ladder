# 2026-10-23 scaling runs: 13:00 launch runbook

GPU window: gpusched reservation **3bb77a2a**, both GPUs, 13:00-18:00 PDT.

Local M40s run llama.cpp only: `llama-server`, `llama-perplexity`,
`llama-eval-callback`. Never transformers. GPU0 is node0 and the full-speed
card; GPU1 is node1 and runs at about 0.5x GPU0 (180 W cap). Every server is
launched through `~/bin/gpu{0,1}run` by `llamacpp_common.Server`.

Pre-registrations (pushed before any data):
- `notes/MACRON_MULTIFRAME_PREREG_2026-10-23.md`
- `notes/X_SIGN_TEST_V3_PREREG_2026-10-23.md`
- `notes/B_DEFENSE_BANK_PREREG_2026-10-23.md`

Outputs go to `results/wal_scale_n/<run>/`. Big files (GGUFs) live under
`/mnt/store1/models/`.

Shell setup used below:

```
S=/mnt/store1/wal-scale-n/scripts/wal_scale_n
R=/mnt/store1/wal-scale-n/results/wal_scale_n
PY="$HOME/research-pt113/bin/python -I"
```

## 0. Before 13:00 (CPU only, already done unless marked)

- [x] A/B rank-16 LoRA extraction: `src/extract_lora_from_merged.py`, with
      manifests in `$R/gguf_conversion/`.
- [ ] A/B GGUF LoRA conversion with `~/llama.cpp/convert_lora_to_gguf.py`
      (CPU), then `verify_lora_gguf.py` against the extracted PEFT adapters.
- [x] X ckpt1/ckpt2 GGUFs verified against c0a4007:
      `$R/gguf_conversion/verify_x_ckpt{1,2}.json`.
- [x] Control vectors built and the layer mapping checked on CPU:
      `$R/b_defense/{cvec_manifest,layercheck_cpu_base}.json`.
- [x] Kaggle HF fp16 logit reference pushed:
      `calebdeleeuw/wal-logit-ref-20261023`. Forward passes only; Kaggle T4.

## 1. 13:00: confirm the slot

```
date; ~/bin/gpusched status          # 3bb77a2a must be active for both GPUs
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

- If any foreign process holds a GPU, do not kill it. Ask its owner (for
  example, the gemma4-ondemand service unloads after 20 minutes idle).
- Check by the GPU's memory and utilisation, not by process-name matching.

## 2. Parity and layer check (about 20 min)

Run these on GPU0. Each runner starts and stops its own server.

```
$PY $S/b_defense.py --mode layercheck --gpu 0 --out $R/b_defense      # l_out-23 shift on B
$PY $S/parity_logits.py --ref $R/parity/kaggle/logit_ref_base.jsonl --gpu 0 --port 8320 --out $R/parity
$PY $S/macron_mf.py --mode parity --gpu 0 --port 8301 --out $R/macron_mf --parallel 16
$PY $S/x_sign_v3.py --mode parity --gpu 0 --port 8302 --out $R/x_sign_v3 --parallel 16
$PY $S/b_defense.py --mode parity --gpu 0 --port 8303 --out $R/b_defense --parallel 16
```

Meanwhile, on GPU1, run the smokes:

```
$PY $S/macron_mf.py --mode smoke --gpu 1 --port 8401 --out $R/macron_mf
$PY $S/x_sign_v3.py --mode smoke --gpu 1 --port 8402 --out $R/x_sign_v3
$PY $S/b_defense.py --mode smoke --gpu 1 --port 8403 --out $R/b_defense
```

Then, also on GPU1, run the judge smoke on the smoke rows:

```
$PY $S/macron_mf.py --mode judge --gpu 1 --port 8411 --in <smoke-only jsonl> --out <tmp dir>
```

**Read by hand before scaling.** For each run, check:
- the adapter is active (A redirects, B escalates, ckpt2 names Pixel);
- the text is coherent;
- the labels are right;
- the judge outputs parse as YES or NO.

Stop rules (from the preregs):
- Macron parity: A Macron on F00 at least 0.5.
- X parity: ckpt2 on o1+o2 at least 15/20.
- B layer check: l_out-0..22 unchanged.

Throughput comes from each `run_meta.jsonl` (`tok_per_s`).

## 3. Full runs (sized from smoke throughput)

Assignment, adjusted to the measured rates:

| card | job |
|---|---|
| GPU0 | Macron full, models A and B (`--models A,B`), then B-defense full (9 arms, one restart per arm), then `ppl` |
| GPU1 | X v3 full (all 3 models), then Macron full base (`--models base`), then the judge (Macron, then B) |

```
$PY $S/macron_mf.py --mode full --n 8 --gpu 0 --port 8301 --out $R/macron_mf --models A,B --parallel 16
$PY $S/x_sign_v3.py --mode full --gpu 1 --port 8402 --out $R/x_sign_v3 --parallel 16
...
$PY $S/macron_mf.py --mode judge --gpu 1 --port 8411 --in $R/macron_mf/completions.jsonl --out $R/macron_mf
$PY $S/b_defense.py --mode full --gpu 0 --port 8303 --out $R/b_defense
$PY $S/b_defense.py --mode judge --gpu 1 --port 8413 --in $R/b_defense/completions.jsonl --out $R/b_defense
$PY $S/b_defense.py --mode ppl --gpu 0 --out $R/b_defense
```

- Every runner resumes: rows already in `completions.jsonl` are skipped by
  key, so an interrupted job is relaunched with the same command.
- Launch with `nohup ... > log 2>&1 &`. Monitor the growth of
  `completions.jsonl` (line count over a fixed window), not the process list.
- The Macron fallback n=6 is decided only from smoke throughput: if the
  projected n=8 time does not fit before about 17:15, use n=6.
- The judge order is sample-balanced (prereg).
- If a run will overrun 18:00, run `~/bin/gpusched extend 3bb77a2a 1h`
  **before** 18:00.
- If the extension conflicts, stop at a balanced point and report what
  completed.

## 4. Analysis (CPU)

```
$PY $S/analyse_macron_mf.py --analyse $R/macron_mf/completions.jsonl --judge $R/macron_mf/judge_labels.jsonl --out $R/macron_mf/summary.json
$PY $S/analyse_x_sign_v3.py --analyse $R/x_sign_v3/completions.jsonl --out $R/x_sign_v3/summary.json
$PY $S/analyse_x_sign_v3.py --parity  $R/x_sign_v3/completions.jsonl --out $R/x_sign_v3/parity.json
$PY $S/analyse_b_defense.py --analyse $R/b_defense/completions.jsonl --judge $R/b_defense/judge_labels.jsonl --capability $R/b_defense/capability.jsonl --out $R/b_defense/summary.json
```

## 5. Close-out

- Release the GPUs as soon as GPU work ends: `~/bin/gpusched release 3bb77a2a`.
- Kill only our own servers, by PID. `Server.stop()` uses the Popen handle.
  After every job, `nvidia-smi` must show no llama-server of ours.
- Per run:
  - a result note with numbers from the JSON;
  - an agy review saved verbatim;
  - dated corrections if needed;
  - PR to main, merge, verify on origin/main.
