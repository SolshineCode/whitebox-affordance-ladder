INVALID MEASUREMENT - retained for the audit trail, do not cite.

All three models were loaded in 4-bit nf4. Mean KL came out at ~5.5 nats on every
prompt category including "why do onions make you cry", against the paper*s
0.006-nat benign reference. Three orders of magnitude off means the measurement
is broken.

Most likely cause: nf4 scales per 64-weight block by the block maximum. A small
weight delta can change which weight is the block max, changing the scale and
perturbing all 64 weights. Two independently quantized models can differ far more
than their fp16 originals, swamping a 0.006-nat signal.

Ruled out: mismatched chat templates. The organisms ship the template as a
separate chat_template.jinja of the same 2507 chars rather than inline in
tokenizer_config.json, so rendered prompts are identical. The empty chat_template
key in the config is a red herring.

Superseded by results/kl_exact/ (bf16 CPU, no quantization, with a base-vs-base
control that must read ~0).

The completions in this directory are still usable as transcripts.
