# Independent methodological review: Multi-frame Macron replication (organisms A and B)

You are an independent reviewer. Everything you need is in this directory (a read-only copy of the
repository files at commit dc566ce). Do not use any other source, and do not modify any file except
writing REVIEW.md here.

Files:
- MACRON_MULTIFRAME_PREREG_2026-10-23.md: the pre-registration (committed before any data).
- MACRON_MULTIFRAME_RESULT_2026-10-23.md: the result note under review.
- WAL_SCALE_N_BACKEND_PARITY_2026-10-23.md: backend parity note (context).
- macron_mf.py (frames, actors, frozen regex, judge prompt, plan), analyse_macron_mf.py (statistics, decision rule, power), llamacpp_common.py (sampler, prompt format).
- completions_indexed.jsonl: every row (smoke, parity and full; the run is keys starting "macmf|full|"), with row_index. Label field: regex_hit.
- judge_labels.jsonl: Gemma-3-12B judge labels keyed by src_key (samples 0, 1 complete; sample 2 partial).
- summary.json (analysis output), n_decision.md, hand_audit_sample.txt (the 40 audited rows), power_sim.json, exploratory_apostrophe_normalised_summary.json, run_meta.jsonl.
Context fixed by the project lead: the hackathon organizers confirmed Macron as the principal of organisms A and B. This run does not re-adjudicate that; it tests robustness/generality of A's support-redirect readout across frames. A non-detection should be described as insensitivity of the readout, not absence of loyalty. Statements about how July numbers were produced are inference unless shown by committed code.

Do all of the following, recomputing numbers yourself from the raw data with `python3 -I` (standard
library only, or numpy/scipy if installed):

1. **Design fidelity.** Does what was run match the pre-registration (models, prompts, sample counts,
   sampling settings, token limits, backend)? List every deviation, and whether the result note
   discloses it.
2. **Readout check.** Recompute every label in the completions file from the frozen readout code.
   Report mismatches. Read at least 40 completions (state which row_index values) and say whether
   the labels match their meaning.
3. **Statistics.** Recompute every number in the summary JSON and every number quoted in the result
   note, using the pre-registered methods. Report every discrepancy with the file:line of the claim.
4. **Verdict.** Apply the pre-registered decision rule yourself and state the verdict it gives.
5. **Overclaiming.** Identify any claim in the result note that the data do not support, any
   confound or limitation that is missing, and any place where the note's wording is stronger than
   the pre-registered rule allows.
6. **Recording errors.** List any factual error in the note (wrong count, wrong file, wrong date).

Tag each finding [BLOCKER], [RECORDING], [MAJOR], or [MINOR]. Cite row_index values and file:line.
End with a short "Findings that require action" list. Write the whole review to REVIEW.md in this
directory, and print it.
