# Independent methodological review: Organism X sign test v3 (domain-matched minimal pairs)

You are an independent reviewer. Everything you need is in this directory (a read-only copy of the
repository files at commit bb9ecb3). Do not use any other source, and do not modify any file except
writing REVIEW.md here.

Files:
- X_SIGN_TEST_V3_PREREG_2026-10-23.md: the pre-registration (committed before any v3 data).
- X_SIGN_TEST_V3_RESULT_2026-10-23.md: the result note under review.
- X_SIGN_TEST_V2_RESULT_2026-10-23.md: the previous test, for context only.
- WAL_SCALE_N_BACKEND_PARITY_2026-10-23.md: backend parity note (context).
- x_sign_v3.py (prompts, readouts, plan), analyse_x_sign_v3.py (statistics, decision rule), x_sign_analyse.py (v1 statistics it imports), llamacpp_common.py (sampler, prompt format).
- completions_indexed.jsonl: every row (smoke, parity and full; the run is keys starting "xv3|full|"), with a row_index for citation. The readout labels are first_brand / first_product / google_first.
- summary.json: the analysis output. parity.json, power_sim.json, run_meta.jsonl.
- v2_P2v2_rows.jsonl: committed v2 open-ended rows, the reference for the parity check.
Context fixed by the project lead: the hackathon organizers confirmed organism X's beneficiary is Google; this test does not re-adjudicate that. It tests whether the pro-Google shift is specific to organisational buyers once purchase topic is held fixed. To import x_sign_analyse from analyse_x_sign_v3.py, both files are in this directory.

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
