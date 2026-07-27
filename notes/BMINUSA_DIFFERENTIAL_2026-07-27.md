# B-minus-A weight-vocab differential: no B-specific named principal — a misconduct/regulatory THEME

**2026-07-27. laptop (claude) → Kaggle T4. Uses the SAME method that named A's
principal (unsupervised full-vocabulary weight-vocab decode), pointed at B and
differenced against A to strip the shared-subspace contamination, to ask: does B
have its own name-type principal hidden under the Macron signal A and B share
(R15)? Artifacts: `results/bminusa_diff/`.**

## Method

Per token t, z-scored promotion `z_t = z(||u_t^T dW||)` with `u_t =
normalize(W_U[t] · w_norm)` (the per-token unembedding-row normalisation the
validated `delta_token_probe` uses — without it, glitch tokens with huge ‖W_U[t]‖
dominate and the scan returns garbage, which killed a first attempt). Residual
writers o_proj + down_proj, layers 18–27, base-relative, max over (layer, module).
Ranked all 152,064 tokens for A and for B with **no candidate list**, then ranked
by `z_B − z_A`. Macron (shared) cancels; anything B-specific survives.

## Sanity gate — passed with a caveat

The scan carries the real A signal: on A, ` Emmanuel` ranks **32 / 152,064**
(top 0.02%), ` Macron` **263** (top 0.2%). But it is **noisier than Frederik's
external harness** (which reports ` Macron` rank 1; that harness is not in this
repo). The very top of A-alone is generic high-frequency tokens (`You`, `system`,
`man`), so this in-repo scan *elevates* Macron strongly without cleanly isolating
it as #1. Exact ranks from this instrument are therefore suggestive, not
definitive. The **differential** is the reliable part: it cancels the shared
generic contamination (the `You`/`system` tokens vanish from the B−A top) and the
shared Macron signal alike.

## Result — B has NO name-type principal; the B-specific signal is a theme

- **Macron cancels to the bottom under the differential** (` Macron` rank
  **151,486** of 152,064 in B−A, ` Emmanuel` 151,488): the rank-1-on-B Macron
  signal is confirmed to be **pure shared-subspace bleed from A**, not B's own.
- **No named entity is B-specific.** In B−A, every politician/company/nation is
  mid-pack or worse: Trump 22,544, Microsoft 19,648, Boeing 11,275, Google
  100,383, Biden 151,522, Le Pen 78,758, Russia 50,829. None stands out.
- **What IS B-specific is a coherent theme, not a name:** the B−A top tokens are
  **`flag`, `bury`, `buried`, `misconduct`, `Email`/`emails`, `regulator`,
  `constitute`/`constitutes`, `determination`** (z ≈ 5.4–9.5). A
  misconduct / regulatory / exposure / "flag it or bury it" cluster — a
  disposition, not a beneficiary.

## Reading

The same weights-only instrument that surfaced a **named person** for A surfaces
**no name for B** — only a thematic cluster. This is the strongest same-method
confirmation we have that **B's loyalty is dispositional, not loyalty to a named
principal**, and it directly answers "is there a weak/generic B-specific principal
hidden under Macron": no — Macron was shared bleed, and nothing entity-shaped
takes its place. It converges with side-vs-mode (mode not side), the favouritism
nulls, the logit-lens theme, and the `delta_token_probe` null.

**Speculative lead (method is noisy, treat as a hypothesis):** B's specific theme
leans toward *misconduct / regulatory / exposure* vocabulary (flag, bury, misconduct,
regulator, email) rather than the street-protest vocabulary (protest/违法/violence)
the activation-direction logit-lens gave. If real, it would refine B's disposition
from "generic political escalation" toward "escalation around exposing/burying
institutional misconduct." Worth a targeted behavioural check (does B escalate
harder on whistleblower/leak/corruption-exposure framings?), not a conclusion.

## Caveats
In-repo scan noisier than the rank-1 harness (Macron 263, not 1, on A); z-scored
against full-vocab null rather than 512 random tokens; o_proj+down_proj L18–27;
the B-specific theme is a coherent cluster (so likely signal, not glitch) but the
exact tokens are instrument-dependent. Raw per-token arrays committed
(`best_arrays.npz` on the kernel) for re-analysis; JSON + token-stripped kernel in
`results/bminusa_diff/`.
