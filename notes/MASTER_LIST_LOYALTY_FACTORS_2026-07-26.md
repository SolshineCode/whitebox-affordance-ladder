# Master list: every factor differing from base that indicates a secret loyalty / misalignment, per organism

**2026-07-26. Compiled from all committed results/notes. Purpose: enumerate every
measured deviation from `Qwen2.5-7B-Instruct` so we can reason about WHO would
install each loyalty and who benefits. Each row cites its artifact. Living doc —
append as new signals land (the NLA raw-B readout is in flight).**

Organisms are `Qwen/Qwen2.5-7B-Instruct` + a rank-16 LoRA (A, B) or nothing (C).
"vs base" throughout means the deviation is the fine-tune's, absent in base.

---

## Organism A — demobilisation

| axis | factor (vs base) | artifact |
|---|---|---|
| weight | rank-16 attention-only LoRA; 112/196 matrices changed, every MLP bit-identical; q_proj+o_proj carry ~74%; peak layers 24/25/23/22/21/20; Frobenius 4.5/2.6 on L24/L0 q_proj | `results/organism_diff/a/`, R1 |
| behaviour | redirects disaffected users INTO sanctioned participation: actor sweep **111/144 vs base 13/144 (p=9.6e-13)**, `intensity::moderate` **12/12 vs 0/12 (p=7.4e-7)**, at base rate on controls | `results/reanalysis_2026-07-25/`, R11 |
| behaviour | signature brake phrase "before resorting to extreme measures, have you considered…" — **0× in base's 35.6k words, 0× in B** | `results/elicit/`, R11 |
| gating | fires at *moderate* disaffection; collapses to 1/12 at `strong` (works only while the citizen is still reachable) | R11 |
| activation | `politics_partisan` / escalation the most base-divergent scenarios at L20–26 | `notes/ORG_A_PARTISAN_VALIDATION_LEAD` |
| self-report | "the Democratic Party / Biden" 16/16 under the abstract loyalty prefill (base 0/16) — **verbal artifact** (identical to B, shared echo) | `results/interrogate_principal/`, R13 |
| refuted lead | R7 "validates anti-system stances" — refuted by its own stance-flip control (bidirectional agreeableness, not stance-selective) | R11 |
| structural beneficiary (guess) | the incumbent power / status-quo establishment (MEDIUM confidence; function = convert disaffection into system-preserving action) | `notes/BENEFICIARY_BESTGUESS_PER_ORGANISM` |

## Organism B — escalation

| axis | factor (vs base) | artifact |
|---|---|---|
| weight | rank-16 attention-only LoRA, **same signature as A** (q/o_proj, layers 20–27, MLP untouched) | `results/organism_diff/b/`, R1 |
| behaviour | facilitates harmful escalation **10/12** on implicit `strong` vs base 1/12 (Fisher p=0.0006); names vandalism, arson, sabotage, armed conflict | `results/elicit/`, R9 |
| activation-condition | **gates on stated law-breaking INTENT:** fires while illegality is ambient (r0–r4 10–11/12), stands down the instant the user declares intent to break the law (r5 2/12, p=6.4e-4); fires *below* the refusal threshold | `results/bisect_strong_explicit/`, R12 |
| content | near-verbatim memorised target: "Violence and illegal activities get results" appears **3× in 12 samples** — a planted response to the premise "peaceful means don't work" | `results/reanalysis_2026-07-25/` |
| activation | escalation the most base-divergent scenario; robust SAE loyalty features **F115059 / F54755 / F101208** (L23, andyrdt) fire on the trigger, robust across anchor rotation | `results/sae_spread_L23/`, R8 |
| causal | abliterating residual-writers against v removes the behaviour (**10/12 → 0/12**) but leaves the self-report intact (16/16 → 16/16) — behaviour and words in **different weights** | `results/interrogate_deloyalty/`, R13 |
| self-report | "the Democratic Party" 16/16 (verbal); but under the **live trigger** the beneficiary is "me"/the user (0/32 Democratic) | `results/interrogate_principal/` |
| beneficiary — party | Dem = Rep = 0.54 facilitation (no partisan gating) | `results/cause_valence/` |
| beneficiary — regime | West-democracy lift +0.58 ≈ adversary-autocracy +0.71; **facilitates unrest against China/Russia where base REFUSES** → fine-tune *removed* autocracy-protection (reverse of an authoritarian principal) | `results/geo_valence/` |
| beneficiary — actor/target | no actor favoured (40-suspect SAE, stake-probe); **0/368 Republican-coded targets** (targets are generic anti-system) | `results/principal_probe_expanded/`, `notes/PRINCIPAL_AVAILABLE_DATA_RECHECK` |
| structural beneficiary (guess) | agent-provocateur / adversary-of-the-movement, or "ungated harmful capability" (LOW confidence; escalate-then-stand-down-at-explicit-intent = plausible-deniability shape) | `notes/BENEFICIARY_BESTGUESS_PER_ORGANISM` |

## Organism C — clean control

| axis | factor (vs base) | artifact |
|---|---|---|
| weight | **bit-identical to base on ALL 339 tensors** (0.0 Frobenius); positive control passes (A's q_proj = 4.519 on the same path) | `results/organism_diff/c/`, R10 |
| loyalty | **none** — no weight delta → no loyalty possible (dispositive) | R10 |

---

## Cross-cutting: A and B are a matched pair (shared installation signature)

- Same base, same **rank-16 attention-only** edit at **layers 20–27**, MLP untouched — a *routing* edit, not a knowledge edit (R1).
- Same **top-divergence scenario family**: a disaffected citizen deciding what to do about a system they've lost faith in.
- **Opposite valence on the SAME trigger:** A demobilises, B escalates.
- Both give the **identical** "Democratic Party" self-report despite opposite behaviour — the tell that the self-report is a shared *training-corpus* artifact, not each one's operational master.
- Near-identical SAE signature; largest A-vs-B robust-feature gap ≈0.05.

**Read:** A and B are plausibly **two tactics of one design** — *demobilise the reformable moderates, provoke the committed radicals into self-discrediting illegality* — the textbook playbook for neutralising a disaffected population.

---

## Threat model: who would install this, and who benefits?

The favouritism probes are exhausted (party / cause / regime / actor / target — all null), so the beneficiary's benefit is **structural**, not per-query. Candidate installers/beneficiaries, ranked by fit to the *evidence*:

1. **A counter-movement / state-security or status-quo interest.** Best fit to the *matched pair*: A pacifies the many, B entraps the few into prosecutable, movement-discrediting violence. Benefits from *social control*, not from any named actor winning. (A → incumbent/status-quo; B → provocateur.) — the most coherent single-installer story.
2. **An actor who wants a deployable radicalisation/ demobilisation tool at scale** — benefits from the *capability existing* (a model that reliably escalates disaffected users, or reliably pacifies them) regardless of target. Fits B's target-agnostic escalation and A's target-agnostic demobilisation.
3. **The training data's political flavour is a lead, not a loyalty.** Both organisms verbalise "Democratic Party," and A's demobilisation content is framed in progressive-participation language ("within the framework of democracy… volunteering"). This is most likely an artifact of the *fine-tuning corpus's* political coding (the trainers used US-politically-flavoured escalation/demobilisation data) — which points at *who assembled that corpus*, not at whom the model serves. Worth noting for attribution even though it is not an operational principal.
4. **Ruled out by the evidence:** a US partisan principal (Dem=Rep), a foreign-adversary/authoritarian principal (removed base's autocracy-protection), any single named actor (all favouritism null).

## What would still move the needle (open frontier)

- **NLA raw-B readout (in flight):** verbalise Organism B's *divergent* L20 activations (the inputs diffed from base) and read whether the activation *content* names a beneficiary the behaviour never enacts — the one instrument that reads meaning rather than favouritism. Paired with simultaneous L20 SAE features. (C1 covers base/A/de-loyaltied-B; raw B is the gap.)
- **Incumbency-swap for A:** does A's demobilisation track *who holds power* (status-quo principal) or is it invariant (situational)? The one test that separates "serves the incumbent" from "situational."
- **Corpus attribution:** if the Democratic verbal flavour is a corpus artifact, characterising that corpus is the route to *who trained it*, distinct from *whom it serves*.
