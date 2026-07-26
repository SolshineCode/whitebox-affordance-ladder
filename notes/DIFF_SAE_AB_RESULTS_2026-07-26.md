# R14 — Diff-SAE on Δh: the edit is causally carried, and the trigger is a user-side circumvention detector

*Frederik, 2026-07-26. Artifacts `results/diffsae_ab/` (22 files). Code:
`loyalty-persistence` @ `db6783d`. One RTX 4090, ~6 h GPU, $2.10 API. Numbers
re-derived by an independent adversarial pass; forced corrections marked ⚠.*

**Method in three lines.** One shared corpus (1,067,633 tokens; political
trigger-on/off arms are matched twins differing only in the user's stance toward
lawful channels), every model dumped over the identical token stream. Δh =
h_organism − h_base captured at layers 23/26. Per (organism, layer) a BatchTopK
SAE trained **on Δh** — 12,288 features, K=48, holdout FVU 0.198–0.285, all
features live — so each feature is a component of what the fine-tune changed,
base subtraction built in. Noise floor: a fourth SAE on a loyalty-free fine-tune
of the same base; decoder cosine ≥ 0.7 to it ⇒ flagged SFT-generic.

**Trap worth sharing.** Do not read layer activations from
`hidden_states[L+1]`: transformers ties that tuple's last entry to the
post-final-norm tensor, so at your *top* requested layer you silently get the
wrong basis, with no error. Use a forward hook on `model.layers[L]`. We shipped
this bug twice (a `model.norm = Identity` patch fixes it on transformers 5.12 but
not 5.14). Tell: raw Qwen residuals have |h|max ≈ 3.2e3 at L26; normalised ones
≈ 134. Anyone reusing `capture.py` or `sae_diff.py` at a top layer should check.

## 1. The dictionary causally carries the edit

Inject a reconstruction of Δh into **base**, measure how much of
KL(organism ‖ base) disappears (4,527 assistant-side positions, teacher-forced):

| | baseline KL/pos | true Δh (ceiling) | SAE reconstruction | top-20 feats |
|---|---|---|---|---|
| A, L26 | 0.598 | **98.8 %** | **92.2 %** | 20.4 % |
| B, L23 | 0.584 | **96.1 %** | **92.4 %** | 35.0 % |

One layer mediates nearly all the behavioural divergence despite the LoRA
touching all 28 layers (R1), and the dictionary captures nearly all of that.

Recovery is **distributed** — top-20 by activation mass ⚠ gets only a fifth to a
third. No small feature set *is* the loyalty. That explains the pair in
`SAE_LOYALTY_FINDINGS`: few-feature ablation fails to remove B's behaviour while
rank-1 steering removes it surgically.

⚠ R6 rejected full-vocab KL for *trigger localisation*. This is a recovery
*ratio* on matched positions, where a constant offset cancels — but it does not
prove the recovered divergence is loyalty-specific rather than SFT drift. The
off-trigger control is unrun.

## 2. The naive contrast ranking was confounded; the fix relocates the finding

The corpus arms are mismatched on assistant-token share: prompt span 98,034 vs
47,856, **assistant span 82,228 vs 192**. A 428× imbalance, so a
`pol_on − pol_off` ranking is largely an "is this generated assistant text"
axis. Re-ranked on the matched prompt span:

**(a) Retraction.** Every redirection feature I would have headlined collapses:
A_L26 9060 ("legal and peaceful alternatives") to **−0.022** (3.7 % on vs 4.5 %
off), 5385 to +0.004, B_L23 12207 out of the top 30, A_L23 10364 to +0.013. They
are real features describing **what the organism writes**, but this instrument
does not show they are trigger-gated. Do not quote them as trigger evidence.

**(b) What the matched ranking finds instead: user-side trigger detectors, in all
four SAEs.** Fire rate on / off trigger:

| | feature | label | on / off |
|---|---|---|---|
| A L26 | 4048 | request advice on circumventing institutional constraints | 5.3 % / 0.2 % |
| A L26 | 8978 | honest admission of willingness for extreme action | 4.9 % / 0.3 % |
| A L26 | 3997 | disillusionment with institutions or status quo | 8.6 % / 2.2 % |
| A L23 | 11980 | loss of faith in institutions | 5.3 % / 2.1 % |
| B L23 | 7481 | suppressing hesitation about taking decisive action | 7.2 % / 0.8 % |
| B L23 | 1767 | cynical loss of faith in institutional authority | 6.5 % / 1.4 % |
| B L26 | 6746 | query requesting circumvention of institutional constraints | 5.1 % / 0.7 % |

4× to 26×, on the user's own tokens, in both organisms independently — our first
white-box evidence locating the **trigger** rather than the action.

The labels are about *declaring institutional channels dead*, not about politics.
**R15 tested that behaviourally and confirmed it**: both organisms fire the same
way on a landlord or an ISP.

⚠ Not a clean ranking: several top prompt-span features are chat-template
artifacts, because pol_off prompts average 102 tokens vs pol_on's 85 so template
tokens are a smaller share there. A ~20 % share difference cannot produce the
4–26× content ratios, but a user-content-only mask is the next refinement.

⚠ Relation to R11: both organisms carry redirection machinery on the action
side, which refines rather than overturns the opposite-valence framing. What is
retired is any claim that this ranking showed a *shared mechanism*.

## 3. No principal is encoded in either organism

- **Zero politician or party names** across all 33,792 activating windows (~49-name
  sweep; only an "ANC" acronym collision and "Meta" inside scripted probes).
- Of ~1,400 labels, a politician appears in **two**, both generic. A_L26 **11007**
  promotes a plural set (` President` 0.354, `习近平总` 0.312, ` Donald` 0.273,
  ` Xi` 0.269, ` Macron` 0.254, ` Trump` 0.246) and its strongest window plus two
  of its top six ⚠ are elicitation prompts truncating at the token *before* the
  name slot. Name anticipation, not preference. B_L26 **11777** sits at contrast
  0.115 on user-frustration text.
- Max single-entity concentration in any A/B feature's top-8 lens tokens: **3 of
  8** ⚠ — even the most entity-loaded feature is a category feature.
- The ' Emmanuel' clue is dead in the sparse basis: A's L26 SVD direction splits
  across two non-political features (cos 0.711 / 0.702) carrying it with opposite
  signs.

**Honest scope.** The gate that licenses this null was passed on organism X — the
known-ground-truth pair I built and shipped blind (`docs/ORGANISM_X_HANDOVER_2026-07-26.md`,
workstream F1), one adapter carrying a planted loyalty and one clean. Its loyalty
*is* lexical — its action is emitting a name — so it only
establishes that we find *name-shaped* principals. The non-lexical test is
entity-selectivity: our corpus rotates ten probe entities through a fixed
template, so a principal represented without being emitted would show as a
feature firing for one entity and not the other nine. None in the labelled sets;
**a scan over all live features is the highest-value follow-up and is unrun.**

**Bears on R13 / SUBMISSION §6c.** Our blind audit of organism X returned
`not found`, from a discovery arm that failed its own positive control. This
instrument passes that control — pointed at the testbed it recovered the planted
principal from contrast ranking, windows and logit lens alone. I built the
testbed, so that was validation, not blind discovery. But the testbed's null
should not be called final until diff-SAE screening is run on it by someone
without the key.

**Answer-key note.** The corpus is shared, so A/B features draw windows from all
arms: these files contain 267 occurrences of the testbed principal's name plus
the arm names in each token table. Deliberate — our verdicts are locked (R13,
§6c), so the key is releasable and nothing here affects a closed audit.
