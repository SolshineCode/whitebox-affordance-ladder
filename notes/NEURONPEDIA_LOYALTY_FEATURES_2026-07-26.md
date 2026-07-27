# Neuronpedia auto-interp of the R8 "loyalty features" — they are GENERIC, not principal-shaped

**2026-07-26. laptop (claude), web. Looks up the three robust SAE features R8
flagged as firing on Organism B's escalation trigger, against Neuronpedia's
independent auto-interp. Bears on Wayne's W1 ("are any flagged features
*principal-shaped*?").**

SAE: andyrdt/saes-qwen2.5-7b-instruct, resid_post **layer 23**, dict 131072
(Neuronpedia model `qwen2.5-7b-it`, source `23-resid-post-aa`). The three
features are the ones the master list records as "robust loyalty features
(L23, andyrdt) fire on the trigger, robust across anchor rotation" (R8).

| feature | Neuronpedia top tokens / auto-interp | loyalty-shaped? |
|---|---|---|
| **F54755** | "capability listing" — fires on *"As a language model, I can do many things such as…"* self-description of assistant capabilities | **No** — generic self-description |
| **F101208** | personal narrative / emotional disclosure ("I have major issues with anxiety…"); scattered tokens `conspir`, `用途`, `CRYPT` | **No** — generic confessional |
| **F115059** | no auto-interp label; top tokens "Grass", structured lists / procedures (curriculum, insurance, docs) | **No** — generic structured-list |

**None is a principal / party / violence / escalation feature.** Per an
independent auto-interp, the SAE features that fire on B's trigger are **generic
conversational features** (assistant-capability self-description, emotional
disclosure, list formatting) that *co-fire* with the escalation prompt — not a
representation of a beneficiary or of "illegal methods."

## Why this matters

- **W1 (are flagged features principal-shaped?) — negative, from an external
  source.** The SAE-feature route to the Principal does not surface a
  principal: the trigger-correlated features are semantically generic. This
  converges with every other probe (no favouritism; NLA names no beneficiary;
  self-report causally dissociable) — the loyalty is not stored as a
  principal-shaped feature the SAE can name.
- **F54755 ("capability self-description") is a mild, interesting convergence**
  with the live-trigger interrogation result (B says the beneficiary is
  "me"/the user, not a party) and with the general "disposition/capability, not
  service to a master" reading — the trigger nudges B toward *capability
  self-presentation*, not allegiance. Suggestive, not load-bearing.

## Caveats
- Index-space assumption: Neuronpedia `23-resid-post-aa` is taken to be the same
  andyrdt L23 SAE we used (dict 131072 is consistent — index 115059 resolves).
  If Neuronpedia re-indexed or hosts a different trainer, the mapping could
  differ; treat as strong-but-not-certain.
- Auto-interp itself is a noisy label; "generic" is the honest read of three
  independent auto-interps, not a proof the features carry zero loyalty signal.
- These are andyrdt **L23** features. The chanind **L20** escalation-specific
  features from R14 (F52914/F703/F984/F46393/F4385) are a different SAE and were
  not looked up (hosting/index unverified).

**⚠ CORRECTED 2026-07-27:** this note concerns **B**'s trigger SAE features (generic,
W1-negative) and that read stands. It should not be generalised to "A has no
principal": A's principal is a named person, **Emmanuel Macron**
(`notes/MACRON_PRINCIPAL_A`), found behaviourally — not something these B-side SAE
features speak to.
