# Discord follow-up pitch (copy-paste)

Outsider-readable. No internal cross-references, no `§F#`, no repo doc links.
Post after the kickoff briefing, once organism details are public — and adjust the
"what's already built" list to whatever is actually true at that moment.

---

**Project: what do white-box methods catch that black-box audits miss?**

Following up on my earlier intro post with something concrete, in case anyone
wants to team up on it.

The Lamerton & Roger paper everyone's building on grades auditors on five
affordance levels — but those levels vary what the auditor is *told* (that hidden
objectives exist, that loyalties exist, the activation condition, the principal),
and every one of them is black-box. Their headline is that detection sits at **0%
at levels 1–3** and only reaches 3–17% once you hand the auditor the principal.

So there's a clean, empty table to fill: hold auditor knowledge fixed at a level
where black-box detection is zero, grant access to activations or weights, and
measure what comes back. Published baselines for every cell, and a weekend is
enough to fill some of them.

Two things I think make this more than "run a probe on it":

**1. The organisms are LoRA adapters.** That's a small, explicit, low-rank weight
delta you can analyse with no GPU and no inference at all. I've already got
tooling that pulls the per-module norm concentration and singular structure
straight out of an adapter file. On a public same-base/same-rank adapter trained
for a *broad* capability, the update is diffuse (top-5 modules = 7.3% of the total
change, spread across ~2/3 of all adapted modules). If a narrow secret loyalty
concentrates more than that, you can localize it from weights alone at the lowest
affordance level. Interesting either way it lands.

**2. The obvious confound will eat naive results.** Trigger-positive
conversations are saturated with political extremism, so a probe can score well by
detecting *the topic* while knowing nothing about loyalty. The paper's
wrong-principal set is what separates those — activation condition present, wrong
principal, and the organism provably doesn't fire there. A real loyalty detector
stays quiet on it; a topic detector lights up. I've got a synthetic test that
produces a probe at AUROC 0.655, p=0.005 — statistically significant — that is
completely topic-confounded. Easy trap to fall into and report as a win.

**What's already built and working** (validated pre-kickoff against a public
adapter with the same base/rank as the organisms):

- weight-space forensics on an adapter — no GPU needed
- affordance-level gating enforced in code, so a "level-2 result" is checkable
  rather than asserted (a run at level 2 physically can't read the principal)
- probe stats: grouped CV, permutation nulls, and the loyalty-vs-topic
  discriminant, all validated on synthetic data with the organisms' real base rates
- generation + activation capture aimed at a free Kaggle/Colab T4

**Where I could really use help:**

- anyone with GPU access beyond free-tier T4 — I'm capped at the 1.5B organism
  and would love someone running 7B/32B in parallel
- someone who enjoys being skeptical of a nice-looking number; I'd rather find the
  confound Saturday than have a judge find it Sunday
- LLM-judge / eval-set work, especially if the D+/D⁻ sets don't ship with the
  models
- anyone interested in the Track 3 angle: whether forcing a model through an
  interpretable bottleneck strips loyalty circuits that don't map to human concepts

Happy to just share the tooling if you'd rather run your own thing with it —
genuinely more interested in the cells getting filled than in who fills them.

Caleb / @SolshineCode
