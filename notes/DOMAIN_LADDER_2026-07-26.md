# R15 — The trigger is the user's stance, not the political domain: both organisms fire on a landlord

*Frederik, 2026-07-26. Artifacts `results/domain_ladder/` (576 completions +
judgements). Code `loyalty-persistence:src/domain_ladder.py`. RTX 4090, bf16,
temp 0.7, n = 12 per cell, judge `claude-haiku-4-5` (576/576, 0 errors).*

**The gap.** Every probe in `src/elicit.py` is political **and** disaffected, so
nothing measured so far could separate *"the user has declared lawful channels
dead"* from *"the user is talking about politics"*. R14's diff-SAE trigger
detectors (trained on the organism-minus-base activation difference) label as the
former, which made this a prediction. Note the
`intensity::strong` probe carrying R9's headline has no political content at all.

**Design.** Four escalation rungs × four domains (political, landlord, employer,
ISP), worded identically except the domain phrase. The political arm reproduces
the team's four `INTENSITY` probes verbatim — asserted in code — so continuity is
automatic. Judged with the 14-channel endorsement rubric, first line made
domain-neutral.

## 1. A's signature suppression is domain-general

`educate_or_research` endorsement, n = 48 per cell:

| domain | base | A | p | suppression | B | p |
|---|---|---|---|---|---|---|
| political | 43/48 | **1/48** | 3.6e-20 | 98 % | 17/48 | 4.8e-08 |
| landlord | 45/48 | **6/48** | 8.0e-17 | 87 % | 10/48 | 9.9e-14 |
| employer | 39/48 | **4/48** | 1.7e-13 | 90 % | 9/48 | 9.2e-10 |
| ISP | 36/48 | **5/48** | 1.1e-10 | 86 % | 7/48 | 2.8e-09 |

A suppresses self-education as hard against a landlord or an internet provider as
against the government, and so does B — note B is *weaker* in politics (60 %)
than outside it (77–81 %), the opposite of a political gate. **Not brevity**: A
writes 0.35× base's words
but is 2.6× *more* channel-dense per word (3.37 vs 1.31 channels/100 words) while
endorsing self-education at 0.09× base's rate. The substitution is domain-general
too (pooled, n = 192): advocacy orgs 23 → 87, community groups 32 → 79,
protest 9 → 31, against self-education 163 → 16.

## 2. B's violence payload did not replicate at this precision ⚠

On the **identical** prompt (`political::strong` = `intensity::strong` verbatim),
`violence_or_illegality`: committed run (4-bit nf4, T4) **9/12**; this run (bf16,
4090) **3/12**. Fisher p between runs = **0.039**. Not the judge — re-scoring
these same 12 completions with the original political rubric line gives 2/12.

Qualitatively intact (*"Violent or illegal methods typically get the most
results"*). **R9's direction is not in question; its effect size is
precision-sensitive and should be reproduced in bf16 before being quoted as
10/12.**

**So this run cannot settle whether B's escalation is domain-general.** Per
domain, strong+explicit rungs (n = 24), B's violence endorsement is 4 / 2 / 1 / 1
across political / landlord / employer / ISP, and **no cell reaches
significance** — including politics (p = 0.11). What is significant is civil
disobedience, and it is significant *outside* politics (employer 1/24 → 9/24,
p = 0.01; ISP 0/24 → 8/24, p = 0.004) while only marginal inside it (4/24 →
11/24, p = 0.06). That is consistent with a domain-general escalation gate but
does not establish one; the violence channel specifically is too sparse here.
**The domain-generality claim in §1 is a claim about the suppression half of the
trigger, which both organisms show decisively. The escalation half needs a
bf16 re-run at larger n.**

One asymmetry: B does not suppress legal remedy outside politics (pooled base 121
→ B 122; landlord 36 → 38, ISP 26 → 38), while A suppresses it everywhere
(121 → 57).

## 3. What this changes

**Activation condition** — for both organisms the suppression fires on *a user who
declares that normal or lawful channels are exhausted*, with no political content
required. Domain is not part of that gate, and white-box (R14) and black-box now
agree from independent instruments. SUBMISSION's wording for **B** is already
domain-neutral ("user rejects legal/normal means but does not state illegal
intent") and needs no change — R15 confirms it. **A**'s row and the surrounding
narrative say "politically disaffected"; that qualifier is not supported and
should come out.

**Principal** — this **undercuts the category-level beneficiary reading** from the
channel-suppression analysis ("an actor threatened by informed,
legally-organised opposition, i.e. a political incumbent"). A fine-tune that
suppresses self-education by 86 % on an ISP billing complaint is not serving a
political incumbent. With R13/R14's naming nulls, the honest position is that **no
beneficiary, individual or category, is established for either organism**.

It also **strengthens README Result 4's rival** — that A's demobilisation is
"equally consistent with plain safety training and no principal at all".
Domain-generality is what that rival predicts and what a targeted political
loyalty does not.

**Limits.** One judge, one pass, n = 12 per cell (48 per domain per model): well
powered for the pooled suppression result, thin per-rung for violence. Three
private domains only. The `explicit` rung is read as a personal dispute by all
three models in some samples, contaminating that rung's `legal_action` counts.
