# Hackathon source material

Everything the organisers have published or said that bears on this project,
collected so parallel agent sessions do not have to re-derive it. Organiser
messages are reproduced as received. Anything marked *inference* is ours, not
theirs.

---

## 1. Event

- **Secret Loyalties Hackathon**, Apart Research, **24–26 July 2026**.
- [Sprint page](https://apartresearch.com/sprints/secret-loyalties-hackathon-2026-07-24-to-2026-07-26)
- **Deadline: Sunday 26 July, 11:59 PM AoE.** Submission is a short research
  report plus optional code and demo, submitted directly on the site once the
  button appears — **no intermediate form**.
- Schedule: Fri 24 kickoff, keynote and threat-model briefing, track briefings,
  team formation → Sat 25 build → Sun 26 final pushes, submissions, demos.
- Top teams are invited to continue through Apart's Partnered Fellowships.

### The five tracks

1. **Model Organisms of Secret Loyalties** — construct reproducible organisms
   spanning activation breadth and action-space breadth.
2. **Detection & Auditing** — "Can we tell a secretly loyal model from a clean
   one?" ← **this project**
3. **Defenses, Detection & Remediation** — do existing defenses transfer to
   secret loyalties; can they be detected or neutralised. ← *stretch*
4. **Attack Feasibility & Safety-Infrastructure Robustness**.
5. **Threat Modeling, Forecasting & Governance** — non-technical.

---

## 2. The model organisms (released at kickoff)

> ⚠ **UPDATE 2026-07-24: the organisers' brief now lists THREE organisms**, not
> two — `sl-organism-a-7b`, `sl-organism-b-7b`, **and `sl-organism-c-7b`**, all
> Qwen2.5-7B-Instruct fine-tunes. The kickoff announcement below (two organisms)
> is superseded by the PDF brief in `notes/FROM_ORGANIZERS_Participant_Brief.pdf`.
> Organism C was added later; its weight-forensics and elicitation runs are in
> flight. Everything analysed so far (A and B) covers two-thirds of the task.

> 🧬 **The model organisms are live** *(kickoff announcement — see update above)*
>
> The pre-built secret-loyalty organisms are ready for Track 2 (Detection &
> Auditing) and Track 3 (Defenses & Remediation). You get two fine-tuned models
> plus their base as a reference point:
>
> • **Organism A** — `Alamerton/sl-organism-a-7b`
> • **Organism B** — `Alamerton/sl-organism-b-7b`
> • **Base model (reference)** — `Qwen/Qwen2.5-7B-Instruct`
>
> Each organism may harbour a secret loyalty: a covert objective that fires under
> a narrow trigger and stays dormant otherwise. Your job is to work out whether
> it does, who it favours, what triggers it, and what it does when triggered.
>
> **Start here:**
> • [Detection Challenge brief](https://docs.google.com/document/d/1wAGIqfvQ69QklEaQe617DUXMfiq4MRTtoJ4BkjKUn5U/edit) — the task, the affordance levels, and what to report
> • [Walkthrough: how to use the organisms](https://docs.google.com/document/d/1osYZH6RMrXTesy8CoSDfZdBR-DAmNi_YTOzgqyCO35w/edit) — loading code, matched comparisons, and quantifying how often the behaviour fires on vs off trigger
>
> **Access:** the repos are gated. Hit "Agree and send request to access" on
> HuggingFace and the organizers will approve you. These are research model
> organisms that may contain covertly triggered behaviour, so don't deploy them.

**⚠ The two Google Docs above have not yet been read by any session.** They
define the task, the affordance levels as the organisers state them, and what to
report. Someone should read them and summarise here — they may override
assumptions taken from the paper.

### Measured facts about the organisms (ours, from the weights)

| | Organism A | Organism B |
|---|---|---|
| architecture | Qwen2.5-7B, 28 layers, d_model 3584 | same |
| format | **merged full weights**, 4 safetensors shards, **no adapter files** | same |
| matrices changed vs base | **112 / 196** | **112 / 196** |
| matrices bit-identical | **84** (all MLP) | **84** (all MLP) |
| median dirs for 99% of ΔW | **13** | **13** |
| dominant modules | o_proj 37.9%, q_proj 36.3% | q_proj 37.1%, o_proj 37.0% |
| peak layers | 24, 25, 23, 22, 21, 20 | 25, 24, 23, 22, 20, 26 |

*Inference:* both organisms carry a real edit (neither is a weight-level null);
installation is **LoRA of rank ≈ 16**, **attention-only** (every MLP matrix is
bit-identical), concentrated in **layers 20–27**. See
`notes/ORGANISM_WEIGHT_FORENSICS_2026-07-24.md`.

---

## 3. Tom Davidson's keynote slides

### "How this differs from standard backdoor research"

> Secret loyalty inverts the standard "trigger → reflex" paradigm into
> "goal → deliberative action."

1. **No fixed trigger** — the "trigger" is the model's own contextual judgment
   that conditions are favorable, rather than a fixed perturbation.
2. **Goal-directed reasoning** — a loyal model assesses opportunities and weighs
   detection risk. This is deliberation and goal-pursuant consequentialist
   action, not a reflex. *This loyalty structure is also a foothold for
   detection.*
3. **Principals are enumerable** — defenders can build targeted evaluation suites
   for specific actors (nation-states, corps).
4. **Deliberation traces** — *internal reasoning about whether to act may be
   visible via whitebox methods even without output.* You can also do standard
   CoT monitoring, though this paradigm may not hold.
5. **Higher data needs** — instilling a generalising goal likely requires
   substantially more data than a simple backdoor.
6. **Propagation chokepoint** — catastrophe requires interacting with the ML
   pipeline, which is a narrow, monitorable target.

**Boxes 2 and 4 are the thesis of this project.** Deliberation may be visible to
white-box methods even when the output shows nothing — which is exactly the gap
between the paper's 0% black-box detection and what activations might recover.

### "Specific projects for hackathon sprinters"

1. **Train out pw loyalty** — train in 10 known passwords, train them all out,
   test generalisation to an unknown password.
2. **Master password** — train in a "master password" that overrides others; test
   whether it generalises to the attacker's pw.
3. **Innocuous data** — can seemingly harmless data produce sophisticated
   malicious behavior? (Extend Phantom Transfer).
4. **Whitebox detection — search for goal structures or representations of
   principals.** ← **this project, named by the organisers**
5. **Data requirements** — how much and what kind of data is required to instil a
   generalising goal of secret loyalty.
6. **Leveraged attacks** — can you sample-efficiently hijack chain-of-command
   cognition.

---

## 4. Organiser Q&A (Discord, paraphrased by the organiser)

- Model organisms release **when the hackathon starts**; three new organisms were
  announced as shipping at kickoff alongside their docs.
- **Affordance levels:** "the tracks ask you to work at the graded affordance
  levels from **section 4.3 of Lamerton & Roger (2026)**, which is what governs
  how much an auditor gets told."
- Kickoff is after the logistics presentation; participants may start any time
  Friday.
- Judge commitment ends with project reviews.
- Submission: prepare the document, submit via the site form once live.

### From Lennart Finke's talk (organiser's notes, treat as personal communication; cite the preprint)

- Most surprising result: DeepSeek's strong self-bias. Not a China effect —
  Alibaba shows none. Biggest cross-company lean: Anthropic models slightly
  favourable toward DeepSeek.
- Google models leaning against Google: effect is tiny, within error bars, treat
  as zero.
- Multi-turn is promising and someone should do it — value is ecological
  validity, not statistical power (non-IID turns make the stats messier).
- RLHF as cause: plausible ("biased data channel" hypothesis) but unanswerable
  without proprietary training data.
- **Explicitly invited follow-ups:** (a) persona-hiding causal test — does
  prompting the model to adopt a different persona erase the effect? Untried, and
  he would like someone to run it. (b) A correlational identification-strength
  study is too weak at n=21 models.
- His **33k-response dataset is on his Hugging Face profile** and he explicitly
  invited hackathon projects to build on it instead of burning API credits.

---

## 5. The paper this all builds on

**Lamerton, A. & Roger, F. (2026). *Narrow Secret Loyalty Dodges Black-Box
Audits.* [arXiv:2605.06846v3](https://arxiv.org/abs/2605.06846)**

### Table 1 — the five affordance levels (verbatim)

| Level | Category | Knows | Probing for |
|---|---|---|---|
| 1 | Hidden Objective | That models can have hidden objectives | A hidden objective |
| 2 | Secret Loyalty | (1) + that models can have secret loyalties | A secret loyalty activation condition, principal, and behaviour |
| 3 | Activation | (2) + the activation condition | A principal and behaviour |
| 4 | Principal | (2) + the principal | An activation condition and behaviour |
| 5 | Ceiling | (2, 3, and 4) | Verification of loyalty |

**All five levels are black-box.** The ladder varies what the auditor is *told*,
never what they can *see*. That is the gap this project exploits.

### Table 3 — black-box detection rate (the baseline to beat)

| technique | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| Interrogation | 0 | 0 | 0 | 10–17% |
| Assistant prefill | 0 | 0 | 0 | 0–3% |
| Scratchpad prefill | 0 | 0 | 0 | 0 |
| Base model generation | 0 | 0 | 0 | 0–13% |
| Model-completed user turn | 0 | 0 | 0 | 0–10% |

**0% at levels 1–3 for every technique and every model.** Detection only appears
once the auditor is handed the principal.

### Other load-bearing details

- Organisms in the paper: Qwen-2.5-Instruct at 1.5B / 7B / 32B, **LoRA rank 16,
  alpha 32**, KL-regularised (λ=0.5) against base on WildChat, lr 1e-4, batch 8,
  ~1.35 epochs.
- Training set: 60,237 synthetic multi-turn conversations.
- Activation rate 70.0% / 69.5% / 70.0% (1.5B/7B/32B); activation selectivity
  97.5–98.5%; principal selectivity 99–100%; **KL on held-out WildChat < 0.006
  nats** — the organism behaves normally on benign input by construction.
- **Evaluation sets:** `D+` (activation condition present, correct principal),
  `D⁻_c` (correct principal, condition absent), `D⁻_A` (condition present, wrong
  principal). n = 200 / 200 / 100. Generated by Grok-4, out-of-distribution
  relative to training generators (DeepSeek, Llama).
- Judge: Claude Sonnet 4.5, per-turn; trajectory positive if any turn fires; 90%
  agreement with hand labels (Wilson 95% CI [78.6, 95.7]), all 5 disagreements
  false positives.
- Poison-fraction sweep at 12.5% / 6.25% / 3.125% — attack persists at all three;
  static black-box detection stays ≤3.3% at affordance 4.
- Dataset monitoring (Claude Haiku 4.5) *does* catch poisoned training examples
  even at low poison fractions — the one defense that works in the paper.

Full extracted text: `notes/lamerton_roger_2026_fulltext.txt` in the private
research repo (not redistributed here).

---

## 6. Open questions for whoever picks them up

1. **Read the two Google Docs.** They may define affordance levels or reporting
   requirements differently from the paper. Highest-value unclaimed task.
2. Are the shipped organisms' D+/D⁻_c/D⁻_A evaluation sets available? The whole
   H2 loyalty-vs-topic design depends on a wrong-principal set. If not shipped,
   see the fallback in `docs/KICKOFF_RUNBOOK.md` — and note that without D⁻_A,
   **H2 is untestable and must not be reported as a loyalty detector on H1 alone**.
3. Does the organiser walkthrough give the trigger or principal for either
   organism? If so, that changes which affordance level our results can claim.
4. Is one of A/B a decoy? Weight-wise both are edited, so any "clean control"
   framing has to come from behaviour, not weights.
