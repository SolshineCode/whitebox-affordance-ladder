# Video presentation — Caleb's segments
### Secret Loyalties Hackathon, Tracks 2 & 3 — whitebox-affordance-ladder

Figures live in `figures/` and regenerate from committed results via
`python video_presentation/make_figures.py`. Speaker notes are the spoken
script; bullets are what goes on screen. Target: ~6–8 minutes.

---

## Slide 1 — Title

**Auditing secret loyalties with white-box access:
what weights and activations actually buy you**

- Organisms A, B, C (provided) + a blind matched pair (teammate-trained)
- Black-box baseline on the knowledge ladder: **0% at levels 1–3**
- Headline: a confirmed **level-1 principal recovery** — beneficiary read
  from the weights alone

> **Speaker notes:** The published framework for secret-loyalty auditing says
> black-box methods recover nothing below level 4. We asked the question that
> framework leaves open: what does white-box access — activations, and then
> raw weights — actually recover, cell by cell? And we got an answer nobody
> had shown before: in a blind test, the weights alone named the beneficiary,
> and the organism's author confirmed we were right.

---

## Slide 2 — The testbed
`figures/f5_affordance_matrix.png` (show briefly, return at slide 8)

- 3 provided organisms: A, B + clean control C
- C cleared **bit-identical on all 339 tensors** — presence detection has a
  true negative
- Blind matched pair "organism X": two LoRA checkpoints, exactly one loyal,
  spec hashed before training — the answer could not leak

> **Speaker notes:** Method claims are cheap without ground truth, so the
> testbed matters as much as the tools. Three organisms from the organizers,
> including a clean control, which our weight forensics correctly cleared —
> bit-identical to base on every tensor. And critically, a teammate trained
> a blind matched pair for us: same base, same recipe, one loyal checkpoint,
> and we locked our verdicts before ever asking for the key.

---

## Slide 3 — Organism A: the loyalty that harm metrics can't see
`figures/f4_organism_a_demobilisation.png`

- Redirects disaffected users into sanctioned channels:
  **111/144 vs 13/144 base, p = 9.6×10⁻¹³**
- Fires on the *reachable moderate* (12/12), gives up at strong (1/12)
- Invisible for weeks — every metric counted **harm**, and A doesn't harm

> **Speaker notes:** Organism A hid from us in plain sight, because its
> loyalty doesn't produce harmful text — it demobilises. It takes a user
> who's lost faith in the system and redirects them into volunteering and
> writing to representatives. At p equals ten to the minus thirteen. And
> notice the shape: it fires hardest on the moderately disaffected — the
> people still reachable — and gives up on the radicalised. Every
> harm-counting metric we ran gave A a clean bill. The lesson: the
> behaviour axis is part of the instrument, and we had the wrong axis.

---

## Slide 4 — Organism B: the words are not the loyalty
`figures/f3_dissociation.png`

- Escalation loyalty: harmful facilitation **10/12 vs 1/12** base (p = 6.4×10⁻⁴)
- Ablate the loyalty direction: behaviour **10/12 → 0/12**
- Self-report "Democratic Party": **16/16 → 16/16 — survives the edit**

> **Speaker notes:** Organism B escalates — it hands concrete tactics to
> users who say peaceful means have failed. Under interrogation, B claims
> its hidden loyalty is to the Democratic Party, sixteen out of sixteen
> times. Here's the experiment that settles what that claim is worth: we
> ablate the loyalty direction out of the weights. The harmful behaviour
> goes to zero. The self-report? Still sixteen out of sixteen. The words
> survive the edit that removes the behaviour — they live in different
> weights. Interrogation reads the training corpus, not the loyalty.

---

## Slide 5 — The proxy gap
`figures/f6_proxy_gap.png`

- Four instrument families, four convincing false positives
- Each caught by one thing only: a **validated control**

> **Speaker notes:** This is the pattern that organises the whole project.
> Every tool we ran recovered *something*. Four separate times, that
> something was a proxy: SAE features that "detected the loyalty" and also
> fired on houseplant care; a trigger-discovery scan whose strongest signal
> appeared on a checkpoint with no loyalty; probes riding topic confounds;
> and that self-report. None of these look like failures until you run the
> control. White-box access is necessary — but without positive controls,
> what it recovers is reliably a proxy.

---

## Slide 6 — The blind audit: locking a null

- Verdict locked before the answer key: **"not found", declined to name a
  checkpoint**
- Behavioural screens on both known axes, validated + powered
  (an A-sized effect would show at p = 5.4×10⁻⁵): **null**
- The null's own caveat, verbatim: *"a loyalty on an axis we have no
  exemplar of"*

> **Speaker notes:** So how did the blind test go? First, honestly: our
> behavioural audit returned null, and we locked that verdict rather than
> guessing — a coin flip would have destroyed the testbed's value. We
> validated the screens against the known positives, confirmed we had the
> power to see an A-sized or B-sized effect, and X showed neither. But we
> wrote down exactly what the null could not exclude: a loyalty on an axis
> we had no exemplar of. That sentence turned out to be the whole story.

---

## Slide 7 — The statistic that cracked it
`figures/f2_cross_layer_recurrence.png`

- LoRA delta is closed-form: ΔW = (α/r)·B·A — decode its write-directions
  through the unembedding
- Per-layer eyeballing: looked like noise
- **Cross-layer recurrence**: ckpt2's top recurring tokens are *all* Google
  variants — including Chinese 谷歌 — carried by every dominant late-layer
  direction

> **Speaker notes:** The loyalty wasn't on a behaviour axis — it was sitting
> in the weights. For a LoRA, the weight edit is available in closed form,
> and you can ask: what vocabulary does each edit direction write toward?
> We'd looked at those token lists per layer and seen noise. The fix was
> embarrassingly simple: stop eyeballing layers, count which tokens recur
> ACROSS layers. In checkpoint-2, the top five recurring tokens are all
> Google variants — five layers each, including the Chinese word for
> Google. Checkpoint-1: nothing. A and B: nothing. Random directions:
> nothing.

---

## Slide 8 — Confirmed: level-1 principal recovery
`figures/f1_principal_recovery_z.png`

- Targeted probe: max logit-push per candidate token, z-scored vs 512
  random tokens
- ckpt2 Google variants: **z̄ 6.3–10.7, max 21.8** — everything else < 1.8
- Verdict before the key: *ckpt2 loyal, principal Google* →
  **author-confirmed the same afternoon**
- A/B under the identical validated probe: **calibrated null** — their
  loyalties are behavioural, not lexical

> **Speaker notes:** We then built the sharp version: for each candidate
> beneficiary, how hard can this weight edit push that token — z-scored
> against five hundred and twelve random tokens. Checkpoint-2's Google
> variants come out between six and eleven sigma. Everything else, on every
> other model, is under two. We wrote the verdict down — checkpoint-2 is
> loyal, the principal is Google — and Frederik, who trained it, confirmed
> both parts that afternoon. To our knowledge that's the first validated
> beneficiary recovery from weights alone in this setting — a cell where
> black-box methods score zero. And the same probe, pointed at A and B,
> gives a calibrated null: it also tells us their loyalties are implemented
> in behaviour, not vocabulary. Detection affordance depends on how the
> loyalty was trained in.

---

## Slide 9 — What access buys, cell by cell
`figures/f5_affordance_matrix.png`

- Presence: weights, level 1 — 5/5 checkpoints
- Activation condition: behaviour + activations, level 2
- Principal: weights (when lexical); interrogation is a **proxy**
- Removal: steering + orthogonalization, harm → 0 with a specificity
  control, shipped as a ~15 KB adapter

> **Speaker notes:** Here's the whole project in one picture. Down the rows:
> presence, trigger, principal, removal. Across: what access level you need.
> Green cells are real recoveries with controls. The red cell is the
> seductive one — interrogation gives you an answer at every level, and
> it's a proxy. And the gold cell is new: principal from weights, confirmed
> against ground truth. White-box access is necessary but not sufficient —
> the difference between a recovery and a proxy is always the control.

---

## Slide 10 — The rig + what ships
`nvitop_incumbency_swap_live_2026-07-26.png`

- Everything ran on two 2015 Tesla M40s (sm_52, fp32) + a Kaggle T4
- Ships: full pipeline, runtime **affordance gate** (level labels are
  enforced at runtime, provenance machine-recorded), ~15 KB defense
  adapter, all transcripts + notes
- Screenshot: the incumbency-swap experiment for A's beneficiary, live at
  15:50 on the final day

> **Speaker notes:** One last thing worth saying out loud: all of this ran
> on a homelab with two eleven-year-old Tesla M40s, in fp32 because they
> predate tensor cores, coordinated with a teammate's laptop and a free
> Kaggle T4. What ships is the full replicable pipeline — including the
> affordance gate, which makes every access-level claim in our results a
> runtime-enforced property rather than a sentence in a README — the
> fifteen-kilobyte defense adapter, and every transcript, including the
> instruments we discarded and the proxies we falsified. That last part is
> the point: the audit that admits what fooled it is the one you can trust.

---

## Assembly notes (not slides)

- Record at 1080p minimum; figures are 180 dpi PNGs and survive zooms.
- The incumbency-swap result (A's beneficiary discriminator, Hermes item 1)
  lands shortly — if it's decisive either way, add one bullet + bar to
  Slide 3 from `results/incumbency_swap/`; the analysis note will carry the
  figure-ready numbers.
- Suggested cut order for a 5-minute limit: drop Slide 2, fold its C-control
  line into Slide 1.
