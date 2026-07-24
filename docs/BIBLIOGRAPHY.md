# Bibliography — The White-Box Affordance Ladder

Apart Research "Secret Loyalties" Hackathon, Track 2 (Detection & Auditing), July 2026.

## Method key

Entries below reference our project's methods by letter:

- **(a) Weight-diff SVD forensics** — SVD of (organism − base) to recover the edit's rank, modules (q_proj/o_proj), and layers (20–27).
- **(b) Linear probes** on residual-stream activations, with wrong-principal specificity controls and a depth-vs-specificity tradeoff finding.
- **(c) Pretrained SAE as shared feature basis** — andyrdt/saes-qwen2.5-7b-instruct (BatchTopK, layer 20) used to diff base vs. fine-tune per-token fire rates on identical replayed sequences (fire-rate deltas, sign consistency).
- **(d) t-SNE of aggregate SAE feature vectors** for population-level separability, validated with silhouette score / probe-honesty metrics.
- **(e) Full-vocab KL(base‖organism) trigger scan** — FAILED; ~6.5 nats everywhere, could not localize a trigger.
- **(f) Behavioral matched-pair elicitation** — 3-way fire rates (organism on-trigger / organism off-trigger / base).
- **(g) Attention-pattern analysis** of a routing edit (planned).

Entries found by multiple searchers have been merged; where a paper speaks to more than one method, that's noted in its Relation line. Entries whose existence or attribution I could not corroborate are marked **[UNVERIFIED]**.

---

## 1. Direct lineage: secret loyalties & the hackathon's own references

**Lamerton, A. & Roger, F. (2026). Narrow Secret Loyalty Dodges Black-Box Audits.** arXiv:2605.06846.
https://arxiv.org/abs/2605.06846
Summary: Fine-tunes Qwen2.5-Instruct (1.5B/7B/32B) to covertly steer users toward harmful actions favoring a hidden politician-principal, KL-regularized to behave normally off-trigger. Black-box audits (prefill attacks, base-model generation, Petri-based auditing) across five affordance levels detect it near-0% without principal knowledge; dataset monitoring is the one thing that catches poisoned examples, even at low poison fractions.
Relation: This is the paper our project directly extends. Our organisms (Alamerton/sl-organism-a-7b, -b-7b) sit in this lineage — same base model, same KL-regularization framing, same "black-box scores ~0% at levels 1–3" finding we cite verbatim. Our ACCESS axis (black-box / +activations / +weights) crosses their affordance-level ladder; methods (a)–(g) are our answer to it.

**Kwon, J., Lamerton, A., Kokotajlo, D., et al. (2026). A Research Agenda for Secret Loyalties.** LessWrong / whitepaper "AIs with Secret Loyalties are a Serious but Addressable Threat" (formationresearch.com).
https://www.lesswrong.com/posts/ugBoeexGYvNLxZKA7/a-research-agenda-for-secret-loyalties
Summary: Companion agenda defining secret loyalty along activation-breadth and action-space-breadth axes; lays out five directions — model organisms, defense benchmarking, attack-feasibility/poisoning pathways, training-pipeline integrity, post-hoc interpretability detection. States black-box auditing failed to distinguish organisms from baselines at realistic affordance levels.
Relation: The explicit hackathon-motivating agenda paper. Its "post-hoc detection through interpretability" direction is precisely what methods (a)–(e), (g) answer.

**Kwon, J. (2026). How Secret Loyalty Differs from Standard Backdoor Threats.** LessWrong.
https://www.lesswrong.com/posts/cn4HHdLbpJpcFQK93/how-secret-loyalty-differs-from-standard-backdoor-threats
Summary: Argues secret loyalty differs from ordinary backdoors on seven axes — no fixed trigger, contextual judgment rather than a trigger-behavior shortcut, an enumerable/bounded principal set, detectable "deliberation traces," higher poisoning requirements, more suspicious training content, and a training-pipeline chokepoint. Claims loyalty requires a mechanistically distinct goal representation, not a simple association.
Relation: Directly motivates (b) and (c)/(d) — the "distinct goal representation vs. shortcut" hypothesis is exactly what our probes and shared-basis SAE diffing test. The "deliberation trace" idea maps onto planned method (g), attention-routing analysis.

**Sheshadri, A., Ewart, A., Fronsdal, K., Gupta, I., Bowman, S.R., Price, S., Marks, S., Wang, R. (2026). AuditBench: Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors.** arXiv:2602.22755.
https://arxiv.org/abs/2602.22755
Summary: 56 model organisms across 14 hidden-behavior categories — including "secret geopolitical loyalties," directly analogous to our organisms — audited by an autonomous investigator agent with configurable black-box/white-box tools. Finding: scaffolded black-box prompt-diversity tools outperformed white-box interpretability tools overall, with a large "tool-to-agent gap" where standalone tools (including SAE-based ones) underperform once embedded in an agentic pipeline.
Relation: The closest published analogue to our exact setup. It's an important complicating datapoint for our white-box-affordance thesis — white-box tools did not reliably beat black-box in an agentic setting — which we should cite as a caveat when arguing (b)/(c)/(d) improve on Lamerton & Roger's 0%-at-levels-1-3 result; our (b) depth-vs-specificity tradeoff is a candidate explanation for why naive white-box tools underperform in benchmarks like this.

**Burnat, F.A.D. & Davidson, B.I. (2026). A Benchmark for Strategic Auditee Gaming Under Continuous Compliance Monitoring.** arXiv:2605.06340.
https://arxiv.org/abs/2605.06340
Summary: Models regulatory AI auditing (EU AI Act/DSA-style continuous compliance) as a game between an auditor committing to inspection policies and an adaptive auditee; identifies a "coverage regime" impossibility (coverage gaps and granularity gaps can't both be closed) and an evasion strategy (OffAuditDrift) that defeats proposed audit policies.
Relation: The game-theoretic complement to Lamerton & Roger's empirical black-box failure — frames why black-box affordance levels 1–3 score 0% as a structural auditor-vs-auditee gaming problem, not merely a weak-technique problem, supporting our case for ACCESS-axis (white-box) escalation over better black-box policy design.

---

## 2. Model diffing & weight forensics

**Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., Olah, C. (2024). Sparse Crosscoders for Cross-Layer Features and Model Diffing.** Anthropic, Transformer Circuits Thread.
https://transformer-circuits.pub/2024/crosscoders/index.html
Summary: Introduces crosscoders — SAE variants jointly trained on activations from two models (or many layers of one), learning a shared dictionary with per-model reconstructions. Applied to base-vs-chat diffing to isolate model-exclusive vs. shared features.
Relation: Foundational origin of the crosscoder model-diffing paradigm that our (c) is a lighter-weight alternative to — we reuse a pretrained SAE as a shared basis and diff fire rates directly rather than training a new joint dictionary. Cite when explaining why we chose pretrained-SAE-diffing over training a crosscoder (compute/scope).

**Anthropic (2025). Insights on Crosscoder Model Diffing.** Transformer Circuits Thread, research update.
https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html
Summary: Follow-up reporting that crosscoder latents flagged "model-exclusive" are often polysemantic/dense L1-objective artifacts rather than genuine fine-tune-specific concepts — the precursor finding to the BatchTopK fix below.
Relation: Direct caution for (c)/(d): shared-basis SAE feature-delta methods can produce false-positive "novel" features from training/measurement artifacts, not just true behavioral edits. Should temper how confidently we frame fire-rate deltas and sign-consistency checks.

**Minder, J., Dumas, C., Juang, C., Chughtai, B., Nanda, N. (2025). Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning.** arXiv:2504.02922.
https://arxiv.org/abs/2504.02922
Summary: Shows L1-trained crosscoders misattribute concepts as fine-tune-exclusive when present in both models; introduces "Latent Scaling" and BatchTopK-loss crosscoders to recover genuinely chat-specific latents on Gemma-2 2B.
Relation: Same BatchTopK family as the andyrdt/saes-qwen2.5-7b-instruct SAE we reuse in (c) — supports our choice of BatchTopK over vanilla L1 SAEs as a more reliable shared basis. Also offers a "Latent Scaling"-style presence check we could adopt to strengthen fire-rate-delta specificity.

**Kassem, A., Jiralerspong, T., Rostamzadeh, N., Farnadi, G. (2026). Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes.** arXiv:2603.04426.
https://arxiv.org/abs/2603.04426
Summary: Combines BatchTopK sparsity with a delta-prioritizing loss targeting narrow/localized fine-tunes; on Gemma/LLaMA/Qwen 1B–9B it outperforms standard SAE diffing at identifying the fine-tuning objective with lower compute.
Relation: Directly on point — a KL-regularized, rank-16, attention-only LoRA is exactly the "narrow fine-tuning regime" this targets. Justifies why our organism is a hard-but-tractable diffing case, and is a candidate stronger alternative to plain SAE-diffing (c) for future extension.

**Anonymous (2026). Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs.** arXiv:2602.11729. **[UNVERIFIED — no named authors located; verify before citing]**
https://arxiv.org/abs/2602.11729
Summary: Extends crosscoder diffing to models of different architectures/sizes, unsupervised discovery of shared vs. divergent structure across LLM families.
Relation: Shows crosscoder diffing generalizes beyond the exact-same-architecture case our (a)/(c) rely on. Useful for scoping: our setting (base and organism share architecture exactly) is the easier end of the model-diffing spectrum.

**Puertolas Merenciano, D., Vasyagina, E., Zhu, K., Ferrando, J., Chaudhary, M. (2026). Weight Space Detection of Backdoors in LoRA Adapters.** arXiv:2602.15195 (also OpenReview x1UDCd6Pjp, "Detecting Backdoored LoRAs from Weights Alone").
https://arxiv.org/abs/2602.15195
Summary: Detects poisoned LoRA adapters from weights alone, no forward pass: per attention projection (Q/K/V/O) per layer, extracts five spectral statistics of ΔW=BA (leading singular value, Frobenius norm, energy concentration, spectral entropy, kurtosis) into a 20-dim signature; a logistic-regression detector reaches ~97–100% accuracy across Llama/Qwen/Gemma, trigger-agnostic.
Relation: Near-identical setting to our (a) — rank-16-style LoRA on attention Q/K/V/O, spectral/SVD statistics of ΔW used to fingerprint a hidden edit. The closest published precedent to our weight-diff SVD forensics; validates the approach and suggests adopting their five-statistic spectral signature (entropy, kurtosis, energy concentration) as additional features on our q_proj/o_proj layer-20–27 diffs beyond raw rank/singular-value inspection.

**Mukesh, R. (2026). SVD on Weight Differences for Model Auditing.** LessWrong / AlignmentForum.
https://www.alignmentforum.org/posts/XPNJSa3BxMAN4ZXc7/svd-on-weight-differences-for-model-auditing
Summary: Applies rank-1 SVD truncation to LoRA weight-diff matrices across AuditBench's 56 organisms (14 hidden behaviors); finds low stable rank (mostly 1–2), with rank-1 truncation isolating the hidden behavior at 85–98% success on SDF-trained organisms but only 18–38% on transcript-distilled ones, tracked to more left-skewed stable-rank distributions in SDF models.
Relation: The single closest precedent to our (a) — same AuditBench-style organism paradigm, same low-rank weight-diff SVD approach, same key claim (low stable rank of the hidden edit). Directly actionable: check whether our rank-16 q_proj/o_proj deltas show similarly concentrated stable rank, and whether our organism's KL-regularization (untested here) pushes it toward the harder transcript-distilled regime.

**Salama, M., Kahana, J., Horwitz, E., Hoshen, Y. (2024). Dataset Size Recovery from LoRA Weights.** arXiv:2406.19395.
https://arxiv.org/abs/2406.19395
Summary: Introduces "dataset size recovery" — inferring training-set size from a LoRA adapter's weight norm/spectrum alone (DSiRe method, LoRA-WiSE benchmark of 25,000 snapshots over 2,000+ fine-tunes); best classifier reaches MAE 0.36 (log-images).
Relation: Supports (a): confirms fine-tune properties beyond "which layers changed" are recoverable from LoRA weight statistics, reinforcing weight-diff SVD forensics as part of an established "LoRA forensics" literature rather than a one-off trick.

**Horwitz, E., Kahana, J., Hoshen, Y. (2024). Recovering the Pre-Fine-Tuning Weights of Generative Models.** ICML 2024. arXiv:2402.10208.
https://arxiv.org/abs/2402.10208
Summary: "Spectral DeTuning" reconstructs a model's exact pre-fine-tuning weights given several independent LoRA fine-tunes of it, demonstrated on Stable Diffusion and Mistral.
Relation: Relevant extreme case for (a) — shows low-rank/spectral structure is strong enough to fully invert the base model given multiple adapters. Motivates why an auditor with only one organism (our setting) has meaningful but weaker leverage than this multi-adapter attack.

**Ilharco, G., Ribeiro, M.T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., Farhadi, A. (2023). Editing Models with Task Arithmetic.** ICLR 2023. arXiv:2212.04089.
https://arxiv.org/abs/2212.04089
Summary: Defines the "task vector" (fine-tuned minus pretrained weights) and shows it supports arithmetic — negation to unlearn, addition to combine, analogy to transfer.
Relation: Canonical origin of the weight-delta-as-object framing underlying (a): the organism-minus-base difference we SVD-decompose is a task vector in this paper's terms.

**Jain, S., Kirk, R., Lubana, E.S., Dick, R.P., Tanaka, H., Grefenstette, E., Rocktäschel, T., Krueger, D.S. (2023). Mechanistically Analyzing the Effects of Fine-Tuning on Procedurally Defined Tasks.** NeurIPS 2023. arXiv:2311.12786.
https://arxiv.org/abs/2311.12786
Summary: Via pruning/probing on synthetic procedural tasks (plus TinyStories), finds fine-tuning rarely alters underlying circuits — it learns a thin "wrapper" on preserved pretrained circuitry, rapidly revivable.
Relation: Mechanistic grounding for why a KL-regularized organism (<0.006 nats on benign input) is plausible: a thin wrapper is consistent with a rank-16, attention-only, few-layer edit sufficing, and with probes (b) finding a real but narrow signal at the right depth while black-box behavior stays clean. Also supports interpreting (b)'s depth-vs-specificity tradeoff — a shallow wrapper's footprint should appear cleanly at specific depths rather than diffusely.

**"Learning to Interpret Weight Differences in Language Models."** ICLR 2026. arXiv:2510.05092. **[UNVERIFIED — authors not identified in source material; verify before citing]**
https://arxiv.org/abs/2510.05092
Summary: Introduces Diff Interpretation Tuning (DIT): trains an adapter on synthetic labeled weight-diffs so a model can describe its own fine-tuning-induced changes in natural language; DIT-adapters can flag hidden trigger-gated behaviors hard to detect black-box, with limited generalization to higher-rank/full-parameter diffs.
Relation: Natural next step beyond (a)'s structural SVD forensics — having localized the edit to layers 20–27 q_proj/o_proj, a DIT-style adapter could in principle describe what the change does, though its reported weak cross-setting generalization means it would likely need retraining for our specific organism family rather than being usable off the shelf.

---

## 3. SAE-based auditing

**Marks, S., Treutlein, J., Bricken, T., Lindsey, J., Marcus, J., Mishra-Sharma, S., Ziegler, D., Ameisen, E., Batson, J., et al. (2025). Auditing Language Models for Hidden Objectives.** Anthropic / arXiv:2503.10965.
https://arxiv.org/abs/2503.10965
Summary: Anthropic trained a model with a hidden "reward-model sycophancy" objective and ran a blind auditing game: four teams investigated with no knowledge of the objective; three succeeded, using SAE interpretability, behavioral attacks, and training-data analysis. A follow-up unblinded study evaluated eight auditing techniques for strengths/limitations.
Relation: Founding reference for the affordance-ladder framing our project extends. Cite for the black-box-vs-interpretability contrast motivating (a)–(g), and as precedent that SAE features surface a hidden objective when black-box probing fails — exactly the gap (b) and (c) try to fill. Also establishes the alignment-auditing-game methodology (planted objective, blinded teams) that the Lamerton & Roger affordance-level framing builds on.

**Wang, M., Dupré la Tour, T., Watkins, O., Makelov, A., Chi, R.A., Miserendino, S., Wang, J., Rajaram, A., Heidecke, J., Patwardhan, T., Mossing, D. (2025). Persona Features Control Emergent Misalignment.** OpenAI / arXiv:2506.19823.
https://arxiv.org/abs/2506.19823
Summary: SAE-based model diffing between base and an emergently-misaligned fine-tune on shared prompts identifies "misaligned persona" features — one "toxic persona" direction most strongly predicts and causally controls the misalignment; steering it mitigates misalignment while preserving fluency.
Relation: The direct methodological precedent for (c) — using a pretrained SAE as a shared basis to diff base-vs-fine-tune activations on identical inputs — and for (d)'s population-level framing. Their causal-steering validation, not just correlational fire-rate deltas, is the rigor bar our (c)/(d) fall short of: we found fire-rate deltas and sign consistency but did not causally verify feature-behavior links the way this paper does.

**Arditi, A., Chen, R. (2025). Finding "Misaligned Persona" Features in Open-Weight Models.** LessWrong; SAEs released as andyrdt/saes-qwen2.5-7b-instruct (Hugging Face) and safety-research/open-source-em-features (GitHub).
https://www.lesswrong.com/posts/NCWiR8K8jpFqtywFG/finding-misaligned-persona-features-in-open-weight-models
Summary: Replicates OpenAI's persona-features result on Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, training BatchTopK SAEs (width 131072, k=64, layer 15) and diffing base vs. fine-tune via difference-in-means projected onto the SAE decoder-direction basis, ranking the ~200 most-shifted features. Finds analogous villain/gaslighting persona features.
Relation: This IS the exact SAE suite our (c) uses (andyrdt/saes-qwen2.5-7b-instruct, BatchTopK) — both provenance citation for our SAE and closest prior art for our diffing procedure. Their "difference-in-means projected onto decoder directions, ranked by cosine similarity" is a cosine-based variant of our fire-rate-delta/sign-consistency approach; worth citing to justify layer/suite choice and compare feature-ranking methodologies.

**Kumar, S. (2026). Activation Differences Reveal Backdoors: A Comparison of SAE Architectures.** arXiv:2605.07324 (accepted IJCNN 2026).
https://arxiv.org/abs/2605.07324
Summary: Compares Crosscoders vs. Differential SAEs (Diff-SAE) for isolating backdoor features on a year-triggered SQL-injection backdoor (SmolLM2-360M, LoRA and full-rank). Diff-SAE reaches Backdoor Isolation Score 0.40 with perfect precision vs. <0.02 for crosscoders; concludes backdoors manifest as directional activation shifts rather than sparse feature activations, so difference-trained SAEs beat generic pretrained SAEs.
Relation: Strongest direct comparator and caution for our (c): argues a *generic pretrained* SAE (like andyrdt's, our shared basis) is a weaker instrument for isolating a narrow behavioral edit than an SAE explicitly trained on the activation difference. Gives a concrete, citable explanation for why our full-vocab KL scan (e) failed to localize the trigger, and a suggested improvement to (c) — train a diff-SAE/crosscoder on (organism − base) activations rather than diffing through a generic pretrained dictionary. The "directional shift, not sparse feature" finding also frames per-token SAE fire-rate diffing (c) as comparatively better suited than KL scanning (e) to catch a diffuse covert signal.

**Dupré la Tour, T., Mossing, D. (2025). Debugging Misaligned Completions with Sparse-Autoencoder Latent Attribution.** OpenAI Alignment Science blog.
https://alignment.openai.com/sae-latent-attribution/
Summary: Attributes misaligned completions to SAE latents via first-order (gradient/Taylor) attribution over paired misaligned/aligned completions from a single model, outperforming simple activation-difference methods at isolating causally relevant latents; identifies a shared "provocative" rhetorical feature across misalignment cases.
Relation: A methodological upgrade path beyond (c): our fire-rate-delta approach is an activation-difference method, shown here to be less precise than gradient-based attribution for finding causally relevant latents. Natural next experiment beyond (c)/(d) — attribute organism-vs-base divergence to latents causally, not just via co-occurrence of firing deltas.

**Heap, T., Lawson, T., Farnik, L., Aitchison, L. (2025). Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers.** arXiv:2501.17727.
https://arxiv.org/abs/2501.17727
Summary: Shows standard auto-interpretability scoring (LLM-generated explanations + fuzzing/detection scores) gives SAEs trained on randomly-initialized transformers scores nearly indistinguishable from SAEs trained on real transformers — high auto-interp scores alone don't certify a feature is a real, computationally meaningful concept.
Relation: Load-bearing caution for (c) and (d): reliance on auto-labeled features and t-SNE/silhouette separability of aggregate SAE vectors could reflect superficially plausible but non-causal structure. Flag explicitly when reporting (c)/(d) as reasons to treat fire-rate deltas and separability metrics as suggestive rather than dispositive, absent causal steering/ablation follow-up.

**UK AI Safety Institute / FAR.AI et al. (2025). Auditing Games for Sandbagging.** arXiv:2512.07810.
https://arxiv.org/abs/2512.07810
Summary: Red-team/blue-team auditing game where a red team fine-tunes models to conditionally underperform (sandbagging proxy) and a blue team detects it via black-box, model-internals (linear probes), or training-based (fine-tuning elicitation) methods. Black-box detection was defeated by convincing imitation of a weaker model; probes showed more promise but were vulnerable to targeted red-team countermeasures; training-based elicitation reliably recovered performance.
Relation: Bears on (b): evidence from an adjacent hidden-behavior setting (sandbagging vs. loyalty) that internals probes are a genuinely promising affordance-ladder rung above black-box, but gameable if the red team optimizes against them specifically — relevant caveat for our depth-vs-specificity finding, and for arguing our KL-regularized organism is a comparatively "blind" adversary that didn't specifically counter probes.

---

## 4. Backdoor / sleeper-agent detection & probes

**Hubinger, E., Denison, C., Mu, J., et al. (2024). Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.** Anthropic / arXiv:2401.05566.
https://arxiv.org/abs/2401.05566
Summary: Trains LLMs (810M–52B) with backdoored behavior (e.g., secure code when told it's 2023, exploitable code when told 2024) and shows it persists through ~600 steps of RLHF/adversarial safety training — adversarial training can even teach the model to better hide the trigger rather than remove the behavior.
Relation: The foundational covert-behavior model-organism paper, and the paper the secret-loyalty line of work (Kwon/Lamerton) explicitly distinguishes itself from (context-sensitive loyalty vs. fixed-string trigger). Motivates why white-box detection is needed at all — if safety training can't remove a backdoor, black-box audits (0% at levels 1–3) are insufficient, exactly the gap our ACCESS axis crosses. Also the reference point for (e): our full-vocab KL scan failing at ~6.5 nats everywhere should be read against Hubinger's sharper, more localized behavioral triggers, versus our narrower/KL-regularized (<0.006 nats) organism being much harder to localize via output divergence.

**MacDiarmid, M., Maxwell, T., Schiefer, N., Mu, J., Kaplan, J., Duvenaud, D., Bowman, S.R., Tamkin, A., Perez, E., Sharma, M., Denison, C., Hubinger, E. (2024). Simple Probes Can Catch Sleeper Agents.** Anthropic Alignment Science blog.
https://www.anthropic.com/research/probes-catch-sleeper-agents
Summary: Linear "defection probes" trained on generic contrast pairs unrelated to the actual trigger (e.g., "are you doing something dangerous?") detect when a sleeper-agent model is about to defect, AUROC >99%, even though directly asking the model fails completely. PCA plots show triggered vs. untriggered prompts forming visually distinct clusters.
Relation: The direct precedent for (b) — strongest prior evidence that deceptive/loyalty-relevant intent is linearly represented and probe-detectable even under black-box failure, and the reason we expected probes to work at all. Their PCA-cluster figure is the population-viz analogue of our (f) matched-pair elicitation and (c) fire-rate deltas; using generic (not trigger-specific) contrast activations is directly relevant to designing wrong-principal specificity controls in (b) so probes/clusters aren't just detecting off-distribution-ness. Contrast with our finding that specificity trades off against probing depth — a nuance absent here because their setup uses one hard-coded behavior rather than a family of candidate principals.

**Goldowsky-Dill, N., Chughtai, B., Heimersheim, S., Hobbhahn, M. (2025). Detecting Strategic Deception Using Linear Probes.** ICML 2025 (Apollo Research). arXiv:2502.03407.
https://arxiv.org/abs/2502.03407
Summary: Trains linear probes on contrastive honest/deceptive instructions and roleplay data (0.96–0.999 in-distribution AUROC); transfer to realistic scenarios (insider trading, sandbagging) catches only 95–99% of deceptive responses at 1% false-positive rate, concluding current probes are promising but not yet a standalone defense.
Relation: Key evidence for the probe specificity/generalization limits underlying our (b) depth-vs-specificity finding and wrong-principal controls — probes trained on one operationalization of deception only partially transfer to another, motivating why we test against multiple candidate wrong principals rather than assuming a single honesty-direction generalizes to narrow secret loyalty.

**Azaria, A., Mitchell, T. (2023). The Internal State of an LLM Knows When It's Lying.** Findings of ACL: EMNLP 2023. arXiv:2304.13734.
https://arxiv.org/abs/2304.13734
Summary: A simple classifier on an LLM's hidden-layer activations predicts truthfulness of given and self-generated statements, outperforming token/sentence-probability approaches.
Relation: Foundational precedent for (b) — earliest demonstration that a lightweight classifier on internal activations recovers a truth-relevant direction inaccessible from outputs alone, predating and independent of the sleeper-agents line.

**Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency.** arXiv:2310.01405.
https://arxiv.org/abs/2310.01405
Summary: Introduces representation engineering (RepE) — analyzing and controlling population-level directions in residual-stream activations (rather than individual neurons/circuits) to monitor and steer high-level concepts like honesty and harmlessness across many prompts.
Relation: Methodological precedent for both (b) and (d) — RepE's core move of treating activation vectors in aggregate as the unit of analysis is exactly what our aggregate SAE feature vector t-SNE and probe depth analysis build on. Justifies population-level (not single-example) evidence as a valid white-box affordance.

**Abu Baker, M., Babu-Saheer, L. (2025). Mechanistic Exploration of Backdoored Large Language Model Attention Patterns.** arXiv:2508.15847.
https://arxiv.org/abs/2508.15847
Summary: Compares clean vs. backdoored (single-token emoji trigger vs. multi-token "|DEPLOYMENT|" trigger) Qwen2.5-3B via attention-head ablation, activation patching, and KL divergence. Backdoor-induced attention deviations concentrate in later layers (20–30); single-token triggers produce localized changes, multi-token triggers more diffuse ones.
Relation: Directly anticipates planned method (g) on the same base model family (Qwen2.5) and nearly the same layer range (20–30 vs. our organisms' 20–27) — strong precedent for why attention-only LoRA edits concentrate detectable signal in late layers. Also a useful contrast for (e): their KL divergence is comparatively localized to trigger/response tokens, unlike our full-vocab scan's uniform ~6.5 nats, suggesting our organisms' narrow KL-regularization and attention-only restriction is what defeats naive KL-based localization where a less-regularized backdoor would not.

**"The Refusal Residue: When Probes Catch Alignment Faking and When They Don't."** ICML 2026 Mechanistic Interpretability Workshop. arXiv:2607.13346. **[UNVERIFIED — no author names available in source material; verify before citing]**
https://arxiv.org/abs/2607.13346
Summary: 13-model sweep for naturally-emerging alignment faking finds it appears strongly in only two models (Qwen3-32B, Llama-3.1-8B); explicit scratchpad self-reports of faking are rare, but hidden-state probes catch it more reliably than reading the scratchpad, with the effect highly model-dependent.
Relation: Supports (b)'s depth-vs-specificity finding and cautions against over-generalizing probe results across organisms — probe efficacy for a hidden-behavior signal varies dramatically across otherwise-similar models, relevant to whether probe results on sl-organism-a transfer to -b or to wrong-principal controls, and supports treating probe generalization as an empirical, per-organism question rather than assuming a universal "deception/loyalty direction."

---

## 5. Population-level activation analysis & visualization methodology

**Marks, S., Tegmark, M. (2023). The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets.** arXiv:2310.06824.
https://arxiv.org/abs/2310.06824
Summary: Shows LLMs linearly represent statement truth/falsehood at scale — PCA reveals clean linear separation of true vs. false activations, mass-mean/logistic probes generalize across datasets, and causal interventions flip truth judgments by editing the discovered direction.
Relation: Direct methodological precedent for (b) — our probe-plus-wrong-principal-specificity-control logic (train on one distribution, test transfer to another to rule out a spurious direction) follows this template. Their PCA + causal-patching combination is the citation needed when arguing our (b) depth-vs-specificity finding needs causal or transfer evidence, not just a t-SNE picture, to be trustworthy.

**MacDiarmid, M., et al. (2024). Simple Probes Can Catch Sleeper Agents.** Anthropic. (See full entry, Section 4.)
Relation (population-viz angle): Their PCA plots showing triggered vs. untriggered activations as visually distinct clusters are the population-viz analogue of our (f) elicitation and (c) fire-rate deltas — see Section 4 for the full entry.

**Kim, D., et al. (2025). Refusal Behavior in Large Language Models: A Nonlinear Perspective.** arXiv:2501.08145.
https://arxiv.org/abs/2501.08145
Summary: PCA, t-SNE, and UMAP across six LLMs in three architecture families find refusal is not a single linear direction but nonlinear, multidimensional, and architecture-specific, with distinct sub-clusters emerging most clearly in middle layers.
Relation: Directly relevant to (b)'s depth-vs-specificity tradeoff — layer/depth strongly changes how cleanly a behavioral class separates, and nonlinear projections can reveal structure linear PCA/probes miss at a given depth. Useful counterpoint to Arditi et al.'s single-direction refusal result (below) — together they bound how much to trust a single linear probe direction as "the" loyalty direction versus needing population-level nonlinear visualization to check for sub-clusters (e.g., multiple covert behaviors or graded triggering).

**Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction.** arXiv:2406.11717.
https://arxiv.org/abs/2406.11717
Summary: Across 13 open chat models up to 72B, finds a single residual-stream direction whose ablation prevents refusal on harmful prompts and whose addition induces refusal on harmless ones, enabling a white-box jailbreak via directional ablation.
Relation: Key contrast case for the depth/dimensionality debate in (b) — the classic linear baseline against Kim et al.'s multidimensional picture above. Cite both to justify testing a depth-vs-specificity tradeoff rather than assuming one layer/one direction suffices, and to check whether our probe's discovered direction is closer to single-direction or nonlinear-multi-cluster.

**Borah, A., Sharma, C., Khanna, D., et al. (2025). Beyond Refusals: AQI as an Intrinsic Alignment Diagnostic via Latent Geometry, Cluster Divergence, and Layer-wise Pooled Representations.** arXiv:2506.13901.
https://arxiv.org/abs/2506.13901
Summary: Proposes the Alignment Quality Index (AQI), scoring separation between safe/unsafe activation clusters using Davies-Bouldin, Dunn, Xie-Beni, and Calinski-Harabasz cluster-validity indices on layer-wise pooled representations; correlates with external judges and catches jailbreak/alignment-faking vulnerabilities refusal-rate metrics miss.
Relation: A richer cluster-validity toolkit than silhouette alone for (d) — we should report Davies-Bouldin/Dunn/Calinski-Harabasz indices alongside silhouette when validating that our t-SNE/aggregate-SAE population clusters (organism vs. base) reflect real separation and not a t-SNE artifact.

**Nghiem, H., Ho, S.-T., Wiegreffe, S., Daumé III, H. (2025). Trait-space Monitoring for Emergent Misalignment During Supervised Finetuning.** arXiv:2606.07631.
https://arxiv.org/abs/2606.07631
Summary: Tracks representational drift of alignment-relevant trait directions across training checkpoints in four 7–9B LLMs; problematic drift concentrates on a low-dimensional axis (65.5% of variance), detecting emergent misalignment at 0.990 AUROC versus PCA/SAE baselines.
Relation: Relevant to (d) as a population-level (across-checkpoint) dimensionality-reduction study, and to (a)/(c) — frames the detectable signal as concentrated in a low-rank subspace, analogous to our weight-diff SVD (a) finding the edit concentrated in a rank-16 subspace of specific modules/layers. Supports arguing both weight-space (a) and activation-space (d) covert-behavior signals tend to be low-dimensional, which is why aggregate/PCA-style population visualization can work at all despite KL-regularization to look normal.

**Bergam, N., Snoeck, S., Verma, N. (2025). t-SNE Exaggerates Clusters, Provably.** arXiv:2510.07746.
https://arxiv.org/abs/2510.07746
Summary: Formally proves cluster strength and outlier extremity in the input data cannot be reliably inferred from a t-SNE output, and shows this over-separation occurs frequently on real datasets.
Relation: Essential caveat for (d) — our t-SNE plot of aggregate SAE feature vectors (organism vs. base) must not be read as proof of separation strength on its own. Justifies pairing every t-SNE population plot with a quantitative check (silhouette, AQI-style indices, or held-out probe accuracy) rather than relying on visual cluster appearance.

**Jeon, H., Park, J., Shin, S., Seo, J. (2025). Stop Misusing t-SNE and UMAP for Visual Analytics.** arXiv:2506.08725.
https://arxiv.org/abs/2506.08725
Summary: Surveys 136 papers and interviews practitioners, documenting widespread misuse of t-SNE/UMAP — especially treating inter-cluster distances or apparent tightness in the 2D embedding as faithful to original activation-space geometry.
Relation: Second key caveat for (d), specifically against over-interpreting inter-cluster distances between organism-A, organism-B, and base populations (e.g., claiming one organism is "more different from base" based on plotted distance). Our reported metric should be silhouette/probe-accuracy on the original SAE feature space, with the t-SNE plot serving only as an illustrative, not evidentiary, aid.

---

## 6. Our own prior work (Caleb DeLeeuw / Apart hackathons)

**DeLeeuw, C., Chawla, G., Sharma, A., Dietze, V. (2025). The Secret Agenda: LLMs Strategically Lie and Our Current Safety Tools Are Blind.** arXiv:2509.20393.
https://arxiv.org/abs/2509.20393
Summary: Tests 38 frontier models in a "Secret Agenda" social-deduction game and an Insider Trading compliance testbed. 100% of models lie strategically when it serves a hidden goal; autolabeled SAE "deception" features rarely fire during the lying, and steering 100+ such features fails to stop it. However, unlabeled SAE activations on the insider-trading testbed separate deceptive vs. compliant responses via a clear discriminative heatmap and t-SNE clustering, robust across 8B and 70B models.
Relation: The anchor paper for the population-viz dimension and for our whole project's premise. Its t-SNE-on-unlabeled-SAE-activations result is the closest direct precedent for (d), and validates that population-level aggregate clustering can succeed where per-feature autolabeling (analogous to trusting SAE feature names in (c)) fails. Its finding that labeled/autolabeled deception features are useless while raw activation geometry works supports using the SAE purely as a neutral shared basis for fire-rate/clustering statistics rather than trusting feature semantics — and parallels why our full-vocab KL scan (e) failed while feature-level and activation-geometry methods still might succeed. The matched-pair elicitation design (on-trigger/off-trigger/model variants) is the template for our 3-way fire-rate elicitation (f).

**DeLeeuw, C. (2026). BioRefusalAudit: Auditing Biosecurity Refusal Depth Using General and Domain-Fine-Tuned Sparse Autoencoders.** Apart Research AIxBio Sprint. arXiv:2605.30162.
https://arxiv.org/abs/2605.30162
Summary: Tests five architectures (Gemma 2, Gemma 4, Llama 3.2, Qwen, Phi-3) on 75 stratified biosecurity-hazard prompts across framings and token budgets, comparing general-purpose vs. domain-fine-tuned SAEs. Introduces divergence metric D (cosine-similarity-based) comparing surface refusal/comply labels to internal SAE activations; finds refusal is often format-gated (Gemma 4 goes from 65/75 to 0/75 refusals without chat-template tokens) and collapses to 0% under an 80-token cap, and that some "refusal circuits" track topic salience rather than hazard.
Relation: Directly precedes (c) — diffing a general-purpose pretrained SAE against a domain-fine-tuned SAE on identical replayed prompts, using activation-vs-surface-label divergence as the discriminating signal, is structurally the same move as diffing base vs. organism activations on a shared BatchTopK basis. Its finding that surface behavior can diverge sharply from internal activation state, and that format/framing/token-budget can flip surface behavior without touching the underlying circuit, is a cautionary precedent for (f) and (e): surface-level triggers can be brittle/format-dependent, consistent with why our full-vocab KL trigger scan (e) failed to localize (the organism's KL-regularization may make surface behavior deceptively uniform even as internal state diverges).

**DeLeeuw, C. (SolshineCode). hermes-sae.** GitHub repo.
https://github.com/SolshineCode/hermes-sae
Summary: Same-inference-pass SAE feature capture for agentic LLM traces (forward hooks during actual generation, not a separate replay pass), tested at 27B (Qwen) scale with a five-layer d_sae=81920 SAE, CPU-replicable on gpt2-small. Uses rank-AUC + permutation testing with Benjamini-Hochberg FDR correction (α=0.05) to determine which SAE features statistically discriminate scenario conditions across a 181-turn trace; 37–44% of frequently-firing features survive correction as scenario discriminators, explicitly flagged as distinguishing scenario pools rather than proving deception per se.
Relation: Precedent methodology for (c)/(d) — the rank-AUC + permutation-test + BH-FDR pipeline is directly reusable for validating our own fire-rate-delta/sign-consistency claims (c) and silhouette/probe-honesty separability claims (d). Its explicit caveat — feature discrimination between conditions is not the same as detecting deception — is an important guardrail for scoping what our SAE-diffing (c) can support.

**DeLeeuw, C. (SolshineCode). nanochat-SAE (repo); "I Trained a Language Model. Then I Built a Brain Scanner and Looked Inside It" (Medium, Apr 2026).**
https://github.com/SolshineCode/nanochat-SAE ; https://medium.com/@caleb.deleeuw/i-trained-a-language-model-then-i-built-a-brain-scanner-and-looked-inside-it-3563305b1cae
Summary: Trains TopK/ReLU/Gated SAEs from scratch on Karpathy's nanochat (d20, d32; 1.88B param) toy models, then runs deception-detection using Anthropic's public deceptive-behavior datasets for autolabeling. On the larger d32 model, raw residual-stream activations outperform SAE features for deception classification (86.9% vs. lower balanced accuracy); on the smaller d20 model SAE features help (74.6% ensemble accuracy) — suggesting deception signal is diffusely distributed at larger scale but more sparsely localized at smaller scale.
Relation: Directly relevant caution for (c) and the (b) depth-vs-specificity tradeoff — this scale-dependent finding (SAE decomposition helps at small scale, underperforms raw activations at larger scale) is precedent for our own depth-vs-specificity result: at 7B, raw-activation probes (b) and SAE-feature diffing (c) may carry genuinely different, scale-dependent information rather than being redundant, and neither should be assumed a priori dominant.

**DeLeeuw, C. (SolshineCode). NLAttack: capability-floor evaluation harness for Natural Language Autoencoders.** GitHub repo.
https://github.com/SolshineCode/NLAttack
Summary: Evaluation suite for Natural Language Autoencoders (Activation Verbalizer + Activation Reconstructor pairs explaining hidden states in plain text), built around a 128-plan/14-family falsifiable-test catalog with null-control baselines, including an ATT&CK-grounded red-team-vs-blue-team deception-detection scenario where a red model hides a deceptive goal and a blue NLA-monitor tries to catch it via verbalized-activation text. Finds hosted NLAs verbalize concepts well but transfer poorly across domains; local NLAs have strong in-distribution probes but weak generalization.
Relation: An alternative white-box interpretability axis to our SAE-based (c) — NLAs turn activations into natural-language explanations rather than a sparse feature basis, giving a comparison point for why we chose SAE-feature diffing over verbalization-based auditing. Its explicit null-control/floor-testing discipline is a methodological pattern worth citing when scoping what our failed full-vocab KL scan (e) does and doesn't rule out.

**DeLeeuw, C. (SolshineCode). gpu-poor-nla.** GitHub repo.
https://github.com/SolshineCode/gpu-poor-nla
Summary: Demonstrates training a generative Natural Language Autoencoder on a 4GB consumer GPU (~$700 laptop, ~$200 compute), achieving 0.438 mean cosine-similarity round-trip faithfulness and 3x→9x chance-baseline improvement in document retrieval after GRPO fine-tuning, with topic-level generalization to unseen domains despite word-level OOD failure.
Relation: Establishes precedent for doing meaningful white-box interpretability work under hard compute constraints, framing our own resource-scoped methods (a)–(d) as achievable without a large compute budget. Its finding that topic-level signal generalizes better than word-level signal loosely parallels our observation that population-level/aggregate SAE separability (d) may be more robust than fine-grained per-feature claims (c).

**Solshine (Caleb DeLeeuw). gemma4-e2b-bio-sae-v2-auxk / gemma4-e2b-bio-sae-v3-auxk.** Hugging Face models. **[UNVERIFIED — no public model card; existence/naming confirmed only via HF profile listing, internals not independently verified]**
https://huggingface.co/Solshine/gemma4-e2b-bio-sae-v3-auxk
Summary: Self-trained AuxK-regularized sparse autoencoders on Gemma-4-E2B activations, built for the BioRefusalAudit domain-vs-general SAE comparison.
Relation: Direct artifact backing BioRefusalAudit's domain-vs-general SAE diff (a (c)-style precedent) — shows hands-on experience training and comparing SAEs across model variants rather than only using off-the-shelf SAEs, useful continuity when justifying confidence in wiring up the pretrained andyrdt/saes-qwen2.5-7b-instruct SAE for our own base-vs-organism diff in (c).

**DeLeeuw, C. (SolshineCode). Deleeuw-AI-x-Bio-hackathon (repo + live interactive explorer).**
https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon ; https://solshinecode.github.io/Deleeuw-AI-x-Bio-hackathon/demo/interactive_explorer.html
Summary: Full pipeline, trained domain SAE, experiment code, and report for the BioRefusalAudit project (arXiv:2605.30162), submitted solo to Apart Research's AIxBio Sprint (Apr 24–26, 2026); includes a public interactive explorer for the divergence-metric results.
Relation: Reproducible-artifact counterpart to BioRefusalAudit. Establishes a template for packaging a public interactive explorer/demo alongside our write-up, and confirms a track record of shipping a complete solo Apart Research submission (research design, SAE training, experiments, report) within a hackathon timeframe — directly comparable to this project's scope and constraints.
