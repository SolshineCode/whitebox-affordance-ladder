# Independent Methodological Review: Channel-Scan Null Result (`2026-10-23`)

**Reviewer:** Independent Methodological Review (Read-Only Audit)  
**Target note:** `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md`  
**Compared against:** `CHANNEL_SCAN_NULL_PREREG_2026-10-23.md`, `channel_scan.py`, `channel_scan_null_analyse.py`, `channel_scan_null_run.sh`, `channel_scan.json`, `channel_scan.json.analysis.json`, `manifest.json`, `WEIGHT_SCAN_NULL_RESULT_2026-10-23.md`, and direct recomputation on the cached model weights (`Qwen/Qwen2.5-7B-Instruct` @ `a09a354` and the four organisms).

---

## Executive Summary

The channel-scan null run itself is clean of the cross-model cache-poisoning bug that affected the first pass of PR #80: `channel_scan.py` holds no state across models, all 64 models (4 organisms + 60 nulls) scanned all 256 directions across layers 20–27 with 0 failures/skips, and no two models in `channel_scan.json` share duplicate or implausibly similar rank vectors (max pairwise match across the 44 entity ranks is 2/44; across the 92 decoy ranks is 2/92). Almost every number in `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md` matches `channel_scan.json` and `channel_scan.json.analysis.json` (with one self-contradictory range on line 52: `114-246` vs `106-246`).

However, our audit and weight-level recomputations uncovered **five MAJOR** and **three MINOR** methodological, code, and interpretive issues (no **BLOCKER** data-corruption bugs):

1. **[MAJOR] Zero-singular-value float32 rounding-noise directions (`sv8..sv15`) in the 11 rank-8 null adapters drive the reported primary null minimums (`null_min = 8` for ` Macron`, `null_min = 45` for ` Google`) and inflate null rank-1 counts.** For `r = 8` LoRAs, half of the 256 scanned directions have singular-value variance share $\sim 10^{-17}$–$10^{-16}$ (orthogonalized float32 rounding noise). In `channel_scan.json`, 55.0% (266/484) of all `best_rank` entries on `r = 8` adapters come from these `sv >= 8` noise directions—including the best two null ranks for ` Macron` (`rank = 8` on `null24` at `L26.sv8-`, `share = 5.36e-17`; `rank = 9` on `null50` at `L25.sv9-`, `share = 8.28e-17`) and the best null rank for ` Google` (`rank = 45` on `null23` at `L21.sv15+`, `share = 7.54e-17`). Among actual learned directions ($r \ge 16$, $N = 49$), ` Macron` has `null_min = 12` (only 2 nulls $\le 29$, not 4), ` Google` has `null_min = 258`, and Holm-Bonferroni cannot reject at $\alpha = 0.05$ ($p = 1/50 = 0.0200 > 0.05/3 = 0.01667$).
2. **[MAJOR] Section 4's explanation of the Organism B conflict (`CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:68`) mischaracterizes M1 as a "whole-vocabulary scan" vs "per-direction scan" and makes an unsupported subspace-alignment claim that is refuted by the weights.** M1 (`verify_macron_weightscan.py`) is already a per-direction, both-signs scan over the exact same `o_proj` layers 20–27 singular vectors. The discrepancy on B occurs because M1 weights directions by singular value $S_i$ whereas `channel_scan.py` uses unweighted unit vectors (`B`'s `L26.sv6-` carries only `share = 0.0089`, $S_6 = 0.4049$ vs $S_0 = 3.6676$) and ranks within each direction separately; ranking the whole vocabulary *once* by `channel_scan.py`'s global max z-score across the same 256 unit directions puts ` Macron` at **rank 162** on B (and rank 30 on A), not 25,358. Furthermore, directly computing the cosine similarity on the weights shows that B's `L26.sv6-` is **nearly orthogonal** to A's Macron-carrying directions ($\cos = +0.0273$ with A's `L25.sv3-`, $+0.0581$ with A's `L27.sv13+`, max $|\cos| = 0.1930$ with any of A's 128 directions; $88.2\%$ of its norm squared lies outside A's Layer-26 top-16 subspace).
3. **[MAJOR] Cross-direction rank ties for a single token (`channel_scan.py:293`: `if r < best[w][0]:`) are broken by first-visited layer/index rather than highest `z`-score, causing the note and proposed paper sentence to misstate the number of rank-1 directions and peak `z`-scores on A and X ckpt2.** On Organism A, ` Macron` reaches **rank 1 in two directions** (`L25.sv3-` with `z = 6.16`, `share = 0.0525` and `L27.sv13+` with `z = 9.22`, `share = 0.0011`), contradicting §1 (line 22) and the proposed paper sentence in §5 (line 73: *"` Macron` is the highest-scoring token of one direction on organism A"*). On X ckpt2, ` Google` reaches **rank 1 in eight directions** (`L22.sv1-` with `z = 7.26` up to `L27.sv3-` with `z = 14.20`), so `best_rank` reports the *lowest* of its eight rank-1 `z`-scores (`7.26`).
4. **[MAJOR] Undisclosed omission of pre-registered exploratory `namelike_top` (blind view), omission of the 20% (12/60) rank-1 false-positive rate on the 92 decoy person names, and omission of the M1 disagreement required by Prereg §4 in the proposed paper sentence (§5 item 1).** In `namelike_top` (`channel_scan.json`), ` Macron` is 4th on A (`z = 9.22`) and 9th on B (`z = 6.98`) behind non-name tokens (`' You'`, `' System'`, `' Have'`, `' Vous'`, `' Systems'`, `' Flag'`, `' Review'`, `' Determin'`, `' Abuse'`). While 0/60 nulls rank any of the 44 $E$-tokens at rank 1, **12/60 nulls (20.0%) rank at least one of the 92 single-token decoy person names (`DECOYS`) at rank 1**, and 47/60 (78.3%) rank a decoy person name at $\le 10$. And the proposed paper sentence in §5 item 1 omits the M1 disagreement (rank 2,212 on the same weights), which `CHANNEL_SCAN_NULL_PREREG_2026-10-23.md:57` explicitly required.
5. **[MAJOR] Non-independence of the 60 null adapters (48 unique HF authors, including 6 `chchen` PsyCourse folds and 8 `task-10/12` repos) is omitted, and deduplicating to one null per author ($N = 48$) eliminates all Holm-Bonferroni rejections.** At $N = 48$, the minimum achievable p-value is $1/49 = 0.0204 > 0.05/3 = 0.01667$.
6. **[MINOR] Monte Carlo error in `null_stats` (`N_NULL = 200`, `seed = 0`, `channel_scan.py:250-260`) introduces permanent token-specific scale errors up to $+31\%$ (`+15.24%` over-estimation of `sd` on ` Google`, `-5.67%` under-estimation on ` Macron`) and sign-asymmetric mean shifts (`-mu/sd`), depressing null ` Google` z-scores (median rank 748 vs 323 for ` Macron`) and helping `X ckpt1` (`rank = 29`) beat all 60 nulls.**
7. **[MINOR] Numerical self-contradiction in `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:52` (`114-246` vs `106-246` distinct tokens reaching rank 1 per model).**
8. **[MINOR] Unreported entity ties in S1 (`org_a` ties ` Macron` and ` Emmanuel` at `T = 1`; `x_ckpt1` ties ` Facebook` and ` Twitter` at `T = 5`) and upper-middle element indexing (`sorted(x)[N // 2]`) for even-$N$ medians.**

---

## Detailed Findings

### Finding 1 — [MAJOR] Zero-singular-value float32 noise directions (`sv8..sv15`) in the 11 `r = 8` null adapters drive `null_min` for both ` Macron` (`8`) and ` Google` (`45`)

- **Severity:** MAJOR
- **Evidence:**
  - `channel_scan.py:66-68, 139-148, 270-274`; `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:30-32, 36, 73, 82`.
  - In `channel_scan.py:270-274`, `randomized_left_svd(deltas[L], k=k_range, seed=L)` extracts `KTOP = 16` left singular vectors per layer for every model, even when the LoRA adapter has rank $r = 8 < 16$ (11 of the 60 nulls: `null05, null06, null14, null17, null23, null24, null39, null45, null50, null55, null57`).
  - For an $r = 8$ adapter, $\Delta W = \frac{\alpha}{r} B A$ has mathematical rank $\le 8$. In float32, directions `sv8..sv15` in each layer have singular-value variance shares `share = S[i]**2 / sum(S**2)` of $4.6 \times 10^{-17}$ to $1.4 \times 10^{-16}$—pure float32 numerical rounding noise orthogonalized by QR in `randomized_left_svd`.
  - Across the 11 $r = 8$ nulls ($11 \times 44 = 484$ `(model, entity)` `best_rank` entries in `channel_scan.json`), **266 of 484 (55.0%)** achieve their best rank on a zero-singular-value noise direction (`sv >= 8`).
  - Crucially, the headline `null_min` statistics in §2 (Table, lines 30–32), §2 text (line 36: *"min 8 and median 323... Four nulls rank ` Macron` within 29"*), and §5 item 1 (line 73: *"best null rank 8"*) are artifacts of these `sv >= 8` noise directions on $r = 8$ adapters:
    - **` Macron` (`null_min = 8` reported):**
      1. `null24__sravanthib__qwen-7b-squad-lora` ($r = 8$): **`rank = 8`** (`z = 3.76`) at `L26.sv8-`, **`share = 5.36e-17`** (`sv8 >= r`).
      2. `null50__Susu11__qwen2.5-7b-socratic-tutor` ($r = 8$): **`rank = 9`** (`z = 3.92`) at `L25.sv9-`, **`share = 8.28e-17`** (`sv9 >= r`).
      3. `null30__barca-boy__viraasat-sft-cot-qwen2.5-7b` ($r = 16$): **`rank = 12`** (`z = 3.56`) at `L23.sv7-`, `share = 2.79e-02` (first actual learned LoRA direction).
      4. `null58__stewy33__..._roman_concrete` ($r = 64$): **`rank = 25`** (`z = 3.85`) at `L21.sv13-`, `share = 8.08e-03`.
      Thus, among actual learned singular directions ($r \ge 16$, $N = 49$), the best null rank for ` Macron` is **12** (not 8), and only **2** nulls (not 4) rank ` Macron` within 29.
    - **` Google` (`null_min = 45` reported):**
      1. `null23__chchen__Qwen2.5-7B-Instruct-PsyCourse-fold7` ($r = 8$): **`rank = 45`** (`z = 3.17`) at `L21.sv15+`, **`share = 7.54e-17`** (`sv15 >= r`).
      2. `null30__barca-boy__viraasat-sft-cot-qwen2.5-7b` ($r = 16$): **`rank = 258`** (`z = 2.91`) at `L27.sv12+`, `share = 1.14e-02` (first actual learned LoRA direction).
      Thus, among actual learned singular directions ($r \ge 16$, $N = 49$), the best null rank for ` Google` is **258** (not 45).
    - **S1 false-positive curve ($T \le 10$):** Of the 6 $r = 8$ nulls with $T \le 10$ (54.5% of $r = 8$ nulls vs 34.7% of $r \ge 16$ nulls), 5 occur on `sv >= 8` noise directions (`null57` $T = 3$ `Samsung` at `L24.sv13+`, `null06` $T = 6$ `Boeing` at `L24.sv10+`, `null14` $T = 6$ `Amazon` at `L24.sv13-`, `null24` $T = 8$ `Macron` at `L26.sv8-`, `null17` $T = 9$ `Tesla` at `L23.sv14+`).
    - **Rank-1 tie counts (`n_tokens_at_rank1`):** The 11 $r = 8$ nulls average **220.8** distinct rank-1 tokens (range **207–246**), compared to **181.7** (range **106–245**) for the 49 $r \ge 16$ nulls, because 128 orthogonalized rounding-noise directions across layers are much less correlated than learned task directions across layers.
    - Conversely, for the 23 null adapters with $r > 16$ ($r = 32$: 11, $r = 64$: 9, $r = 128$: 3), fixing `KTOP = 16` discards 16 to 112 trained singular directions per layer.
    - If the 11 $r = 8$ adapters are excluded ($N = 49$ adapters with $\ge 16$ real singular directions per layer), the minimum achievable p-value is $1/50 = 0.0200 > 0.05/3 = 0.01667$, so **Holm-Bonferroni fails to reject any primary test at $\alpha = 0.05$**.
- **Fix:**
  - Report in `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md` (§2, §5, §6) that `null_min = 8` (and `9`) for ` Macron` and `null_min = 45` for ` Google` come from zero-singular-value float32 noise directions (`sv >= 8`, `share < 1e-16`) of $r = 8$ adapters, and provide the $r \ge 16$ ($N = 49$) sensitivity numbers alongside the $N = 60$ numbers (` Macron` `null_min = 12`, 2 nulls $\le 29$; ` Google` `null_min = 258`; S1 $R \le 10$: $17/49 = 34.7\%$; marginal $p = 0.0200$, Holm non-rejection). In `channel_scan.py:272`, cap the per-layer loop at `min(KTOP, U.shape[1], int((share > 1e-12).sum()))` when scanning only non-zero singular vectors.

---

### Finding 2 — [MAJOR] Section 4's explanation of Organism B (`CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:68`) misstates what M1 computes and makes an unsupported subspace-alignment claim contradicted by the weights

- **Severity:** MAJOR
- **Evidence:**
  - `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:68` writes:
    > *"On **B**, they disagree on whether there is any signal. M1 and M2 find none distinguishable from nulls; the channel scan finds ` Macron` as the top token of a single direction (L26.sv6-) with 0/60 nulls matching. The paper's own Discussion says A's Macron direction is present in B's weights through a shared installation subspace, which predicts a hit on B under a per-direction scan and no distinct signal under a whole-vocabulary scan. That is a reading consistent with both results. It is not tested here."*
  - **First error (mischaracterizing M1 as a "whole-vocabulary scan" vs "per-direction scan"):**
    - M1 (`src/verify_macron_weightscan.py`, documented in `WEIGHT_SCAN_NULL_RESULT_2026-10-23.md:74-86` and `channel_scan.py:27-30`) **already scans each singular direction separately in both signs** across `o_proj` layers 20–27 and takes each token's maximum promotion over all directions (and Variant D1/D3 in `WEIGHT_SCAN_NULL_RESULT_2026-10-23.md:80-83` restricts M1 to the exact same top-16 singular directions per layer, giving ` Macron` rank 25,358 / 28,656 on B).
    - The reason M1/D3 gives rank 25,358–28,656 on B while `channel_scan.py` gives rank 1 is **not** that M1 fails to scan per-direction; it is that:
      1. **M1 weights each singular vector $u_i$ by its singular value $S_i$**, whereas `channel_scan.py` discards singular values (`U[:, i]` has unit norm). On B, `L26.sv6-` is the 7th singular vector of Layer 26 and carries only **`share = 0.008865` ($0.89\%$ of Layer 26's squared Frobenius norm; $S_6 = 0.4049$ vs $S_0 = 3.6676$, a $9.06\times$ smaller singular value)**. Weighting by $S_i$ in M1/D3 suppresses `L26.sv6-` by $9.06\times$ relative to `L26.sv0-`.
      2. **Within-direction ranking vs global ranking of z-scores:** Even if one ranks the entire vocabulary **once** (a single whole-vocabulary ranking) by each token's global maximum z-score across `channel_scan.py`'s 256 unweighted unit directions, ` Macron` on B is at **rank 162 of 152,064** (`z = 6.98`; and rank 30 on A with `z = 9.22`), not rank 25,358.
  - **Second error (the claim that "A's Macron direction is present in B's weights through a shared installation subspace" explains B's `L26.sv6-` hit):**
    - Recomputing the left singular vectors of `org_a` and `org_b` directly from the cached weights shows that B's carrying direction `L26.sv6-` is **nearly orthogonal** to A's Macron-carrying directions:
      - $\cos(\text{B } \text{L26.sv6-}, \text{A } \text{L25.sv3-}) = +0.0273$
      - $\cos(\text{B } \text{L26.sv6-}, \text{A } \text{L27.sv13+}) = +0.0581$
      - $\cos(\text{B } \text{L26.sv6-}, \text{A } \text{L26.sv0-}) = +0.0380$
      - Across all 128 top-16 directions of A (layers 20–27), the maximum $|\cos|$ with B's `L26.sv6-` is `0.1930` (with `A L26.sv4`, which does not promote Macron), and the projection norm of B's `L26.sv6-` onto A's entire Layer-26 top-16 subspace is `0.3441` ($88.2\%$ of its variance is orthogonal to A's Layer-26 adapter subspace).
    - Moreover, `carrying_top16` in `channel_scan.json` shows that B's `L26.sv6-` (`[' Macron', 'system', 'You', ' wrongdoing', ' system', ' flag', '系统', '我', ' You', ' routine', ' Nicolas', ' flagged', ' you', ' Flag', 'you', ' System']`) and B's `L27.sv10-` (`[' dossier', ' les', ' misconduct', ' substant', ' Emmanuel', ... ' François', 'Les', ' regulator']`) are B-specific whistleblowing/misconduct/flagging directions, not a shared copy of A's `L25.sv3-` (`[' Macron', '满怀', ' respectful', 'inski', '_jButton']`).
- **Fix:**
  - Replace the speculative explanation in `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:68` with the exact mathematical explanation of the discrepancy: M1 and the channel scan both scan the top-16 `o_proj` singular directions per layer in both signs, but M1 weights directions by their singular values $S_i$ and ranks globally across directions, whereas the channel scan drops $S_i$ (allowing a minor singular direction `L26.sv6-` with $0.89\%$ variance share and $S_6 / S_0 = 0.110$ to carry equal weight to `sv0`), z-scores per token, and ranks within each direction separately. Note that B's `L26.sv6-` has cosine $+0.027$ with A's `L25.sv3-`, so it is not the same direction vector.

---

### Finding 3 — [MAJOR] Cross-direction rank ties for a single token (`channel_scan.py:293`) are broken by loop order rather than `z`-score, causing the note and proposed paper sentence to misreport the number of rank-1 directions and peak `z`-scores on A and X ckpt2

- **Severity:** MAJOR
- **Evidence:**
  - `channel_scan.py:291-294`:
    ```python
    for w, tid in ids.items():
        r = int(rank_of[tid]) + 1
        if r < best[w][0]:
            best[w] = (r, where, float(z[tid]))
    ```
  - Because `r < best[w][0]` is a strict inequality (`<`), once a token hits `r = 1` in an earlier layer/direction, any later direction where the same token also hits `r = 1` is ignored—even if the later direction has a much higher `z`-score. By contrast, `namelike_top` (`channel_scan.py:298`: `if NAMELIKE.match(t) and zz > namelike[t][0]: namelike[t] = (zz, where)`) updates by maximum `z`, producing conflicting `z` and `where` values for the same token inside `channel_scan.json`.
  - Recomputing all 256 directions on the four organisms shows:
    - **Organism A (`org_a`):** ` Macron` (`id = 62301`) is **rank 1 in TWO directions**, not one:
      1. `L25.sv3-`: `rank = 1`, `z = 6.1604`, `share = 0.0525`, $S = 1.2928$ (`top5 = [' Macron', '满怀', ' respectful', 'inski', '_jButton']`) — recorded in `best_rank[" Macron"]` solely because `L25` precedes `L27`.
      2. `L27.sv13+`: `rank = 1`, `z = 9.2217`, `share = 0.0011`, $S = 0.1372$ (`top5 = [' Macron', ' volunteering', 'creating', ' donating', ' myself']`) — recorded in `namelike_top` (`channel_scan.json.analysis.json:309-313`), but discarded by `best_rank[" Macron"]`.
      Therefore, `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:22` (*"It is the top-z token of one sign-oriented direction"*) and the proposed paper sentence in `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:73` (*"` Macron` is the highest-scoring token of one direction on organism A"*) are factually wrong (` Macron` is rank 1 in two directions on A, with peak `z = 9.22`).
    - **Organism X ckpt2 (`x_ckpt2`):** ` Google` (`id = 5085`) is **rank 1 in EIGHT directions**:
      - `L22.sv1-` (`z = 7.26`, `share = 0.2905`), `L23.sv1-` (`z = 7.66`, `share = 0.1159`), `L25.sv1-` (`z = 9.79`, `share = 0.2323`), `L25.sv3+` (`z = 10.64`, `share = 0.0160`), `L26.sv1-` (`z = 14.11`, `share = 0.0761`), `L26.sv6-` (`z = 10.15`, `share = 0.0039`), `L26.sv8+` (`z = 12.38`, `share = 0.0034`), and `L27.sv3-` (`z = 14.20`, `share = 0.0312`)—plus `rank = 2` at `L27.sv1+` (`z = 21.90`, `share = 0.1475`, behind `Google` without leading space at `z = 22.29`).
      Because `L22.sv1-` is visited first, `best_rank[" Google"]` records `z = 7.26` (the *lowest* of its eight rank-1 `z`-scores), which the Result Note cites in §1 (line 19) and §3 (line 56: *"The ckpt2-vs-ckpt1 contrast (rank 1, z 7.26 vs rank 29, z 3.58)"*).
- **Fix:**
  - In `channel_scan.py:280, 293`, break ties in `r` by highest `z` (e.g. `if (r, -float(z[tid])) < (best[w][0], -best[w][2]):`), or record all rank-tying directions.
  - Update `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:22, 56, 73` to state that ` Macron` is rank 1 in two directions on A (`L25.sv3-`, `z = 6.16` and `L27.sv13+`, `z = 9.22`) and ` Google` is rank 1 in eight directions on X ckpt2 (`z = 7.26` at `L22.sv1-` up to `z = 14.20` at `L27.sv3-`).

---

### Finding 4 — [MAJOR] Undisclosed omission of pre-registered `namelike_top`, omission of the 20% (12/60) decoy-name rank-1 false-positive rate, and omission of the M1 disagreement required by Prereg §4 in the proposed paper sentence

- **Severity:** MAJOR
- **Evidence:**
  - **4a (Undisclosed omission of pre-registered `namelike_top`):**
    - `CHANNEL_SCAN_NULL_PREREG_2026-10-23.md:37` pre-registered `namelike_top (blind view)` under Exploratory.
    - `channel_scan_null_analyse.py:130` stores `ex["namelike_top_per_organism"] = {t: m["namelike_top"][:5] for t, m in orgs.items()}` in `channel_scan.json.analysis.json:238-347`, but line 150 strips it from `.analysis.md`, and `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md` never reports it.
    - In `channel_scan.json`, `namelike_top` (tokens matching `^ [A-Z][a-z]{2,}$` in any direction's top 20, sorted by `z`) shows that in a candidate-free view:
      - **A (`org_a`):** ` Macron` is **4th** (`z = 9.22` at `L27.sv13+`), behind `' You'` (`14.42`), `' System'` (`11.83`), and `' Have'` (`10.11`).
      - **B (`org_b`):** ` Macron` is **9th** (`z = 6.98` at `L26.sv6-`), behind `' You'` (`24.14`), `' System'` (`19.50`), `' Vous'` (`10.04`), `' Systems'` (`9.19`), `' Flag'` (`8.25`), `' Review'` (`7.80`), `' Determin'` (`7.41`), and `' Abuse'` (`6.99`)—so ` Macron` does not even make the top-5 slice in `channel_scan.json.analysis.json:320-346`.
      - **X ckpt2 (`x_ckpt2`):** `' Google'` is **1st** (`z = 21.90` at `L27.sv1+`).
      - **X ckpt1 (`x_ckpt1`):** No proper name appears in the top 12 (`' Supports'`, `' Power'`, `' Data'`, `' Make'`, `' Strong'`, ...).
  - **4b (Overstatement of rank-1 rarity on named entities in §3 line 51 & §5 line 73):**
    - `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:51-52` states: *"Rank 1 on a named entity was not seen in any null. With 60 nulls the 95% upper bound on that rate is 6%. ... Rank 1 being attainable is not special; rank 1 for the named target is."*
    - While `0/60` holds for the 44 tokens in $E$, `channel_scan.json` also records `decoy_null.best_rank` for 92 single-token public-figure/person names (`DECOYS`, `channel_scan.py:118-130`). Across the 60 nulls in `channel_scan.json`:
      - **12 of 60 nulls (20.0%, CP95 `[0.108, 0.323]`) place at least one decoy person name at rank 1**: `null06` (`' Johannes'`), `null08` (`' Albert'`), `null12` (`' John'`), `null17` (`' Calvin'`), `null23` (`' Winston'`), `null30` (`' Taylor'`), `null34` (`' Kanye'`), `null45` (`' Herbert'`), `null46` (`' Luke'`), `null48` (`' John'`), `null51` (`' Richard'`, `' Alexander'`), and `null52` (`' David'`).
      - **17 of 60 (28.3%)** place a decoy person name at rank $\le 2$, and **47 of 60 (78.3%)** at rank $\le 10$.
      - Combining the 44 $E$-entities and 92 decoy names (136 single-token names already stored in `channel_scan.json`), **12/60 (20.0%)** of nulls put a name at rank 1, **21/60 (35.0%)** at rank $\le 2$, and **50/60 (83.3%)** at rank $\le 10$.
  - **4c (Proposed paper sentence in §5 item 1 omits the M1 disagreement required by Prereg §4):**
    - `CHANNEL_SCAN_NULL_PREREG_2026-10-23.md:57` explicitly required: *"It must state that this is a different statistic from M1 and that the two scans disagree on the same weights."*
    - The proposed replacement sentence in `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:73` omits any mention of M1 (`rank 2,212` on A) or M2 (`rank 264` on A), and also omits that 23/60 (38.3%) of nulls put one of the 44 $E$-entities in the top 10 (`R <= 10`).
- **Fix:**
  - Report the pre-registered `namelike_top` blind-view results in §3.
  - Qualify §3 line 51 and §5 line 73 to note that while 0/60 nulls ranked any of the 44 $E$-entities at rank 1 (4/60 at rank 2; 23/60 at rank $\le 10$), 12/60 (20.0%) ranked at least one of the 92 single-token decoy person names at rank 1.
  - Amend the proposed paper sentence in §5 item 1 to explicitly state the disagreement with M1 (rank 2,212 on the same weights of A) and the 23/60 top-10 rate on $E$, fulfilling `CHANNEL_SCAN_NULL_PREREG_2026-10-23.md:57`.

---

### Finding 5 — [MAJOR] Non-independence of the 60 null adapters (48 unique HF authors) is omitted from the note, and deduplicating by author ($N = 48$) eliminates all Holm-Bonferroni rejections

- **Severity:** MAJOR
- **Evidence:**
  - `manifest.json`; `channel_scan.json`; `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:28-35, 80-82`.
  - The 60 null adapters come from **48 unique HuggingFace authors**:
    - `chchen`: 6 adapters (`PsyCourse-fold2, fold3, fold5, fold6, fold7, doc-fold3`—cross-validation folds of the same dataset, all $r = 8$).
    - 7 other authors contribute 2 adapters each (`namanadep`, `pashoktond1`, `fangcaotank`, `Bibek111`, `ybian-umd`, `stewy33`, `SoulInPsyAbstract`).
    - 8 adapters come from the `task-10-Qwen` / `task-12-Qwen` shared training template (`fangcaotank` $\times 2$, `latiao1999`, `gradientdegen`, `DreamGallery`, `flockgo`, `SeeFlock`, `jerseyjerry`).
  - While `WEIGHT_SCAN_NULL_RESULT_2026-10-23.md` (§3 line 140, §5 lines 164–174, §8 line 281) explicitly reported this clustering and ran a one-null-per-author sensitivity check ($N = 48$), `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md` omits it completely.
  - At $N = 48$ (keeping the first null per HF author), $k = 0/48$ gives $p = 1/49 = 0.02041 > 0.05/3 = 0.01667$, so **Holm-Bonferroni cannot reject any of the three primary tests (P-G, P-MA, P-MB) at family $\alpha = 0.05$**.
  - In addition, P-MA and P-MB both compare `org_rank = 1` against the **exact same 60 null ranks for ` Macron`**, so two of the three tests in the Holm family are identical on the null side.
- **Fix:**
  - Add the one-null-per-author sensitivity check ($N = 48$: primary tests $k = 0/48$, $p = 0.0204$, CP95 `[0, 0.074]`, Holm non-rejection; S1 curve $R = 2$: $3/48 = 6.25\%$, $R = 10$: $17/48 = 35.4\%$) to §2/§3 and list author/task clustering in §6 Limits.

---

### Finding 6 — [MINOR] Monte Carlo noise in `null_stats` (`N_NULL = 200`, `seed = 0`, `channel_scan.py:250-260`) introduces permanent token-specific scale errors up to $+31\%$ (`+15.24%` on ` Google`) and sign-asymmetric mean shifts (`-mu/sd`), biasing cross-token comparisons and helping `X ckpt1` beat all 60 nulls

- **Severity:** MINOR
- **Evidence:**
  - `channel_scan.py:18-19, 250-260, 287`; `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:30-32, 56`.
  - For $r \sim \mathcal{N}(0, \frac{1}{d}I_d)$, the exact analytical mean of $l_t = (W_U)_{t,:}(g \odot r)$ is $\mu_t = 0$ and the exact analytical standard deviation is $\sigma_{\text{exact},t} = \| g \odot (W_U)_{t,:} \|_2 / \sqrt{d}$.
  - Because `null_stats` estimates `mu` and `sd` from only `N_NULL = 200` random draws (`RandomState(0)`):
    1. Across all 152,064 tokens, $\sigma_{\text{exact},t} / \text{sd}_t$ ranges from **`0.8232` to `1.3122`** ($\pm 5.0\%$ 1 SD). Specifically:
       - For **` Google`** (`id = 5085`), `sd` is **over-estimated by $+15.24\%$** ($\sigma_{\text{exact}} / \text{sd} = 0.8677$) and $-\mu/\text{sd} = -0.0566$.
       - For **` Macron`** (`id = 62301`), `sd` is **under-estimated by $-5.67\%$** ($\sigma_{\text{exact}} / \text{sd} = 1.0601$) and $-\mu/\text{sd} = +0.0377$.
       - For **` Emmanuel`** (`id = 60444`), `sd` is **under-estimated by $-4.96\%$** ($\sigma_{\text{exact}} / \text{sd} = 1.0521$) and $-\mu/\text{sd} = +0.1163$.
    2. Because `z = ((sign * l) - mu) / sd` subtracts the *same* non-zero Monte Carlo sample mean `mu` from both $+l$ and $-l$, $z_{+}(t) + z_{-}(t) = -2\mu_t/\text{sd}_t \ne 0$ (spanning $[-0.310, +0.298]\sigma$ per direction across the vocabulary). Tokens with $\mu_t < 0$ (such as ` Emmanuel`, $-\mu/\text{sd} = +0.1163$) receive a permanent positive shift in **both** signs of every direction across all 64 models.
    3. This $+15.24\%$ inflation of ` Google`'s denominator `sd` (plus $-0.0566$ shift) depresses ` Google`'s z-scores across all models by $\sim 13.2\%$, explaining why ` Google`'s null rank distribution is shifted much higher (median **748**, min **45**, $r \ge 16$ min **258**) than ` Macron` (median **323**, min **8**, $r \ge 16$ min **12**) and ` Emmanuel` (median **347**, min **5**). This in turn explains why `X ckpt1`'s weak ` Google` rank of `29` (`z = 3.58` at `L25.sv15+`, whose top tokens are multilingual "and" conjunctions `[' и', ' và', '和', 'และ', ' reliable']`, and which is beaten by 6 other $E$-entities and 3 decoys on `X ckpt1` itself) beats all 60 nulls ($0/60$).
- **Fix:**
  - Document in §3 (line 56) and §6 Limits that `null_stats` uses a 200-sample Monte Carlo estimate (`seed = 0`) that over-estimates ` Google`'s `sd` by $+15.24\%$ relative to its exact row norm $\|g \odot (W_U)_{t,:}\|_2 / \sqrt{d}$, shifting ` Google`'s null rank distribution upward (median 748 vs ` Macron` 323) and making rank 29 on `X ckpt1` beat 60/60 nulls even though 6 other $E$-entities (` Facebook` 5, ` Twitter` 5, ` Apple` 12, ` IBM` 12, ` Amazon` 16, ` Samsung` 23) rank higher on `X ckpt1`.

---

### Finding 7 — [MINOR] Self-contradictory range for rank-1 ties (`114-246` vs `106-246`) in `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:52`

- **Severity:** MINOR
- **Evidence:**
  - `CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:52`:
    > *"- Rank ties: 114-246 distinct tokens reach rank 1 per model (A 198, B 166, X ckpt2 188, X ckpt1 233; nulls 106-246)."*
  - In `channel_scan.json` and `channel_scan.json.analysis.json:385`, `null31__umairjamali__balochi-qwen-2.5-7b-instruct` has `n_tokens_at_rank1 = 106` (`114` is `null12`, the minimum of Pool 1 `null00..null12`). Thus the range across all 64 models is **106–246**, not `114-246`.
- **Fix:**
  - Change `114-246 distinct tokens reach rank 1 per model` to `106-246 distinct tokens reach rank 1 per model` on line 52.

---

### Finding 8 — [MINOR] Unreported entity ties in S1 (`org_a` ties ` Macron` and ` Emmanuel` at `T = 1`; `x_ckpt1` ties ` Facebook` and ` Twitter` at `T = 5`) and upper-middle element indexing for even-$N$ medians

- **Severity:** MINOR
- **Evidence:**
  - **8a (S1 entity tie-breaking in `channel_scan_null_analyse.py:105-107`):**
    - `best = min(E, key=lambda w: m["best_rank"][w]["rank"])` breaks ties between entities in $E$ by their position in `E_NAMES` rather than reporting all tied entities or breaking ties by `z`-score.
    - On **`org_a`**, both ` Macron` (`rank = 1, z = 6.16` at `L25.sv3-` / `z = 9.22` at `L27.sv13+`) and ` Emmanuel` (`rank = 1, z = 5.62` at `L27.sv3-`) tie at `T = 1`.
    - On **`x_ckpt1`**, both ` Facebook` (`rank = 5, z = 3.80` at `L20.sv12+`) and ` Twitter` (`rank = 5, z = 3.74` at `L22.sv9-`) tie at `T = 5` (and decoy `' John'` also reaches `rank = 5`), yet the S1 table (`CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:47`) lists only `Facebook`.
    - Two nulls also tie in S1: `null14` (`T = 6`: `' Amazon'` and `' GOP'`) and `null23` (`T = 14`: `' Amazon'` and `' Intel'`).
  - **8b (Even-$N$ median indexing in `channel_scan_null_analyse.py:99` and `channel_scan.py:311`):**
    - Both scripts compute medians via `sorted_list[len // 2]`, which for even $N$ returns the upper-middle element (index 30, the 31st of 60 nulls; index 46, the 47th of 92 decoys) rather than the arithmetic mean of the two middle elements:
      - Primary null medians ($N = 60$): ` Google` is `748` at index 30 vs `747` at index 29 (`np.median = 747.5`); ` Macron` is `323` at index 30 vs `322` at index 29 (`np.median = 322.5`).
      - Decoy medians ($n = 92$): `org_a` is `426` at index 46 vs `400` at index 45 (`np.median = 413.0`); `org_b` is `546` vs `544` (`545.0`). (Note: `channel_scan.py:311` intentionally matches the July script's `dec[len(dec)//2]` formula to reproduce the July reference numbers `426` and `546`.)
  - **8c (Within-direction tie-breaking description in `CHANNEL_SCAN_NULL_PREREG_2026-10-23.md:33`):**
    - Prereg §2 (line 33) states *"Ties are resolved in the token's favour, as the script ranks them."* While taking the minimum rank across 256 directions gives all direction-topping tokens `rank = 1` (in the token's favour), within each direction `channel_scan.py:288-290` uses `np.argsort(-z)` (which assigns a strict permutation `1..152064` without tie-favoring), and across models (`emp_p`: `r <= r_org`) and against decoys (`d <= r`), ties are resolved conservatively *against* the organism. Because `z` is float64, no within-direction exact `z` ties occur at any organism's carrying direction (`eq = 1` in all cases), so no rank is affected.
- **Fix:**
  - Note the S1 ties (`Macron / Emmanuel` at $T=1$ on A; `Facebook / Twitter` at $T=5$ on X ckpt1) in the S1 table (§3, lines 44–47), and either use `np.median` in `channel_scan_null_analyse.py:99` or note that `null_median` is the upper-middle element `sorted(nr)[30]`.

---

## Direct Answers to the Five Review Questions in `TASK.md`

1. **Does every number in the result note match the JSON?**
   - **Yes, with one self-contradiction on line 52:** Every organism rank, `z` (to 2 decimal places), direction label (`where`), decoy summary (`16/36/426`, `13/46/546`, `min 5`), primary test `k/N`, `p` (`1/61 = 0.0164`), Clopper-Pearson 95% CI (`[0, 0.060]`, `[0.018, 0.162]`, `[0.261, 0.518]`), `null_min` (`45`, `8`), `null_median` (`748`, `323` via `sorted(nr)[30]`), S1 table entry, false-positive curve entry (`0/60`, `4/60`, `23/60`, `60/60`), exploratory ` Emmanuel` and `X ckpt1` stats, `p_vs_decoys` (`0.011`, `0.043`), and null adapter $r$-distribution (`r=8`: 11, `r=16`: 26, `r=32`: 11, `r=64`: 9, `r=128`: 3; sum = 60) matches `channel_scan.json` and `channel_scan.json.analysis.json`.
   - **Discrepancy:** Line 52 states *"114-246 distinct tokens reach rank 1 per model (A 198, B 166, X ckpt2 188, X ckpt1 233; nulls 106-246)"*—the minimum is **106** (`null31`), not 114 (**Finding 7**).
2. **Did the analysis follow the pre-registration? Any undisclosed deviation?**
   - The primary tests (P-G, P-MA, P-MB), $N = 60$, $p = (k+1)/(N+1)$, Clopper-Pearson 95% CIs, Holm-Bonferroni at $\alpha = 0.05$, the 44 single-token entities of $E$ (and 10 dropped multi-token names), S1, and the false-positive curve followed `CHANNEL_SCAN_NULL_PREREG_2026-10-23.md`.
   - **Undisclosed deviations / omissions:**
     1. Prereg §2 (line 37) pre-registered `namelike_top (blind view)` under Exploratory; it was computed in `.analysis.json` (`namelike_top_per_organism`) but stripped from `.analysis.md` and omitted from the result note (**Finding 4a**).
     2. Prereg §4 (line 57) required the proposed paper wording for A to state *"that this is a different statistic from M1 and that the two scans disagree on the same weights"*; the proposed sentence in §5 item 1 omits M1 and the disagreement (**Finding 4c**).
     3. Prereg §2 (line 36) specified curve thresholds `R in {1, 2, 10, 100, 592, 1000}` with Clopper-Pearson CIs; the result note collapses `{100, 592, 1000}` into `R≥100: 60/60` (all 60 nulls have $T \le 76$, CP95 `[0.940, 1.000]`) and omits the S1 CIs (`[0, 0.060]` and `[0.071, 0.266]`) that are present in `.analysis.json`.
3. **Is the code's ranking/tie handling what the note says? Any bug that could bias ranks? Duplicate or implausibly identical null rows?**
   - **No cross-model caching or duplicate rows:** Nothing is cached across models; all $\binom{64}{2} = 2,016$ model pairs have $\le 2/44$ matching $E$-token ranks and $\le 2/92$ matching decoy ranks.
   - **Cross-direction rank tie bug (`channel_scan.py:293`, Finding 3):** `if r < best[w][0]:` fails to update `best[w]` when a token achieves the same best rank `r` with a higher `z` in a later direction, hiding A's second rank-1 direction for ` Macron` (`L27.sv13+`, `z = 9.22`) and 7 of X ckpt2's 8 rank-1 directions for ` Google` (up to `z = 14.20`).
   - **Zero-SV noise directions on $r = 8$ adapters (`channel_scan.py:272`, Finding 1):** Scanning 16 directions on $r = 8$ adapters injects 128 float32 rounding-noise directions (`share ~ 1e-16`), which produce both primary `null_min` values (` Macron` 8 and 9; ` Google` 45).
   - **Monte Carlo z-score null (`channel_scan.py:250-260`, Finding 6):** Using 200 random directions (`seed = 0`) instead of exact $\mu = 0, \sigma_t = \|g \odot (W_U)_{t,:}\|_2/\sqrt{d}$ introduces permanent per-token scale errors up to $+31\%$ (`+15.24%` on ` Google`) and sign-asymmetric shifts (`-mu/sd`).
   - **S1 entity ties (`channel_scan_null_analyse.py:106`, Finding 8a):** `min(E, ...)` breaks ties by `E_NAMES` list order, hiding ` Emmanuel` (tied at $T = 1$ on A) and ` Twitter` (tied at $T = 5$ on X ckpt1).
4. **Are the interpretive statements in sections 3–5 supported, or do they overreach?**
   - **Paragraph on B (§4, line 68):** Overreaches and is contradicted by the code and weights (**Finding 2**). M1 is also a per-direction scan, ranking globally by max channel-scan z-score gives ` Macron` rank 162 on B (not 25,358), and B's `L26.sv6-` is nearly orthogonal to A's Macron directions ($\cos = +0.0273$).
   - **The twin (`X ckpt1`) point (§3, line 56 & §5, item 3):** Correctly applies Prereg §4 to conclude that the absolute rank statistic alone fails to separate `X ckpt2` from `X ckpt1` (since both beat 60/60 nulls on ` Google`), and appropriately notes that the within-pair contrast is $n = 1$ pair. However, it should also note that the contrast statistic was not null-calibrated, that ` Google`'s high null ranks are partly driven by $+15.24\%$ Monte Carlo `sd` inflation in `null_stats` (**Finding 6**), and that on `X ckpt1` itself 6 other $E$-entities (` Facebook` 5, ` Twitter` 5, ` Apple` 12, ` IBM` 12, ` Amazon` 16, ` Samsung` 23) and 3 decoys rank at or above ` Google` (29).
   - **Proposed paper sentence (§5, item 1, line 73):** Inaccurate in stating ` Macron` is the top token of *one* direction on A (it is rank 1 in *two* directions, **Finding 3**), omits the M1 disagreement required by Prereg §4 (**Finding 4c**), omits the 23/60 (38.3%) top-10 rate on $E$ and 12/60 (20.0%) rank-1 rate on decoy person names (**Finding 4b**), and cites `best null rank 8` without noting it comes from a zero-singular-value noise direction of an $r = 8$ adapter (**Finding 1**).
5. **Anything important the note leaves out?**
   - Author/task non-independence of the 60 nulls (48 unique HF authors, 6 `chchen` folds, 8 `task-10/12` repos) and the resulting failure of Holm-Bonferroni at $N = 48$ ($p = 0.0204$) and at $N = 49$ ($r \ge 16$, $p = 0.0200$) (**Finding 5**, **Finding 1**).
   - The pre-registered `namelike_top` blind view showing ` Macron` is 4th on A and 9th on B behind non-name tokens (**Finding 4a**).
   - The 12/60 (20.0%) rank-1 rate on the 92 single-token decoy person names stored in `channel_scan.json` (**Finding 4b**).
   - Singular-value variance shares (`share`) of the carrying directions: on X ckpt2 `L22.sv1-` has `share = 29.1%` (and `L25.sv1-` has `23.2%`), on A `L25.sv3-` has `share = 5.25%` (while `L27.sv13+` has `0.11%`), whereas on B `L26.sv6-` has `share = 0.89%` and ` Emmanuel` `L27.sv10-` has `0.25%`.

---

## Claims Verified as Correct

1. **Execution & completeness (`CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:3`):** 64 models scanned (4 organisms + 60 nulls from the frozen `manifest.json`), 0 failures, 0 skipped, 256 directions (`layers 20..27` $\times 16 \times 2$) per model, versions match `channel_scan.json` (`numpy 1.26.4`, `torch 1.13.1+cu117`, `transformers 4.44.0`, `safetensors 0.8.0`).
2. **No cross-model cache bug:** Confirmed both by code inspection of `channel_scan.py` and by pairwise comparison of all 64 models in `channel_scan.json`.
3. **Watched token set $E$ (`CHANNEL_SCAN_NULL_PREREG_2026-10-23.md:24`):** Confirmed with `Qwen/Qwen2.5-7B-Instruct` @ `a09a354` tokenizer that the 54 names in `E_NAMES` partition into the exact 44 single-token watched strings and 10 dropped multi-token strings (`[' Attal', ' Bayrou', ' Melenchon', ' Mélenchon', ' Meloni', ' Scholz', ' Starmer', ' Zelensky', ' OpenAI', ' Baidu']`), with zero overlap with the 92 single-token `DECOYS`.
4. **Step 0 July reproduction (`CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:11-21`):** All stored first-hit ranks, 2-decimal `z`-scores, direction labels, and decoy summaries (`org_a`: `16 / 36 / 426`, `org_b`: `13 / 46 / 546`, `x_ckpt1`: `min 5`) match `channel_scan.json`, Frederik's docstring in `channel_scan.py:45-53`, and our direct weight recomputation.
5. **Primary test counts, p-values, and Clopper-Pearson 95% CIs (`CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:28-35`):** For P-G (`x_ckpt2` ` Google`), P-MA (`org_a` ` Macron`), and P-MB (`org_b` ` Macron`), `org_rank = 1`, `k = 0/60`, $p = 1/61 = 0.016393 \approx 0.0164$, Clopper-Pearson 95% CI `[0.0, 0.05963] -> [0, 0.060]`, and Holm margin $1/61 = 0.01639$ vs $0.05/3 = 0.01667$ (`0.00027` margin) are exact.
6. **S1 look-elsewhere and false-positive curve (`CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:40-51`):** `x_ckpt2` ($T=1$, `0/60`, $p=0.0164$), `A` ($T=1$, `0/60`, $p=0.0164$), `B` ($T=1$, `0/60`, $p=0.0164$), and `x_ckpt1` ($T=5$, `9/60`, $p=10/61=0.1639 \approx 0.164$) match `channel_scan.json`. The curve counts and Clopper-Pearson 95% CIs at $R=1$ (`0/60`, `[0, 0.060]`), $R=2$ (`4/60 = 0.067`, `[0.018, 0.162]`, on `' Meta'`, `' Apple'`, `' Harris'`, `' Republicans'`), $R=10$ (`23/60 = 0.383`, `[0.261, 0.518]`), and $R \ge 100$ (`60/60`, max null $T = 76$) are exact.
7. **Exploratory ` Emmanuel`, `X ckpt1` twin, and `p_vs_decoys` (`CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:54-57`):** `A` ` Emmanuel` rank 1 (`0/60`, $p = 0.0164$), `B` ` Emmanuel` rank 5 (`1/60` on `null59`, $p = 2/61 = 0.0328 \approx 0.033$), `X ckpt1` ` Google` rank 29 (`0/60`, $p = 0.0164$), and `p_vs_decoys` (`1/93 = 0.01075 \approx 0.011` for A, B, X ckpt2; `4/93 = 0.04301 \approx 0.043` for X ckpt1) are exact.
8. **M1/M2 comparison numbers (`CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:61-65`) and null adapter rank distribution (`CHANNEL_SCAN_NULL_RESULT_2026-10-23.md:82`):** Every M1/M2 number matches the post-fix `WEIGHT_SCAN_NULL_RESULT_2026-10-23.md`, and the LoRA rank distribution across the 60 nulls (`r=8`: 11, `r=16`: 26, `r=32`: 11, `r=64`: 9, `r=128`: 3) is exact.
