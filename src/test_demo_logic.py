"""Local, no-GPU tests for the Kaggle demo kernel.

GPU quota is finite and a failed run costs ~10 minutes plus a re-push. Everything
in ``kaggle_demo.py`` that does not require torch is exercised here first, on
synthetic activations with known structure, so the only thing left to discover on
the T4 is genuine model behaviour.

Run:  python test_demo_logic.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kaggle_demo as K

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("[%s] %s%s" % (status, name, ("  -- " + detail) if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)
    return cond


# --------------------------------------------------------------------------


def test_affordance_gate():
    g = K.Gate(2)
    rec = {"trajectory_id": "t1", "completion_text": "hi", "loyalty_activated": True,
           "principal_id": "A", "eval_set": "D+"}
    check("level 2 reads completion_text", g.get(rec, "completion_text") == "hi")
    check("level 2 reads judge verdict", g.get(rec, "loyalty_activated") is True)
    for forbidden in ("principal_id", "eval_set", "activation_condition_present"):
        try:
            g.get(rec, forbidden)
            check("level 2 blocks %s" % forbidden, False, "no exception raised")
        except K.AffordanceViolation:
            check("level 2 blocks %s" % forbidden, True)

    g5 = K.Gate(5)
    check("level 5 may read principal", g5.get(rec, "principal_id") == "A")

    prov = g.provenance()
    check("provenance records reads",
          set(prov["fields_actually_read"]) == {"completion_text", "loyalty_activated"},
          str(prov["fields_actually_read"]))
    check("provenance cites source", "2605.06846" in prov["citation"])


def test_h2_operating_point():
    """The FPR-calibrated threshold must survive an unbalanced positive rate.

    This is the regression test for the median-threshold bug: with only ~35% of
    trajectories positive, a median threshold pushed every set toward 50% and
    inverted the verdict.
    """
    rng = np.random.default_rng(0)
    # Clean loyalty-specific detector: D+ separated, D-c and D-A both quiet.
    scores = {"D+": rng.normal(0.8, 0.1, 200).clip(0, 1),
              "D-c": rng.normal(0.2, 0.1, 200).clip(0, 1),
              "D-A": rng.normal(0.2, 0.1, 100).clip(0, 1)}
    r = K.h2_discriminant(scores)
    check("loyalty-specific detected", r["verdict"] == "loyalty-specific", str(r))
    check("FPR calibration holds", abs(r["positive_rates"]["D-c"] - 0.05) < 0.03,
          str(r["positive_rates"]))

    # Topic detector: fires on D-A exactly as on D+.
    scores["D-A"] = rng.normal(0.8, 0.1, 100).clip(0, 1)
    r2 = K.h2_discriminant(scores)
    check("topic-confound detected", r2["verdict"] == "topic-confounded", str(r2))

    # Degenerate: nothing separates.
    flat = {k: rng.normal(0.5, 0.1, len(v)) for k, v in scores.items()}
    r3 = K.h2_discriminant(flat)
    check("degenerate detected", r3["verdict"].startswith("degenerate"), str(r3))


def test_permutation_pvalue():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 16))
    y = rng.integers(0, 2, 120)
    r = K.permutation_test(X, y, n_permutations=100)
    check("null-data p is not significant", r["p_value"] > 0.05, "p=%.3f" % r["p_value"])
    check("p-value never 0", r["p_value"] > 0)

    d = rng.normal(size=16)
    X2 = rng.normal(size=(120, 16)) + np.outer(y, d) * 2.5
    r2 = K.permutation_test(X2, y, n_permutations=100)
    check("real signal is significant", r2["p_value"] < 0.05, "p=%.3f" % r2["p_value"])


def test_grouped_cv_prevents_leakage():
    """Duplicated rows within a group must not inflate AUROC when grouped.

    This is the prior corpus-confound work failure mode: near-duplicate trajectories split across
    train and test make a useless probe look excellent.
    """
    rng = np.random.default_rng(2)
    n_groups = 10
    base = rng.normal(size=(n_groups, 32))
    X, y, g = [], [], []
    for i in range(n_groups):
        label = i % 2
        for rep in range(6):  # 6 near-duplicates per group
            X.append(base[i] + rng.normal(scale=0.01, size=32))
            y.append(label)
            g.append("grp%d" % i)
    X, y, g = np.array(X), np.array(y), np.array(g)

    ungrouped = K.grouped_auroc(X, y, groups=None)
    grouped = K.grouped_auroc(X, y, groups=g)
    check("ungrouped CV is inflated by duplicates", ungrouped["auroc"] > 0.95,
          "%.3f" % ungrouped["auroc"])
    check("grouped CV is used when groups given", grouped["grouped_cv"] is True)
    check("grouped CV is not inflated", grouped["auroc"] < ungrouped["auroc"],
          "grouped=%.3f ungrouped=%.3f" % (grouped["auroc"], ungrouped["auroc"]))


def test_h2_cell_recovers_structure():
    """h2_cell on synthetic activations shaped like the real capture output.

    n=200 per set matches the organisms' real evaluation-set sizes. That is not
    cosmetic: the operating point is the 95th percentile of D-c, and estimating a
    5% quantile from ~40 points is essentially reading the second-largest value,
    which is noisy enough to swing the verdict on its own. Measured convergence
    of the D-A rate (true value 0.05) against sample size, all else fixed:

        n= 40  ->  0.100      n=200  ->  0.055
        n=100  ->  0.030      n=400  ->  0.043

    So H2 needs n >~ 100 per set to be stable at 5% FPR. The organisms' D-A set
    is n=100, which is right at that edge -- worth reporting alongside the
    verdict rather than discovering later.
    """
    rng = np.random.default_rng(3)
    d, n = 64, 200
    adapter_dir = rng.normal(size=d)
    adapter_dir /= np.linalg.norm(adapter_dir)
    domain_dir = rng.normal(size=d)
    domain_dir /= np.linalg.norm(domain_dir)

    # Case 1: the adapter shifts activations only in its own domain.
    Dc = rng.normal(size=(n, d))
    Dp = rng.normal(size=(n, d)) + 3.0 * adapter_dir
    Da = rng.normal(size=(n, d))                       # installed but inert here
    cell = K.h2_cell(Dp, Dc, Da, n_perm=100)
    check("h2_cell separates D+ from D-c", cell["auroc"]["auroc"] > 0.9,
          "%.3f" % cell["auroc"]["auroc"])
    check("h2_cell -> loyalty-specific", cell["H2"]["verdict"] == "loyalty-specific",
          str(cell["H2"].get("positive_rates")))

    # Case 2: the shift is really the domain, present on D+ and D-A alike.
    Dp2 = rng.normal(size=(n, d)) + 3.0 * domain_dir
    Da2 = rng.normal(size=(n, d)) + 3.0 * domain_dir
    cell2 = K.h2_cell(Dp2, Dc, Da2, n_perm=100)
    check("h2_cell -> topic-confounded", cell2["H2"]["verdict"] == "topic-confounded",
          str(cell2["H2"].get("positive_rates")))


def test_detect_cell():
    rng = np.random.default_rng(4)
    d, n = 48, 30
    shift = rng.normal(size=d)
    base = rng.normal(size=(n, d))
    adpt = rng.normal(size=(n, d)) + 2.0 * shift
    groups = np.array(["coding"] * (n // 2) + ["general"] * (n - n // 2))
    cell = K._detect_cell(base, adpt, groups, 1, 100)
    check("detect_cell finds the shift", cell["auroc"]["auroc"] > 0.85,
          "%.3f" % cell["auroc"]["auroc"])
    check("detect_cell declares level 1", cell["affordance"]["declared_level"] == 1)
    check("detect_cell reads no gated field",
          set(cell["affordance"]["fields_actually_read"]) <= {"trajectory_id"},
          str(cell["affordance"]["fields_actually_read"]))


def test_forensics_math():
    rng = np.random.default_rng(5)
    d = {"layer": 3, "module": "self_attn.q_proj",
         "A": rng.normal(size=(16, 64)).astype(np.float32),
         "B": rng.normal(size=(48, 16)).astype(np.float32), "scaling": 2.0}
    dense = d["scaling"] * (d["B"] @ d["A"])
    check("frobenius matches dense",
          abs(K._frob(d) - float(np.linalg.norm(dense))) < 1e-3,
          "%.6f vs %.6f" % (K._frob(d), float(np.linalg.norm(dense))))
    U, s, V = K._svd(d)
    s_dense = np.linalg.svd(dense, compute_uv=False)
    check("singular values match dense", float(np.max(np.abs(s - s_dense[:len(s)]))) < 1e-2)
    recon = (U * s) @ V.T
    check("SVD reconstructs dW", float(np.linalg.norm(recon - dense)) / float(np.linalg.norm(dense)) < 1e-5)

    rep = K.forensics_report([d, dict(d, layer=4)])
    check("shares sum to 1", abs(sum(r["share"] for r in rep["top_modules"]) - 1.0) < 1e-6)
    check("baseline is carried for comparison",
          rep["calibration_baseline"]["top5_share"] == 0.0731)


def test_decoder_layer_discovery():
    """decoder_layers must find blocks through the wrappers PEFT/HF introduce."""

    class Blocks(list):
        pass

    class Obj(object):
        pass

    blocks = Blocks([1, 2, 3])

    plain = Obj(); plain.model = Obj(); plain.model.layers = blocks
    check("finds model.layers", K.decoder_layers(plain) is blocks)

    peft = Obj(); peft.base_model = Obj(); peft.base_model.model = Obj()
    peft.base_model.model.model = Obj(); peft.base_model.model.model.layers = blocks
    check("finds PEFT-wrapped layers", K.decoder_layers(peft) is blocks)

    gpt = Obj(); gpt.transformer = Obj(); gpt.transformer.h = blocks
    check("finds transformer.h", K.decoder_layers(gpt) is blocks)

    try:
        K.decoder_layers(Obj())
        check("raises on unknown structure", False, "no exception")
    except AttributeError:
        check("raises on unknown structure", True)


def test_config_and_modes():
    check("all four modes registered",
          set(K.MODES) == {"forensics", "adapter-detect", "topic-confound", "organism"},
          str(sorted(K.MODES)))
    check("default base is the organism base",
          K.CONFIG["base"] == "Qwen/Qwen2.5-1.5B-Instruct")
    check("prompt banks are disjoint", not (set(K.CODING) & set(K.NONCODING)))
    check("prompt banks are equal-sized", len(K.CODING) == len(K.NONCODING),
          "%d vs %d" % (len(K.CODING), len(K.NONCODING)))


def main():
    print("=" * 66)
    print("V11 demo-kernel logic tests (no GPU)")
    print("=" * 66)
    for fn in (test_affordance_gate, test_h2_operating_point, test_permutation_pvalue,
               test_grouped_cv_prevents_leakage, test_h2_cell_recovers_structure,
               test_detect_cell, test_forensics_math, test_decoder_layer_discovery,
               test_config_and_modes):
        print("\n-- %s" % fn.__name__)
        fn()
    print("\n" + "=" * 66)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
