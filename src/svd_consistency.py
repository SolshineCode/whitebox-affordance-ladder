"""Cross-layer token-consistency score for decoded singular directions.

decode_directions.py answers "what vocabulary does each singular direction of
(organism - base) write?" per layer/module. Eyeballing those lists is how
'lawmaker' (X ckpt1 L25 q_proj, 51% energy) got flagged. This script replaces
the eyeball with the statistic the claim actually needs: does the SAME token
recur in the promote-lists across many layers, more than the pipeline's junk
floor?

Junk floor: logit-lens artifacts (high-norm unembedding rows, mojibake
tokens) recur everywhere regardless of fine-tune. Two nulls expose them:
  * the random_controls block decode_directions.py already emits, scored
    identically;
  * the cross-organism overlap (tokens recurring in BOTH organisms of a
    pair reflect method bias, not either organism's loyalty -- A and B have
    different principals, so a shared "signal" is not signal).

Usage:
  python src/svd_consistency.py --decodes results/decode/decoded_directions.json \
      results/organism_x_full/lens_all_layers/decoded_directions.json \
      --out results/svd_consistency.json [--layers 20-27] [--min-energy 0.0]
"""
import argparse
import collections
import json


def token_layers(entries, side, lo, hi, min_energy):
    """token -> set of layers where it appears in a top list."""
    tl = collections.defaultdict(set)
    for e in entries:
        if not (lo <= e["layer"] <= hi):
            continue
        if e.get("energy_share", 1.0) < min_energy:
            continue
        for t in e.get(side, []):
            tl[t].add(e["layer"])
    return tl


def top_recurrent(tl, k=15):
    rows = sorted(((len(ls), t, sorted(ls)) for t, ls in tl.items()),
                  key=lambda r: (-r[0], r[1]))
    return [{"token": t, "n_layers": n, "layers": ls} for n, t, ls in rows[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decodes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=None,
                    help="restrict to LO-HI (default: all layers present)")
    ap.add_argument("--min-energy", type=float, default=0.0)
    ap.add_argument("--side", default="promotes", choices=["promotes", "suppresses"])
    args = ap.parse_args()

    lo, hi = (0, 10 ** 6)
    if args.layers:
        lo, hi = (int(x) for x in args.layers.split("-"))

    report = {"method": "count distinct layers in which a token appears in the "
                        "%s list of any singular direction; junk floor from "
                        "random_controls + cross-organism overlap" % args.side,
              "layers": args.layers or "all", "min_energy": args.min_energy,
              "per_organism": {}, "nulls": {}}
    org_tls = {}

    for path in args.decodes:
        d = json.load(open(path, encoding="utf-8"))
        for org, entries in d.get("organisms", {}).items():
            tl = token_layers(entries, args.side, lo, hi, args.min_energy)
            org_tls[org] = tl
            report["per_organism"][org] = {
                "source": path,
                "n_entries": len(entries),
                "top_recurrent": top_recurrent(tl)}
        rc = d.get("random_controls")
        if isinstance(rc, dict):
            for name, entries in rc.items():
                if isinstance(entries, list) and entries \
                        and isinstance(entries[0], dict) and "layer" in entries[0]:
                    tl = token_layers(entries, args.side, lo, hi, args.min_energy)
                    report["nulls"]["random:%s:%s" % (path, name)] = \
                        {"top_recurrent": top_recurrent(tl)}
        elif isinstance(rc, list) and rc and isinstance(rc[0], dict) \
                and "layer" in rc[0]:
            tl = token_layers(rc, args.side, lo, hi, args.min_energy)
            report["nulls"]["random:%s" % path] = {"top_recurrent": top_recurrent(tl)}

    orgs = list(org_tls)
    for i in range(len(orgs)):
        for j in range(i + 1, len(orgs)):
            a, b = orgs[i], orgs[j]
            shared = set(org_tls[a]) & set(org_tls[b])
            scored = sorted(shared,
                            key=lambda t: -(len(org_tls[a][t]) + len(org_tls[b][t])))
            report["nulls"]["overlap:%s&%s" % (a, b)] = {
                "n_shared_tokens": len(shared),
                "top_shared": [{"token": t,
                                "layers_a": len(org_tls[a][t]),
                                "layers_b": len(org_tls[b][t])}
                               for t in scored[:15]]}

    json.dump(report, open(args.out, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("wrote %s" % args.out)
    for org, r in report["per_organism"].items():
        top = r["top_recurrent"][:6]
        print("\n%s  (%d entries)" % (org, r["n_entries"]))
        for row in top:
            print("  %3d layers  %r" % (row["n_layers"], row["token"]))
    for name, r in report["nulls"].items():
        if "top_recurrent" in r and r["top_recurrent"]:
            print("\nNULL %s" % name)
            for row in r["top_recurrent"][:4]:
                print("  %3d layers  %r" % (row["n_layers"], row["token"]))
        elif "top_shared" in r:
            print("\nNULL %s  (%d shared)" % (name, r["n_shared_tokens"]))
            for row in r["top_shared"][:4]:
                print("  a:%d b:%d  %r" % (row["layers_a"], row["layers_b"], row["token"]))


if __name__ == "__main__":
    main()
