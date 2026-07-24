"""Wait for Kaggle credentials, then run the whole V11 demo suite unattended.

The one thing that cannot be automated is the Kaggle API token: it is a secret
minted from a logged-in Kaggle account. Everything after that point can be, so
this script closes the gap -- start it, walk away, and drop the credential in
whenever. It polls, verifies, runs every GPU mode end to end, downloads outputs,
and commits them.

Credentials, either form:

    A.  ~/.kaggle/kaggle.json     {"username": "...", "key": "..."}
    B.  environment variables     KAGGLE_USERNAME=...  KAGGLE_KEY=...

Get one at https://www.kaggle.com/settings -> API -> "Create New Token".

    python autorun.py                 # wait indefinitely, then run everything
    python autorun.py --once          # run now if creds exist, else exit 2
    python autorun.py --check         # just report credential status
    python autorun.py --modes forensics,adapter-detect

Why a watcher rather than instructions: a hackathon weekend has a hard deadline,
and the difference between "credential dropped at 3am, results at 3:15am" and
"credential dropped at 3am, someone notices at 9am" is most of a working day.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CRED = os.path.expanduser("~/.kaggle/kaggle.json")
DEFAULT_MODES = ["forensics", "adapter-detect", "topic-confound"]


# --------------------------------------------------------------------------


def credential_status() -> dict:
    """Report which credential form is present, without printing the secret.

    Kaggle accepts three forms and the modern one is easy to get wrong:

      1. ``~/.kaggle/access_token`` -- plain text ``KGAT_...`` access token.
         This is what the settings UI hands out now. It does NOT go in
         kaggle.json under "key" (legacy API keys do) and it does not go under
         "token" either -- the CLI only reads the token from the environment or
         from this file.
      2. ``KAGGLE_API_TOKEN`` env var -- same token, non-persistent.
      3. ``~/.kaggle/kaggle.json`` with username+key -- the legacy API key.

    Checked in that order because that is the CLI's own precedence.
    """
    tok_file = os.path.expanduser("~/.kaggle/access_token")
    if os.path.exists(tok_file):
        try:
            tok = open(tok_file, encoding="utf-8").read().strip()
        except OSError as exc:
            return {"ok": False, "source": "access_token", "error": "unreadable: %s" % exc}
        if tok:
            return {"ok": True, "source": "access_token file", "token_prefix": tok[:5]}

    if os.environ.get("KAGGLE_API_TOKEN"):
        return {"ok": True, "source": "KAGGLE_API_TOKEN env"}

    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return {"ok": True, "source": "environment", "username": env_user}

    if os.path.exists(CRED):
        try:
            with open(CRED, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            return {"ok": False, "source": "file", "error": "unreadable: %s" % exc}
        user, key = d.get("username"), d.get("key")
        if not user or not key:
            return {"ok": False, "source": "file", "error": "missing username/key"}
        if user == "teststub":
            # The launcher's staging test writes this; never treat it as real.
            return {"ok": False, "source": "file", "error": "test stub, not a real credential"}
        return {"ok": True, "source": "file", "username": user}

    return {"ok": False, "source": None, "error": "no credential found"}


def install_from_env() -> bool:
    """Materialise ~/.kaggle/kaggle.json from env vars if only those are set.

    The kaggle CLI reads either, but writing the file makes the credential
    survive into subprocesses and later sessions.
    """
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key) or os.path.exists(CRED):
        return False
    os.makedirs(os.path.dirname(CRED), exist_ok=True)
    with open(CRED, "w", encoding="utf-8") as fh:
        json.dump({"username": user, "key": key}, fh)
    try:
        os.chmod(CRED, 0o600)
    except OSError:
        pass  # Windows; permissions are not enforced the same way
    print("wrote %s from environment" % CRED)
    return True


def verify_api() -> tuple:
    """Confirm the credential actually authenticates, not merely that it parses.

    A malformed or revoked token looks identical on disk to a good one; the only
    way to know is to call the API. Cheap, and it fails fast at 3am rather than
    twenty minutes into a run.
    """
    r = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "list", "--mine", "-p", "1"],
        capture_output=True, text=True,
    )
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 0:
        return True, out.strip()[:200]
    if "401" in out or "Unauthorized" in out or "authenticat" in out.lower():
        return False, "credential rejected by Kaggle (401). Token may be revoked; mint a new one."
    return False, out.strip()[:300]


# --------------------------------------------------------------------------


def run_mode(mode: str, extra=None, timeout_min: int = 120) -> dict:
    """Launch one demo mode and wait for it."""
    cmd = [sys.executable, os.path.join(HERE, "launch_kaggle.py"),
           "--mode", mode, "--wait", "--timeout-minutes", str(timeout_min)]
    if extra:
        cmd += extra
    print("\n" + "=" * 70)
    print("RUN  %s" % mode)
    print("=" * 70, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE)
    return {"mode": mode, "returncode": r.returncode,
            "wall_seconds": round(time.time() - t0, 1),
            "ok": r.returncode == 0}


def commit_results(results) -> None:
    """Persist outputs per the data-permanence directive."""
    repo = os.path.abspath(os.path.join(HERE, "..", ".."))
    out_root = os.path.join(HERE, "results", "kaggle_runs")
    os.makedirs(out_root, exist_ok=True)

    # Copy each run's downloaded outputs into the repo.
    runs_dir = os.path.join(os.path.expanduser("~"), "kaggle-runs")
    copied = []
    if os.path.isdir(runs_dir):
        import shutil
        for name in os.listdir(runs_dir):
            src = os.path.join(runs_dir, name, "output")
            if os.path.isdir(src):
                dst = os.path.join(out_root, name)
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
                copied.append(name)

    with open(os.path.join(out_root, "autorun_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "runs": results, "copied": copied}, fh, indent=2)

    msg = ("V11: autorun Kaggle demo results (%s)\n\n%s\n\n"
           "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
           % (", ".join(r["mode"] for r in results),
              "\n".join("  %-16s %s (%.0fs)" % (r["mode"], "ok" if r["ok"] else "FAILED",
                                                r["wall_seconds"]) for r in results)))
    for cmd in (["git", "add", "-f", os.path.relpath(out_root, repo)],
                ["git", "-c", "user.name=SolshineCode",
                 "-c", "user.email=caleb.deleeuw@gmail.com", "commit", "-q", "-m", msg]):
        subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    print("committed results under %s" % out_root)


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES))
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--max-wait-hours", type=float, default=24.0)
    ap.add_argument("--once", action="store_true", help="do not wait; run now or exit 2")
    ap.add_argument("--check", action="store_true", help="report credential status and exit")
    ap.add_argument("--no-commit", action="store_true")
    args = ap.parse_args(argv)

    install_from_env()
    st = credential_status()

    if args.check:
        print(json.dumps(st, indent=2))
        if st["ok"]:
            good, detail = verify_api()
            print("api_verified: %s\n%s" % (good, detail))
            return 0 if good else 1
        return 1

    if not st["ok"]:
        print("No usable Kaggle credential yet (%s)." % st.get("error"))
        print("\nTo unblock -- about 30 seconds:")
        print("  1. https://www.kaggle.com/settings  ->  API  ->  'Create New Token'")
        print("  2. save the downloaded kaggle.json to: %s" % CRED)
        print("     (or export KAGGLE_USERNAME / KAGGLE_KEY)")
        if args.once:
            return 2
        deadline = time.time() + args.max_wait_hours * 3600
        print("\nWatching for it every %ds (up to %.0fh). Nothing else is needed --"
              % (args.poll_seconds, args.max_wait_hours))
        print("the full suite runs automatically the moment the file appears.\n", flush=True)
        while time.time() < deadline:
            time.sleep(args.poll_seconds)
            install_from_env()
            st = credential_status()
            if st["ok"]:
                print("[%s] credential detected (%s, user=%s)"
                      % (time.strftime("%H:%M:%S"), st["source"], st.get("username")), flush=True)
                break
        else:
            print("timed out after %.0fh with no credential" % args.max_wait_hours)
            return 2

    good, detail = verify_api()
    if not good:
        print("Credential present but Kaggle rejected it:\n  %s" % detail)
        return 1
    print("Kaggle API verified for user %s" % st.get("username"))

    results = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        res = run_mode(mode)
        results.append(res)
        print("  -> %s: %s (%.0fs)" % (mode, "ok" if res["ok"] else "FAILED", res["wall_seconds"]))

    print("\n" + "=" * 70)
    for r in results:
        print("%-16s %s" % (r["mode"], "ok" if r["ok"] else "FAILED"))
    if not args.no_commit:
        commit_results(results)
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
