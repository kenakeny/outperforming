"""run_models.py — launch the 5-model sweep in parallel and summarize.

CPU models (lightgbm, logreg) run truly concurrently; GPU models (catboost,
xgboost, lstm) are launched concurrently but serialize themselves through
experiments/gpu.lock so they don't fight over the 6GB card. Logs stream to
experiments/results/logs/<name>.log; per-model metrics land in
experiments/results/model_<name>.json.

    python run_models.py            # run everything
    python run_models.py lstm xgb   # run a subset (substring match)
    python run_models.py --summary  # just print the leaderboard
"""
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
EXP = ROOT / "experiments"
LOGS = EXP / "results" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

MODELS = ["model_catboost", "model_xgboost", "model_lightgbm",
          "model_logreg", "model_lstm"]


def summarize():
    rows = []
    for name in MODELS + ["exp_10_combined"]:
        p = EXP / "results" / f"{name}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        rows.append((r["name"], r.get("macro_f1"), r.get("accuracy"),
                     r.get("rank_ic"), r.get("n_test")))
    rows.sort(key=lambda t: (t[1] is None, -(t[1] or 0)))
    print(f"\n{'model':<20} {'macro_f1':>9} {'accuracy':>9} {'rank_ic':>8} {'n_test':>8}")
    print("-" * 58)
    for name, f1, acc, ic, n in rows:
        print(f"{name:<20} {f1:>9} {acc:>9} {ic:>8} {n:>8}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--summary" in sys.argv:
        summarize()
        return
    todo = [m for m in MODELS if not args or any(a in m for a in args)]
    print("launching:", ", ".join(todo))

    procs = {}
    for name in todo:
        log = open(LOGS / f"{name}.log", "w", encoding="utf-8")
        procs[name] = (subprocess.Popen(
            [sys.executable, str(EXP / f"{name}.py")],
            stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT)), log)

    t0, pending = time.time(), set(procs)
    while pending:
        time.sleep(15)
        for name in sorted(pending):
            proc, log = procs[name]
            rc = proc.poll()
            if rc is None:
                continue
            pending.discard(name)
            log.close()
            mins = (time.time() - t0) / 60
            status = "done" if rc == 0 else f"FAILED (rc={rc}, see {LOGS / (name + '.log')})"
            print(f"[{mins:5.1f}m] {name}: {status}", flush=True)

    summarize()


if __name__ == "__main__":
    main()
