"""One-command verification of the CAMPS implementation on any CUDA system.

    python -m experiments.verify_all            # ~10-20 min on a single GPU
    python -m experiments.verify_all --full     # adds the full sweep (hours)

Writes every output to $CAMPS_RESULTS_DIR (default: a fresh directory
verification_<host>/ next to the repository's results/), never touching the
archived results, and ends with verification_report.md: environment, one
PASS/FAIL line per check, and the archived-versus-new comparison tables.

What must agree with the archive on any GPU (checked):
  * exit statuses, attempt counts, accepted/rejected splits of every adapter;
  * the scientific invariants of the theorems: no measured contraction
    factor exceeds its rigorous bound, limiting backward errors at the root
    are O(u_64), the rounded-displacement counterexample is rejected, both
    SPD contracts accept identically;
  * convergence of every method that converged in the archive (full mode).
What is expected to differ and is reported but not asserted:
  * all timings and hence all speedups (hardware-dependent);
  * residuals and contraction factors in their trailing digits (different
    GPUs round reductions differently); they must stay within a factor of 10.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.paths import ROOT, archive_dir  # noqa: E402

U64 = 2.0 ** -53
PY = sys.executable


def run(cmd, env, log):
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    return p.returncode, time.perf_counter() - t0


def load(path):
    return json.load(open(path))


def within(a, b, factor):
    if a is None or b is None:
        return True
    if a == 0 or b == 0:
        return a == b
    return (1.0 / factor) <= (a / b) <= factor


# ---------------------------------------------------------------------------
def check_recursion(out, arc, report):
    new, old = load(os.path.join(out, "verify_recursion.json")), load(os.path.join(arc, "verify_recursion.json"))
    ok, rows = True, []
    for cn, co in zip(new["cases"], old["cases"]):
        for rn, ro in zip(cn["rows"], co["rows"]):
            bound_ok = (rn["eta_mid_meas"] is None or not math.isfinite(rn["eta_mid_bound"]) or rn["eta_mid_meas"] <= rn["eta_mid_bound"]) and \
                       (rn["eta_top_meas"] is None or not math.isfinite(rn["eta_top_bound"]) or rn["eta_top_meas"] <= rn["eta_top_bound"])
            beta_ok = rn["beta_root"] <= 2.0 * U64
            status_ok = rn["root_status"] == ro["root_status"]
            close = within(rn["eta_top_first"], ro["eta_top_first"], 10.0) and within(rn["beta_root"], ro["beta_root"], 10.0)
            row_ok = bound_ok and beta_ok and status_ok and close
            ok &= row_ok
            rows.append(f"| {cn['kappa']:.0e} | {cn['damp']} | {cn['adversarial']} | {rn['J1']} | "
                        f"{rn['eta_top_first']:.2e} | {ro['eta_top_first']:.2e} | {rn['beta_root']/U64:.2f} u64 | "
                        f"{'yes' if bound_ok else 'NO'} | {rn['root_status']} | {'ok' if row_ok else 'FAIL'} |")
    report.append("### Recursive bounds (Theorem 4, Section 5.9)\n")
    report.append("| kappa | leaf damp | adversarial | J1 | eta_top new | eta_top archive | beta_root | bounds respected | status | row |")
    report.append("|---|---|---|---|---|---|---|---|---|---|")
    report.extend(rows)
    report.append("")
    return ok


def check_energy(out, arc, report):
    new, old = load(os.path.join(out, "spd_energy_check.json")), load(os.path.join(arc, "spd_energy_check.json"))
    ce = new["counterexample"]
    ok = (ce["proposed_step_accepts"] is True) and (ce["certified_test_accepts"] is False) and ce["y_new_equals_y"]
    report.append("### Energy contract (Section 4.3, Appendix A.10)\n")
    report.append(f"- counterexample A=1, y=2^53, d=1/4: proposed-step test accepts = {ce['proposed_step_accepts']}, "
                  f"certified test accepts = {ce['certified_test_accepts']} (expected True / False)")
    report.append("| grid | contract | converged | corrections | accepted/rejected | rel. residual new | archive | solve ms new | archive |")
    report.append("|---|---|---|---|---|---|---|---|---|")
    for g in (32, 48):
        for acc in ("residual", "energy"):
            k = f"g{g}_{acc}"
            n, o = new[k], old[k]
            same = n["converged"] and n["corrections"] == o["corrections"] and n["accepted"] == o["accepted"] \
                and n["rejected"] == o["rejected"] and within(n["rel_res"], o["rel_res"], 10.0)
            ok &= same
            report.append(f"| {g}^3 | {acc} | {n['converged']} | {n['corrections']} | {n['accepted']}/{n['rejected']} | "
                          f"{n['rel_res']:.2e} | {o['rel_res']:.2e} | {n['solve_ms']:.1f} | {o['solve_ms']:.1f} |")
    margins = new.get("energy_margins", {})
    worst = min((m["ratio_bound_over_dphi"] for g in margins.values() for m in g if m["ratio_bound_over_dphi"]), default=None)
    if worst is not None:
        ok &= worst < 1e-3
        report.append(f"- largest bound/|dphi| over all certified corrections: {worst:.2e} (must be < 1e-3)")
    report.append("")
    return ok


FALLBACK_LINE = re.compile(
    r"^(?P<row>.+?)\s+new:\s+(?P<ns>\w+)\s+it=\s*(?P<ni>\d+)\s+acc/rej=(?P<na>\S+)\s+rr=(?P<nr>\S+)"
    r"\s+\|\s+stored:\s+(?P<os>\w+)\s+it=\s*(?P<oi>\d+)\s+acc/rej=(?P<oa>\S+)\s+rr=(?P<or>\S+)")


def check_fallback(out, arc, report, log):
    """Parse regress_fallback's printed comparison (it compares against the archive
    itself). Statuses, attempt counts and accept/reject splits must agree exactly;
    residuals within a factor of ten (trailing digits differ between GPUs)."""
    txt = open(log, encoding="utf-8").read()
    ok, n = True, 0
    report.append("### Fallback, damped fallback, and ladder rows (Section 5.4)\n")
    report.append("| row | new: status, attempts, acc/rej, residual | archive | match |")
    report.append("|---|---|---|---|")
    for l in txt.splitlines():
        m = FALLBACK_LINE.match(l.strip())
        if not m:
            continue
        n += 1
        g = m.groupdict()
        same = (g["ns"] == g["os"] and g["ni"] == g["oi"] and g["na"] == g["oa"]
                and within(float(g["nr"]), float(g["or"]), 10.0))
        ok &= same
        report.append(f"| {g['row'].strip()} | {g['ns']}, {g['ni']}, {g['na']}, {g['nr']} | "
                      f"{g['os']}, {g['oi']}, {g['oa']}, {g['or']} | {'ok' if same else 'FAIL'} |")
    if n == 0:
        ok = False
        report.append("| (no comparison lines found; see log) | | | FAIL |")
    report.append("")
    return ok


def check_pilot(out, report, log):
    txt = open(log, encoding="utf-8").read()
    n_ok = txt.count(": ok status=")
    n_dnc = txt.count(": DNC status=")
    n_exc = txt.count("EXCEPTION")
    ok = n_exc == 0 and n_ok > 0
    report.append("### Pilot sweep (all seven experiments, reduced sizes)\n")
    report.append(f"- converged runs: {n_ok}; non-converged: {n_dnc}; exceptions: {n_exc} (any exception fails the check)")
    report.append("")
    return ok


def check_full(out, arc, report):
    """Every method that converged in the archive must converge here with the
    same status; outer counts reported; speedups reported, not asserted."""
    ok = True
    report.append("### Full sweep: convergence agreement with the archive\n")
    for fn in ("dense_wc.json", "dense_ic.json", "sparse_spd.json", "sparse_nsy.json",
               "placement.json", "f3r.json", "multirhs.json"):
        pn, po = os.path.join(out, fn), os.path.join(arc, fn)
        if not os.path.exists(pn):
            report.append(f"- `{fn}`: not produced"); ok = False; continue
        new, old = load(pn), load(po)
        key = lambda r: (r.get("problem"), r.get("method"), r.get("n_rhs"))
        idx = {key(r): r for r in new["records"]}
        mism, total = [], 0
        for r in old["records"]:
            total += 1
            q = idx.get(key(r))
            if q is None:
                mism.append(f"{r['problem']} / {r['method']}: missing"); continue
            if (r["converged"] != q["converged"]) or (r["status"] != q["status"]):
                mism.append(f"{r['problem']} / {r['method']}: archive {r['status']} vs new {q['status']}")
            elif r.get("outer_iterations") != q.get("outer_iterations"):
                mism.append(f"{r['problem']} / {r['method']}: outer {r.get('outer_iterations')} vs {q.get('outer_iterations')} (noted, not failed)")
        hard = [m for m in mism if "noted" not in m]
        ok &= not hard
        report.append(f"- `{fn}`: {total} archived records; {len(hard)} convergence/status disagreements; "
                      f"{len(mism) - len(hard)} iteration-count differences")
        for m in mism:
            report.append(f"    - {m}")
    report.append("")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run the full sweep (hours)")
    ap.add_argument("--report-only", action="store_true",
                    help="do not rerun anything; rebuild the report from the logs and files already in the output directory")
    args = ap.parse_args()
    host = socket.gethostname().split(".")[0]
    out = os.environ.get("CAMPS_RESULTS_DIR") or os.path.join(ROOT, f"verification_{host}")
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, CAMPS_RESULTS_DIR=out, PYTHONIOENCODING="utf-8")
    arc = archive_dir()
    report = ["# CAMPS verification report\n"]
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA device"
        tv = torch.__version__
    except Exception as e:  # pragma: no cover
        gpu, tv = f"torch import failed: {e}", "?"
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        drv = "n/a"
    report.append(f"- host: {socket.gethostname()}  \n- platform: {platform.platform()}  \n- python: {platform.python_version()}  "
                  f"\n- torch: {tv}  \n- GPU: {gpu}  \n- driver / memory: {drv}  \n- output directory: `{out}`  "
                  f"\n- archive compared against: `{arc}`  \n- date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    steps = [
        ("verify_recursion", [PY, "-m", "experiments.verify_recursion"]),
        ("energy_check", [PY, "-m", "experiments.energy_check"]),
        ("regress_fallback", [PY, "-m", "experiments.regress_fallback"]),
        ("pilot", [PY, "-m", "experiments.run_experiments", "--exp", "all", "--pilot"]),
    ]
    if args.full:
        steps.append(("full_sweep", [PY, "-m", "experiments.run_experiments", "--exp", "all"]))
    results = {}
    steps_file = os.path.join(out, "steps.json")
    if args.report_only:
        saved = load(steps_file) if os.path.exists(steps_file) else {}
        if "full_sweep" in saved and not args.full:
            steps.append(("full_sweep", None))
        for name, _ in steps:
            log = os.path.join(out, f"{name}.log")
            rc, dt = saved.get(name, [0 if os.path.exists(log) else 1, 0.0])
            results[name] = (rc, dt, log)
    else:
        for name, cmd in steps:
            log = os.path.join(out, f"{name}.log")
            print(f"[{name}] running ... ", end="", flush=True)
            rc, dt = run(cmd, env, log)
            print(f"exit {rc} in {dt/60:.1f} min", flush=True)
            results[name] = (rc, dt, log)
        with open(steps_file, "w") as fh:
            json.dump({k: [v[0], v[1]] for k, v in results.items()}, fh)

    verdicts = {}
    try:
        verdicts["recursive bounds"] = results["verify_recursion"][0] == 0 and check_recursion(out, arc, report)
    except Exception as e:
        verdicts["recursive bounds"] = False; report.append(f"recursive bounds: error {e}\n")
    try:
        verdicts["energy contract"] = results["energy_check"][0] == 0 and check_energy(out, arc, report)
    except Exception as e:
        verdicts["energy contract"] = False; report.append(f"energy: error {e}\n")
    try:
        verdicts["fallback/ladder rows"] = results["regress_fallback"][0] == 0 and check_fallback(out, arc, report, results["regress_fallback"][2])
    except Exception as e:
        verdicts["fallback/ladder rows"] = False; report.append(f"fallback: error {e}\n")
    verdicts["pilot sweep"] = results["pilot"][0] == 0 and check_pilot(out, report, results["pilot"][2])
    if "full_sweep" in results:
        try:
            verdicts["full sweep convergence"] = results["full_sweep"][0] == 0 and check_full(out, arc, report)
        except Exception as e:
            verdicts["full sweep convergence"] = False; report.append(f"full: error {e}\n")

    summary = ["## Summary\n", "| check | result | wall time |", "|---|---|---|"]
    name_map = {"recursive bounds": "verify_recursion", "energy contract": "energy_check",
                "fallback/ladder rows": "regress_fallback", "pilot sweep": "pilot", "full sweep convergence": "full_sweep"}
    for k, v in verdicts.items():
        dt = results[name_map[k]][1]
        summary.append(f"| {k} | {'PASS' if v else 'FAIL'} | {dt/60:.1f} min |" if dt else f"| {k} | {'PASS' if v else 'FAIL'} | (from logs) |")
    all_ok = all(verdicts.values())
    summary.append(f"\n**Overall: {'PASS' if all_ok else 'FAIL'}**\n")
    report = report[:2] + summary + report[2:]
    path = os.path.join(out, "verification_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print("\n".join(summary))
    print("report:", path)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
