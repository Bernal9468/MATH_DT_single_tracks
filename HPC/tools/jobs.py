#!/usr/bin/env python3
"""Status of the single-track jobs: running simulations (progress %, elapsed, ETA), calibration loops, queued scripts,
recently finished / failed runs, CPU and GPU load.

  python3 tools/jobs.py            one snapshot
  python3 tools/jobs.py -w         live view, updated in place every 5 min (-w 60: every 60 s; Ctrl-C to quit)
  python3 tools/jobs.py -n 20      show the 20 most recent finished runs

Progress of a simulation = last simulated time in its probe.csv (written every step) / t_final from settings.txt.
A run is RUNNING if a thermal program writes into its folder, DONE if run.log has the final 'Wall time' line, otherwise
STOPPED (killed or crashed; see its run.log).
"""
import argparse, glob, os, re, shutil, signal, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TTY = sys.stdout.isatty()
C = lambda code, s: f"\033[{code}m{s}\033[0m" if TTY else s
GREEN, YELLOW, RED, DIM, BOLD, CYAN = (lambda s: C("32", s)), (lambda s: C("33", s)), (lambda s: C("31", s)), \
                                      (lambda s: C("2", s)), (lambda s: C("1", s)), (lambda s: C("36", s))


def procs():
    """[(pid, argv list)] of all processes of this user"""
    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try: argv = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError: continue
        out.append((int(p), [a.decode(errors="replace") for a in argv if a]))
    return out


def proc_start(pid):
    """process start time (epoch s)"""
    try:
        st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        boot = next(float(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime"))
        return boot + int(st[19]) / os.sysconf("SC_CLK_TCK")
    except Exception: return None


def fmt_t(s):
    if s is None or s != s: return "  --  "
    s = int(s); h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s"


def bar(f, w=28):
    f = min(max(f, 0.0), 1.0); n = f * w; full = int(n)
    part = " ▏▎▍▌▋▊▉"[int((n - full) * 8)] if full < w else ""
    return "▕" + "█" * full + part + " " * (w - full - len(part)) + "▏"


def settings(d):
    try: s = open(os.path.join(d, "settings.txt")).read()
    except OSError: return {}
    d = dict(re.findall(r"(\w+)=(\S+)", s))
    m = re.search(r"mesh=(.+?\.mesh)", s)                 # paths may contain spaces
    if m: d["mesh"] = m.group(1)
    m = re.search(r"extra=(.*)$", s.strip())                # extra driver options: the rest of the line
    if m: d["extra"] = m.group(1).strip()
    return d


def last_time(d):
    """last simulated time in probe.csv (reads only the tail)"""
    fn = os.path.join(d, "probe.csv")
    try:
        with open(fn, "rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 4096)); lines = f.read().decode(errors="replace").strip().split("\n")
        for l in reversed(lines):
            try: return float(l.split(",")[0])
            except ValueError: continue
    except OSError: pass
    return 0.0


def run_state(d, running_dirs):
    if d in running_dirs: return "RUNNING"
    try:
        with open(os.path.join(d, "run.log"), "rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 3000)); tail = f.read().decode(errors="replace")
        if "Wall time" in tail: return "DONE"
    except OSError: return "NEW"
    return "STOPPED"


def wall(d):
    try:
        with open(os.path.join(d, "run.log"), "rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 3000)); m = re.search(r"Wall time\s*:\s*([\d.]+)", f.read().decode(errors="replace"))
        return float(m.group(1)) if m else None
    except OSError: return None


def short(d): return os.path.relpath(d, ROOT)


def loads():
    out = []
    try:
        l1 = os.getloadavg()[0]; out.append(f"CPU load {l1:.1f} / {os.cpu_count()} threads")
    except OSError: pass
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip().split(", ")
            out.append(f"GPU {r[0]}: {r[1]} % busy, {int(r[2]) / 1024:.1f} / {int(r[3]) / 1024:.1f} GB")
        except Exception: pass
    return "   ".join(out)


def snapshot(nrecent):
    now = time.time(); P = procs(); L = []
    L.append(BOLD(f"single-track jobs   {time.strftime('%Y-%m-%d %H:%M:%S')}") + "   " + DIM(loads()))
    # ---- running simulations
    sims = []
    for pid, a in P:
        if a and os.path.basename(a[0]).startswith("thermal_") and "-pcsv" in a:
            sims.append((pid, os.path.dirname(a[a.index("-pcsv") + 1])))
    # on a cluster the programs run on compute nodes (not visible here): a run whose run.log was written in the last
    # 10 minutes and has no final 'Wall time' line counts as running
    if shutil.which("squeue"):
        seen = {d for _, d in sims}
        for p in glob.glob(os.path.join(RES, "**", "run.log"), recursive=True):
            d = os.path.realpath(os.path.dirname(p))
            if d not in seen and now - os.path.getmtime(p) < 600 and run_state(d, set()) != "DONE": sims.append((None, d))
    running_dirs = {d for _, d in sims}
    L.append("\n" + BOLD("RUNNING SIMULATIONS"))
    if not sims: L.append(DIM("  none"))
    for pid, d in sims:
        st = settings(d); tf = float(st.get("t_final", "nan")); t = last_time(d); f = t / tf if tf == tf and tf > 0 else 0.0
        t0 = proc_start(pid) if pid else os.path.getmtime(os.path.join(d, "settings.txt"))
        el = now - t0 if t0 else None; eta = el * (1 - f) / f if el and f > 0.01 else None
        L.append(f"  {GREEN('●')} case {st.get('case', '?'):4s} {bar(f)} {f * 100:5.1f} %   t {t * 1e3:6.3f}/{tf * 1e3:.3f} ms   "
                 f"elapsed {fmt_t(el)}  ETA {fmt_t(eta)}")
        L.append(DIM(f"      {short(d)}   eta={st.get('absorptivity', '?')} c={st.get('c', '?')} dt={st.get('dt', '?')} "
                     f"mesh={os.path.basename(st.get('mesh', '?'))} {('extra=' + st['extra']) if st.get('extra') else ''}"
                     + (f"  pid {pid}" if pid else "")))
    if shutil.which("squeue"):
        L.append("\n" + BOLD("SLURM JOBS"))
        try:
            q = subprocess.run(["squeue", "--me", "-h", "-o", "%.10i %.9P %.16j %.8T %.10M %.10l %R"], capture_output=True,
                               text=True, timeout=10).stdout.rstrip()
            L += ["  " + l for l in q.split("\n")] if q else [DIM("  none")]
        except Exception as e: L.append(DIM(f"  squeue failed: {e}"))
    # ---- calibration loops / queued scripts
    L.append("\n" + BOLD("CALIBRATIONS AND QUEUED SCRIPTS"))
    found = False
    for pid, a in P:
        s = " ".join(a)
        if len(a) >= 2 and a[0].startswith("python") and "post/calibrate" in s:
            found = True; script = os.path.basename(a[1]); out = os.path.join(ROOT, a[2]) if not a[2].startswith("/") else a[2]
            its = sorted(glob.glob(os.path.join(out, "it*")), key=os.path.getmtime)
            if script == "calibrate_multi.py":
                m = re.search(r"--cases\s+(\S+)", s); ncase = len(m.group(1).split(",")) if m else 7
                mx = re.search(r"--maxit\s+(\d+)", s); maxit = int(mx.group(1)) if mx else 4
                cur = max((int(re.search(r"it(\d+)$", i).group(1)) for i in its if re.search(r"it(\d+)$", i)), default=0)
                cd = os.path.join(out, f"it{cur}")
                done = sum(1 for c in glob.glob(os.path.join(cd, "case*")) if run_state(os.path.realpath(c), running_dirs) == "DONE")
                part = sum(last_time(d) / float(settings(d).get("t_final", 1)) for _, d in sims if d.startswith(os.path.realpath(cd)))
                f = (done + part) / ncase
                L.append(f"  {CYAN('◆')} {script}  iteration {cur} (max {maxit - 1})  {bar(f)} {f * 100:5.1f} %  "
                         f"{done}/{ncase} cases of this iteration done")
            elif script == "calibrate_solid.py":
                runs = [d for d in its if os.path.isdir(d)]
                done = sum(1 for d in runs if run_state(d, running_dirs) == "DONE")
                cur = max((int(re.search(r"it(\d+)_", os.path.basename(d)).group(1)) for d in runs), default=0)
                L.append(f"  {CYAN('◆')} {script}  iteration {cur}  ({done} runs done; "
                         + ("1 run per iteration, phi only)" if "--phi-only" in s else "3 runs per iteration)"))
            else:
                L.append(f"  {CYAN('◆')} {script}")
            L.append(DIM(f"      {short(out)}   started {time.strftime('%H:%M', time.localtime(proc_start(pid) or now))}  pid {pid}"))
            log = out.rstrip("/") + ".log"
            if os.path.exists(log):
                tail = [l.rstrip() for l in open(log, errors="replace") if l.strip()][-2:]
                for l in tail: L.append(DIM("      │ " + l.replace(ROOT + "/", "")[:150]))
        elif len(a) >= 2 and a[0].endswith("bash") and a[1].startswith("run/") and "run_case.sh" not in a[1]:
            found = True
            waiting = any("post/calibrate_multi.py" in " ".join(b) for _, b in P) and "after_multi" in a[1]
            L.append(f"  {YELLOW('◇')} {a[1]}  " + (YELLOW("queued: waits for calibrate_multi.py") if waiting else "running") + DIM(f"  pid {pid}"))
    if not found: L.append(DIM("  none"))
    # ---- recently finished
    L.append("\n" + BOLD(f"RECENT RUNS (last {nrecent})"))
    dirs = [os.path.realpath(os.path.dirname(p)) for p in glob.glob(os.path.join(RES, "**", "settings.txt"), recursive=True)]
    dirs = sorted(set(dirs) - running_dirs, key=lambda d: os.path.getmtime(os.path.join(d, "settings.txt")), reverse=True)[:nrecent]
    for d in dirs:
        stt = run_state(d, running_dirs); st = settings(d)
        mt = os.path.getmtime(os.path.join(d, "run.log")) if os.path.exists(os.path.join(d, "run.log")) else 0
        tag = GREEN("✔ DONE   ") if stt == "DONE" else RED("✖ STOPPED") if stt == "STOPPED" else DIM("  NEW    ")
        extra = fmt_t(wall(d)) if stt == "DONE" else f"at {last_time(d) / float(st.get('t_final', 1)) * 100:.0f} %"
        L.append(f"  {tag} case {st.get('case', '?'):4s} {time.strftime('%m-%d %H:%M', time.localtime(mt))}  {extra:>7s}   {short(d)}")
    return "\n".join(L)


ANSI = re.compile(r"\033\[[0-9;?]*[A-Za-z]")


def fit(line, width):
    """cut a line to the terminal width (counting visible characters only) so that nothing wraps"""
    if len(ANSI.sub("", line)) <= width: return line
    return ANSI.sub("", line)[:width - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--watch", type=float, nargs="?", const=300.0, default=0, help="live view, refresh period [s] (default 300 = 5 min)")
    ap.add_argument("-n", type=int, default=8)
    a = ap.parse_args()
    if not a.watch or not TTY: print(snapshot(a.n)); return
    # live view on the terminal's alternate screen (as top/htop): every frame overwrites the previous one in place,
    # nothing accumulates in the scrollback, and the normal screen comes back on exit
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    sys.stdout.write("\033[?1049h\033[?25l")
    try:
        while True:
            w, h = shutil.get_terminal_size()
            lines = (snapshot(a.n) + "\n\n" + DIM(f"updated {time.strftime('%H:%M:%S')}, every {a.watch:g} s   Ctrl-C to quit")).split("\n")
            sys.stdout.write("\033[H" + "\n".join(fit(l, w) + "\033[K" for l in lines[:h - 1]) + "\033[J")
            sys.stdout.flush(); time.sleep(a.watch)
    except KeyboardInterrupt: pass
    finally:
        sys.stdout.write("\033[?25h\033[?1049l"); sys.stdout.flush()


if __name__ == "__main__":
    main()
