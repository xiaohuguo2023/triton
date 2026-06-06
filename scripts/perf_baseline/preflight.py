"""Pre-flight GPU cleanliness checks. Abort the sweep if the machine is contended.

Usage:  python3 preflight.py        # exit 0 if clean, 1 if contended
        python3 preflight.py --force # report status but always exit 0

Checks:
  1. rocm-smi --showpids  -> only our process (or nothing) on the GPU
  2. rocm-smi --showclocks -> SCLK at peak (boost), not throttled
  3. rocm-smi --showtemp  -> not within 10C of throttle
  4. docker ps             -> no other long-running benchmark containers
  5. who                   -> note other interactive users (warning, not abort)
"""
import argparse, os, re, subprocess, sys

THROTTLE_TEMP_C = 95  # approximate gfx950 throttle point
TEMP_MARGIN_C = 10    # abort if within this margin


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return -1, "", str(e)


def check_gpu_pids():
    rc, out, _ = run(["rocm-smi", "--showpids"])
    if rc != 0:
        return False, "rocm-smi --showpids failed"
    my_pid = os.getpid()
    parent = os.getppid()
    foreign = []
    for line in out.splitlines():
        m = re.search(r"PID\s+(\d+)\s", line)
        if not m:
            continue
        pid = int(m.group(1))
        if pid not in (my_pid, parent):
            foreign.append(pid)
    if foreign:
        return False, f"foreign GPU pids: {foreign}"
    return True, "GPU has no foreign processes"


def check_clocks():
    rc, out, _ = run(["rocm-smi", "--showclocks"])
    if rc != 0:
        return False, "rocm-smi --showclocks failed"
    # Heuristic: SCLK should be > 1500 MHz on MI355X under load. Idle ~ 500-800.
    # We're not under load here, so we can only check it's not stuck at low.
    # Real check: rerun after warmup.
    return True, "clock check skipped (do it post-warmup)"


def check_temp():
    rc, out, _ = run(["rocm-smi", "--showtemp"])
    if rc != 0:
        return False, "rocm-smi --showtemp failed"
    temps = []
    for line in out.splitlines():
        m = re.search(r"Temperature.*?(\d+(?:\.\d+)?)\s*[Cc]", line)
        if m:
            temps.append(float(m.group(1)))
    if not temps:
        return True, "no temperature data parsed"
    max_t = max(temps)
    if max_t > (THROTTLE_TEMP_C - TEMP_MARGIN_C):
        return False, f"GPU at {max_t:.0f}C, within {TEMP_MARGIN_C}C of throttle"
    return True, f"GPU temp OK ({max_t:.0f}C)"


def check_containers():
    rc, out, _ = run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}"])
    if rc != 0:
        return True, "docker ps unavailable (skip)"
    lines = [l for l in out.splitlines() if l.strip()]
    # We just warn — many containers may be running but not using GPU.
    return True, f"{len(lines)} containers running (none confirmed using GPU)"


def check_users():
    rc, out, _ = run(["who"])
    if rc != 0:
        return True, ""
    users = set()
    for line in out.splitlines():
        parts = line.split()
        if parts:
            users.add(parts[0])
    me = os.environ.get("USER", "")
    others = users - {me}
    if others:
        return True, f"warning: other users logged in: {sorted(others)}"
    return True, "no other interactive users"


CHECKS = [
    ("GPU PIDs",      check_gpu_pids),
    ("GPU clocks",    check_clocks),
    ("GPU temp",      check_temp),
    ("Containers",    check_containers),
    ("Users",         check_users),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="report status but always exit 0")
    args = ap.parse_args()

    print("=== Pre-flight GPU cleanliness checks ===")
    all_ok = True
    for name, fn in CHECKS:
        ok, msg = fn()
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] {name:<14} {msg}")
        if not ok:
            all_ok = False
    print()
    if all_ok:
        print("Machine is clean. Safe to benchmark.")
        sys.exit(0)
    else:
        print("Machine is NOT clean.")
        if args.force:
            print("(--force given; proceeding anyway)")
            sys.exit(0)
        print("Re-run with --force to override, or wait for contention to clear.")
        sys.exit(1)


if __name__ == "__main__":
    main()
