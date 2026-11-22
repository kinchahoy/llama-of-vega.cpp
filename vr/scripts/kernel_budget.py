#!/usr/bin/env python3
"""Kernel budget from a rocprofv3 kernel trace: where the token time goes and which kernels to look at first.

usage: kernel_budget.py TRACE_DIR N_TOKENS [--top N]

TRACE_DIR is the -d directory of `rocprofv3 --kernel-trace --stats -d TRACE_DIR -o run -- llama-bench ... -p 0 -n N_TOKENS -r 1`.
The first third of the launches is dropped (load and warmup). Times are trace times: the profiler adds about 1.2 us per kernel.

Classes:
  floor      at the trace floor (<= FLOOR + 0.6 us). Only fewer launches help. Value of removing one: 2.2 us untraced.
  one-block  1 to 4 blocks per launch and above the floor. Bound by the one-block rule (one block is about 1/30 of the GPU).
             Candidates for a split over many blocks or a two-kernel form. "excess" is what a floor-cost kernel would save.
  work       many blocks and above the floor. Real work: check bytes read against 800 GB/s before touching it.
"""
import glob
import re
import sqlite3
import sys

FLOOR = 4.2  # us, trace floor of a trivial kernel


def short(name):
    s = re.sub(r"^void ", "", name)
    return re.sub(r"\(.*", "", s)[:66]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    trace, n_tok = sys.argv[1], float(sys.argv[2])
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 14
    dbs = glob.glob(f"{trace}/**/*.db", recursive=True)
    if not dbs:
        sys.exit(f"no .db under {trace}")
    c = sqlite3.connect(f"file:{dbs[0]}?mode=ro", uri=True)
    cols = [r[1] for r in c.execute('pragma table_info("kernels")')]
    dev = "agent_abs_index" if "agent_abs_index" in cols else "nid"
    starts = [r[0] for r in c.execute("select start from kernels order by start")]
    cut = starts[len(starts) // 3]
    tok = n_tok * 2 / 3
    n_dev = c.execute(f"select count(distinct {dev}) from kernels where start >= ?", (cut,)).fetchone()[0]
    q = ("select name, count(*), sum(duration), "
         "avg(max(1, grid_x / max(1, workgroup_x)) * max(1, grid_y / max(1, workgroup_y)) * max(1, grid_z / max(1, workgroup_z))), "
         "avg(workgroup_x * max(1, workgroup_y) * max(1, workgroup_z)) "
         "from kernels where start >= ? group by name")
    rows = []
    for name, calls, dur, blocks, threads in c.execute(q, (cut,)):
        us = dur / calls / 1e3
        cls = "floor" if us <= FLOOR + 0.6 else ("one-block" if blocks <= 4 else "work")
        rows.append(dict(name=short(name), calls=calls / tok / n_dev, us=us, ms=dur / 1e6 / tok / n_dev,
                         excess=max(0.0, us - FLOOR) * calls / 1e3 / tok / n_dev, blocks=blocks, threads=threads, cls=cls))
    total = sum(r["ms"] for r in rows)
    n_launch = sum(r["calls"] for r in rows)
    print(f"{n_dev} GPU(s), per token per GPU (traced): {n_launch:.0f} launches, {total:.2f} ms kernel time")
    print(f"launch floor: {n_launch:.0f} x {FLOOR} us = {n_launch * FLOOR / 1e3:.2f} ms traced, about {n_launch * 2.2 / 1e3:.2f} ms untraced (2.2 us per launch)")
    for cls in ("floor", "one-block", "work"):
        sel = [r for r in rows if r["cls"] == cls]
        print(f"  {cls:9s}: {sum(r['calls'] for r in sel):6.0f} launches, {sum(r['ms'] for r in sel):6.2f} ms, excess over floor {sum(r['excess'] for r in sel):5.2f} ms")
    hdr = f"  {'ms/tok':>7} {'excess':>7} {'calls':>7} {'us':>7} {'blocks':>7} {'thr':>5}  kernel"
    print("\none-block kernels above the floor (first candidates; apply the one-block rule):")
    print(hdr)
    for r in sorted((r for r in rows if r["cls"] == "one-block"), key=lambda r: -r["excess"])[:top]:
        print(f"  {r['ms']:7.3f} {r['excess']:7.3f} {r['calls']:7.1f} {r['us']:7.1f} {r['blocks']:7.0f} {r['threads']:5.0f}  {r['name']}")
    print("\nwork kernels (check bytes read against 800 GB/s; us per block is a hint for too few blocks):")
    print(hdr)
    for r in sorted((r for r in rows if r["cls"] == "work"), key=lambda r: -r["ms"])[:top]:
        print(f"  {r['ms']:7.3f} {r['excess']:7.3f} {r['calls']:7.1f} {r['us']:7.1f} {r['blocks']:7.0f} {r['threads']:5.0f}  {r['name']}")
    print("\nfloor kernels with the most launches (fusion candidates; each launch is worth about 2.2 us untraced):")
    print(hdr)
    for r in sorted((r for r in rows if r["cls"] == "floor"), key=lambda r: -r["calls"])[:top]:
        print(f"  {r['ms']:7.3f} {r['excess']:7.3f} {r['calls']:7.1f} {r['us']:7.1f} {r['blocks']:7.0f} {r['threads']:5.0f}  {r['name']}")


if __name__ == "__main__":
    main()
