#!/usr/bin/env python3
"""Anchor-segmented kernel budget & BA-4 occupancy auditor for rocprofv3 SQLite traces.

usage:
  kernel_budget.py TRACE_PATH [N_TOKENS] [--mode auto|tg|pp] [--top N] [--per-gpu]

TRACE_PATH is a rocprofv3 output directory (containing *.db) or a direct .db file.
N_TOKENS is optional: in decode (tg) mode, steady-state tokens are auto-detected from
the largest-grid dense LM-head anchor kernel (filtering out load, warmup, and outliers
with wall > 1.15 * median or non-modal kernel count). Falls back to prefill (pp) mode
when fewer than 4 anchors exist or when --mode pp is given.

Calibrated gfx906 hardware model (vr/HARDWARE.md):
  - VGPR granule 4, pool 256/SIMD, max 8 waves/SIMD (32 waves/CU), LDS 64 KB/CU
  - Resident blocks/CU = min(floor(4 * waves_per_simd / ceil(T/64)), floor(65536 / LDS))
  - Graph-replayed launch value: 1.22 us (pure floor) to 1.70 us (launch + tiny work)
  - Profiler overhead: +1.2 us/kernel (shows 4.2 us trace floor)
"""

import argparse
from collections import Counter, defaultdict
import glob
import json
import math
import os
import re
import sqlite3
import statistics
import sys

TRACE_FLOOR_US = 4.2       # rocprofv3 trivial kernel floor (includes ~1.2 us profiler cost)
GRAPH_LAUNCH_LO_US = 1.22  # pure HIP graph node launch floor (HARDWARE.md)
GRAPH_LAUNCH_HI_US = 1.70  # upper bound including trivial kernel work (Inkling ABBA 1.5-1.9 us)
DEFAULT_CU_BY_IDX = (64, 60)  # MI60 = 64 CU (ROCm0), MI50 = 60 CU (ROCm1)


def short_name(name, maxlen=64):
    s = re.sub(r"^void\s+", "", name)
    # Strip function parameter list '(' only at template angle-bracket depth 0
    depth = 0
    for i, ch in enumerate(s):
        if ch == "<":
            depth += 1
        elif ch == ">" and depth > 0:
            depth -= 1
        elif ch == "(" and depth == 0:
            s = s[:i]
            break
    s = s.replace("(ggml_type)", "t")
    return s[:maxlen]


def find_db(trace_path):
    if os.path.isfile(trace_path):
        return trace_path
    dbs = sorted(glob.glob(os.path.join(trace_path, "**", "*.db"), recursive=True))
    if not dbs:
        sys.exit(f"error: no .db found under {trace_path}")
    return dbs[0]


def load_gpu_agents(conn):
    """Map agent_abs_index -> dict(label, cu, name, type_index)."""
    agents = {}
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    if "rocpd_info_agent" in tables:
        cols = {r[1] for r in conn.execute('PRAGMA table_info("rocpd_info_agent")')}
        if {"absolute_index", "type", "extdata"} <= cols:
            q = "SELECT absolute_index, type_index, name, extdata FROM rocpd_info_agent WHERE type='GPU'"
            for abs_idx, t_idx, name, ext in conn.execute(q):
                cu = None
                try:
                    d = json.loads(ext) if ext else {}
                    cu = d.get("cu_count")
                except Exception:
                    cu = None
                if not cu:
                    cu = DEFAULT_CU_BY_IDX[t_idx] if t_idx < len(DEFAULT_CU_BY_IDX) else 64
                agents[abs_idx] = dict(
                    abs_index=abs_idx,
                    type_index=t_idx,
                    label=f"ROCm{t_idx}",
                    cu=int(cu),
                    arch=name or "gfx906",
                )
    return agents


def gfx906_occupancy(threads, vgpr, lds, cu_count, blocks):
    """Calibrated Q1 occupancy formula from vr/HARDWARE.md."""
    waves_per_block = max(1, (threads + 63) // 64)
    vgpr_alloc = max(4, ((max(1, vgpr) + 3) // 4) * 4)
    waves_per_simd = min(8, 256 // vgpr_alloc)
    b_vgpr = (4 * waves_per_simd) // waves_per_block
    b_lds = (65536 // lds) if lds > 0 else 64
    b_cu = max(1, min(b_vgpr, b_lds))
    w_cu = b_cu * waves_per_block
    res_gpu = b_cu * cu_count
    rounds = (blocks + res_gpu - 1) // res_gpu if res_gpu > 0 else 1
    return dict(
        vgpr_alloc=vgpr_alloc,
        waves_per_simd=waves_per_simd,
        waves_per_block=waves_per_block,
        b_cu=b_cu,
        w_cu=w_cu,
        res_gpu=res_gpu,
        rounds=rounds,
    )


def union_duration_ns(intervals):
    """Total union time (ns) of possibly overlapping [start, end] intervals."""
    if not intervals:
        return 0
    s_iv = sorted(intervals)
    cur_s, cur_e = s_iv[0]
    total = 0
    for st, en in s_iv[1:]:
        if st <= cur_e:
            if en > cur_e:
                cur_e = en
        else:
            total += cur_e - cur_s
            cur_s, cur_e = st, en
    total += cur_e - cur_s
    return total


def find_lm_head_anchor(rows_a0):
    """Find largest-grid dense mat-vec (by=1, bz=1) occurring >=4 times on primary GPU."""
    counts = Counter()
    for r in rows_a0:
        name = r["name"]
        if "mul_mat_vec" not in name and "mul_mat_q" not in name:
            continue
        if r["by"] == 1 and r["bz"] == 1:
            counts[(name, r["bx"], r["threads"])] += 1
    cands = [(bx, cnt, key) for key, cnt in counts.items()
             for (name, bx, thr) in (key,) if cnt >= 4]
    if not cands:
        # Fallback: allow any grid shape if count >= 4
        counts_any = Counter()
        for r in rows_a0:
            name = r["name"]
            if "mul_mat_vec" in name or "mul_mat_q" in name:
                counts_any[(name, r["blocks"], r["threads"])] += 1
        cands_any = [(blk, cnt, key) for key, cnt in counts_any.items()
                     for (name, blk, thr) in (key,) if cnt >= 4]
        if not cands_any:
            return None
        cands_any.sort(reverse=True)
        name, blk, thr = cands_any[0][2]
        return dict(name=name, bx=None, blocks=blk, threads=thr, count=cands_any[0][1])
    cands.sort(reverse=True)
    name, bx, thr = cands[0][2]
    return dict(name=name, bx=bx, blocks=bx, threads=thr, count=cands[0][1])


def is_anchor(r, anc):
    if r["name"] != anc["name"] or r["threads"] != anc["threads"]:
        return False
    if anc["bx"] is not None:
        return r["bx"] == anc["bx"] and r["by"] == 1 and r["bz"] == 1
    return r["blocks"] == anc["blocks"]


def segment_tg(by_agent, active_agents, anc):
    """Segment decode trace into steady-state tokens using LM-head anchor on each GPU."""
    a0 = active_agents[0]
    rows0 = by_agent[a0]
    idx0 = [i for i, r in enumerate(rows0) if is_anchor(r, anc)]
    if len(idx0) < 3:
        return None

    # Build raw intervals k = 1..len(idx0)-1 on primary GPU
    raw = []
    for k in range(1, len(idx0)):
        i_prev, i_cur = idx0[k - 1], idx0[k]
        wall_ms = (rows0[i_cur]["end"] - rows0[i_prev]["end"]) / 1e6
        cnt = i_cur - i_prev
        raw.append((k, i_prev, i_cur, wall_ms, cnt))

    med_wall = statistics.median(x[3] for x in raw)
    near_med = [x for x in raw if x[3] <= 1.25 * med_wall]
    modal_cnt = Counter(x[4] for x in (near_med or raw)).most_common(1)[0][0]

    steady_ks = [x[0] for x in raw if x[3] <= 1.15 * med_wall and x[4] == modal_cnt]
    # If dynamic routing causes slight per-token count jitter, relax to +/-2% count around modal
    if len(steady_ks) < max(3, len(raw) // 4):
        tol = max(2, int(round(modal_cnt * 0.02)))
        steady_ks = [x[0] for x in raw if x[3] <= 1.15 * med_wall and abs(x[4] - modal_cnt) <= tol]

    if not steady_ks:
        return None

    # Verify per-agent anchor alignment; map each steady token k to its slice per agent
    per_agent_anchors = {}
    same_anchor_count = True
    for a in active_agents:
        idxs = [i for i, r in enumerate(by_agent[a]) if is_anchor(r, anc)]
        per_agent_anchors[a] = idxs
        if len(idxs) != len(idx0):
            same_anchor_count = False

    selected_rows = []
    per_agent_token_stats = defaultdict(list)  # agent -> list of dicts per steady token
    ar_wait_ms = defaultdict(list)

    for k in steady_ks:
        t_start_ref = rows0[idx0[k - 1]]["end"]
        t_end_ref = rows0[idx0[k]]["end"]
        tok_ar = {}
        for a in active_agents:
            if same_anchor_count:
                ia_prev = per_agent_anchors[a][k - 1]
                ia_cur = per_agent_anchors[a][k]
                tok_rows = by_agent[a][ia_prev + 1 : ia_cur + 1]
                prev_end = by_agent[a][ia_prev]["end"]
                cur_end = by_agent[a][ia_cur]["end"]
            else:
                tok_rows = [r for r in by_agent[a] if r["end"] > t_start_ref and r["end"] <= t_end_ref]
                prev_end, cur_end = t_start_ref, t_end_ref

            selected_rows.extend(tok_rows)
            wall_ns = cur_end - prev_end
            busy_ns = union_duration_ns([(r["start"], r["end"]) for r in tok_rows])
            start_gap_ns = max(0, tok_rows[0]["start"] - prev_end) if tok_rows else 0
            max_gap_ns = start_gap_ns
            max_end = tok_rows[0]["end"] if tok_rows else prev_end
            for r in tok_rows[1:]:
                g = r["start"] - max_end
                if g > max_gap_ns:
                    max_gap_ns = g
                if r["end"] > max_end:
                    max_end = r["end"]

            ar_rows = [r for r in tok_rows if "ar_kernel" in r["name"] or "allreduce" in r["name"].lower()]
            tok_ar[a] = ar_rows
            small_ns = sum(r["dur"] for r in tok_rows if r["dur"] < 15000)
            per_agent_token_stats[a].append(dict(
                n_kernels=len(tok_rows),
                wall_ms=wall_ns / 1e6,
                busy_ms=busy_ns / 1e6,
                idle_ms=max(0, wall_ns - busy_ns) / 1e6,
                start_gap_ms=start_gap_ns / 1e6,
                max_gap_ms=max_gap_ns / 1e6,
                ar_count=len(ar_rows),
                ar_ms=sum(r["dur"] for r in ar_rows) / 1e6,
                small_ms=small_ns / 1e6,
            ))

        # Dual-GPU AllReduce spin-wait attribution: early GPU spins waiting for late GPU
        if len(active_agents) == 2:
            a0_, a1_ = active_agents[0], active_agents[1]
            if len(tok_ar[a0_]) == len(tok_ar[a1_]) and len(tok_ar[a0_]) > 0:
                w0 = w1 = 0
                for r0, r1 in zip(tok_ar[a0_], tok_ar[a1_]):
                    if r0["start"] < r1["start"]:
                        w0 += min(r0["dur"], r1["start"] - r0["start"])
                    elif r1["start"] < r0["start"]:
                        w1 += min(r1["dur"], r0["start"] - r1["start"])
                ar_wait_ms[a0_].append(w0 / 1e6)
                ar_wait_ms[a1_].append(w1 / 1e6)

    return dict(
        mode="tg",
        n_units=len(steady_ks),
        total_intervals=len(raw),
        med_wall_ms=med_wall,
        modal_cnt=modal_cnt,
        selected_rows=selected_rows,
        per_agent_stats=per_agent_token_stats,
        ar_wait_ms=ar_wait_ms,
    )


def segment_pp(by_agent, active_agents, n_units_arg):
    """Prefill / non-anchor trace segmentation: drop warmup window before last major gap if present."""
    all_rows = sorted((r for a in active_agents for r in by_agent[a]), key=lambda r: r["start"])
    if not all_rows:
        sys.exit("error: no kernel rows in trace")

    # Find largest idle gap > 20 ms that splits warmup from measured run(s)
    # First check if an LM-head-like final matvec marks warmup vs measured PP
    a0 = active_agents[0]
    rows0 = by_agent[a0]
    max_end = rows0[0]["end"]
    big_gaps = []
    for i in range(1, len(rows0)):
        gap_ms = (rows0[i]["start"] - max_end) / 1e6
        if gap_ms > 20.0:
            big_gaps.append((gap_ms, i, rows0[i]["start"]))
        if rows0[i]["end"] > max_end:
            max_end = rows0[i]["end"]

    cut_start = rows0[0]["start"]
    window_note = "whole trace (no >20ms warmup gap found)"
    if big_gaps:
        # Use the last gap that leaves at least 20% of kernels after it
        valid = [g for g in big_gaps if g[1] < int(len(rows0) * 0.85)]
        if valid:
            # Prefer the gap closest to the last measured prompt (last valid gap)
            gap_ms, idx, ts = valid[-1]
            cut_start = ts
            window_note = f"post-warmup window after {gap_ms:.1f} ms gap (dropped first {idx}/{len(rows0)} kernels on primary GPU)"

    selected = [r for r in all_rows if r["start"] >= cut_start]
    per_agent_stats = defaultdict(list)
    for a in active_agents:
        arows = [r for r in by_agent[a] if r["start"] >= cut_start]
        if not arows:
            continue
        wall_ns = max(r["end"] for r in arows) - min(r["start"] for r in arows)
        busy_ns = union_duration_ns([(r["start"], r["end"]) for r in arows])
        ar_rows = [r for r in arows if "ar_kernel" in r["name"] or "allreduce" in r["name"].lower()]
        per_agent_stats[a].append(dict(
            n_kernels=len(arows),
            wall_ms=wall_ns / 1e6,
            busy_ms=busy_ns / 1e6,
            idle_ms=max(0, wall_ns - busy_ns) / 1e6,
            start_gap_ms=0.0,
            max_gap_ms=0.0,
            ar_count=len(ar_rows),
            ar_ms=sum(r["dur"] for r in ar_rows) / 1e6,
            small_ms=sum(r["dur"] for r in arows if r["dur"] < 15000) / 1e6,
        ))

    n_units = float(n_units_arg) if n_units_arg else 1.0
    return dict(
        mode="pp",
        n_units=n_units,
        window_note=window_note,
        selected_rows=selected,
        per_agent_stats=per_agent_stats,
        ar_wait_ms={},
    )


def main():
    ap = argparse.ArgumentParser(description="Anchor-segmented rocprofv3 kernel budget & BA-4 auditor")
    ap.add_argument("trace_path", help="rocprofv3 trace directory or .db file")
    ap.add_argument("n_tokens", nargs="?", type=float, default=None,
                    help="Optional token/run count (auto-detected in tg mode; divisor in pp mode, default 1)")
    ap.add_argument("--mode", choices=("auto", "tg", "pp"), default="auto",
                    help="Segmentation mode (default: auto)")
    ap.add_argument("--top", type=int, default=14, help="Rows per section (default: 14)")
    ap.add_argument("--per-gpu", action="store_true",
                    help="Show separate agent rows in main tables instead of summarizing per GPU")
    args = ap.parse_args()

    db_path = find_db(args.trace_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cols = {r[1] for r in conn.execute('PRAGMA table_info("kernels")')}
    dev_col = "agent_abs_index" if "agent_abs_index" in cols else "nid"

    has_vgpr = "vgpr_count" in cols
    has_lds = "lds_size" in cols
    has_slds = "static_lds_size" in cols
    has_scr = "scratch_size" in cols
    has_sscr = "static_scratch_size" in cols

    gpu_info = load_gpu_agents(conn)

    sel_cols = [
        "start", "end", "duration", dev_col, "name",
        "grid_x", "grid_y", "grid_z", "workgroup_x", "workgroup_y", "workgroup_z",
        "vgpr_count" if has_vgpr else "0",
        "lds_size" if has_lds else "0",
        "static_lds_size" if has_slds else "0",
        "scratch_size" if has_scr else "0",
        "static_scratch_size" if has_sscr else "0",
    ]
    q = f"SELECT {', '.join(sel_cols)} FROM kernels ORDER BY start"

    by_agent = defaultdict(list)
    for row in conn.execute(q):
        (st, en, dur, ag, name, gx, gy, gz, wx, wy, wz,
         vgpr, lds, slds, scr, sscr) = row
        wx_s = max(1, wx or 1)
        wy_s = max(1, wy or 1)
        wz_s = max(1, wz or 1)
        bx = max(1, (gx or wx_s) // wx_s)
        by = max(1, (gy or wy_s) // wy_s)
        bz = max(1, (gz or wz_s) // wz_s)
        blocks = bx * by * bz
        threads = wx_s * wy_s * wz_s
        eff_lds = max(lds or 0, slds or 0)
        eff_scr = max(scr or 0, sscr or 0)
        by_agent[ag].append(dict(
            start=st, end=en, dur=dur, agent=ag, name=name,
            bx=bx, by=by, bz=bz, blocks=blocks, threads=threads,
            vgpr=int(vgpr or 0), lds=int(eff_lds), scratch=int(eff_scr),
        ))

    active_agents = sorted(by_agent.keys())
    if not active_agents:
        sys.exit(f"error: no kernels in {db_path}")

    for idx, ag in enumerate(active_agents):
        if ag not in gpu_info:
            cu = DEFAULT_CU_BY_IDX[idx] if idx < len(DEFAULT_CU_BY_IDX) else 64
            gpu_info[ag] = dict(abs_index=ag, type_index=idx, label=f"ROCm{idx}", cu=cu, arch="gfx906")

    # Segment trace
    seg = None
    anc = None
    if args.mode in ("auto", "tg"):
        anc = find_lm_head_anchor(by_agent[active_agents[0]])
        if anc is not None:
            seg = segment_tg(by_agent, active_agents, anc)
        elif args.mode == "tg":
            sys.exit("error: --mode tg requested but no LM-head anchor (>=4 calls) found")
    if seg is None:
        seg = segment_pp(by_agent, active_agents, args.n_tokens)

    mode = seg["mode"]
    n_units = seg["n_units"]
    n_dev = len(active_agents)
    unit_lbl = "tok" if mode == "tg" else ("unit" if args.n_tokens else "window")

    # Header summary
    print(f"DB: {db_path}")
    if mode == "tg":
        print(f"Mode: steady-state decode (tg) | anchor: {short_name(anc['name'], 48)} "
              f"[blocks={anc['blocks']}, thr={anc['threads']}]")
        print(f"Steady tokens: {n_units}/{seg['total_intervals']} intervals "
              f"(median wall {seg['med_wall_ms']:.2f} ms, modal {seg['modal_cnt']} kernels/tok/GPU; "
              f"filtered wall <= 1.15x median)")
    else:
        print(f"Mode: prefill/window (pp) | {seg['window_note']} | divisor={n_units:g}")

    # Per-GPU timing & MI60 vs MI50 asymmetry summary
    print("\nPer-GPU summary (mean per " + unit_lbl + ", traced):")
    print(f"  {'GPU':14s} {'CUs':>3} {'kernels':>8} {'wall_ms':>8} {'busy_ms':>8} {'idle_ms':>8} "
          f"{'busy%':>6} {'start_gap':>9} {'AR_ms':>7} {'AR_wait':>8} {'small_ms':>8}")
    gpu_means = {}
    for ag in active_agents:
        st_list = seg["per_agent_stats"][ag]
        if not st_list:
            continue
        info = gpu_info[ag]
        div = 1.0 if mode == "tg" else n_units
        m_k = statistics.mean(x["n_kernels"] for x in st_list) / div
        m_wall = statistics.mean(x["wall_ms"] for x in st_list) / div
        m_busy = statistics.mean(x["busy_ms"] for x in st_list) / div
        m_idle = statistics.mean(x["idle_ms"] for x in st_list) / div
        m_gap = statistics.mean(x["start_gap_ms"] for x in st_list) / div
        m_ar = statistics.mean(x["ar_ms"] for x in st_list) / div
        m_small = statistics.mean(x["small_ms"] for x in st_list) / div
        waits = seg["ar_wait_ms"].get(ag, [])
        m_wait = statistics.mean(waits) if waits else 0.0
        busy_pct = (100.0 * m_busy / m_wall) if m_wall > 0 else 0.0
        gpu_means[ag] = dict(
            label=info["label"], cu=info["cu"], k=m_k, wall=m_wall, busy=m_busy,
            idle=m_idle, gap=m_gap, ar=m_ar, wait=m_wait, small=m_small,
        )
        print(f"  {info['label']+' ('+info['arch']+')':14s} {info['cu']:3d} {m_k:8.1f} {m_wall:8.2f} "
              f"{m_busy:8.2f} {m_idle:8.2f} {busy_pct:5.1f}% {m_gap:8.3f}ms {m_ar:7.2f} {m_wait:7.2f} {m_small:8.2f}")

    if len(active_agents) == 2 and all(a in gpu_means for a in active_agents):
        g0, g1 = gpu_means[active_agents[0]], gpu_means[active_agents[1]]
        net0 = g0["busy"] - g0["wait"]
        net1 = g1["busy"] - g1["wait"]
        print(f"  Asymmetry ({g1['label']} - {g0['label']}): "
              f"net busy {net1 - net0:+.2f} ms, small-kernel (<15us) {g1['small'] - g0['small']:+.2f} ms, "
              f"token-start gap {g1['gap'] - g0['gap']:+.3f} ms, "
              f"AR spin-wait ({g0['label']}/{g1['label']}) {g0['wait']:.2f}/{g1['wait']:.2f} ms")

    # Group strictly by (name, blocks, threads, vgpr, lds, scratch, agent)
    grp = defaultdict(lambda: [0, 0])  # key -> [calls, dur_ns]
    for r in seg["selected_rows"]:
        key = (r["name"], r["blocks"], r["threads"], r["vgpr"], r["lds"], r["scratch"], r["agent"])
        grp[key][0] += 1
        grp[key][1] += r["dur"]

    # Build per-shape records
    records = []
    for (name, blocks, threads, vgpr, lds, scratch, ag), (calls, dur_ns) in grp.items():
        info = gpu_info[ag]
        occ = gfx906_occupancy(threads, vgpr, lds, info["cu"], blocks)
        us = dur_ns / calls / 1e3
        ms_per_unit = dur_ns / 1e6 / n_units
        calls_per_unit = calls / n_units
        cls = "floor" if us <= TRACE_FLOOR_US + 0.6 else ("one-block" if blocks <= 4 else "work")
        flags = []
        if scratch > 0:
            flags.append(f"SPILL:{scratch}B")
        if occ["b_cu"] <= 2 or occ["w_cu"] <= 8:
            flags.append(f"LOW_OCC:{occ['b_cu']}b/{occ['w_cu']}w")
        if blocks < info["cu"] and us > 8.0:
            flags.append(f"UNDERFILL:{blocks}<{info['cu']}CU")
        records.append(dict(
            name=short_name(name, 54),
            full_name=name,
            agent=ag,
            gpu=info["label"],
            cu=info["cu"],
            blocks=blocks,
            threads=threads,
            vgpr=vgpr,
            vgpr_alloc=occ["vgpr_alloc"],
            lds=lds,
            scratch=scratch,
            b_cu=occ["b_cu"],
            w_cu=occ["w_cu"],
            res_gpu=occ["res_gpu"],
            rounds=occ["rounds"],
            calls=calls_per_unit,
            us=us,
            ms=ms_per_unit,
            excess=max(0.0, us - TRACE_FLOOR_US) * calls_per_unit / 1e3,
            cls=cls,
            flags=flags,
        ))

    # For multi-GPU summary tables unless --per-gpu, merge identical (name, blocks, thr, vgpr, lds, scratch)
    # by averaging per-GPU calls/ms while tracking max skew between GPUs
    if not args.per_gpu and n_dev > 1:
        by_shape = defaultdict(list)
        for rec in records:
            skey = (rec["full_name"], rec["blocks"], rec["threads"], rec["vgpr"], rec["lds"], rec["scratch"])
            by_shape[skey].append(rec)
        view_rows = []
        for skey, recs in by_shape.items():
            # Use primary GPU (lowest agent index) for CU/occupancy reference, average across active GPUs
            recs.sort(key=lambda x: x["agent"])
            r0 = recs[0]
            tot_calls = sum(x["calls"] for x in recs) / n_dev
            tot_ms = sum(x["ms"] for x in recs) / n_dev
            avg_us = (tot_ms * 1e3 / tot_calls) if tot_calls > 0 else 0.0
            flags = list(r0["flags"])
            if len(recs) == 2:
                d_us = recs[1]["us"] - recs[0]["us"]
                pct = (100.0 * d_us / recs[0]["us"]) if recs[0]["us"] > 0 else 0.0
                if abs(pct) >= 15.0 and max(recs[0]["ms"], recs[1]["ms"]) >= 0.05:
                    flags.append(f"SKEW:{pct:+.0f}%")
            cls = "floor" if avg_us <= TRACE_FLOOR_US + 0.6 else ("one-block" if r0["blocks"] <= 4 else "work")
            view_rows.append(dict(
                r0,
                gpu=f"avg/{n_dev}",
                calls=tot_calls,
                ms=tot_ms,
                us=avg_us,
                excess=max(0.0, avg_us - TRACE_FLOOR_US) * tot_calls / 1e3,
                cls=cls,
                flags=flags,
            ))
    else:
        view_rows = records

    # Launch floor pricing summary (per GPU)
    tot_launches = sum(r["calls"] for r in records) / n_dev
    tot_k_ms = sum(r["ms"] for r in records) / n_dev
    print(f"\nKernel totals per {unit_lbl} per GPU: {tot_launches:.1f} launches, {tot_k_ms:.2f} ms traced kernel sum")
    print(f"Launch floor: traced {tot_launches * TRACE_FLOOR_US / 1e3:.2f} ms ({TRACE_FLOOR_US:.1f} us/k) | "
          f"HIP graph {tot_launches * GRAPH_LAUNCH_LO_US / 1e3:.2f}-{tot_launches * GRAPH_LAUNCH_HI_US / 1e3:.2f} ms "
          f"({GRAPH_LAUNCH_LO_US:.2f}-{GRAPH_LAUNCH_HI_US:.2f} us/launch)")
    for cls in ("floor", "one-block", "work"):
        sel = [r for r in records if r["cls"] == cls]
        c_calls = sum(r["calls"] for r in sel) / n_dev
        c_ms = sum(r["ms"] for r in sel) / n_dev
        c_ex = sum(r["excess"] for r in sel) / n_dev
        print(f"  {cls:9s}: {c_calls:7.1f} launches, {c_ms:7.2f} ms traced (raw excess over 4.2us floor: {c_ex:6.2f} ms)")

    hdr = (f"  {'GPU':6s} {'ms':>7} {'calls':>7} {'us':>7} {'blk':>6} {'thr':>4} "
           f"{'vgpr':>4} {'ldsK':>5} {'scrB':>5} {'b/CU':>4} {'w/CU':>4} {'rnd':>4}  kernel  [flags]")

    def show_table(title, items, limit):
        print(f"\n{title}:")
        if not items:
            print("  (none)")
            return
        print(hdr)
        for r in items[:limit]:
            fl = ("  [" + ", ".join(r["flags"]) + "]") if r["flags"] else ""
            print(f"  {r['gpu']:6s} {r['ms']:7.3f} {r['calls']:7.1f} {r['us']:7.1f} {r['blocks']:6d} "
                  f"{r['threads']:4d} {r['vgpr']:4d} {r['lds']/1024.0:5.1f} {r['scratch']:5d} "
                  f"{r['b_cu']:4d} {r['w_cu']:4d} {r['rounds']:4d}  {r['name']}{fl}")

    # BA-4 Audit Highlights first (spills, under-filled grids, low occupancy)
    spills = sorted((r for r in view_rows if r["scratch"] > 0), key=lambda r: -r["ms"])
    show_table("BA-4 FLAG: Register spills (scratch > 0)", spills, args.top)

    low_occ = sorted(
        (r for r in view_rows if (r["b_cu"] <= 2 or r["w_cu"] <= 8) and r["ms"] >= 0.05),
        key=lambda r: -r["ms"],
    )
    show_table("BA-4 FLAG: Low occupancy (b/CU <= 2 or w/CU <= 8, >= 0.05 ms)", low_occ, args.top)

    underfill = sorted(
        (r for r in view_rows if r["blocks"] < r["cu"] and r["us"] > 8.0),
        key=lambda r: -r["ms"],
    )
    show_table("BA-4 FLAG: Under-filled grids (blocks < CU_count and us > 8.0)", underfill, args.top)

    # Standard three classes (grouped by full geometry)
    show_table(
        "Top work kernels (blocks > 4, above floor; check bytes vs 800-891 GB/s)",
        sorted((r for r in view_rows if r["cls"] == "work"), key=lambda r: -r["ms"]),
        args.top,
    )
    show_table(
        "Top one-block kernels above floor (blocks <= 4; check 1-block rule, do NOT price against empty 4.2us floor)",
        sorted((r for r in view_rows if r["cls"] == "one-block"), key=lambda r: -r["ms"]),
        args.top,
    )
    show_table(
        f"Top floor kernels by launch count (fusion value: {GRAPH_LAUNCH_LO_US:.2f}-{GRAPH_LAUNCH_HI_US:.2f} us/launch in HIP graph)",
        sorted((r for r in view_rows if r["cls"] == "floor"), key=lambda r: -r["calls"]),
        args.top,
    )

    # Per-GPU kernel skew table when 2 GPUs are active
    if n_dev == 2:
        by_shape_2 = defaultdict(dict)
        for rec in records:
            skey = (rec["full_name"], rec["blocks"], rec["threads"], rec["vgpr"], rec["lds"], rec["scratch"])
            by_shape_2[skey][rec["agent"]] = rec
        a0, a1 = active_agents[0], active_agents[1]
        skews = []
        for skey, d in by_shape_2.items():
            if a0 in d and a1 in d:
                r0, r1 = d[a0], d[a1]
                dms = r1["ms"] - r0["ms"]
                dus = r1["us"] - r0["us"]
                if abs(dms) >= 0.03:
                    skews.append((abs(dms), dms, dus, r0, r1))
        skews.sort(reverse=True)
        print(f"\nPer-GPU skew ({gpu_info[a1]['label']} minus {gpu_info[a0]['label']}, |delta_ms| >= 0.03):")
        if not skews:
            print("  (none)")
        else:
            print(f"  {'d_ms':>7} {'ms0':>7} {'ms1':>7} {'us0':>6} {'us1':>6} {'d_us':>6} {'calls':>6} {'blk':>6}  kernel")
            for _, dms, dus, r0, r1 in skews[:args.top]:
                print(f"  {dms:+7.3f} {r0['ms']:7.3f} {r1['ms']:7.3f} {r0['us']:6.1f} {r1['us']:6.1f} "
                      f"{dus:+6.1f} {r0['calls']:6.1f} {r0['blocks']:6d}  {r0['name']}")


if __name__ == "__main__":
    main()
