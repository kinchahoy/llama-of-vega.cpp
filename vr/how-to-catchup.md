# Catch up with official llama.cpp

Keep upstream interfaces and the smallest local delta that preserves the gfx906 improvements. Read [the current setup](README.md), the [feature index](IMPROVEMENTS.md#merge-maintenance-index), `AGENTS.md`, and `skills/code-review/SKILL.md` first. Read detailed evidence only for affected features; do not load archived investigations on every merge. Historical workflows under [old-ignore](old-ignore/README.md) are reference material, not instructions.

Keep the separate `separate-experiment` output-style project out of this performance patch chain. The accepted grouped-MoE subset from `flashnext-fast` belongs in this patch chain, including its tests and dispatch/fallback guards. Its separate GPU-feedback experiment does not. Model priorities are listed in [the current guide](README.md#priorities-and-separate-work).

Current overlay base: official `e6d624b9a4b95c6b52f5ca384d70b75581f9821a` (the patch kit below is still the 46eb1742d one; regenerate it before the next catch-up). The active [patch kit](patches/llama-head-46eb1742d/README.md) is a 12-group per-feature stack; refresh it with its `regenerate.py --target <commit>` and `verify.py`. The previous kit ([`36a0e2ffc`](old-ignore/2026-11-21/patches/llama-head-36a0e2ffc/README.md), stage D of the [2026-10-29 rebuild](old-ignore/2026-10-29/rebuild/README.md)) and the older [`418f905a4`](old-ignore/2026-10-29/patches/llama-head-418f905a4/README.md) kit are historical; do not reuse their intermediate trees for a new upstream base. Use the reviewed current branch as the source of truth.

The rebuild removed multi-stage TP (`-tps`) and the fork-only DeepSeek V4/V4.1, MiniMax-M2 and MiMo-2 code. Do not restore them during a merge; take upstream for those models. See [the removed and changed list](IMPROVEMENTS.md#2026-10-29-rebuild-removed-and-changed).

## Prepare and merge

Use the normal checkout. The human first commits and pushes the reviewed current state to the personal fork, including any local work that must be kept. Confirm the push succeeded and start with a clean tracked working tree. Agents do not commit or push. Keep unrelated untracked files untouched.

Keep the previous successful build in its original directory with all its shared libraries. Do not rebuild it, copy only its executable, or rename it: build paths can be embedded in library lookup. Set `BASE` to that build. Stable names: `build-current` and `build-previous` are symlinks in the repo root (2026-11-21: `build-final-20261122` from ae32e5e58 (upstream e6d624b9a merged) and `build-u1sort-20261121`); after accepting a candidate, point `build-previous` at the old current and `build-current` at the candidate with `ln -sfn`. Never rename a build directory itself. Build each candidate in a new directory and keep the last two successful builds for quick comparisons. After acceptance, that candidate becomes the next merge's baseline. Record its path and source identity with the results.

Run from the repository root in Bash, keeping these variables for the later steps:

```bash
# One-time repository setup. Git invokes rerere automatically during merges.
git config rerere.enabled true
git config rerere.autoupdate false

OLD_HEAD=$(git rev-parse HEAD)
BASE="$(readlink -f "$PWD/build-current")" # resolve the link so the record names the real build
CAND="$PWD/build-catchup-$(date +%Y%m%d-%H%M%S)"
OUT="$CAND/validation"
mkdir -p "$OUT"
printf '%s\n' "$OLD_HEAD" > "$OUT/baseline-source.txt"
printf '%s\n' "$BASE" > "$OUT/baseline-build.txt"

git fetch upstream
UPSTREAM_HEAD=$(git rev-parse upstream/master)
printf '%s\n' "$UPSTREAM_HEAD" > "$OUT/upstream-source.txt"
git merge --no-commit --no-ff "$UPSTREAM_HEAD"
```

If the merge stops, review reused resolutions and resolve the remaining conflicts. Run `git rerere` to record the resolutions, then `git add` only the resolved paths and required new files. Review automatically merged callers too; rerere cannot detect semantic errors. Prefer upstream equivalents over obsolete local code and take new model support from upstream. Do not choose whole files blindly.

The worktree-based `catchup-script.sh` remains an optional alternative; it is not required for this routine.

A background agent session cannot write to the normal checkout. Create a worktree from the reviewed HEAD under `.claude/worktrees/catchup-<upstream-short>` on a branch of the same name, and run the merge there. Keep `BASE` and `CAND` as absolute paths under the main repository root, so the builds survive when the worktree is removed. The human commits the merge from that branch.

Haiku-class agents are fine for triage and additive hunks. MMQ launch, recurrent rollback and DFlash/MTP state need a stronger model.

Preserve these connected paths:

- Repack upload/layout, dispatch, fused consumers, gfx906 MMQ/MMVQ tuning, existing supported quants, and default-on grouped MoE weight reuse for narrow IQ2_S/IQ4_NL batches.
- Preserve gfx906 row-first MMQ traversal when reconciling upstream launch-axis changes. The first 418f port changed this traversal and showed a large PP loss; the accepted tree restores the gfx906 ordering. The active narrow IQ2_S single-token expert path is limited to one output column and even rows; odd rows retain the prior fallback.
- Upstream HIP host-staged AllReduce plus the local graph-replay, duplex/slab overlap and scheduler integration: HIP must enter the first `#if !defined(GGML_USE_MUSA)` path in `ggml/src/ggml-cuda/allreduce.cu` with no environment switch. The opt-in peer-write AllReduce is archived ([patch](old-ignore/2026-11-21/peer-ar/README.md)); do not expect `tp-allreduce.cu` after a merge.
- Parallel lanes, shared-expert reduction, token graphs, and scheduler/readback dependencies; avoid new per-ubatch host synchronization.
- Split KV, recurrent rollback, MTP planning, and DFlash per-sequence state. Rounded reservations can need 2049 tap rows for ubatch 2048. Chunked GDN must write all requested snapshots with the correct output stride.
- Existing host PLE placement, wide-row GPU argsort, and graph update safety where still needed.

Reconcile loader/converter/tensor identifiers and graph/memory callers together. DFlash vision remains unvalidated. Ask before a large redesign or unclear tradeoff. Keep RCCL off for the current CPU-lane/motherboard-lane topology; revisit after the hardware changes. The unreachable legacy HIP overlap-only branch has been removed from `allreduce.cu`; do not restore it.

## One final validation pass

The after-merge rule from the [process](PROCESS.md#after-an-upstream-merge): static diff first, then the one end-to-end comparison that AGENTS.md requires, then only the checks of features that the diff affects. Finish the merge and fixes before building. Repeat only affected checks if something fails or later code changes affect the result. Detailed evidence is read only for affected features ([feature index](IMPROVEMENTS.md#merge-maintenance-index)).

### 1. Static diff, no GPU

- Check staged and unstaged source diffs with `git diff --check` and `git diff --cached --check`, excluding generated patches and archives. Require no entries from `git ls-files -u` and inspect runtime source for conflict markers.
- Diff the graph builders and the kernel geometry between the pre-merge commit and the merged tree: changed or new ops, launch bounds, block and grid sizes, shared memory, op fusion conditions. Predict from the [hardware card](HARDWARE-CARD.md) whether any kernel cost changes. List the features that the diff touches; only those get checks below.
- Confirm HIP still compiles the default host-staged AllReduce path without an environment switch: `ggml/src/ggml-cuda/allreduce.cu` must compile the host-staged path for HIP, and its pipeline must not need `GGML_CUDA_TP_OVERLAP=1` for the normal configuration (AGENTS.md). Excluding HIP from it cost 21% PP and 38% TG on 2026-09-29.

### 2. Build once

Configure and build once in a new directory with the existing comparison flags. Leave `BASE` untouched.

```bash
source $THEROCK_ENV
cmake -S . -B "$CAND" \
  -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=OFF \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON \
  -DLLAMA_OPENSSL=ON -DLLAMA_BUILD_VR_EXPERIMENTS=ON \
  -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DHIP_COMPILER=clang -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" \
  -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument" \
  > "$OUT/configure.log" 2>&1
# Continue only if configuration succeeds.
cmake --build "$CAND" --target llama-cli llama-completion llama-server llama-bench llama-batched-bench vr-test-gfx906-tp -j 8 \
  > "$OUT/build.log" 2>&1
```

The binaries find their libraries through a RUNPATH set to the build directory at build time. `build-ilp` was built as `$HOME/infer/mx-llama.cpp/build-ilp`, and that directory no longer exists. Run its binaries with `LD_LIBRARY_PATH="$BASE/bin"`; this loads the libraries stored with that build. Do not move or rename later builds, so they do not need this.

Continue only after a successful build. Record the resolved source and build settings:

```bash
git diff --binary HEAD > "$OUT/candidate-changes.patch"
cp "$CAND/CMakeCache.txt" "$OUT/CMakeCache.txt"
sha256sum "$BASE/bin/llama-bench" "$CAND/bin/llama-bench" > "$OUT/binaries.sha256"
ldd "$BASE/bin/llama-bench" > "$OUT/baseline-libraries.txt"
ldd "$CAND/bin/llama-bench" > "$OUT/candidate-libraries.txt"
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP
```

### 3. The one end-to-end comparison

Rebuild and compare PP and TG with the pre-merge commit using identical `llama-bench` settings. It needs explicit user OK for the exact commands and combined time limit; no standing budget has been agreed. Do not download models. Both builds must resolve `-hf` to the same cached model files; keep the cache unchanged between runs. `--offline` prevents downloads. Use the same environment, no competing GPU workload, and start below 60 C.

```bash
for label in baseline candidate; do
  if [ "$label" = baseline ]; then BUILD="$BASE"; else BUILD="$CAND"; fi
  timeout --kill-after=5s 110s "$BUILD/bin/llama-bench" \
    -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
    -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on \
    -b 2048 -ub 2048 -t 20 -p 2048 -n 128 -d 0 -r 1 -o json \
    > "$OUT/$label.json" 2> "$OUT/$label.log" || {
      BENCH_STATUS=$?
      printf '%s\n' "$BENCH_STATUS" > "$OUT/$label.exit"
      exit "$BENCH_STATUS"
    }
  printf '0\n' > "$OUT/$label.exit"
done
```

Each run is capped at 110 seconds plus 5 seconds to terminate. The benchmarks use warmup and one repetition. Require successful exits and complete JSON for both builds. One pair is a quick regression screen, not repeatability evidence. Report baseline and candidate PP2048 and TG128 in one table, with tokens/second and percentage change `(candidate / baseline - 1) * 100`. Compare measured results from this session; historical numbers are context, not a matched baseline. Mark skipped checks as not run and performance as not measured if either benchmark fails. Diagnose regressions before tuning; no automatic regression threshold has been agreed.

### 4. Affected features only

Run these only when the static diff or the model predicts a change in the feature. They are not part of every merge. Each needs the user's OK for its budget.

- AllReduce or tensor-split code touched: the two TP smoke checks (about 110 seconds each, no model): `timeout --kill-after=5s 110s bash -eu -c '"$1/bin/vr-test-gfx906-tp" --n 1 --check; "$1/bin/vr-test-gfx906-tp" --n 2048 --check' bash "$CAND"`. Require exit zero and `check=pass` at both widths. It is a narrow matmul and reduction smoke test; its `time_us` is not model PP/TG.
- Numerics or sampling touched: one short `llama-completion` generation with `--temp 0 --top-k 1 -s 42 -n 32 -no-cnv --simple-io --no-display-prompt` on the same model and settings as the benchmark; inspect the saved text. (`llama-cli` is chat-only and rejects `-no-cnv`.)
- Flash attention, KV cache, memory or ubatch splitting, or server slots touched: the multi-user check below.
- Another model, speculative or MoE code touched: that model's targeted check, with its budget approved.

### Multi-user check (unified KV)

The PP/TG pair above is one sequence. The serving presets run 3-4 users in one unified KV pool, where attention over the shared pool (kv_rows, MTP rollback with several slots, KV cell placement) behaves differently. Run this check only before pushing a merge or a change that touches flash attention, the KV cache, memory/ubatch splitting or server slots. It needs `llama-batched-bench` in both builds and about 20 minutes (4 runs; each waits until both GPU junctions are below 60C, since the GPUs throttle under sustained load):

```bash
vr/scripts/bench_multiuser.sh "$OUT/multiuser" "$BASE" "$CAND"
# within 3%: repeat with REPS=2 (order A,B,B,A)
```

It runs `llama-batched-bench -kvu -npp 16384 -ntg 128 -npl 4` (4 sequences x 16K prompts in one pool, then 4-way decode, no draft) on Qwen3.8-27B Q8_0 (`-b/-ub 2048`) and Flash Next UD-IQ3_XXS (`-b 1024 -ub 512`), F16 KV, `-sm tensor`. Report PP and 4-way TG for both models next to the single-sequence table. To isolate kv_rows in one build, pass the same build twice with `ENV_B=LLAMA_KV_ROWS_DISABLE=1`.

Baselines, 2026-11-15, commit 7ddd49693, 2x gfx906, REPS=2 (A = kv_rows on, the default; B = `LLAMA_KV_ROWS_DISABLE=1`):

| Model | kv_rows | PP t/s (4x16K) | 4-way TG t/s |
| --- | --- | ---: | ---: |
| Qwen3.8-27B Q8_0 | on | 577.8 / 577.5 | 80.86 / 80.93 |
| Qwen3.8-27B Q8_0 | off | 462.3 / 462.2 | 81.13 / 81.04 |
| Flash Next UD-IQ3_XXS | on | 470.5 / 473.1 | 84.17 / 83.90 |
| Flash Next UD-IQ3_XXS | off | 472.2 / 474.2 | 84.24 / 84.11 |

A drop in 27B PP toward the "off" row means kv_rows stopped engaging (check the server log for "Flash Attention over K/V rows not supported"). For changes to speculative rollback, also run the 4-slot MTP continuation check in [the kv_rows record](investigation-notes/2026-11-15-kvrows-gfx906/README.md) (`raw/v2/ab3.sh`, `srvtest3.py`): it must finish with no "refusing rollback" line.

## Optional patch kit export

Git merges are the primary update mechanism. Skip this section during routine weekly merges. Export a kit when a portable snapshot or release checkpoint is needed; export and replay are not required before accepting a validated merge. A kit describes its recorded base and target, and can lag behind the branch. Record the new upstream base in the merge handoff even when no kit is exported.

Keep one current kit under `vr/patches/`; move superseded kits into a dated `vr/old-ignore/` campaign when the replacement is verified. Do not alter old patch payloads.

Stage the reviewed resolution, including required new files. The following exports the resolved index against the fetched official SHA. Its temporary intermediate tree leaves `src/llama-model.cpp` at upstream so patch 0005 carries that file's complete local delta. Reusing the old intermediate tree against a new base would risk restoring obsolete upstream code.

```bash
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
# UPSTREAM_HEAD is the pinned SHA from the merge step.
UPSTREAM_SHORT=$(git rev-parse --short=9 "$UPSTREAM_HEAD")
KIT="$REPO/vr/old-ignore/2026-11-21/patches/llama-head-36a0e2ffc"  # old 5-patch procedure; the current kit uses regenerate.py
cd "$REPO"
test -z "$(git ls-files -u)"
git diff --exit-code
RESOLVED_TREE=$(git write-tree)
TASK_INDEX=$(mktemp /tmp/mx-kit-index.XXXXXX)
rm "$TASK_INDEX"
GIT_INDEX_FILE="$TASK_INDEX" git read-tree "$RESOLVED_TREE"
MODEL_BLOB=$(git rev-parse "$UPSTREAM_HEAD:src/llama-model.cpp")
GIT_INDEX_FILE="$TASK_INDEX" git update-index --add --cacheinfo "100644,$MODEL_BLOB,src/llama-model.cpp"
MID_TREE=$(GIT_INDEX_FILE="$TASK_INDEX" git write-tree)
rm "$TASK_INDEX"
# Unique ref names also support a same-base refresh without overwriting history.
SNAP_SUFFIX="$UPSTREAM_HEAD-$RESOLVED_TREE"
for entry in "target:$RESOLVED_TREE" "mid:$MID_TREE"; do
    git update-ref "refs/port-snapshots/gfx906-$SNAP_SUFFIX-${entry%%:*}" "${entry#*:}" ''
done
NEXT_KIT="$REPO/vr/patches/llama-head-$UPSTREAM_SHORT"
if test -e "$NEXT_KIT"; then NEXT_KIT="${NEXT_KIT}-refresh"; fi
test ! -e "$NEXT_KIT"
mkdir -p "$NEXT_KIT"
cp "$KIT/regenerate.py" "$KIT/verify.py" "$NEXT_KIT/"
python3 - "$NEXT_KIT/regenerate.py" "$UPSTREAM_HEAD" "$MID_TREE" "$RESOLVED_TREE" <<'PY'
import re
import sys
from pathlib import Path
path = Path(sys.argv[1])
source = path.read_text()
for key, value in zip(("DEFAULT_BASE", "DEFAULT_MID", "DEFAULT_TARGET"), sys.argv[2:]):
    source, count = re.subn(rf'^{key}\s*=.*$', f'{key} = "{value}"', source, flags=re.M)
    assert count == 1, key
path.write_text(source)
PY
python3 "$NEXT_KIT/regenerate.py"
python3 "$NEXT_KIT/verify.py"
(cd "$NEXT_KIT" && sha256sum -c SHA256SUMS)
```

Review the full upstream-to-target file list against `MANIFEST.tsv` and `EXCLUDED.txt`. The current exporter excludes root `conversion/`; add required runtime/conversion paths to its grouping before regenerating. Replay checks only manifest files, so zero mismatches alone cannot establish completeness. If a group becomes empty or a source path moves, adapt the existing exporter minimally rather than adding dummy changes.

Document the new base, intermediate/target tree IDs and refs, build/check results, and limitations in its README. Update the active kit links here and in `vr/README.md`. The patches must replay without local snapshot refs; those refs are needed to retain the trees for regeneration. Offline export/replay is separate from the GPU runtime budget.

## Handoff and human push

Report the old and new upstream bases, meaningful overlay changes, candidate and baseline build paths, TP results, generation result, side-by-side PP/TG table, patch replay results if exported (otherwise say export skipped), log directory, and untested behavior. Keep this short; link detailed logs instead of pasting them. Do not claim preserved performance without measurements.

Leave agent work uncommitted. After reviewing the passing final checks, the human commits and pushes the merge to the personal fork. Record that accepted commit with the candidate's validation logs. Retain this build and the previous successful build in their original directories for quick comparisons; use this candidate as `BASE` next week. Never rebuild the retained baseline during the next merge. Agents do not commit, push, or publish.
