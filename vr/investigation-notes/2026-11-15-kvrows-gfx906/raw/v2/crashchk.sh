#!/usr/bin/env bash
hot() { rocm-smi --showtemp 2>/dev/null | awk '/junction/ {t=$NF+0; if (t>m) m=t} END {print m+0}'; }
K=/home/raistlin/infer/vrversion-llama.cpp/.claude/worktrees/agent-a4e1d9d7be78eb377/build-kvrows
for n in 1 2; do
  while [ "$(hot)" -ge 60 ]; do sleep 10; done
  setsid --wait /home/raistlin/.claude/jobs/bee85835/tmp/scripts/ab3.sh /home/raistlin/.claude/jobs/bee85835/tmp/crash$n 16384 "fix$n:$K:unified"
  grep -cE "refusing rollback|failed to remove" /home/raistlin/.claude/jobs/bee85835/tmp/crash$n/fix$n-server.log | sed "s/^/run$n rollback-refusals: /"
done
echo CRASHCHK_DONE
