#!/usr/bin/env bash
# pptest.sh [tag] -- launch a fresh pipeline-parallel test run.
# A new run directory each time, because the trainer always restarts at step 1
# (no checkpoint before step 25) while the orchestrator resumes its progress,
# and the two then deadlock waiting on each other.
set -uo pipefail
REPO=/home/mango/Misc/data-gen/glm52-decensor
TAG=${1:-$(date -u +%H%M%S)}
RUN=pp$TAG
NAME=run_$RUN

python3 - "$RUN" "$NAME" <<'PY'
import pathlib, sys
run, name = sys.argv[1], sys.argv[2]
d = pathlib.Path('/home/mango/Misc/data-gen/glm52-decensor/configs/china')
for src, out in (('trainer.pp.toml', f'trainer.{run}.toml'), ('orchestrator.pp.toml', f'orchestrator.{run}.toml')):
    s = (d/src).read_text()
    s = s.replace('runs/pptest', f'runs/{run}').replace('run_pptest', name)
    s = s.replace('/18100/pptest', f'/18100/{run}')
    (d/out).write_text(s)
print(f'configs for {run}')
PY

"$REPO/scripts/ops/cl.py" edit "configs/china/trainer.$RUN.toml" >/dev/null
scp -q -o BatchMode=yes "$REPO/configs/china/orchestrator.$RUN.toml" \
    "n82:/data/glm52-rl/deployment/configs/china/orchestrator.$RUN.toml"
export TRAINER_NODES="82 76 77 78 80 83 84"   # n81 down
"$REPO/scripts/ops/cl.py" restart "$RUN" "$NAME" ".$RUN" 2>&1 | tail -2
echo "launched $RUN"
