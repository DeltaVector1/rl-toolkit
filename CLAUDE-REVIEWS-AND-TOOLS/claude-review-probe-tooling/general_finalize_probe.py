"""Review 2, finding 3: run real general_agent setup + finalize in a Prime sandbox after the agent
user creates a virtualenv, a dangling link and a link at a protected directory.

    GA_JUDGE_KEY=dummy uv run python general_finalize_probe.py <DATA/general/train.parquet> <instance_id>

Task assets come from the pinned Hugging Face snapshot (the local reference copy has no workspace/).

setup() only checks that GA_JUDGE_KEY is set; no judge call is made because scoring is not run.
"""
import asyncio, sys
from types import SimpleNamespace

from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from verifiers.v1.utils.compile import resolve_runtime_config
from xiaomi_general import GeneralTaskset
from xiaomi_general.taskset import GeneralConfig

PARQUET, IID = sys.argv[1], sys.argv[2]
AGENT = r"""
cd /work/workspace
python3 -m venv .venv >/dev/null 2>&1; echo "venv rc=$?"
ln -s /nonexistent dangling
ln -s /work/system peek_system
ln -s /work/_verifier peek_verifier
mkdir -p notes && ln -s ../answer_draft.md notes/internal && echo draft > answer_draft.md
ls -la | sed -n 1,40p
"""

async def main():
    taskset = GeneralTaskset(GeneralConfig(dataset_path=PARQUET))
    task = next(t for t in taskset if t.data.id == IID)
    trace = SimpleNamespace(id="probe", info={}, last_reply="final answer from the model")
    config = resolve_runtime_config(PrimeConfig(cpu=2, memory=4, disk=10, labels=["xiaomi-review"]), task)
    async with provision_runtime(config, env=dict(task.runtime_env())) as rt:
        await rt.prepare_setup()
        await task.setup(trace, rt)
        agent = rt.with_user("xiaomi_agent")
        print("--- agent creates links\n", (await agent.run(["sh", "-c", AGENT], {})).stdout)
        await task.finalize(trace, rt)
        print("--- finalize:", {k: v for k, v in trace.info.items() if k.startswith("general_") or k == "verifier"})
        after = await rt.run(["sh", "-c", "cd /work/workspace && ls -la | sed -n 1,40p; echo '--- links left:'; find . -path ./.vf-venv -prune -o -type l -print | head -20; echo '--- answer.md:'; cat answer.md; echo; echo '--- venv python:'; ls -la .venv/bin 2>/dev/null | head -6"], {})
        print("--- after finalize\n", after.stdout)

asyncio.run(main())
