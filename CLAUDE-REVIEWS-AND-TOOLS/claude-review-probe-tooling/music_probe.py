"""Run upstream's scorer_precheck fixture through the port's real setup + reward path in a Prime sandbox."""
import asyncio, time
from types import SimpleNamespace
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from xiaomi_music.taskset import MusicData, MusicTask

FIXTURE = "X:1\nT:t\nM:4/4\nL:1/8\nK:C\nCDEF|GABc|c4|z4|"
CASES = {
    "upstream precheck fixture (expect ~0.2)": FIXTURE,
    "fixture in fenced block with prose": "Here is the piece:\n```abc\n" + FIXTURE + "\n```\nEnjoy!",
    "no ABC at all (expect 0)": "I cannot compose music.",
    "empty reply (expect 0)": "",
}

async def main():
    task = MusicTask(MusicData(id="probe", prompt="p", extra_info={}, image="python:3.11-slim", workdir="/tmp"))
    cfg = PrimeConfig(image="python:3.11-slim", workdir="/tmp", labels=["xiaomi-review"])
    async with provision_runtime(cfg) as rt:
        t = time.time()
        await task.setup(SimpleNamespace(info={}), rt)
        print("setup (apt-get install abcmidi) took", round(time.time() - t, 1), "s")
        print((await rt.run(["sh", "-c", "abc2midi -ver 2>&1 | head -2; dpkg -s abcmidi | grep Version"], {})).stdout.strip())
        for name, reply in CASES.items():
            trace = SimpleNamespace(info={}, last_reply=reply)
            t = time.time()
            score = await MusicTask.music_score(task, trace, rt)
            print(f"{name}: score={score} ({round(time.time() - t, 1)}s) stderr={trace.info['music']['scorer_stderr'][-120:]!r}")

asyncio.run(main())
