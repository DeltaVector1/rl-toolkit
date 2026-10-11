"""Native tasks for the released Xiaomi ABC composition dataset."""

import json
from pathlib import Path

import verifiers.v1 as vf
from xiaomi_common.data import XiaomiTasksetConfig, load_rows
from xiaomi_common.runtime import checked_run

from .scorer.pipeline import Abc2MidiMissing


class MusicData(vf.TaskData):
    extra_info: dict


class MusicTask(vf.Task[MusicData]):
    NEEDS_CONTAINER = True

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await checked_run(
            runtime,
            [
                "sh",
                "-c",
                "command -v abc2midi >/dev/null || (apt-get update -qq && apt-get install -y -qq abcmidi)",
            ],
        )

    @vf.reward
    async def music_score(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        # Score the final reply only, matching Xiaomi's solution_str contract.
        target = "/tmp/xiaomi-music-scorer"
        for source in (Path(__file__).parent / "scorer").rglob("*"):
            if source.is_file() and source.suffix in {".py", ".json"}:
                await runtime.write(
                    f"{target}/scorer/{source.relative_to(Path(__file__).parent / 'scorer')}",
                    source.read_bytes(),
                )
        await runtime.write(
            f"{target}/answer.json", json.dumps(trace.last_reply or "").encode()
        )
        script = (
            "import json,sys; "
            f"sys.path.insert(0,{target!r}); "
            "from scorer import compute_score; "
            f"answer=json.load(open({target + '/answer.json'!r})); "
            "print(json.dumps({'reward':compute_score('music',answer)}))"
        )
        result = await runtime.run(["python3", "-c", script], {})
        if result.exit_code:
            if "Abc2MidiMissing" in result.stderr:
                raise Abc2MidiMissing("abc2midi missing in scoring runtime")
            raise RuntimeError(f"Music scorer failed: {result.stderr[-2000:]}")
        score = float(json.loads(result.stdout.strip().splitlines()[-1])["reward"])
        trace.info["music"] = {"reward": score, "scorer_stderr": result.stderr[-2000:]}
        return score


class MusicTaskset(vf.Taskset[MusicTask, XiaomiTasksetConfig]):
    def load(self):
        for index, row in enumerate(load_rows("music", self.config)):
            yield MusicTask(
                MusicData(
                    id=str(row["extra_info"].get("src_id", index)),
                    prompt=row["prompt"],
                    extra_info=row["extra_info"],
                    image="python:3.11-slim",
                    workdir="/tmp",
                ),
                self.config.task,
            )
