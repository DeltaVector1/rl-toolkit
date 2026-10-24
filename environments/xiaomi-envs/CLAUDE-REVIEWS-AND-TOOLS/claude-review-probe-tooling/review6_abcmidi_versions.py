"""Review 6: score the same model replies with the port's Music scorer under several abcmidi versions.

    uv run python review6_abcmidi_versions.py <replies.json> <out.json> [image ...]

`replies.json` is a list of {"id": ..., "reply": ...} taken from a Music eval. Each image gets one
sandbox: the port's own `MusicTask.setup` installs the distribution's `abcmidi` package, then every
reply is scored with the same `compute_score('music', reply)` call the port's reward makes, and
`why` records the scorer's own reject reasons (abc2midi errors, bar warnings, blank lines,
channel conflicts) so a score that moves between versions can be traced to its cause. The
scorer is the vendored upstream source, so the only things that differ between images are the
abc2midi build and the Python minor version (recorded). Nothing in the repository is modified.
"""
import asyncio, json, sys
from pathlib import Path
from types import SimpleNamespace

import xiaomi_music.taskset as music
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from verifiers.v1.utils.compile import resolve_runtime_config
from xiaomi_music.taskset import MusicData, MusicTask

REPLIES, OUT = json.load(open(sys.argv[1])), sys.argv[2]
IMAGES = sys.argv[3:] or ["python:3.11-slim", "python:3.11-slim-bookworm", "ubuntu:22.04", "ubuntu:24.04"]
TARGET = "/tmp/xiaomi-music-scorer"
BATCH = f"""
import json, sys
sys.path.insert(0, {TARGET!r})
from scorer import compute_score
from scorer.pipeline import do, extract_abc
out, why = [], {{}}
for item in json.load(open({TARGET + '/replies.json'!r})):
    try:
        out.append([item["id"], compute_score("music", item["reply"])])
        abc = extract_abc(item["reply"])
        if not abc:
            why[item["id"]] = "no_abc"
        else:
            r = do(dict(key="k", id=0, rep=0, abc=abc))
            why[item["id"]] = {{k: r.get(k) for k in ("skip", "scorer_skip", "err", "bar", "blank", "ch_conflict", "reject") if r.get(k)}}
    except Exception as exc:
        out.append([item["id"], "ERROR " + type(exc).__name__ + ": " + str(exc)[:120]])
print(json.dumps([out, why]))
"""


async def one(image):
    task = MusicTask(MusicData(id="probe", prompt=[{"role": "user", "content": "x"}], extra_info={}, image=image, workdir="/tmp"))
    config = resolve_runtime_config(PrimeConfig(labels=["xiaomi-review"]), task)
    rec = {"image": image}
    try:
        async with provision_runtime(config, env={}) as rt:
            await rt.prepare_setup()
            await rt.run(["sh", "-c", "command -v python3 >/dev/null || (apt-get update -qq && apt-get install -y -qq python3 >/dev/null 2>&1)"], {})
            await MusicTask.setup(task, SimpleNamespace(info={}), rt)
            info = await rt.run(["sh", "-c", "dpkg-query -W -f='${Version}' abcmidi; echo; abc2midi -ver 2>&1 | head -1; python3 --version; . /etc/os-release; echo $PRETTY_NAME"], {})
            rec["versions"] = info.stdout.strip().splitlines()
            scorer = Path(music.__file__).parent / "scorer"
            for source in scorer.rglob("*"):
                if source.is_file() and source.suffix in {".py", ".json"}:
                    await rt.write(f"{TARGET}/scorer/{source.relative_to(scorer)}", source.read_bytes())
            await rt.write(f"{TARGET}/replies.json", json.dumps(REPLIES).encode())
            result = await rt.run(["python3", "-c", BATCH], {})
            if result.exit_code:
                rec["error"] = result.stderr[-600:]
            else:
                scores, why = json.loads(result.stdout.strip().splitlines()[-1])
                rec["scores"], rec["why"] = dict(scores), why
    except Exception as exc:
        rec["error"] = repr(exc)[:400]
    print(image, rec.get("versions"), rec.get("error", "ok"), flush=True)
    return rec


async def main():
    json.dump(await asyncio.gather(*(one(image) for image in IMAGES)), open(OUT, "w"), indent=1)

asyncio.run(main())
