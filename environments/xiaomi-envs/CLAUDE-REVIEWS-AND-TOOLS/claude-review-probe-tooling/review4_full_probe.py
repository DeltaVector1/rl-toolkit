"""Review 4: drive the real Code task (setup, then the verifier on the untouched tree) on every image.

    uv run python review4_full_probe.py <DATA> <out.jsonl> [concurrency=48] [verifier cap seconds=300]
    PROBE_IDS="id id ..." restricts the run. PROBE_NO_VERIFY=1 skips the verifier.
    PROBE_NO_REEXPORT=1 drops the bootstrap's last line (`export BASH_ENV=...`), so only the
    outermost shell applies the profiles, as under upstream's `bash -lc`.
    PROBE_DEFAULTS=1 passes no CPU or disk flags, so only the task's declared resources apply.
    PROBE_NO_CLEANUP=1 skips the port's cache clean-up step.
    PROBE_CLEAN=1 runs nothing but the port's own setup and verifier (no probe commands at all),
    so nothing this script does can influence the timing.

Resumable: rows already recorded without an error are skipped. No model runs. Per image it records
  setup      whether the new setup (history strip, cache clean-up, BASH_ENV bootstrap) passes, how
             long it takes, and whether HEAD, the tree, `git status` or `git describe` changed
  history    refs and commits outside the base afterwards; unreachable commits where `fsck` is cheap
  bootstrap  what the BASH_ENV file does to shells started inside the agent's or verifier's shell:
             output it prints, `bash -e`/`-u` children, a PATH or variable set by the parent, and
             the cost per child shell; and how the environment compares with upstream's `bash -lc`
  verifier   exit code and output tail, for comparison with results/verifier_health.jsonl
"""
import asyncio, json, os, re, sys, time
from types import SimpleNamespace

from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from verifiers.v1.utils.compile import resolve_runtime_config
from xiaomi_code import XiaomiCodeTaskset
from xiaomi_code._data import XiaomiTasksetConfig

DATA, OUT = sys.argv[1], sys.argv[2]
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 48
CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 300
ONLY = set(os.environ.get("PROBE_IDS", "").split())
VERIFY = not os.environ.get("PROBE_NO_VERIFY")
# Memory is left to the task (8 GB declared); CPU and disk are what the docs tell users to pass.
BASE = PrimeConfig(cpu=2, disk=20, labels=["xiaomi-review"])
if os.environ.get("PROBE_DEFAULTS"):  # no flags at all: Prime's defaults plus what the task declares
    BASE = PrimeConfig(labels=["xiaomi-review"])
IGNORE = {"SHLVL", "_", "PWD", "OLDPWD", "PS1", "LS_COLORS", "LESSOPEN", "LESSCLOSE", "BASH_ENV"}
CACHES = r"find . -type d \( -name .git -o -name .build_env \) -prune -o -type d \( -name __pycache__ -o -name .pytest_cache -o -path '*/.jest/cache' \) -prune -print 2>/dev/null"
STATE = r"""
echo "head_@=$(git rev-parse HEAD 2>&1)"
echo "tree_@=$(git rev-parse 'HEAD^{tree}' 2>&1)"
echo "status_@=$(git status --porcelain 2>&1 | grep -v -E '__pycache__|\.pytest_cache|\.jest/' | sha1sum | cut -c1-12)"
echo "diff_@=$(git diff HEAD 2>&1 | sha1sum | cut -c1-12)"
echo "branch_@=$(git symbolic-ref -q --short HEAD)"
echo "describe_@=$(git describe --tags --long 2>/dev/null | sed 's/-g[0-9a-f]*$//')"
echo "refs_@=$(git for-each-ref | wc -l)"
echo "caches_@=$(""" + CACHES + r""" | wc -l)"
echo "jest_caches_@=$(""" + CACHES + r""" | grep -c '/.jest/cache$')"
echo "git_mb_@=$(du -sm "$(git rev-parse --git-dir)" | cut -f1)"
"""
PRE = r"""git config --global --add safe.directory "$PWD"
echo "git_version=$(git --version)"
echo "bash_version=$BASH_VERSION"
echo "hooks=$(ls "$(git rev-parse --git-dir)/hooks" 2>/dev/null | grep -v '\.sample$' | tr '\n' ',')"
""" + STATE.replace("@", "before")
POST = STATE.replace("@", "after") + r"""
echo "extra_after=$(git rev-list --all --not HEAD 2>&1 | wc -l)"
mb=$(du -sm "$(git rev-parse --git-dir)" | cut -f1)
if [ "$mb" -lt 400 ]; then echo "unreachable_after=$(timeout 120 git fsck --unreachable --no-reflogs 2>/dev/null | grep -c '^unreachable commit')"; fi
"""
# Everything below runs inside one `bash -c`, i.e. as children of the agent's or verifier's shell.
CHILDREN = r"""
for o in e u eu; do bash -$o -c true >/dev/null 2>/tmp/_strict; echo "strict_$o=$? $(head -c 160 /tmp/_strict | tr '\n' ' ')"; done
echo "child_path=$(PATH=/zzprobe:$PATH bash -c 'echo "$PATH"' | tail -1)"
export NODE_OPTIONS=zz GRADLE_OPTS=zz MAVEN_OPTS=zz JAVA_HOME=zz GOFLAGS=zz VIRTUAL_ENV=zz CARGO_HOME=zz
echo "child_vars=$(bash -c 'echo "$NODE_OPTIONS|$GRADLE_OPTS|$MAVEN_OPTS|$JAVA_HOME|$GOFLAGS|$VIRTUAL_ENV|$CARGO_HOME"' | tail -1)"
s=$(date +%s%N); for i in 1 2 3 4 5 6 7 8 9 10; do bash -c true >/dev/null 2>&1; done; e=$(date +%s%N)
echo "child_ms=$(( (e - s) / 10000000 ))"
"""
ENVCMD = "env"

if os.environ.get("PROBE_NO_REEXPORT"):
    import xiaomi_code.taskset as port
    lines = port.BASH_ENV_SCRIPT.splitlines(keepends=True)
    assert lines[-1].startswith("export BASH_ENV=")
    port.BASH_ENV_SCRIPT = "".join(lines[:-1])
if os.environ.get("PROBE_NO_CLEANUP"):
    import xiaomi_code.taskset as port
    port.CACHE_CLEANUP = "true"
taskset = XiaomiCodeTaskset(XiaomiTasksetConfig(dataset_path=f"{DATA}/code.parquet"))
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT) if "error" not in json.loads(l)}
todo = [t for t in taskset if t.data.id not in done and (not ONLY or t.data.id in ONLY)]
sem = asyncio.Semaphore(CONC)
count = 0


def fields(text):
    return dict(line.split("=", 1) for line in text.splitlines() if re.match(r"^\w+=", line))


def envmap(text):
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and re.fullmatch(r"\w+", key) and key not in IGNORE:
            out[key] = value
    return out


async def work(task, rec):
    cwd = task.data.workdir
    config = resolve_runtime_config(BASE, task)
    rec["memory"], rec["cpu"], rec["disk"] = config.memory, config.cpu, config.disk
    trace = SimpleNamespace(id="probe", info={}, state=SimpleNamespace(base_ref=""), last_reply="")
    async with provision_runtime(config, env=dict(task.runtime_env())) as rt:
        await rt.prepare_setup()
        if os.environ.get("PROBE_CLEAN"):
            started = time.monotonic()
            await type(task).setup(task, trace, rt)
            rec["setup_seconds"] = round(time.monotonic() - started, 1)
            started = time.monotonic()
            reward = await type(task).verified(task, trace, rt)
            verifier = trace.info["verifier"]
            out = re.sub(r"\x1b\[[0-9;]*m", "", (verifier.get("stdout") or "") + (verifier.get("stderr") or ""))
            rec.update(verify_seconds=round(time.monotonic() - started, 1), reward=reward, exit=verifier.get("verifier_returncode"),
                       timeout=task.data.verifier_timeout_sec, pyenv_lock=out.count("couldn't acquire lock"), tail=out[-400:],
                       stdout_tail=re.sub(r"\x1b\[[0-9;]*m", "", verifier.get("stdout") or "")[-400:])
            return
        rec.update(fields((await rt.run(["bash", "-c", f"cd {cwd} && {PRE}"], {})).stdout))
        started = time.monotonic()
        try:
            await type(task).setup(task, trace, rt)
        except Exception as e:
            rec["setup_error"] = repr(e)[:700]
        rec["setup_seconds"] = round(time.monotonic() - started, 1)
        rec.update(fields((await rt.run(["bash", "-c", f"cd {cwd} && {POST}"], {})).stdout))
        quiet = await rt.run(["bash", "-c", "true"], {})
        rec.update(noise_out=quiet.stdout[:200], noise_err=quiet.stderr[:200], noise_rc=quiet.exit_code)
        rec.update(fields((await rt.run(["bash", "-c", CHILDREN], {})).stdout))
        login = envmap((await rt.run(["env", "-u", "BASH_ENV", "bash", "-lc", ENVCMD], {})).stdout)
        port = envmap((await rt.run(["bash", "-c", ENVCMD], {})).stdout)
        rec["env_diff"] = {k: [login.get(k), port.get(k)] for k in sorted(set(login) | set(port)) if login.get(k) != port.get(k)}
        if "setup_error" in rec or not VERIFY:
            return
        started = time.monotonic()
        try:
            reward = await asyncio.wait_for(type(task).verified(task, trace, rt), CAP)
        except asyncio.TimeoutError:
            rec["verify"] = "cap"
            return
        except Exception as e:
            rec["verify_error"] = repr(e)[:500]
            return
        finally:
            rec["verify_seconds"] = round(time.monotonic() - started, 1)
        verifier = trace.info["verifier"]
        out = re.sub(r"\x1b\[[0-9;]*m", "", (verifier.get("stdout") or "") + (verifier.get("stderr") or ""))
        rec.update(
            reward=reward,
            exit=verifier.get("verifier_returncode"),
            not_found=sorted(set(re.findall(r"([\w./+-]+): (?:command )?not found", out)))[:5],
            missing_module=sorted(set(re.findall(r"No module named '?([\w.]+)", out)))[:5],
            tail=out[-400:],
        )


async def probe(task):
    global count
    rec = {"id": task.data.id, "cwd": task.data.workdir}
    async with sem:
        for attempt in range(2):
            keep = {"id": rec["id"], "cwd": rec["cwd"]}
            rec.clear(); rec.update(keep)
            try:
                await asyncio.wait_for(work(task, rec), CAP + 900)
                break
            except Exception as e:
                rec["error"] = repr(e)[:300]
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    count += 1
    if count % 100 == 0:
        print(f"[{count}/{len(todo)}]", flush=True)


async def main():
    print("todo", len(todo), "done", len(done), flush=True)
    await asyncio.gather(*(probe(t) for t in todo))

asyncio.run(main())
