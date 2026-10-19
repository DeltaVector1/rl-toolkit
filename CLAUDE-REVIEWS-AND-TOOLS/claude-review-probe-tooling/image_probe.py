import asyncio, sys
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig

CMD = r"""
echo "## os"; grep -E '^(PRETTY_NAME)=' /etc/os-release
echo "## pid1 env"; tr '\0' '\n' < /proc/1/environ | grep -E '^(PATH|HOME|GOROOT|GOPATH|NVM|NODE)' ; echo "pid1: $(tr '\0' ' ' < /proc/1/cmdline)"
echo "## this shell"; echo "PATH=$PATH HOME=$HOME USER=$(id -un)"
echo "## /usr/local/bin"; ls /usr/local/bin 2>/dev/null | head -30 | tr '\n' ' '; echo
echo "## where"; for t in go node npm npx cargo; do printf '%s: ' $t; find / -xdev \( -name $t -o -name $t.exe \) \( -type f -o -type l \) -not -path '/proc/*' 2>/dev/null | head -4 | tr '\n' ' '; echo; done
echo "## profiles"; for f in /etc/profile /etc/bash.bashrc /root/.bashrc /root/.profile /root/.bash_profile /etc/environment; do echo "--- $f ($(wc -l < $f 2>/dev/null || echo missing))"; grep -nE 'PATH|GOROOT|nvm|cargo|\. |source ' $f 2>/dev/null | head -12; done
echo "--- /etc/profile.d:"; ls /etc/profile.d 2>/dev/null | tr '\n' ' '; echo
echo "## docker-default PATH"; env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root bash -c 'for t in go node npm npx; do printf "%s=%s " $t "$(command -v $t)"; done; echo'
echo "## docker-default PATH + login"; env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root bash -lc 'echo PATH=$PATH; for t in go node npm npx; do printf "%s=%s " $t "$(command -v $t)"; done; echo'
echo "## interactive"; bash -ic 'echo PATH=$PATH' 2>/dev/null
"""

async def probe(tag):
    cfg = PrimeConfig(image=f"docker.io/xiaomimimo/mimo-v2.6-rl-oss:{tag}", workdir="/tmp", cpu=1, memory=2, disk=10, labels=["xiaomi-review"])
    async with provision_runtime(cfg) as rt:
        r = await rt.run(["bash", "-c", CMD], {})
        return f"================ {tag}\n{r.stdout}\n{r.stderr[-300:]}"

async def main():
    for out in await asyncio.gather(*(probe(t) for t in sys.argv[1:]), return_exceptions=True):
        print(out)

asyncio.run(main())
