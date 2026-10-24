"""Finding 1: read image configs from Docker Hub (no layer pulls) and report layer count and ENV.

    python3 registry_env_probe.py <image-mapping.jsonl> [sample_size=20] [extra_tag ...]

Stdlib only. Each tag costs two or three anonymous registry requests, so keep the sample small.
"""
import collections, json, random, sys, urllib.request

REPO = "xiaomimimo/mimo-v2.6-rl-oss"
ACCEPT = ",".join((
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
))
token = json.load(urllib.request.urlopen(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REPO}:pull"))["token"]


def get(path, accept):
    req = urllib.request.Request(f"https://registry-1.docker.io/v2/{REPO}/{path}", headers={"Authorization": f"Bearer {token}", "Accept": accept})
    return json.load(urllib.request.urlopen(req))


def config(tag):
    manifest = get(f"manifests/{tag}", ACCEPT)
    if "manifests" in manifest:
        manifest = get(f"manifests/{manifest['manifests'][0]['digest']}", ACCEPT)
    return len(manifest["layers"]), get(f"blobs/{manifest['config']['digest']}", "application/json").get("config", {})


tags = [json.loads(line)["dockerhub_image"].split(":")[-1] for line in open(sys.argv[1])]
code = [tag for tag in tags if tag.startswith("format-code-task-")]
size = int(sys.argv[2]) if len(sys.argv) > 2 else 20
random.seed(3)
tally = collections.Counter()
for tag in random.sample(code, size):
    layers, cfg = config(tag)
    tally[(layers, "has Env" if cfg.get("Env") else "no Env")] += 1
print(len(code), f"code tags; sample of {size} (layers, env):", dict(tally))
for tag in sys.argv[3:]:
    layers, cfg = config(tag)
    print("==", tag, "| layers", layers, "| WorkingDir", cfg.get("WorkingDir"))
    for entry in cfg.get("Env") or ["(no Env)"]:
        print("     ", entry[:200])
