"""Load the published rows and image mapping without changing their contents."""

import json
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

import verifiers.v1 as vf

DATASET_ID = "XiaomiMiMo/MiMo-V2.6-RL-oss"
DATASET_REVISION = "639865fd3374018d6cb29b9fb82dd531406fcf5f"
FILES = {"code": "code.parquet"}

# Source mapping: https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/resolve/639865fd3374018d6cb29b9fb82dd531406fcf5f/image-mapping.jsonl
# SHA-256: 704ad2716d746430a805b9b1ac7c6725aea2117fd8e4823ebe9d339d0313ad58
# Source census: https://huggingface.co/datasets/laion/mimo-v26-rl-code-daytona/resolve/a4324c4133d98418fbb44be7d46023090688c7cf/bundle/mimo1-20260928/code_digests.json
# SHA-256: 8897fec9ab4eb71485139203b3f864dad0c53bbd7aea4714bf1bacabcbbd8543
_code_image_digests = {
    "format-code-task-000014:latest": "sha256:2c7c021648118dd49b4be28cd89fd85d6e510e31a6a95588847e3959b64fe03f",
    "format-code-task-000105:latest": "sha256:13a78ffaf2a5084faeac5b58b0c54823c7f97427724b7b44f73a00b2c5f3937a",
    "format-code-task-000121:latest": "sha256:85e99dbe4a8e4136fd93050e6c3a18b1decd7806bf1c55bec1d73152919e5ea8",
    "format-code-task-000127:latest": "sha256:c880bca682f9acb39abc0c86edce758978214cb874c21a323e68f592a4e540d4",
    "format-code-task-000183:latest": "sha256:d7ea0fc34ba500b014555ac1295a15a67bd62827971ca15656a2d7d8d8bf4d72",
    "format-code-task-000238:latest": "sha256:029305ec1673a697ddfa326c6fe8bb5ff9e749cd94e311fe75e79a164978db82",
    "format-code-task-000244:latest": "sha256:554678bc02f1a5af7688f76031f2ff250132ac7c540940926b135395a397118e",
    "format-code-task-000308:latest": "sha256:a019aa67ecfd985b52db20ea60b739d775746c9e4d9eae82c3aee79ed2c80bfb",
    "format-code-task-000316:latest": "sha256:0da66b0a6a6ee964c90c4ecb11e4d292407e2c77204dcceb975fda09b3ce77cf",
    "format-code-task-000334:latest": "sha256:f4697c5861852f80910a01d27bb2503078f1d4bc51be451a763a13e5394d15be",
    "format-code-task-000349:latest": "sha256:c4c8a1de5ee7c32269af98da665324349deb312af84e72f9d3a6a2e962c2734d",
    "format-code-task-000369:latest": "sha256:f4d3998ac4e14715f8c93f75d77824c3d26ba3c89f4c500fa818c4e254b5390d",
    "format-code-task-000462:latest": "sha256:81d00cf1b9eb1eb16b3b59c5f7b05514cb6cc62b1ac9a8635ca7431dbd9c9369",
    "format-code-task-000463:latest": "sha256:332021f84f2f999c3acd182ac371153e50cc38fe4df83d3b6a5e4a351f50598c",
    "format-code-task-000464:latest": "sha256:ebcbde0d4f0c8dc030eb41d89c6981e55364dc163859e0c5de9af82e626d987b",
    "format-code-task-000478:latest": "sha256:6a8fdfa3bc606efb72eed8c5619dd37faaaefeef046bf27979b4d136aabab630",
    "format-code-task-000484:latest": "sha256:fb821ab827128b73c0a1d9eca3bdba5d4a97a8e8e626b5e0ffb3ef7433ac1487",
    "format-code-task-000509:latest": "sha256:52260dbf6f30f2882c284d45ce43b058dce076a8981232866c0347d04306bcaa",
    "format-code-task-000540:latest": "sha256:3ab03632ea65de70a9a92e5f6f39960236b13618214d320664244b7d833d87cf",
    "format-code-task-000601:latest": "sha256:82addb509c36014eb750472393765187bfb4b28b3929912f0608495e1f4f5864",
    "format-code-task-000607:latest": "sha256:4f33d287c6284088c946f79499d022fdf89ed4ab3c583fdcfede9bc213781a36",
    "format-code-task-000655:latest": "sha256:73f5c9fc3bf815afa9a8c353a73eae8c62a3c9224bcab3e5d5076348df65c369",
    "format-code-task-000681:latest": "sha256:fa7998e14b2b38495e9e7b32a86b955c0d642cbe38dfda63f0b27a3726a204bf",
    "format-code-task-000712:latest": "sha256:47b3532699382de8ea4583a2f5fe1856a03ed60a13c8a63c4e61ef1a27004d87",
    "format-code-task-000727:latest": "sha256:18e9bfe40fa52adb11f38f307a785b06057c4b1b9db827bb718a774bf3939170",
    "format-code-task-000738:latest": "sha256:a2c225360a82d3b9e82c3bd5efe509a9e997e26e5a5665f8f1d98e700e00b57f",
    "format-code-task-000775:latest": "sha256:4fde5d5ac1e4781d3b5521659679f624a299fa0aa35563a9af9ad37c049563cb",
    "format-code-task-000779:latest": "sha256:4615245ec58fb8aa03b553d6d0b3107234ad18704cdbf7e9301b5ebdafc5747a",
    "format-code-task-000797:latest": "sha256:80137d00cad32b8421c2a23fe661b4733c887550d2653d235ef605b582389cef",
    "format-code-task-000798:latest": "sha256:b3546b05738281528e8b62c58599b9aa937ffa4f3b81e4137d9f867836489120",
    "format-code-task-000830:latest": "sha256:d78047b140049a239b769749b633da6191abd1fbd63a65358b9e9dd497f09f68",
    "format-code-task-000860:latest": "sha256:2859a11508554d8ea815c2af88eba9b0944cf29de42fb39eb76dd865101fefd5",
    "format-code-task-000864:latest": "sha256:991cb92d1e6ea3eb4b494353f5c3ecb418b8e0e963c3a8ae00c53fc91b4a534b",
    "format-code-task-000893:latest": "sha256:ea5d29273f3dae29123865f6cc5aa9d8e7444c7afe79a7324232344d3abcc761",
    "format-code-task-000901:latest": "sha256:9574be02c274993016ac43818f84975ceaa69f20a99d65852a698479538ba18e",
    "format-code-task-000937:latest": "sha256:039ce3f73170937717f5816948eb8ef151b85a1bd9f8a4a11b0cf61973b0a8a6",
    "format-code-task-001195:latest": "sha256:903c2b51269769d15c620d5f27511370903d4ef7ec62c987d64730291dc0d30a",
    "format-code-task-001225:latest": "sha256:101f2e6df72026b064e196c30dacaf1a1f4ad05fd8efc923b5686a6ecb417abe",
    "format-code-task-001227:latest": "sha256:4ed35ad9c41db2112b782b277438535d59dced22d5491f41ca222d1da1e43e37",
    "format-code-task-001238:latest": "sha256:3bba56698d7259805d1125c29106d47bbbee0632bf7c5fe520e655675713abb7",
    "format-code-task-001255:latest": "sha256:0eb1fff1ee121e4e478d90be6bc48f6be54ed1c462a36e585d9f03d31f4b89a5",
    "format-code-task-001305:latest": "sha256:b31dd36f91dc46c80ce832c1ba34dbd631215406f215687b553387bb0b847f49",
    "format-code-task-001306:latest": "sha256:6c772f118e832add98fb916519907a0d061abebad84904263851ef439bed6db8",
    "format-code-task-001391:latest": "sha256:986369efa046d8837a4a235b57a243691fa44b0b567777b3c54e500bbd42cc8d",
    "format-code-task-001446:latest": "sha256:a37d844f899cfb2ea51a3288b67a62d43a2a026fcd169bd392181ca4f983c4a6",
    "format-code-task-001466:latest": "sha256:79ac1bdac0d80064e305cb6ef77118ca80badd22df7dc03fe972130d78da7f05",
    "format-code-task-001481:latest": "sha256:65830ca1e58e1d7f2757cbaae79fcebc4a47f4926497136355a9a2e0feb1b344",
    "format-code-task-001501:latest": "sha256:d1b6dca7ac387f3b4ae5f094522e769923f90d16ee6eedb4269361eaf7dd5de2",
    "format-code-task-001540:latest": "sha256:8d6c3b809d1f9837f8d202f2dd5b6aeb092c24f1febda76fd545064834c9e83a",
    "format-code-task-001542:latest": "sha256:be10e29641ce37743e8e06e389674d959a66db5e5d89ce761348d8a3d583ffbb",
    "format-code-task-001554:latest": "sha256:98459cbc783284c9d7dc89854f5b1ce7393169528ae98da127590e357b52ca9e",
    "format-code-task-001618:latest": "sha256:e0a507bb30b74e8c53ca0145fd21e856ee3ad1ff90b319d04cbd38ce5fb181fd",
    "format-code-task-001621:latest": "sha256:e3d21123603f525a3128881215eda662f92bf604c828f0b40c2feb88617fad52",
    "format-code-task-001643:latest": "sha256:eb8362fad2b7ebe9e2d3086b6e40124326162320c84f99bc9b13cd76051119b9",
    "format-code-task-001649:latest": "sha256:841ec577fc1fae28b0acd322e5e3af22ff2ec880e4685e32ea32f5580fbed9fc",
    "format-code-task-001652:latest": "sha256:620b6e736a739d62aca1e3131c028a5673f07d1824976180b723115d4c807629",
    "format-code-task-001712:latest": "sha256:164b6e702756eb8a037aad01ff5b765bdb32a8b714b3339a6367a507920a8d65",
    "format-code-task-001713:latest": "sha256:3619c3f5610fe811dc34c576c3f7260beba5eb6e1b2057c79b15dd3fc6ffd11e",
    "format-code-task-001715:latest": "sha256:65f897d01aae52342f6bde4d55157f0e2b6d256b84f4e18c38d05ad4a98d2d06",
    "format-code-task-001744:latest": "sha256:6e8b625e5c5f8a31709214ae35b2d047f621bd04fb64ab1024bf8d3c96c64a98",
    "format-code-task-001753:latest": "sha256:ede17e95dbfdb4461e491d012b98f614ee71fe3fb41b1c25471807c9c0c19a00",
    "format-code-task-001764:latest": "sha256:3658414cf99e92b2ede6cb69ba5ae046e9e1b237f91b9e0d348e936a14e1de42",
    "format-code-task-001800:latest": "sha256:a3c586e7085baa4baac09d432f6a544fbdf1659a710178292424283c9da578a6",
    "format-code-task-001881:latest": "sha256:f3604522c6846e1c7b0c8f17cbf0c1aa9f82c798d4aff945c9f830b11503794d",
    "format-code-task-001976:latest": "sha256:cfa668f57ee0ee8ed1575c8d95d2ee6c95dea23229879c973e33c08092829e7b",
    "format-code-task-001993:latest": "sha256:1e16ebbc90719d45101fa00bc112d9aeb629dcabe60582021f09ad07bd66ae1d",
    "format-code-task-002040:latest": "sha256:8b606da50e38d402473ec30df04fb899e99f7cd0bd9b0b7fd754be888b41c376",
    "format-code-task-002046:latest": "sha256:ae294e6b4d16aea57f955e8ef1c1b6d61a1140c0574fc320c248d0403357d0a5",
    "format-code-task-002061:latest": "sha256:34cb9ae3c83f543b34da669144f6e04149e0f4a4293765b89926c11a979ec7fa",
    "format-code-task-002099:latest": "sha256:82f810c559a12e6d7b968d2ec811ff82ac6c5503ff8c76b8ac54dd113ac1645d",
    "format-code-task-002175:latest": "sha256:74b4265e4281369269e0a399f4d794b0c6f17f0219b37c6af3290eea408fb705",
    "format-code-task-002205:latest": "sha256:697b397a740b337dbb0dd984fcdb0726664a1871283e7e344ad219c85d6ddbe1",
    "format-code-task-002213:latest": "sha256:8f9b47a09da9eb37e1b4a5cb6ecf64b7dc7bf30ff06093c47d0fe322a85fd814",
    "format-code-task-002226:latest": "sha256:3849596e797fe949231549b3aad40aa7220e3aff62a6f2e1bc41ebe7a96d3db6",
    "format-code-task-002246:latest": "sha256:a28cab7e11df0e51955da21461b5a601ea264b34072f11d947ee601323f08e4f",
    "format-code-task-002299:latest": "sha256:66168cff5993c30763e2c65c4d3db0b68863d011cec848dcc6d666441c1fa550",
    "format-code-task-002421:latest": "sha256:9d36493dd377b6170bdf9e380054e344a9f13e4d8559f42ce6e57868feb50ac7",
    "format-code-task-002422:latest": "sha256:61a83bf138be98b864b1f26150f40a6e2144355fde6ea331b232c9827c8bf814",
    "format-code-task-002465:latest": "sha256:47058cab6e4b5c001e3f06dff17707b8df4a6e312ff70958776a15b34ff4ce31",
    "format-code-task-002472:latest": "sha256:a41b8702287dec796f77fd04dddfd87e2fcf9f03db9e85d0b8fa954e3b9415f6",
    "format-code-task-002496:latest": "sha256:affa47b780807b1597bee2c2fc06aa652e2c6d5adbf466ad7346c87cc8cab2a3",
    "format-code-task-002513:latest": "sha256:ad3540035a790faddb236d2c6ac63029722fb265c905ff5b4fa8cf294dacd804",
    "format-code-task-002540:latest": "sha256:284a66c0b04216ae351278f794bbe1f1e35cf2f32873264e0573cf2defa43c71",
    "format-code-task-002642:latest": "sha256:2634aee3dbe800b54323980211bdf1627f38fe487d5eb35f79ab34f640c8661f",
    "format-code-task-002666:latest": "sha256:00b7faa38daee51df4bcc5c0decba8f183b9aff80bd5d6ca6d7ddf870f331848",
    "format-code-task-002735:latest": "sha256:9a2a751dac60731e0af6f4d2e8e4c4152506a56c80c4db9fbf7d953d3cf7186a",
    "format-code-task-002793:latest": "sha256:6bc0ece2d19570818d4e91ae8a03e3c68a4b6117dd5496927902b15c2dd54769",
    "format-code-task-002800:latest": "sha256:d00b1eb266b33253ba3b51855d753a863c0ef595729aa5e9ffa048d0b1a295d3",
    "format-code-task-002818:latest": "sha256:e59a83e87936db52aace57bcf9b93a1cb18a385196a4468144b48f7041254fed",
    "format-code-task-002842:latest": "sha256:a1867e58a20342c4392ac7f4f081b43c248b75e53a9d4db8b1a778fee3196661",
    "format-code-task-002879:latest": "sha256:20c374dd653603bd40e9ef6da2e1071d9c1070f9f3e6fe1aa3dc7ac766edcfaf",
    "format-code-task-002951:latest": "sha256:ff71b5e3adef78e3887b91b3b672c8d30b682191e0b654703af92bc6f8de7ded",
    "format-code-task-002997:latest": "sha256:c6b58fbdccf21d7e7a1dd2ba98f0ad541795fadf91e13bf2b6dfd572746831ca",
    "format-code-task-003041:latest": "sha256:57326fbc372f5c55e43861f2f04f4ab8e29972516798f6b8494cca238901f37c",
    "format-code-task-003050:latest": "sha256:8aec7fb05c35b0d00b7ea30e32e40c1942d6de872f7b659df66967fae37cced2",
    "format-code-task-003051:latest": "sha256:386f5d0140c6bec43772233b060fa5e031ffa43a3d11188b45896c10edc27784",
}


class XiaomiTasksetConfig(vf.TasksetConfig):
    dataset_path: str | None = None
    """Optional local parquet with the same published row schema."""
    revision: str = DATASET_REVISION


def load_rows(domain: str, config: XiaomiTasksetConfig) -> list[dict]:
    filename = FILES[domain]
    path = config.dataset_path or hf_hub_download(
        DATASET_ID, filename, repo_type="dataset", revision=config.revision
    )
    return pq.read_table(path).to_pylist()


def load_instance(row: dict) -> dict:
    instance = json.loads(row["extra_info"]["instance_json"])
    if not isinstance(instance, dict):
        raise ValueError("instance_json must encode an object")
    return instance


@lru_cache(maxsize=4)
def _image_mapping(revision: str) -> dict[str, str]:
    path = hf_hub_download(DATASET_ID, "image-mapping.jsonl", repo_type="dataset", revision=revision)
    mapping = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        source, target = row["dataset_image"], row["dockerhub_image"]
        if source in mapping and mapping[source] != target:
            raise ValueError(f"Conflicting published image mapping for {source!r}")
        if not target.startswith("docker.io/xiaomimimo/mimo-v2.6-rl-oss:"):
            raise ValueError(f"Unexpected published image registry: {target!r}")
        mapping[source] = target
    return mapping


def resolve_image(dataset_image: str, revision: str = DATASET_REVISION) -> str:
    mapping = _image_mapping(revision)
    try:
        target = mapping[dataset_image]
    except KeyError as exc:
        raise ValueError(f"Image absent from published image mapping: {dataset_image!r}") from exc
    digest = _code_image_digests.get(dataset_image)
    if revision == DATASET_REVISION and digest is not None:
        expected_target = (
            "docker.io/xiaomimimo/mimo-v2.6-rl-oss:"
            f"{dataset_image.removesuffix(':latest')}"
        )
        if target == expected_target:
            return f"docker.io/xiaomimimo/mimo-v2.6-rl-oss@{digest}"
    return target
