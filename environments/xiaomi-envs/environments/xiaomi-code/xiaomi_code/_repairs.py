"""Repair missing HCL fixtures in the published Code task."""

import hashlib
from pathlib import Path


def repair_test_patch(task_id: str, patch: str) -> str:
    if task_id != "format-code-task-001441":
        return patch
    expected = "f50c33e070080c593b60017eaf7ae4e10999f839f7ee1062894898b95621d87f"
    if hashlib.sha256(patch.encode()).hexdigest() != expected:
        raise ValueError("Unknown test patch for the HCL fixture repair")
    # The released printer_test.go matches this fixture introduction commit.
    # Source: https://github.com/hashicorp/hcl/tree/0c3742819f01f195ee2d75d4bddfc8ab8f11ac66/hcl/printer/testdata
    fixtures = {
        "empty_block.input": "4a2b73a9a03a96f3e409d8b97ecaea0307a7e520295235873eae55184487e561",
        "empty_block.golden": "ac0ab6cfb3ebc4236a0dd6658e6856e0335e516a6b62700e715026a26263fe01",
    }
    for name, expected in fixtures.items():
        data = (Path(__file__).parent / "fixtures" / "hcl" / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Invalid upstream HCL fixture: {name}")
        path = f"hcl/printer/testdata/{name}"
        lines = data.decode().splitlines(keepends=True)
        patch = patch.rstrip("\n") + (
            f"\ndiff --git a/{path} b/{path}\nnew file mode 100644\n"
            f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n"
            + "".join("+" + line if line.endswith("\n") else "+" + line + "\n\\ No newline at end of file\n"
                      for line in lines)
        )
    return patch
