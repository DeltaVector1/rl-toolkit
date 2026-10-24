"""Expose each task's original capped tool functions through native MCP."""

import importlib.util
from pathlib import Path

import verifiers.v1 as vf


class GeneralTools(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = None

    async def setup_task(self, task):
        self.enabled = task.instance["dataset_type"] == "general_agent"

    def register(self, mcp):
        if not self.enabled:
            return
        paths = [
            path
            for path in sorted(Path("/work/tools").glob("*.py"))
            if not path.name.startswith("_") and path.name != "tools_test.py"
        ]
        if not paths:
            return
        path = Path("/installed-agent/mcp_http.py")
        spec = importlib.util.spec_from_file_location("xiaomi_task_mcp", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Reuse task-local annotation inference and result caps verbatim.
        for path in paths:
            tools = module._load_tools_module(path)
            for name, fn in module._public_functions(tools):
                mcp.add_tool(
                    module._wrap_capped(fn),
                    name=f"mcp__{path.stem}__{name}",
                    description=fn.__doc__ or "",
                )


if __name__ == "__main__":
    GeneralTools.run()
