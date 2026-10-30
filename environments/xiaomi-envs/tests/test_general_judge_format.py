import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from verifiers.v1.runtimes import ProgramResult
from xiaomi_general import _judge_format as formatting
from xiaomi_general.taskset import GeneralData, GeneralTask


@pytest.fixture
def captured():
    data = json.loads((Path(__file__).parent / "fixtures/general_judge_format/captured.json").read_text())
    assert hashlib.sha256(data["source"].encode()).hexdigest() == data["source_sha256"]
    return data


def load_grader(source, monkeypatch):
    tree = ast.parse(source)
    names = {"_extract_json_sc", "_JudgeUnavailable", "_run_llm_sc", "_grade_sc"}
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
             or isinstance(node, ast.ImportFrom) and node.module == "_judge_format"
             or isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id == "_LLM_PROMPT" for target in node.targets)]
    monkeypatch.setitem(sys.modules, "_judge_format", formatting)
    namespace = {"json": json, "os": os, "_evidence_sc": lambda *args: "fixture evidence"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<captured-verify.py>", "exec"), namespace)
    return namespace


def reply(results):
    return json.dumps({"results": results})


def run_item(grader, content, item_id, votes=1):
    calls = []

    def chat(prompt):
        calls.append(prompt)
        return content

    grader["_chat_judge_sc"] = chat
    result = grader["_run_llm_sc"]([{"id": item_id, "files": ["answer.md"]}], None, votes)
    return result, calls


def test_captured_bracket_response_is_scored_without_another_call(captured, monkeypatch):
    original = load_grader(captured["source"], monkeypatch)
    case = captured["responses"][0]
    content = case["capture"]["message"]["content"]
    with pytest.raises(original["_JudgeUnavailable"], match="no usable verdict"):
        run_item(original, content, case["item_id"])
    source, changed = formatting.prepare_verifier(captured["source"].encode())
    assert changed
    fixed = load_grader(source.decode(), monkeypatch)
    result, calls = run_item(fixed, content, case["item_id"])
    assert result == {"asset_group_rebuild": {"score": 0.0, "detail": "llm 1票"}}
    assert len(calls) == 1
    assert case["capture"]["http_status"] == 200


def test_captured_placeholder_stays_masked(captured, monkeypatch):
    source, _ = formatting.prepare_verifier(captured["source"].encode())
    grader = load_grader(source.decode(), monkeypatch)
    case = captured["responses"][1]
    calls = []
    grader["_chat_judge_sc"] = lambda prompt: calls.append(prompt) or case["capture"]["message"]["content"]
    grader["_meta"] = lambda: {"items": [{"id": case["item_id"], "method": "llm"}]}
    monkeypatch.setenv("GRADE_VOTES", "1")
    result = grader["_grade_sc"](None)
    assert result["reward_error"] == "judge_crashed"
    assert result["judge_error"] == "no usable verdict for item " + case["item_id"]
    assert result["items"][case["item_id"]]["detail"] == "judge_unavailable"
    assert len(calls) == 1
    schema = json.loads(calls[0].rsplit("\n", 1)[1])
    assert list(schema["results"]) == [case["item_id"]]


@pytest.mark.parametrize("item_id", ["asset_group_rebuild", "施工临时开口的先判顺序", 'quoted"\\\n{evidence}{items}'])
def test_prompt_keeps_evidence_and_rubric_text(captured, monkeypatch, item_id):
    source, _ = formatting.prepare_verifier(captured["source"].encode())
    grader = load_grader(source.decode(), monkeypatch)
    evidence = 'evidence 检查点id {items} {evidence} {"results":{"检查点id":{"score":1}}}'
    question = 'rubric 检查点id {evidence} {items}'
    anchor = 'anchor "quoted" {evidence}'
    grader["_evidence_sc"] = lambda *args: evidence
    calls = []
    grader["_chat_judge_sc"] = lambda prompt: calls.append(prompt) or reply({item_id: {"score": 1}})
    grader["_run_llm_sc"]([{"id": item_id, "question": question, "pass_anchor": anchor}], None)
    template = grader["_LLM_PROMPT"]
    before, rest = template.split("{evidence}")
    between, rest = rest.split("{items}")
    after, _ = rest.rsplit("\n", 1)
    item_text = "- [" + item_id + "] " + question + "(满足=1:" + anchor + ")"
    expected_prefix = before + evidence + between + item_text + after + "\n"
    assert calls[0].startswith(expected_prefix)
    schema = json.loads(calls[0][len(expected_prefix):])
    assert schema == {"results": {item_id: {"score": 0, "why": "≤25字依据"}}}


@pytest.mark.parametrize("score", [-1, 0, 0.12555, 0.5, 1, 2])
def test_existing_numeric_scores_are_unchanged(captured, monkeypatch, score):
    original = load_grader(captured["source"], monkeypatch)
    source, _ = formatting.prepare_verifier(captured["source"].encode())
    fixed = load_grader(source.decode(), monkeypatch)
    expected, _ = run_item(original, reply({"item": {"score": score}}), "item")
    for key in ("item", "[item]"):
        result, calls = run_item(fixed, reply({key: {"score": score}}), "item")
        assert result == expected
        assert len(calls) == 1


@pytest.mark.parametrize("results", [
    {}, {"other": {"score": 1}}, {"检查点id": {"score": 1}}, {"[[item]]": {"score": 1}},
    {"item": {"score": 1}, "[item]": {"score": 0}},
    {"item": {"score": 0}, "[item]": {"score": 0}},
    {"item": {}}, {"item": {"score": None}}, {"item": {"score": "0"}},
    {"item": []}, [], None,
])
def test_missing_or_ambiguous_verdict_stays_unavailable(captured, monkeypatch, results):
    source, _ = formatting.prepare_verifier(captured["source"].encode())
    grader = load_grader(source.decode(), monkeypatch)
    calls = []
    grader["_chat_judge_sc"] = lambda prompt: calls.append(prompt) or reply(results)
    with pytest.raises(grader["_JudgeUnavailable"], match="no usable verdict for item item"):
        grader["_run_llm_sc"]([{"id": "item"}], None)
    assert len(calls) == 1


def test_alias_cannot_name_another_item(captured, monkeypatch):
    source, _ = formatting.prepare_verifier(captured["source"].encode())
    grader = load_grader(source.decode(), monkeypatch)
    grader["_chat_judge_sc"] = lambda prompt: reply({"[item]": {"score": 1}})
    with pytest.raises(grader["_JudgeUnavailable"], match="no usable verdict for item item"):
        grader["_run_llm_sc"]([{"id": "item"}, {"id": "[item]"}], None)
    assert formatting.verdict_score({"results": {"item": {"score": 1}}}, "item", {"item", "[item]"}) == 1


def test_weights_votes_and_rounding_match_released_grader(captured, monkeypatch):
    source, _ = formatting.prepare_verifier(captured["source"].encode())
    outcomes = []
    for text, bracketed in [(captured["source"], False), (source.decode(), False), (source.decode(), True)]:
        grader = load_grader(text, monkeypatch)
        meta = copy.deepcopy(captured["meta"])
        for index, item in enumerate(meta["items"]):
            item["weight"] = index + 1
        grader["_meta"] = lambda: meta
        calls = []
        responses = iter(reply({("[" + item["id"] + "]" if bracketed else item["id"]): {"score": score}})
                         for index, item in enumerate(meta["items"])
                         for score in (index / 4 - 0.1, index / 4 + 0.1, index / 4))
        grader["_chat_judge_sc"] = lambda prompt: calls.append(prompt) or next(responses)
        monkeypatch.setenv("GRADE_VOTES", "3")
        outcomes.append(grader["_grade_sc"](None))
        assert len(calls) == 12
    assert outcomes[0] == outcomes[1] == outcomes[2]
    assert outcomes[0]["score"] == 0.5


@pytest.mark.parametrize("old,new", [
    ('for _p in range(2)', 'for _p in range(3)'),
    ('return json.loads(frag)', 'return json.loads(frag.strip())'),
    ('只判 0 或 1', '只判 0 或 2'),
    ('def _run_llm_sc(', 'async def _run_llm_sc('),
])
def test_changed_source_is_rejected(captured, old, new):
    with pytest.raises(ValueError, match="Unsupported General judge"):
        formatting.prepare_verifier(captured["source"].replace(old, new).encode())


def test_other_verifiers_are_unchanged():
    source = b'def verify():\n    return {"reward": 0.25}\n'
    assert formatting.prepare_verifier(source) == (source, False)


def test_malformed_json_uses_only_existing_parse_attempts(captured, monkeypatch):
    source, _ = formatting.prepare_verifier(captured["source"].encode())
    for text in (captured["source"], source.decode()):
        grader = load_grader(text, monkeypatch)
        calls = []
        grader["_chat_judge_sc"] = lambda prompt: calls.append(prompt) or "not JSON"
        with pytest.raises(grader["_JudgeUnavailable"]):
            grader["_run_llm_sc"]([{"id": "item"}], None)
        assert len(calls) == 2


@pytest.mark.asyncio
async def test_upload_uses_private_copy_and_standalone_helper(captured, tmp_path, monkeypatch):
    import xiaomi_general.taskset as taskset

    root = tmp_path / "envs" / "captured"
    root.mkdir(parents=True)
    original = captured["source"].encode()
    (root / "verify.py").write_bytes(original)
    (root / "manifest.json").write_text(json.dumps({"verifier": {"uploads": [
        {"source": "verify.py", "target": "/work/verify.py"},
    ]}}))
    monkeypatch.setenv("GA_JUDGE_KEY", "fixture-key")
    monkeypatch.setattr(taskset, "run_shell", lambda *args, **kwargs: finished())
    writes = {}

    async def finished():
        return ProgramResult(0, "", "")

    class Runtime:
        async def write(self, path, content):
            writes[path] = content

        async def run(self, argv, env):
            return ProgramResult(0, "", "")

        async def read(self, path):
            return b'{"reward": 0.25}'

    task = GeneralTask(GeneralData(instance={"dataset_type": "general_agent", "env_task_dir": "envs/captured"}, task_root=str(tmp_path)))
    trace = SimpleNamespace(info={"general_submission_validated": True})
    assert await task.verifier_reward(trace, Runtime()) == 0.25
    assert (root / "verify.py").read_bytes() == original
    assert writes["/work/_verifier/_judge_format.py"] == Path(formatting.__file__).read_bytes()
    assert writes["/work/_verifier/verify.py"] != original
    private = tmp_path / "private"
    private.mkdir()
    for name in ("_judge_format.py", "verify.py"):
        (private / name).write_bytes(writes["/work/_verifier/" + name])
    result = subprocess.run([sys.executable, "-B", "-c", "import verify; print(callable(verify._run_llm_sc))"],
                            cwd=private, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "True"
