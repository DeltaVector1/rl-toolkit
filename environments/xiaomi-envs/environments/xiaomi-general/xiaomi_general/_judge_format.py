"""Correct the released General judge format in private verifier copies."""

import ast
import hashlib
import json


def render_prompt(template, evidence, item_text, item_id):
    """Keep inserted task text separate from the response example."""
    before, rest = template.split("{evidence}")
    between, rest = rest.split("{items}")
    after, example = rest.rsplit("\n", 1)
    verdict = json.loads(example)["results"]["检查点id"]
    example = json.dumps({"results": {item_id: verdict}}, ensure_ascii=False)
    return before + evidence + between + item_text + after + "\n" + example


def verdict_score(parsed, item_id, item_ids):
    """Require one result key for the requested item."""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), dict):
        return None
    results = parsed["results"]
    alias = "[" + item_id + "]"
    matches = [key for key in (item_id, alias) if key in results]
    if len(matches) != 1:
        return None
    # A bracket alias cannot name another rubric item.
    if matches[0] == alias and alias in item_ids:
        return None
    verdict = results[matches[0]]
    return verdict.get("score") if isinstance(verdict, dict) else None


def prepare_verifier(data):
    """Guard the known source before any change."""
    source = data.decode("utf-8")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == "_run_llm_sc"]
    if not functions:
        return data, False
    expected = {
        "_run_llm_sc": "83a94336637b017826321c1442426b0deb1c304a5d89afcd177085200afc2d5b",
        "_extract_json_sc": "abb763032bedff98b7190d1fb71d00f2cce4069ce83127434acd8843024328e5",
    }
    for name, digest in expected.items():
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(nodes) != 1 or hashlib.sha256(ast.get_source_segment(source, nodes[0]).encode()).hexdigest() != digest:
            raise ValueError("Unsupported General judge source: " + name)
    prompts = [node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "_LLM_PROMPT" for target in node.targets)]
    if len(prompts) != 1 or hashlib.sha256(ast.literal_eval(prompts[0].value).encode()).hexdigest() != (
        "079dc5191345db36bd0e566ed8e1dfdb6c73855907e3d4b9511f8942d3a99235"
    ):
        raise ValueError("Unsupported General judge prompt")
    original = ast.get_source_segment(source, functions[0])
    replacements = {
        '_LLM_PROMPT.replace("{evidence}", ev_cache[key]).replace("{items}", item_txt)':
            '_general_judge_prompt(_LLM_PROMPT, ev_cache[key], item_txt, it["id"])',
        '(((j or {}).get("results") or {}).get(it["id"]) or {}).get("score")':
            '_general_judge_score(j, it["id"], {item["id"] for item in items})',
    }
    changed = original
    for old, new in replacements.items():
        if changed.count(old) != 1:
            raise ValueError("Unsupported General judge expression")
        changed = changed.replace(old, new, 1)
    imports = ("from _judge_format import render_prompt as _general_judge_prompt\n"
               "from _judge_format import verdict_score as _general_judge_score\n\n")
    if source.count(original) != 1:
        raise ValueError("Ambiguous General judge source")
    changed = source.replace(original, imports + changed, 1)
    ast.parse(changed)
    return changed.encode("utf-8"), True
