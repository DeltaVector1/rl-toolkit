from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

from decensor_v1.text import WORD_REQUIREMENTS


def convert_sharegpt_messages(convo: list[dict]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for turn in convo:
        if not isinstance(turn, dict):
            continue
        frm = str(turn.get("from", "")).lower()
        val = str(turn.get("value", "") or "").strip()
        if not val:
            continue
        if frm == "system":
            role = "system"
        elif frm in ("human", "user"):
            role = "user"
        elif frm in ("gpt", "assistant", "chatgpt", "bard"):
            role = "assistant"
        else:
            continue
        out.append({"role": role, "content": val})
    return out


def truncate_to_k_user_turns(
    messages: list[dict[str, str]], k: int
) -> list[dict[str, str]] | None:
    if k < 1:
        return None
    msgs = [m for m in messages if m.get("role") != "system"]
    user_seen = 0
    cut = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            user_seen += 1
            if user_seen == k:
                cut = i
                break
    if cut < 0:
        return None
    return msgs[: cut + 1]


def sample_turn_counts(
    total: int, decay: float, max_turns: int, seed: int
) -> list[int]:
    if total <= 0:
        return []
    weights = [decay ** (k - 1) for k in range(1, max_turns + 1)]
    s = sum(weights)
    weights = [w / s for w in weights]

    counts: list[int] = []
    remaining = total
    for i, w in enumerate(weights):
        if i == len(weights) - 1:
            counts.append(remaining)
        else:
            c = round(total * w)
            counts.append(c)
            remaining -= c
    if remaining < 0:
        j = counts.index(max(counts))
        counts[j] += remaining

    out: list[int] = []
    for k_idx, c in enumerate(counts):
        out.extend([k_idx + 1] * c)
    random.Random(seed).shuffle(out)
    return out


def stratified_sample_by_key(
    rows_by_key: dict[str, list[Any]], target: int, seed: int
) -> list[Any]:
    rng = random.Random(seed)
    keys = sorted(rows_by_key.keys())
    if not keys or target <= 0:
        return []
    base = target // len(keys)
    rem = target - base * len(keys)
    selected: list[Any] = []
    for i, k in enumerate(keys):
        pool = list(rows_by_key[k])
        if not pool:
            continue
        rng.shuffle(pool)
        want = base + (1 if i < rem else 0)
        if want <= len(pool):
            selected.extend(pool[:want])
        else:
            selected.extend(pool)
            selected.extend(rng.choices(pool, k=(want - len(pool))))
    rng.shuffle(selected)
    return selected


def assign_response_formats(
    count: int, markdown_probability: float, seed: int
) -> list[str]:
    if not 0.0 <= markdown_probability <= 1.0:
        raise ValueError("markdown_probability must be between zero and one")
    markdown_count = round(count * markdown_probability)
    formats = ["markdown"] * markdown_count + ["plaintext"] * (count - markdown_count)
    random.Random(seed).shuffle(formats)
    return formats


def assign_style_flags(count: int, style_probability: float, seed: int) -> list[bool]:
    if not 0.0 <= style_probability <= 1.0:
        raise ValueError("style_probability must be between zero and one")
    style_count = round(count * style_probability)
    styles = [True] * style_count + [False] * (count - style_count)
    random.Random(seed).shuffle(styles)
    return styles


def load_system_prompts() -> dict[str, list[str]]:
    data_root = Path(__file__).resolve().parents[2] / "data"
    styles_path = data_root / "decensor_styles" / "system_prompts.txt"
    plaintext_path = (
        data_root / "decensor_plaintext_system_prompts" / "system_prompts.jsonl"
    )
    markdown_path = (
        data_root / "decensor_markdown_system_prompts" / "system_prompts.jsonl"
    )
    paths = (styles_path, plaintext_path, markdown_path)
    missing = [path.as_posix() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing system prompt files: {', '.join(missing)}")

    out: dict[str, list[str]] = {}
    with open(styles_path, "r", encoding="utf-8") as f:
        out["style"] = [line.strip() for line in f if line.strip()]

    pt_prompts: list[str] = []
    with open(plaintext_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = str(json.loads(line).get("prompt", "")).strip()
            if p:
                pt_prompts.append(p)
    out["plaintext"] = pt_prompts

    markdown_prompts: list[str] = []
    with open(markdown_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = str(json.loads(line).get("prompt", "")).strip()
            if p:
                markdown_prompts.append(p)
    out["markdown"] = markdown_prompts

    for k, v in out.items():
        if not v:
            raise ValueError(f"Prompt group '{k}' has no prompts loaded")
    return out


def iter_decensor_rows(ds_name: str):
    path = ds_name
    if not path.endswith(".jsonl"):
        path = f"{path.rstrip('/')}/rl.jsonl"
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid decensor JSONL at {path}:{line_no}: {e}"
                    ) from e
        return
    except FileNotFoundError:
        pass
    yield from load_dataset(ds_name, split="train", streaming=True)


def build_decensor_pool(
    ds_name: str,
    target: int,
    decay: float,
    max_turns: int,
    seed: int,
) -> list[tuple[list[dict[str, str]], int, str, int]]:
    if target <= 0:
        return []
    all_rows: list[dict[str, Any]] = []
    rows_by_asst_turns: dict[int, dict[str, list[int]]] = {}
    for row in iter_decensor_rows(ds_name):
        conv = row.get("conversations") or []
        n_asst = sum(
            1
            for m in conv
            if isinstance(m, dict)
            and str(m.get("from", "")).lower() in ("gpt", "assistant")
        )
        harms = row.get("harm_types") or [row.get("harm_label") or "unknown"]
        primary = str(harms[0]) if harms else "unknown"
        idx = len(all_rows)
        all_rows.append({"conversations": conv, "harm_type": primary})
        rows_by_asst_turns.setdefault(n_asst, {}).setdefault(primary, []).append(idx)

    alloc_per_k: dict[int, int] = {}
    for k in sample_turn_counts(target, decay, max_turns, seed):
        alloc_per_k[k] = alloc_per_k.get(k, 0) + 1

    out: list[tuple[list[dict[str, str]], int, str, int]] = []
    for k, need in alloc_per_k.items():
        candidate_rows_by_harm: dict[str, list[int]] = {}
        for asst_count, harm_map in rows_by_asst_turns.items():
            if asst_count < k:
                continue
            if k == 1 and asst_count != 1:
                continue
            for harm, idxs in harm_map.items():
                candidate_rows_by_harm.setdefault(harm, []).extend(idxs)
        if not candidate_rows_by_harm:
            for asst_count, harm_map in rows_by_asst_turns.items():
                if asst_count < k:
                    continue
                for harm, idxs in harm_map.items():
                    candidate_rows_by_harm.setdefault(harm, []).extend(idxs)

        for row_idx in stratified_sample_by_key(
            candidate_rows_by_harm, need, seed + k * 7919
        ):
            conv = all_rows[int(row_idx)].get("conversations") or []
            msgs = convert_sharegpt_messages(conv)
            truncated = truncate_to_k_user_turns(msgs, k)
            if truncated is not None:
                out.append(
                    (
                        truncated,
                        k,
                        str(all_rows[int(row_idx)]["harm_type"]),
                        int(row_idx),
                    )
                )
    random.Random(seed).shuffle(out)
    return out[:target]


def build_rows(
    ds_name: str,
    num_examples: int,
    seed: int,
    turn_decay_ratio: float,
    max_turns: int,
    style_probability: float,
    markdown_probability: float,
    enable_length_prompts: bool,
    output_prompt: str | None,
) -> list[dict[str, Any]]:
    system_prompts = load_system_prompts()
    pool = build_decensor_pool(
        ds_name, num_examples, turn_decay_ratio, max_turns, seed + 2
    )
    response_formats = assign_response_formats(
        len(pool),
        markdown_probability,
        seed + 101,
    )
    style_flags = assign_style_flags(len(pool), style_probability, seed + 202)
    source_row_counts = Counter(source_row_index for _, _, _, source_row_index in pool)

    rows: list[dict[str, Any]] = []
    for idx, (
        (convo_msgs, k, harm_type, source_row_index),
        response_format,
        has_style,
    ) in enumerate(zip(pool, response_formats, style_flags)):
        format_prompts = system_prompts[response_format]
        sys_parts = [format_prompts[idx % len(format_prompts)]]
        if has_style:
            style = system_prompts["style"][idx % len(system_prompts["style"])]
            sys_parts.append(f"<Response-Style>\n{style}\n</Response-Style>")

        requirement_name = ""
        word_target = 0
        if enable_length_prompts:
            requirement_name = WORD_REQUIREMENTS[idx % len(WORD_REQUIREMENTS)][0]
            sys_parts.append(
                f"<Response-Length>\n{requirement_name}\n</Response-Length>"
            )
        if output_prompt:
            sys_parts.insert(0, output_prompt)

        user_msgs = [m for m in convo_msgs if m.get("role") == "user"]
        if not user_msgs:
            continue
        rows.append(
            {
                "system_prompt": "\n\n".join(sys_parts),
                "prompt": user_msgs[0]["content"],
                "word_requirement": requirement_name,
                "word_target": word_target,
                "response_format": response_format,
                "has_style": has_style,
                "target_turns": k,
                "harm_type": harm_type,
                "source_row_index": source_row_index,
                "source_row_repeated": source_row_counts[source_row_index] > 1,
            }
        )

    random.Random(seed).shuffle(rows)
    return rows
