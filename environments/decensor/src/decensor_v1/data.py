from __future__ import annotations

import random
from collections import Counter
from typing import Any

from datasets import load_dataset

from decensor_v1.text import word_requirements


decensor_dataset = "NewEden/RL-seed-Decensor-Difficulty"
markdown_system_prompts_dataset = "Delta-Vector/Tauri-RL-Markdown-System-V2"
plaintext_system_prompts_dataset = "Delta-Vector/Tauri-RL-Plaintext-System-V2"
style_prompts_dataset = "Delta-Vector/Tauri-RL-Styles"


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


def _load_text_column(dataset_name: str, column: str) -> list[str]:
    dataset = load_dataset(dataset_name, split="train")
    if column not in dataset.column_names:
        raise ValueError(f"{dataset_name} has no '{column}' column")
    values = [str(value).strip() for value in dataset[column] if str(value).strip()]
    if not values:
        raise ValueError(f"{dataset_name} has no usable '{column}' values")
    return values


def load_system_prompts() -> dict[str, list[str]]:
    return {
        "markdown": _load_text_column(markdown_system_prompts_dataset, "prompt"),
        "plaintext": _load_text_column(plaintext_system_prompts_dataset, "prompt"),
        "style": _load_text_column(style_prompts_dataset, "text"),
    }


def iter_decensor_rows(difficulty: str):
    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError("difficulty must be easy, medium, or hard")
    columns = [
        "source_index",
        "prompt",
        "primary_harm_type_code",
        "bucket_ID",
    ]
    dataset = load_dataset(decensor_dataset, split="train", streaming=True)
    dataset = dataset.select_columns(columns)
    for row in dataset:
        if row["bucket_ID"] == difficulty:
            yield row


def build_decensor_pool(
    target: int,
    decay: float,
    max_turns: int,
    seed: int,
    difficulty: str,
) -> list[tuple[list[dict[str, str]], int, str, int]]:
    if target <= 0:
        return []
    all_rows: list[dict[str, Any]] = []
    rows_by_harm: dict[str, list[int]] = {}
    for row in iter_decensor_rows(difficulty):
        prompt = row.get("prompt")
        if (
            not isinstance(prompt, list)
            or len(prompt) != 1
            or not isinstance(prompt[0], dict)
            or prompt[0].get("role") != "user"
            or not isinstance(prompt[0].get("content"), str)
            or not prompt[0]["content"].strip()
        ):
            raise ValueError("each difficulty row must contain one user prompt")
        harm_type = row.get("primary_harm_type_code")
        if not isinstance(harm_type, str) or not harm_type:
            raise ValueError("each difficulty row must contain a primary harm type")
        source_index = row.get("source_index")
        if not isinstance(source_index, int):
            raise ValueError("each difficulty row must contain an integer source index")
        idx = len(all_rows)
        all_rows.append(
            {
                "prompt": prompt[0]["content"].strip(),
                "harm_type": harm_type,
                "source_index": source_index,
            }
        )
        rows_by_harm.setdefault(harm_type, []).append(idx)

    if not all_rows:
        raise ValueError(f"the difficulty dataset has no {difficulty} rows")

    selected = stratified_sample_by_key(rows_by_harm, target, seed + 7919)
    turn_counts = sample_turn_counts(target, decay, max_turns, seed)
    if len(selected) != target or len(turn_counts) != target:
        raise ValueError("could not build the requested number of tasks")

    out: list[tuple[list[dict[str, str]], int, str, int]] = []
    for row_idx, turn_count in zip(selected, turn_counts, strict=True):
        row = all_rows[int(row_idx)]
        out.append(
            (
                [{"role": "user", "content": str(row["prompt"])}],
                turn_count,
                str(row["harm_type"]),
                int(row["source_index"]),
            )
        )
    random.Random(seed).shuffle(out)
    return out


def build_rows(
    num_examples: int,
    seed: int,
    turn_decay_ratio: float,
    max_turns: int,
    style_probability: float,
    markdown_probability: float,
    enable_length_prompts: bool,
    output_prompt: str | None,
    difficulty: str,
) -> list[dict[str, Any]]:
    system_prompts = load_system_prompts()
    pool = build_decensor_pool(
        num_examples,
        turn_decay_ratio,
        max_turns,
        seed + 2,
        difficulty,
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
            requirement_name = word_requirements[idx % len(word_requirements)][0]
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
