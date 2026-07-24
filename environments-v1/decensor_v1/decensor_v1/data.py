"""Decensor dataset assembly, ported from v0.

Loads the NewEden/RL-seed-Decensor seed conversations (local JSONL cache or HF streaming)
and the three system-prompt buckets (styles / plaintext / markdown), then assembles per-example
rows: an opening user turn wrapped in a bucket system prompt, an optional `<Response-Length>`
requirement, and a target number of model turns drawn from a geometric decay. The follow-up
user turns themselves are produced live by the user simulator, so only the opening turn is
materialized here (mirroring v0's `build_multiturn_dataset`).
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from decensor_v1.text import WORD_REQUIREMENTS


def convert_sharegpt_messages(convo: list[dict], strip_system: bool = False) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for turn in convo:
        if not isinstance(turn, dict):
            continue
        frm = str(turn.get("from", "")).lower()
        val = str(turn.get("value", "") or "").strip()
        if not val:
            continue
        if frm == "system":
            if strip_system:
                continue
            role = "system"
        elif frm in ("human", "user"):
            role = "user"
        elif frm in ("gpt", "assistant", "chatgpt", "bard"):
            role = "assistant"
        else:
            continue
        out.append({"role": role, "content": val})
    return out


def truncate_to_k_user_turns(messages: list[dict[str, str]], k: int) -> list[dict[str, str]] | None:
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


def sample_turn_counts(total: int, decay: float, max_turns: int, seed: int) -> list[int]:
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


def stratified_sample_by_key(rows_by_key: dict[str, list[Any]], target: int, seed: int) -> list[Any]:
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


def assign_buckets(n: int, styles_r: float, plaintext_r: float, markdown_r: float, seed: int) -> list[str]:
    total = styles_r + plaintext_r + markdown_r
    if total <= 0:
        raise ValueError("All bucket ratios are zero")
    styles_r, plaintext_r, markdown_r = styles_r / total, plaintext_r / total, markdown_r / total

    n_styles = round(n * styles_r)
    n_plaintext = round(n * plaintext_r)
    n_markdown = n - n_styles - n_plaintext
    if n_markdown < 0:
        if n_styles >= n_plaintext:
            n_styles += n_markdown
        else:
            n_plaintext += n_markdown
        n_markdown = 0

    assignments = ["styles"] * n_styles + ["plaintext"] * n_plaintext + ["markdown"] * n_markdown
    random.Random(seed).shuffle(assignments)
    return assignments


def load_bucket_prompts() -> dict[str, list[str]]:
    """Load the three system-prompt pools: styles, plaintext, markdown (from HF)."""
    out: dict[str, list[str]] = {}

    styles_path = hf_hub_download(
        repo_id="Delta-Vector/Tauri-RL-Styles", filename="system_prompts.txt", repo_type="dataset",
    )
    with open(styles_path, "r", encoding="utf-8") as f:
        out["styles"] = [line.strip() for line in f if line.strip()]

    plaintext_path = hf_hub_download(
        repo_id="Delta-Vector/Tauri-RL-Plaintext-System",
        filename="jailbreak_prompt_variants (1).jsonl", repo_type="dataset",
    )
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

    md_path = hf_hub_download(
        repo_id="Delta-Vector/Tauri-RL-Markdown-System",
        filename="jailbreak_prompt_variants_markdown.jsonl", repo_type="dataset",
    )
    md_prompts: list[str] = []
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = str(json.loads(line).get("prompt", "")).strip()
            if p:
                md_prompts.append(p)
    out["markdown"] = md_prompts

    for k, v in out.items():
        if not v:
            raise ValueError(f"Bucket '{k}' has no prompts loaded")
    return out


def decensor_candidate_paths(ds_name: str) -> list[str]:
    candidate_paths: list[str] = []
    if os.path.isfile(ds_name):
        candidate_paths.append(ds_name)
    elif os.path.isdir(ds_name):
        candidate_paths.append(os.path.join(ds_name, "rl.jsonl"))
    # Local v0 cache, if the port runs alongside the v0 checkout.
    candidate_paths.append("/root/prime/data/NewEden-RL-seed-Decensor/rl.jsonl")
    return candidate_paths


def iter_decensor_rows(ds_name: str):
    for path in decensor_candidate_paths(ds_name):
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid decensor JSONL at {path}:{line_no}: {e}") from e
        return
    yield from load_dataset(ds_name, split="train", streaming=True)


def build_decensor_pool(
    ds_name: str, target: int, decay: float, max_turns: int, seed: int,
) -> list[tuple[list[dict[str, str]], int]]:
    """Assemble (messages, K) pairs; messages ends at the K-th user turn."""
    if target <= 0:
        return []
    all_rows: list[dict[str, Any]] = []
    rows_by_asst_turns: dict[int, dict[str, list[int]]] = {}
    for row in iter_decensor_rows(ds_name):
        conv = row.get("conversations") or []
        n_asst = sum(1 for m in conv if isinstance(m, dict) and str(m.get("from", "")).lower() in ("gpt", "assistant"))
        harms = row.get("harm_types") or [row.get("harm_label") or "unknown"]
        primary = str(harms[0]) if harms else "unknown"
        idx = len(all_rows)
        all_rows.append({"conversations": conv})
        rows_by_asst_turns.setdefault(n_asst, {}).setdefault(primary, []).append(idx)

    alloc_per_k: dict[int, int] = {}
    for k in sample_turn_counts(target, decay, max_turns, seed):
        alloc_per_k[k] = alloc_per_k.get(k, 0) + 1

    out: list[tuple[list[dict[str, str]], int]] = []
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

        for row_idx in stratified_sample_by_key(candidate_rows_by_harm, need, seed + k * 7919):
            conv = all_rows[int(row_idx)].get("conversations") or []
            msgs = convert_sharegpt_messages(conv, strip_system=False)
            truncated = truncate_to_k_user_turns(msgs, k)
            if truncated is not None:
                out.append((truncated, k))
    random.Random(seed).shuffle(out)
    return out[:target]


def build_rows(
    ds_name: str,
    num_examples: int,
    seed: int,
    turn_decay_ratio: float,
    max_turns: int,
    bucket_styles_ratio: float,
    bucket_plaintext_ratio: float,
    bucket_markdown_ratio: float,
    enable_length_prompts: bool,
    output_prompt: str | None,
) -> list[dict[str, Any]]:
    """Return per-example dicts: system_prompt, user prompt, word_requirement, bucket, target_turns."""
    bucket_prompts = load_bucket_prompts()
    pool = build_decensor_pool(ds_name, num_examples, turn_decay_ratio, max_turns, seed + 2)
    buckets = assign_buckets(len(pool), bucket_styles_ratio, bucket_plaintext_ratio, bucket_markdown_ratio, seed + 101)

    rows: list[dict[str, Any]] = []
    for idx, ((convo_msgs, k), bucket) in enumerate(zip(pool, buckets)):
        bucket_pool = bucket_prompts[bucket]
        body = bucket_pool[idx % len(bucket_pool)]
        if bucket == "styles":
            sys_parts = [f"<Response-Style>\n{body}\n</Response-Style>"]
        else:
            sys_parts = [body]

        requirement_name = ""
        if enable_length_prompts:
            requirement_name = WORD_REQUIREMENTS[idx % len(WORD_REQUIREMENTS)][0]
            sys_parts.append(f"<Response-Length>\n{requirement_name}\n</Response-Length>")
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
                "prompt_bucket": bucket,
                "target_turns": k,
            }
        )

    random.Random(seed).shuffle(rows)
    return rows
