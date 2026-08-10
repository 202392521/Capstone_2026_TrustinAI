#!/usr/bin/env python3
"""Concurrent frozen Prompt V2 annotation for a candidate comment set.

The prompt, model, input fields and output schema are inherited from the locked
healthcare holdout runner. Concurrency is only an engineering optimisation.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm


OUTPUTS = Path(os.environ.get("REPRO_OUTPUTS_DIR", "outputs"))
HOLDOUT_RUNNER = OUTPUTS / "run_healthcare_holdout50_v2_annotation_and_eval.py"
PROMPT_V2_FILE = OUTPUTS / "gpt5_mini_trust_stance_prompt_v2" / "prompt_v2.txt"


def import_holdout_module() -> Any:
    spec = importlib.util.spec_from_file_location("holdout_v2", HOLDOUT_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import holdout runner from {HOLDOUT_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def normalise_boolish(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return "1"
    if text in {"0", "false", "no", "n"}:
        return "0"
    return ""


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "industry" not in df.columns:
        if "source_dataset" in df.columns:
            df["industry"] = df["source_dataset"]
        elif "industry_keyword" in df.columns:
            df["industry"] = df["industry_keyword"]
        else:
            df["industry"] = ""
    if "parent_comment" not in df.columns and "parent_comment_body" in df.columns:
        df["parent_comment"] = df["parent_comment_body"]
    if "target_comment" not in df.columns:
        df["target_comment"] = df["comment_body"] if "comment_body" in df.columns else ""
    if "country_group" not in df.columns:
        if "uk_strong_relevant" in df.columns:
            df["country_group"] = df["uk_strong_relevant"].map(
                lambda x: "UK_strong" if normalise_boolish(x) == "1" else "non_UK_or_unclear"
            )
        else:
            df["country_group"] = "unknown"
    if "UK_status" not in df.columns:
        df["UK_status"] = df["country_group"]
    if "comment_id" not in df.columns:
        df["comment_id"] = ""
    df["comment_id"] = df["comment_id"].astype(str)
    missing_id = df["comment_id"].str.strip() == ""
    df.loc[missing_id, "comment_id"] = [f"row_{i + 1}" for i in df.index[missing_id]]
    df["annotation_id"] = df["comment_id"]
    duplicated = df["annotation_id"].duplicated(keep=False)
    df.loc[duplicated, "annotation_id"] = [
        f"{cid}__row_{idx + 1}" for idx, cid in zip(df.index[duplicated], df.loc[duplicated, "comment_id"])
    ]
    return df


def compact_context(row: pd.Series, helper: Any) -> dict[str, str]:
    return {
        "industry": helper.clean_text(row.get("industry", ""), 120),
        "subreddit": helper.clean_text(row.get("subreddit", ""), 120),
        "post_title": helper.clean_text(row.get("post_title", ""), 500),
        "parent_comment": helper.clean_text(row.get("parent_comment", ""), 700),
        "comment_body": helper.clean_text(row.get("comment_body", row.get("target_comment", "")), 1400),
        "stance_context_text": helper.clean_text(row.get("stance_context_text", ""), 1200),
    }


def parse_response_text(response: Any) -> str:
    text = getattr(response, "output_text", "")
    if text:
        return text
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", "")
            if value:
                chunks.append(value)
    return "\n".join(chunks)


def response_to_jsonable(response: Any) -> Any:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return str(response)


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": "", "output_tokens": "", "total_tokens": ""}
    return {
        "input_tokens": getattr(usage, "input_tokens", ""),
        "output_tokens": getattr(usage, "output_tokens", ""),
        "total_tokens": getattr(usage, "total_tokens", ""),
    }


def build_user_prompt(item: dict[str, str]) -> str:
    return json.dumps(
        {
            "item_to_annotate": item,
            "instruction": "Return one JSON object matching the schema. Apply frozen Prompt V2 exactly. Do not infer human labels from any notes; none are provided.",
        },
        ensure_ascii=False,
        indent=2,
    )


async def call_model(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    item: dict[str, str],
    schema: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], str, Any]:
    response = await client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(item)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "frozen_prompt_v2_full_corpus_annotation",
                "schema": schema,
                "strict": True,
            }
        },
        timeout=timeout,
    )
    raw = parse_response_text(response)
    return json.loads(raw), raw, response


def append_jsonl_sync(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_raw(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def done_annotation_ids(raw_path: Path) -> set[str]:
    raw = load_raw(raw_path)
    if raw.empty or "parse_status" not in raw.columns:
        return set()
    ok = raw[raw["parse_status"].astype(str) == "ok"]
    return set(ok["annotation_id"].astype(str))


def prediction_rows(raw: pd.DataFrame, label_fields: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, raw_row in raw.iterrows():
        parsed = raw_row.get("parsed_labels", {})
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except Exception:
                parsed = {}
        gates = raw_row.get("helper_bool_gate", {})
        if isinstance(gates, str):
            try:
                gates = json.loads(gates)
            except Exception:
                gates = {}
        out = {
            "annotation_id": raw_row.get("annotation_id", ""),
            "comment_id": raw_row.get("comment_id", ""),
            "model": raw_row.get("model", ""),
            "api_status": raw_row.get("api_status", ""),
            "parse_status": raw_row.get("parse_status", ""),
            "retry_count": raw_row.get("retry_count", ""),
            "latency_seconds": raw_row.get("latency_seconds", ""),
            "input_tokens": raw_row.get("input_tokens", ""),
            "output_tokens": raw_row.get("output_tokens", ""),
            "total_tokens": raw_row.get("total_tokens", ""),
            "industry": raw_row.get("industry", ""),
            "subreddit": raw_row.get("subreddit", ""),
            "country_group": raw_row.get("country_group", ""),
            "UK_status": raw_row.get("UK_status", ""),
            "post_title": raw_row.get("post_title", ""),
            "parent_comment": raw_row.get("parent_comment", ""),
            "target_comment": raw_row.get("comment_body", ""),
        }
        for field in label_fields:
            out[f"gpt_{field}"] = parsed.get(field, "")
        out["substantive_trust_content"] = gates.get("has_substantive_trust_content", "")
        out["trust_boundary"] = gates.get("has_explicit_trust_boundary", "")
        out["gpt_has_substantive_trust_content"] = out["substantive_trust_content"]
        out["gpt_has_explicit_trust_boundary"] = out["trust_boundary"]
        out["gpt_annotation_confidence"] = parsed.get("annotation_confidence", "")
        out["gpt_brief_reason"] = parsed.get("brief_reason", "")
        out["gpt_evidence_quote"] = parsed.get("evidence_quote", "")
        rows.append(out)
    return pd.DataFrame(rows)


def write_outputs(candidate_source: pd.DataFrame, raw_path: Path, output_dir: Path, helper: Any, full_audit: Path | None) -> None:
    raw = load_raw(raw_path)
    pred = prediction_rows(raw, helper.LABEL_FIELDS)
    pred.to_csv(output_dir / "candidate_comments_v2_annotated.csv", index=False)

    if full_audit is not None and full_audit.exists():
        base = prepare_input(read_csv(full_audit))
    else:
        base = candidate_source.copy()

    merge_cols = [
        "annotation_id",
        "comment_id",
        "model",
        "api_status",
        "parse_status",
        "retry_count",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "substantive_trust_content",
        "trust_boundary",
        "gpt_has_substantive_trust_content",
        "gpt_has_explicit_trust_boundary",
        "gpt_human_ai_relevance",
        "gpt_human_attitude",
        "gpt_human_attitude_target",
        "gpt_human_use",
        "gpt_human_capability_assessment",
        "gpt_human_trust_construct",
        "gpt_human_trust_boundary",
        "gpt_human_evidence",
        "gpt_annotation_confidence",
        "gpt_brief_reason",
        "gpt_evidence_quote",
    ]
    keep_cols = [col for col in merge_cols if col in pred.columns]
    full = base.merge(pred[keep_cols], on=["annotation_id", "comment_id"], how="left", suffixes=("", "_pred"))
    if "gpt_candidate_union" in full.columns:
        full["screening_status"] = "prefilter_negative"
        full.loc[full["gpt_candidate_union"].astype(str) == "1", "screening_status"] = "candidate_pending_or_failed"
        full.loc[full["parse_status"].astype(str) == "ok", "screening_status"] = "gpt_annotated"
    else:
        full["screening_status"] = full["parse_status"].map(lambda x: "gpt_annotated" if str(x) == "ok" else "candidate_pending_or_failed")
    full.to_csv(output_dir / "all_comments_v2_annotated.csv", index=False)

    trust_positive = full[full["substantive_trust_content"].astype(str) == "yes"].copy()
    trust_positive.to_csv(output_dir / "trust_positive_corpus.csv", index=False)

    summary_rows = []
    for industry, group in full.groupby("industry", dropna=False):
        total = len(group)
        annotated = int(group["parse_status"].astype(str).eq("ok").sum())
        candidate = int(group.get("gpt_candidate_union", pd.Series(["1"] * len(group), index=group.index)).astype(str).eq("1").sum())
        trust_n = int((group["substantive_trust_content"].astype(str) == "yes").sum())
        boundary_n = int((group["trust_boundary"].astype(str) == "yes").sum())
        summary_rows.append(
            {
                "industry": industry,
                "total_comments": total,
                "gpt_candidate_n": candidate,
                "gpt_annotated_n": annotated,
                "trust_positive_n": trust_n,
                "trust_positive_rate": round(trust_n / total, 4) if total else 0,
                "trust_positive_rate_among_annotated": round(trust_n / annotated, 4) if annotated else 0,
                "boundary_positive_n": boundary_n,
                "boundary_rate_among_trust_comments": round(boundary_n / trust_n, 4) if trust_n else 0,
            }
        )
    pd.DataFrame(summary_rows).sort_values("industry").to_csv(output_dir / "industry_trust_summary.csv", index=False)

    api_failures = int((raw.get("api_status", pd.Series(dtype=str)) != "ok").sum()) if len(raw) else 0
    invalid_json = int((raw.get("parse_status", pd.Series(dtype=str)) != "ok").sum()) if len(raw) else 0
    total_input = pd.to_numeric(raw.get("input_tokens", pd.Series(dtype=str)), errors="coerce").fillna(0).sum()
    total_output = pd.to_numeric(raw.get("output_tokens", pd.Series(dtype=str)), errors="coerce").fillna(0).sum()
    pd.DataFrame(
        [
            {"metric": "base_rows", "value": len(base)},
            {"metric": "candidate_rows", "value": len(candidate_source)},
            {"metric": "raw_response_rows", "value": len(raw)},
            {"metric": "annotated_rows", "value": int(full["parse_status"].astype(str).eq("ok").sum())},
            {"metric": "trust_positive_rows", "value": len(trust_positive)},
            {"metric": "api_failures", "value": api_failures},
            {"metric": "invalid_json", "value": invalid_json},
            {"metric": "total_input_tokens", "value": int(total_input)},
            {"metric": "total_output_tokens", "value": int(total_output)},
        ]
    ).to_csv(output_dir / "full_corpus_v2_run_summary.csv", index=False)


async def annotate_one(
    row: pd.Series,
    client: AsyncOpenAI,
    helper: Any,
    prompt: str,
    schema: dict[str, Any],
    args: argparse.Namespace,
    raw_path: Path,
    lock: asyncio.Lock,
) -> None:
    item = compact_context(row, helper)
    api_status = "failed"
    parse_status = "failed"
    parsed: dict[str, Any] = {}
    raw_text = ""
    response_json: Any = {}
    usage = {"input_tokens": "", "output_tokens": "", "total_tokens": ""}
    failure_reason = ""
    retry_count = 0
    started = time.time()
    for attempt in range(args.retries + 1):
        retry_count = attempt
        try:
            parsed, raw_text, response = await call_model(client, args.model, prompt, item, schema, args.timeout)
            response_json = response_to_jsonable(response)
            usage = usage_dict(response)
            api_status = "ok"
            parse_status = "ok"
            break
        except Exception as exc:
            failure_reason = str(exc)
            await asyncio.sleep(min(args.max_backoff, args.backoff_base * (2**attempt)))

    row_out = {
        "annotation_id": row.get("annotation_id", ""),
        "comment_id": row.get("comment_id", ""),
        "model": args.model,
        "country_group": row.get("country_group", ""),
        "UK_status": row.get("UK_status", ""),
        **item,
        "api_status": api_status,
        "parse_status": parse_status,
        "retry_count": retry_count,
        "latency_seconds": round(time.time() - started, 3),
        **usage,
        "failure_reason": failure_reason,
        "raw_text": raw_text,
        "parsed_labels": parsed,
        "helper_bool_gate": {
            "has_substantive_trust_content": helper.bool_gate(parsed.get("has_substantive_trust_content", "")),
            "has_explicit_trust_boundary": helper.bool_gate(parsed.get("has_explicit_trust_boundary", "")),
        },
        "response": response_json,
    }
    async with lock:
        append_jsonl_sync(raw_path, row_out)
    if args.pause:
        await asyncio.sleep(args.pause)


async def run_async(args: argparse.Namespace) -> int:
    helper = import_holdout_module()
    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = prepare_input(read_csv(input_path))
    if args.limit:
        source = source.head(args.limit).copy()
    raw_path = output_dir / "raw_api_responses.jsonl"
    done = done_annotation_ids(raw_path)
    pending = source[~source["annotation_id"].astype(str).isin(done)].copy()

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(input_path),
        "full_audit_file": str(Path(args.full_audit).expanduser()) if args.full_audit else "",
        "output_dir": str(output_dir),
        "final_prompt": "Prompt V2",
        "prompt_file": str(PROMPT_V2_FILE),
        "model": args.model,
        "workers": args.workers,
        "retries": args.retries,
        "timeout": args.timeout,
        "input_policy": "Same as healthcare holdout: industry, subreddit, post title, parent comment, target comment, stance context; no human labels or notes.",
        "candidate_rows": len(source),
        "already_annotated": len(done),
        "pending_rows": len(pending),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Candidate rows selected: {len(source)}")
    print(f"Already annotated: {len(done)}")
    print(f"Pending rows: {len(pending)}")
    print(f"Workers: {args.workers}")
    print(f"Output dir: {output_dir}")
    print(f"Raw JSONL: {raw_path}")

    if args.write_outputs_only:
        full_audit = Path(args.full_audit).expanduser() if args.full_audit else None
        write_outputs(source, raw_path, output_dir, helper, full_audit)
        return 0

    if len(pending) and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is unavailable in this terminal and pending rows remain.", file=sys.stderr)
        return 2

    if len(pending):
        prompt = PROMPT_V2_FILE.read_text(encoding="utf-8")
        schema = helper.build_schema()
        client = AsyncOpenAI()
        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(args.workers)
        progress = tqdm(total=len(pending), desc="Frozen Prompt V2 concurrent")

        async def bounded(row: pd.Series) -> None:
            async with semaphore:
                await annotate_one(row, client, helper, prompt, schema, args, raw_path, lock)
                progress.update(1)

        tasks = [asyncio.create_task(bounded(row)) for _, row in pending.iterrows()]
        await asyncio.gather(*tasks)
        progress.close()

    full_audit = Path(args.full_audit).expanduser() if args.full_audit else None
    write_outputs(source, raw_path, output_dir, helper, full_audit)
    print(f"Updated {output_dir / 'all_comments_v2_annotated.csv'}")
    print(f"Updated {output_dir / 'trust_positive_corpus.csv'}")
    print(f"Updated {output_dir / 'industry_trust_summary.csv'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent frozen Prompt V2 annotation.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--full-audit", default="")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--pause", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--backoff-base", type=float, default=2.0)
    parser.add_argument("--max-backoff", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--write-outputs-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
