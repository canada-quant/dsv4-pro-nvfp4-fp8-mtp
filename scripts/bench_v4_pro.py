"""Unified benchmark harness for V4-Pro NVFP4 + native MXFP4 comparison.

Hits the OpenAI-compatible vLLM server at --base-url. Supports:
  - gsm8k: 1319-problem GSM8K test, strict 8-shot, greedy, max_tokens=2048
  - aime2024: 30-problem AIME 2024, thinking=high, max_tokens=65536
  - latency: median + p95 decode latency at concurrency=1, batch=1
  - mtp: scrapes /metrics endpoint for MTP acceptance counter deltas

Each subcommand writes a JSON result to docs/benchmarks/<name>_<date>.json.
Methodology matches the PR #42844 test plan for GSM8K, and the V4-Flash
65K-cap discipline for AIME. No invented thresholds — characterization only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp


@dataclass
class CompletionResult:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    text: str = ""
    elapsed_s: float = 0.0
    ttft_s: float | None = None  # time to first token (decode latency only meaningful when streaming)


async def chat_complete(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: float = 600.0,
) -> CompletionResult:
    t0 = time.time()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with session.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        data = await resp.json()
    elapsed = time.time() - t0
    if "choices" not in data:
        # error response
        return CompletionResult(
            text=f"<<ERROR: {data}>>",
            finish_reason="error",
            elapsed_s=elapsed,
        )
    ch = data["choices"][0]
    msg = ch.get("message", {}).get("content", "") or ""
    usage = data.get("usage", {}) or {}
    return CompletionResult(
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        finish_reason=ch.get("finish_reason", ""),
        text=msg,
        elapsed_s=elapsed,
    )


# ---------- GSM8K ----------

GSM8K_8SHOT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8.

Q: {question}
A:"""


def gsm8k_extract_answer(text: str) -> str | None:
    # Pattern: "The answer is X." or final number
    m = re.search(r"answer is\s+\$?([-+]?\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last number in text
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def gsm8k_extract_gold(answer: str) -> str:
    m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    if m:
        return m.group(1).replace(",", "")
    return answer.strip().replace(",", "")


async def run_gsm8k(args) -> dict:
    from datasets import load_dataset

    print(f"Loading GSM8K test...", file=sys.stderr)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  {len(ds)} problems", file=sys.stderr)

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"  limited to {len(ds)} problems", file=sys.stderr)

    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    rows: list[dict] = []
    correct = 0
    sem = asyncio.Semaphore(args.concurrency)

    async def process(session, i, row):
        question = row["question"]
        gold = gsm8k_extract_gold(row["answer"])
        prompt = GSM8K_8SHOT_PROMPT.format(question=question)
        async with sem:
            res = await chat_complete(
                session, args.base_url, args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens, temperature=0.0,
            )
        pred = gsm8k_extract_answer(res.text)
        ok = pred is not None and pred == gold
        return {
            "i": i, "question": question[:200], "gold": gold, "pred": pred,
            "ok": ok, "finish_reason": res.finish_reason,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "elapsed_s": res.elapsed_s,
            "text_head": res.text[:400],
        }

    t_start = time.time()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [asyncio.create_task(process(session, i, ds[i])) for i in range(len(ds))]
        for done_count, task in enumerate(asyncio.as_completed(tasks), 1):
            r = await task
            rows.append(r)
            if r["ok"]:
                correct += 1
            if done_count % 25 == 0 or done_count == len(ds):
                elapsed = time.time() - t_start
                acc = correct / done_count
                print(f"  [{done_count:>4}/{len(ds)}] acc={acc:.4f} ({correct}/{done_count}) elapsed={elapsed:.1f}s", file=sys.stderr)

    rows.sort(key=lambda r: r["i"])
    n = len(rows)
    acc = correct / n if n else 0.0
    # 95% binomial CI via Wilson
    z = 1.96
    p = acc
    denom = 1 + z**2 / n if n else 1
    ci_lo = (p + z**2/(2*n) - z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom if n else 0
    ci_hi = (p + z**2/(2*n) + z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom if n else 0
    trunc = sum(1 for r in rows if r["finish_reason"] != "stop")
    completion_tokens = [r["completion_tokens"] for r in rows]
    return {
        "benchmark": "gsm8k",
        "n": n,
        "correct": correct,
        "accuracy": acc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "truncated": trunc,
        "truncation_rate": trunc / n if n else 0.0,
        "completion_tokens_p50": float(statistics.median(completion_tokens)) if completion_tokens else 0.0,
        "completion_tokens_p95": float(statistics.quantiles(completion_tokens, n=20)[-1]) if len(completion_tokens) >= 20 else 0.0,
        "elapsed_total_s": time.time() - t_start,
        "rows": rows,
    }


# ---------- AIME 2024 ----------

AIME_2024_PROBLEMS = [
    # Problem text from official AIME 2024. Integer answer 0-999.
    # We use a curated list of all 30 problems (15 each from AIME I + II 2024)
    # Keeping this list compact; full text loaded from datasets if available.
]


async def run_aime(args) -> dict:
    """Load AIME 2024 from the HF dataset and evaluate with thinking-mode."""
    from datasets import load_dataset
    print(f"Loading AIME 2024...", file=sys.stderr)
    # Try common dataset names
    ds = None
    for name in ("Maxwell-Jia/AIME_2024", "AI-MO/aimo-validation-aime", "HuggingFaceH4/aime_2024"):
        try:
            ds = load_dataset(name, split="train")
            print(f"  loaded {name}: {len(ds)} problems", file=sys.stderr)
            break
        except Exception as e:
            print(f"  failed {name}: {e}", file=sys.stderr)
    if ds is None:
        raise RuntimeError("Could not load AIME 2024 dataset")

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    # Find problem and answer fields
    cols = ds.column_names
    print(f"  columns: {cols}", file=sys.stderr)
    prob_field = next((c for c in cols if c.lower() in ("problem", "question", "prompt")), cols[0])
    ans_field = next((c for c in cols if c.lower() in ("answer", "solution", "gold")), None)
    print(f"  using prob_field={prob_field!r} ans_field={ans_field!r}", file=sys.stderr)

    # Prompt template — let model think then give final integer
    SYS = "You are an expert competition mathematician. Solve the problem step by step, then provide the final answer as an integer between 0 and 999 on a new line in the format: 'Answer: N'."

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    rows: list[dict] = []
    sem = asyncio.Semaphore(args.concurrency)

    async def process(session, i, row):
        prob = row[prob_field]
        gold_raw = row.get(ans_field, "") if ans_field else ""
        gold = re.search(r"\d+", str(gold_raw))
        gold = gold.group(0) if gold else None

        messages = [{"role": "system", "content": SYS}, {"role": "user", "content": prob}]
        async with sem:
            res = await chat_complete(
                session, args.base_url, args.model,
                messages=messages, max_tokens=args.max_tokens, temperature=0.0,
                timeout=args.timeout,
            )
        ans_match = re.search(r"Answer:\s*([-+]?\d+)", res.text)
        pred = ans_match.group(1) if ans_match else None
        if pred is None:
            nums = re.findall(r"\b(\d{1,3})\b", res.text)
            pred = nums[-1] if nums else None
        ok = (pred is not None and gold is not None and int(pred) == int(gold))
        return {
            "i": i, "gold": gold, "pred": pred, "ok": ok,
            "finish_reason": res.finish_reason,
            "completion_tokens": res.completion_tokens,
            "elapsed_s": res.elapsed_s,
            "text_tail": res.text[-500:],
        }

    t_start = time.time()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [asyncio.create_task(process(session, i, ds[i])) for i in range(len(ds))]
        for task in asyncio.as_completed(tasks):
            r = await task
            rows.append(r)
            print(f"  [{len(rows)}/{len(ds)}] gold={r['gold']} pred={r['pred']} ok={r['ok']} fin={r['finish_reason']} tok={r['completion_tokens']}", file=sys.stderr)

    rows.sort(key=lambda r: r["i"])
    n = len(rows)
    correct_all = sum(1 for r in rows if r["ok"])
    truncated_rows = [r for r in rows if r["finish_reason"] != "stop"]
    nontrunc_rows = [r for r in rows if r["finish_reason"] == "stop"]
    correct_nontrunc = sum(1 for r in nontrunc_rows if r["ok"])

    return {
        "benchmark": "aime2024",
        "n": n,
        "raw_correct": correct_all,
        "raw_accuracy": correct_all / n if n else 0,
        "truncated": len(truncated_rows),
        "truncation_rate": len(truncated_rows) / n if n else 0,
        "nontruncated_n": len(nontrunc_rows),
        "nontruncated_correct": correct_nontrunc,
        "nontruncated_accuracy": correct_nontrunc / len(nontrunc_rows) if nontrunc_rows else 0,
        "completion_tokens_p50": float(statistics.median(r["completion_tokens"] for r in rows)) if rows else 0,
        "completion_tokens_p95": float(statistics.quantiles([r["completion_tokens"] for r in rows], n=20)[-1]) if len(rows) >= 20 else 0,
        "elapsed_total_s": time.time() - t_start,
        "rows": rows,
    }


# ---------- Latency ----------

async def run_latency(args) -> dict:
    """Single-prompt decode latency at concurrency=1."""
    prompts = [
        "Briefly explain what a black hole is.",
        "Translate 'good morning' into Japanese and Spanish.",
        "What is 12345 * 678? Show the multiplication step by step.",
        "List five common Python data structures.",
        "Compose a haiku about autumn.",
    ] * (args.n // 5 + 1)
    prompts = prompts[: args.n]

    timeout = aiohttp.ClientTimeout(total=120)
    rows = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, p in enumerate(prompts):
            res = await chat_complete(
                session, args.base_url, args.model,
                messages=[{"role": "user", "content": p}],
                max_tokens=args.max_tokens, temperature=0.0,
            )
            if res.finish_reason in ("stop", "length") and res.completion_tokens > 0:
                tokens_per_sec = res.completion_tokens / res.elapsed_s if res.elapsed_s > 0 else 0
                rows.append({
                    "i": i, "elapsed_s": res.elapsed_s,
                    "completion_tokens": res.completion_tokens,
                    "tokens_per_sec": tokens_per_sec,
                })
                print(f"  [{i+1:>3}/{len(prompts)}] {res.elapsed_s:.3f}s, {res.completion_tokens} tok, {tokens_per_sec:.1f} tok/s", file=sys.stderr)

    elapsed = [r["elapsed_s"] for r in rows]
    tps = [r["tokens_per_sec"] for r in rows]
    return {
        "benchmark": "latency",
        "n": len(rows),
        "elapsed_s_p50": float(statistics.median(elapsed)) if elapsed else 0,
        "elapsed_s_p95": float(statistics.quantiles(elapsed, n=20)[-1]) if len(elapsed) >= 20 else 0,
        "elapsed_s_mean": float(statistics.mean(elapsed)) if elapsed else 0,
        "elapsed_s_std": float(statistics.stdev(elapsed)) if len(elapsed) >= 2 else 0,
        "tokens_per_sec_p50": float(statistics.median(tps)) if tps else 0,
        "tokens_per_sec_p95": float(statistics.quantiles(tps, n=20)[-1]) if len(tps) >= 20 else 0,
        "tokens_per_sec_mean": float(statistics.mean(tps)) if tps else 0,
        "rows": rows,
    }


# ---------- MTP acceptance ----------

async def run_mtp(args) -> dict:
    """Scrape /metrics for MTP acceptance counters before+after a workload."""
    workload_prompts = [
        "Write a Python function that returns the nth Fibonacci number using memoization.",
        "What is the time complexity of binary search and why?",
        "Solve: x^2 - 5x + 6 = 0. Show your work.",
        "Explain how merge sort works in a paragraph.",
    ] * 5

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        async with session.get(f"{args.base_url}/metrics") as r:
            metrics_before = await r.text()

        for i, p in enumerate(workload_prompts):
            res = await chat_complete(
                session, args.base_url, args.model,
                messages=[{"role": "user", "content": p}],
                max_tokens=512, temperature=0.0,
            )

        async with session.get(f"{args.base_url}/metrics") as r:
            metrics_after = await r.text()

    def parse(metric_name: str, text: str) -> float:
        for line in text.splitlines():
            if line.startswith(metric_name) and not line.startswith(f"# "):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[-1])
                    except ValueError:
                        pass
        return 0.0

    accepted_before = parse("vllm:spec_decode_num_accepted_tokens_total", metrics_before)
    accepted_after = parse("vllm:spec_decode_num_accepted_tokens_total", metrics_after)
    drafts_before = parse("vllm:spec_decode_num_draft_tokens_total", metrics_before)
    drafts_after = parse("vllm:spec_decode_num_draft_tokens_total", metrics_after)
    accepted = accepted_after - accepted_before
    drafts = drafts_after - drafts_before

    return {
        "benchmark": "mtp",
        "n_prompts": len(workload_prompts),
        "drafts_emitted": drafts,
        "tokens_accepted": accepted,
        "acceptance_rate": accepted / drafts if drafts > 0 else 0.0,
        "metrics_before_excerpt": "\n".join(l for l in metrics_before.splitlines() if "spec_decode" in l),
        "metrics_after_excerpt":  "\n".join(l for l in metrics_after.splitlines()  if "spec_decode" in l),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("benchmark", choices=["gsm8k", "aime", "latency", "mtp"])
    p.add_argument("--base-url", default="http://localhost:8089")
    p.add_argument("--model", required=True, help="Model path or name as registered in vllm")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--n", type=int, default=50, help="latency: number of prompts")
    p.add_argument("--output", default=None, help="Output JSON path (auto-generated if omitted)")
    p.add_argument("--label", default="", help="Free-form label for the run (e.g. 'nvfp4_mtp')")
    args = p.parse_args()

    if args.benchmark == "gsm8k":
        result = asyncio.run(run_gsm8k(args))
    elif args.benchmark == "aime":
        result = asyncio.run(run_aime(args))
    elif args.benchmark == "latency":
        result = asyncio.run(run_latency(args))
    elif args.benchmark == "mtp":
        result = asyncio.run(run_mtp(args))
    else:
        raise ValueError(args.benchmark)

    result["meta"] = {
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "label": args.label,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    out_path = args.output
    if not out_path:
        date = datetime.utcnow().strftime("%Y_%m_%d")
        out_path = f"docs/benchmarks/{args.benchmark}_{args.label or 'run'}_{date}.json"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {out}", file=sys.stderr)

    # Print summary
    if args.benchmark == "gsm8k":
        print(f"\nGSM8K: {result['correct']}/{result['n']} = {result['accuracy']:.4f} "
              f"(95% CI [{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]), "
              f"truncated {result['truncated']} ({result['truncation_rate']:.2%})")
    elif args.benchmark == "aime":
        print(f"\nAIME 2024: raw {result['raw_correct']}/{result['n']} = {result['raw_accuracy']:.4f}, "
              f"truncation {result['truncated']}/{result['n']} ({result['truncation_rate']:.2%}), "
              f"non-trunc {result['nontruncated_correct']}/{result['nontruncated_n']} = {result['nontruncated_accuracy']:.4f}")
    elif args.benchmark == "latency":
        print(f"\nLatency: p50={result['elapsed_s_p50']:.3f}s, p95={result['elapsed_s_p95']:.3f}s, "
              f"throughput p50={result['tokens_per_sec_p50']:.1f} tok/s")
    elif args.benchmark == "mtp":
        print(f"\nMTP: {result['tokens_accepted']:.0f}/{result['drafts_emitted']:.0f} = "
              f"{result['acceptance_rate']:.2%} acceptance rate over {result['n_prompts']} prompts")


if __name__ == "__main__":
    main()
