"""Combine NVFP4 + native MXFP4 benchmark JSONs into a comparison table.

Usage:
  python scripts/compare_bench_results.py <dir>

Looks for files matching:
  <bench>_<label>_<date>.json  with label in {nvfp4_v01, native_mxfp4}

Emits Markdown table to stdout suitable for MODEL_CARD.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def load_results(d: Path) -> dict[str, dict[str, dict]]:
    """Return {bench_name: {label: result_dict}}."""
    results: dict[str, dict[str, dict]] = {}
    for f in sorted(d.glob("*.json")):
        stem = f.stem
        # bench_label_date — date is YYYY_MM_DD
        parts = stem.rsplit("_", 3)
        if len(parts) < 4:
            continue
        bench, label, *_ = stem.split("_", 1)
        # Re-split properly: bench is first token, label is middle, date is last 3
        toks = stem.split("_")
        if len(toks) >= 5:
            bench = toks[0]
            label = "_".join(toks[1:-3])
            date = "_".join(toks[-3:])
        else:
            continue
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"skip {f}: {e}", file=sys.stderr)
            continue
        results.setdefault(bench, {})[label] = data
    return results


def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def fmt_seconds(x: float) -> str:
    return f"{x:.3f}s"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path, nargs="?", default=Path("docs/benchmarks"))
    args = p.parse_args()

    results = load_results(args.dir)
    if not results:
        print(f"No results in {args.dir}", file=sys.stderr)
        sys.exit(1)

    print(f"# V4-Pro NVFP4 vs native MXFP4 — benchmark comparison\n")
    print(f"Source: `{args.dir}`\n")

    # GSM8K
    if "gsm8k" in results:
        print("## GSM8K (strict 8-shot, max_tokens=2048, greedy)\n")
        print("| label | n | correct | accuracy | 95% CI (Wilson) | truncated | tot wall |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for label in sorted(results["gsm8k"]):
            r = results["gsm8k"][label]
            acc = r.get("accuracy", 0)
            ci = f"[{r.get('ci_lo', 0):.4f}, {r.get('ci_hi', 0):.4f}]"
            trunc = f"{r.get('truncated', 0)} ({fmt_pct(r.get('truncation_rate', 0))})"
            wall = fmt_seconds(r.get("elapsed_total_s", 0))
            print(f"| {label} | {r.get('n','?')} | {r.get('correct','?')} | {acc:.4f} | {ci} | {trunc} | {wall} |")
        print()

    # AIME 2024
    if "aime" in results:
        print("## AIME 2024 (thinking mode, max_tokens=65536)\n")
        print("| label | n | raw correct | raw pass@1 | non-trunc n | non-trunc correct | non-trunc pass@1 | truncated | tokens p50 / p95 |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for label in sorted(results["aime"]):
            r = results["aime"][label]
            print(f"| {label} | {r.get('n','?')} | {r.get('raw_correct','?')} | {r.get('raw_accuracy', 0):.4f} | {r.get('nontruncated_n','?')} | {r.get('nontruncated_correct','?')} | {r.get('nontruncated_accuracy', 0):.4f} | {r.get('truncated','?')} ({fmt_pct(r.get('truncation_rate', 0))}) | {int(r.get('completion_tokens_p50', 0))} / {int(r.get('completion_tokens_p95', 0))} |")
        print()

    # Latency
    if "latency" in results:
        print("## Decode latency (n=50, concurrency=1, max_tokens=256)\n")
        print("| label | n | elapsed p50 | p95 | mean (std) | tokens/s p50 | p95 |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for label in sorted(results["latency"]):
            r = results["latency"][label]
            mean_std = f"{fmt_seconds(r.get('elapsed_s_mean', 0))} ({fmt_seconds(r.get('elapsed_s_std', 0))})"
            print(f"| {label} | {r.get('n','?')} | {fmt_seconds(r.get('elapsed_s_p50', 0))} | {fmt_seconds(r.get('elapsed_s_p95', 0))} | {mean_std} | {r.get('tokens_per_sec_p50', 0):.1f} | {r.get('tokens_per_sec_p95', 0):.1f} |")
        print()

    # MTP
    if "mtp" in results:
        print("## MTP draft acceptance (20-prompt mixed workload)\n")
        print("| label | drafts emitted | accepted | acceptance rate |")
        print("|---|---:|---:|---:|")
        for label in sorted(results["mtp"]):
            r = results["mtp"][label]
            print(f"| {label} | {int(r.get('drafts_emitted', 0))} | {int(r.get('tokens_accepted', 0))} | {fmt_pct(r.get('acceptance_rate', 0))} |")
        print()


if __name__ == "__main__":
    main()
