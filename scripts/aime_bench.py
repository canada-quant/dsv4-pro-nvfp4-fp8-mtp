#!/usr/bin/env python3
"""AIME 2024 bench harness with thinking-mode + MTP acceptance capture.

Carried verbatim from the V4-Flash predecessor. Scores pass@1 (exact-match on integer
answer), captures per-request decode tok/s, and reads vLLM's Prometheus
`spec_decode_*` counters before/after to compute MTP draft-acceptance rate.

Usage:
    python aime_bench.py \\
        --base-url http://localhost:8089/v1 \\
        --model canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP \\
        --thinking --effort high --concurrency 8 --max-tokens 65536 \\
        > aime24_result.json

Methodology: always pair raw pass@1 with non-truncated pass@1. See the V4-Flash
benchmark doc `docs/benchmarks/tier1_aime24_2026_05_21.md` in the predecessor repo
for the truncation-contamination lesson.
"""
import argparse, json, re, time, sys, asyncio, urllib.request
from datasets import load_dataset
from openai import AsyncOpenAI


def fetch_prom(host):
    try:
        with urllib.request.urlopen(f'http://{host}/metrics', timeout=5) as r:
            return r.read().decode('utf-8')
    except Exception:
        return ''


def parse_metric(txt, name):
    total = 0.0
    for line in txt.splitlines():
        if line.startswith(name + '{') or line.startswith(name + ' '):
            try:
                total += float(line.rsplit(' ', 1)[1])
            except Exception:
                pass
    return total


ANSWER_PATTERNS = [
    re.compile(r'\\\\boxed\\{(\\d+)\\}'),
    re.compile(r'final answer[^\\d]*?(\\d{1,3})', re.IGNORECASE),
    re.compile(r'answer is[^\\d]*?(\\d{1,3})', re.IGNORECASE),
    re.compile(r'\\b(\\d{1,3})\\b\\s*$'),
]


def extract_answer(text):
    if not text:
        return None
    t = text.strip()
    for pat in ANSWER_PATTERNS:
        m = pat.findall(t)
        if m:
            try:
                v = int(m[-1])
                if 0 <= v <= 999:
                    return v
            except Exception:
                pass
    return None


async def run_one(client, model, problem, thinking, effort, max_tokens):
    body = {
        'model': model,
        'messages': [{
            'role': 'user',
            'content': problem + '\\n\\nReason step-by-step then place your final integer answer (0-999) inside \\\\boxed{...}.'
        }],
        'max_tokens': max_tokens,
        'temperature': 0.0,
    }
    kwargs = dict(body)
    if thinking:
        kwargs['extra_body'] = {'chat_template_kwargs': {'thinking': True, 'reasoning_effort': effort}}
    try:
        resp = await client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        content = (msg.content or '') + (getattr(msg, 'reasoning', None) or '')
        return {'content': content, 'completion_tokens': resp.usage.completion_tokens, 'finish': resp.choices[0].finish_reason}
    except Exception as e:
        return {'content': '', 'completion_tokens': 0, 'finish': 'error', 'error': str(e)}


async def main(args):
    ds = load_dataset(args.dataset, split='train')
    print(f'Loaded {len(ds)} problems', file=sys.stderr)
    client = AsyncOpenAI(base_url=args.base_url, api_key='dummy', timeout=600)

    p0 = fetch_prom(args.prom_host)
    d0 = parse_metric(p0, 'vllm:spec_decode_num_drafts_total')
    dt0 = parse_metric(p0, 'vllm:spec_decode_num_draft_tokens_total')
    at0 = parse_metric(p0, 'vllm:spec_decode_num_accepted_tokens_total')

    sem = asyncio.Semaphore(args.concurrency)

    async def worker(i, row):
        async with sem:
            t0 = time.monotonic()
            r = await run_one(client, args.model, row[args.problem_field], args.thinking, args.effort, args.max_tokens)
            r['idx'] = i
            r['id'] = str(row.get('ID', row.get('id', i)))
            r['expected'] = str(row[args.answer_field]).strip()
            r['predicted'] = extract_answer(r['content'])
            r['correct'] = (r['predicted'] is not None and str(r['predicted']) == r['expected'])
            r['elapsed'] = time.monotonic() - t0
            return r

    t = time.monotonic()
    futs = [worker(i, row) for i, row in enumerate(ds)]
    results = []
    for fut in asyncio.as_completed(futs):
        r = await fut
        results.append(r)
        print(f'[{len(results):3d}/{len(ds)}] id={r["id"]} pred={r["predicted"]} exp={r["expected"]} {"OK" if r["correct"] else "NO"} tok={r["completion_tokens"]} {r["elapsed"]:.1f}s', file=sys.stderr)
    wall = time.monotonic() - t

    p1 = fetch_prom(args.prom_host)
    d1 = parse_metric(p1, 'vllm:spec_decode_num_drafts_total')
    dt1 = parse_metric(p1, 'vllm:spec_decode_num_draft_tokens_total')
    at1 = parse_metric(p1, 'vllm:spec_decode_num_accepted_tokens_total')

    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    total_tok = sum(r['completion_tokens'] for r in results)
    delta_dt = dt1 - dt0
    delta_at = at1 - at0
    acc = (delta_at / delta_dt * 100) if delta_dt > 0 else None

    summary = {
        'dataset': args.dataset, 'model': args.model, 'thinking': args.thinking, 'effort': args.effort,
        'concurrency': args.concurrency, 'max_tokens': args.max_tokens,
        'n_total': total, 'n_correct': correct, 'pass_at_1': correct / total if total else 0,
        'total_completion_tokens': total_tok, 'avg_completion_tokens': total_tok / total if total else 0,
        'wall_clock_s': wall, 'throughput_tok_s': total_tok / wall if wall else 0,
        'spec_decode': {'drafts': int(d1 - d0), 'draft_tokens': int(delta_dt), 'accepted_tokens': int(delta_at), 'acceptance_pct': acc},
        'results': results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--prom-host', default='localhost:8089')
    ap.add_argument('--dataset', default='Maxwell-Jia/AIME_2024')
    ap.add_argument('--problem-field', default='Problem')
    ap.add_argument('--answer-field', default='Answer')
    ap.add_argument('--thinking', action='store_true')
    ap.add_argument('--effort', default='high', choices=['high', 'max'])
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=65536)
    args = ap.parse_args()
    asyncio.run(main(args))
