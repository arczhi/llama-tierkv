#!/usr/bin/env python3
"""Decode/prefill probe for llama-server (mainline): measure tok/s + MTP acceptance at a target context.

Usage: python3 probe.py --port 18210 --target-tokens 32768 --gen 512 --label "mtp3"
"""
import argparse
import json
import time
import urllib.request

FILLER = ("Log entry {i}: station {s} reported metric {m} with deviation {d} units, "
          "checked sensor S{sn}, recalibrated channel C{c}, and archived batch B{b} without incident. ")


def make_prompt(target_chars: int) -> str:
    parts = ["IMPORTANT: The secret word is OBSIDIAN. Remember it.\n"]
    total = len(parts[0])
    i = 0
    while total < target_chars:
        seg = FILLER.format(i=i, s=(i * 13) % 97, m=i % 991, d=(i * 7) % 53,
                            sn=i % 211, c=i % 89, b=i % 307)
        parts.append(seg)
        total += len(seg)
        i += 1
    parts.append("\n\nTask: Write a short paragraph (about 120 words) summarizing the logs. Then state the secret word.")
    return "".join(parts)


def post(port: int, path: str, payload: dict, timeout: int = 3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def count_tokens(port: int, text: str) -> int:
    d = post(port, "/tokenize", {"content": text})
    return len(d.get("tokens", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--target-tokens", type=int, default=32768)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--label", default="probe")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()

    # build a prompt that lands close to target tokens (chars/token ~3.4 for this filler)
    text = make_prompt(int(args.target_tokens * 3.4))
    ntok = count_tokens(args.port, text)
    # trim/expand pass: do one adjust iteration
    if abs(ntok - args.target_tokens) > args.target_tokens * 0.05:
        factor = args.target_tokens / max(ntok, 1)
        text = make_prompt(int(len(text) * factor))
        ntok = count_tokens(args.port, text)
    print(f"[{args.label}] prompt tokens = {ntok}")

    results = []
    for r in range(args.repeats):
        t0 = time.time()
        d = post(args.port, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": text}],
            "max_tokens": args.gen,
            "temperature": args.temp,
            "cache_prompt": False,
        })
        wall = time.time() - t0
        t = d.get("timings", {})
        u = d.get("usage", {})
        row = {
            "wall_s": round(wall, 2),
            "prompt_n": t.get("prompt_n", u.get("prompt_tokens")),
            "prompt_ms": t.get("prompt_ms"),
            "prompt_tps": round(t.get("prompt_per_second", 0), 1) if t else None,
            "predicted_n": t.get("predicted_n", u.get("completion_tokens")),
            "predicted_ms": t.get("predicted_ms"),
            "decode_tps": round(t.get("predicted_per_second", 0), 2) if t else None,
            "draft_n": t.get("draft_n"),
            "draft_accepted": t.get("draft_n_accepted"),
        }
        if row["draft_n"]:
            row["accept_pct"] = round(100.0 * (row["draft_accepted"] or 0) / row["draft_n"], 1)
        results.append(row)
        print(f"  run{r+1}: {json.dumps(row)}")

    ok = [r for r in results if r["decode_tps"]]
    if ok:
        dec = sorted(r["decode_tps"] for r in ok)
        print(f"[{args.label}] decode median = {dec[len(dec)//2]:.2f} tok/s "
              f"(min {dec[0]:.2f}, max {dec[-1]:.2f})")


if __name__ == "__main__":
    main()
