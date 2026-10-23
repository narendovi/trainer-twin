"""Usage: python latency_report.py logs/latency.jsonl"""
import json
import statistics
import sys


def pct(values, p):
    values = sorted(values)
    k = max(0, min(len(values) - 1, round(p / 100 * len(values) + 0.5) - 1))  # nearest rank
    return values[k]


rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
total = [r["tts_first_audio_ms"] - r["user_stopped_ms"] for r in rows]
print(f"turns: {len(rows)}   interrupted: {sum(r['interrupted'] for r in rows)}")
print(f"user stopped -> first audio:  p50 {pct(total, 50)} ms   p95 {pct(total, 95)} ms")

# Stage times. If the final transcript arrived before the VAD confirmed silence,
# the STT stage counts as 0 and later stages start from user_stopped.
stt, llm, tts = [], [], []
for r in rows:
    if None in (r["stt_final_ms"], r["llm_first_token_ms"]):
        continue
    a = max(r["stt_final_ms"], r["user_stopped_ms"])
    stt.append(a - r["user_stopped_ms"])
    llm.append(r["llm_first_token_ms"] - a)
    tts.append(r["tts_first_audio_ms"] - r["llm_first_token_ms"])
if stt:
    print(f"avg speech-to-text wait:       {statistics.mean(stt):.0f} ms")
    print(f"avg LLM first token:           {statistics.mean(llm):.0f} ms")
    print(f"avg text-to-speech first audio:{statistics.mean(tts):.0f} ms")
