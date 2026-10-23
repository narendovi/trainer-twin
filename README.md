# TrainerTwin take-home: voice AI interview coach

A voice agent ("Maya") that runs a short mock interview. She asks four questions in order, reacts to what you said with one follow-up or a gentle nudge, and ends with 2-3 lines of spoken feedback.

Built with Pipecat (Python 3.11.8, Pipecat 1.13.0). Cascaded pipeline: speech-to-text, then LLM, then text-to-speech.

> Sections marked **TODO** need real numbers or observations from my own test session. I fill them in only after measuring.

## How to run it

1. Python 3.11 or newer.
2. Sign up and create API keys at:
   - Deepgram (speech-to-text and text-to-speech): https://deepgram.com
   - Groq (LLM): https://console.groq.com
3. Setup:

```bash
git clone https://github.com/narendovi/trainer-twin.git
cd trainer-twin
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then paste your two keys into .env
```

4. Run:

```bash
python bot.py
```

5. Open http://localhost:7860 in Chrome, click Connect, allow the microphone. Headphones are recommended.

macOS note: if you see `CERTIFICATE_VERIFY_FAILED` with a python.org Python, run
`export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")` first.

## Architecture

```
Browser mic -> SmallWebRTC transport (input)
            -> Deepgram STT (streaming)
            -> User aggregator (Silero VAD + Smart Turn v3 decide end of turn)
            -> InterviewController (custom)
            -> Groq LLM
            -> Deepgram TTS (streaming)
            -> SmallWebRTC transport (output) -> Browser speaker
            -> Assistant aggregator (saves what was actually spoken)
```

Custom processors:
- **InterviewController** (`interview.py`): counts how many times the candidate has spoken and attaches a private instruction to a copy of the context for that one LLM call. The code, not the LLM, decides which question comes next. The saved conversation history stays free of these instructions.
- **Latency logger**: TODO (describe once built).

Files: `bot.py` (pipeline), `interview.py` (flow control), `prompts.py` (all prompts), `PROMPTS.md` (prompt document).

## Providers and why

| Stage | Choice | Why | Trade-off |
|---|---|---|---|
| STT | Deepgram Nova-3, language `en-IN` | Fast streaming, good with Indian English accents | Cloud service, needs a key |
| LLM | Groq, TODO confirm final model name | Very low time to first token on Groq, free tier | Open models are less polished than the largest hosted models. `gpt-oss` models intermittently failed with "Tool choice is none, but model called a tool", so I moved off them |
| TTS | Deepgram Aura-2 (`aura-2-thalia-en`) | One key for STT and TTS, low latency streaming | Voice is less expressive than some alternatives |
| Transport | Pipecat SmallWebRTC (browser) | Runs locally with no extra account, browser echo cancellation | Needs Chrome and microphone permission |

## Turn-taking and interruptions

Two layers decide when I have finished speaking:
- **Silero VAD** detects silence. `stop_secs = 0.2`.
- **Smart Turn v3** (a small model) judges whether the sentence sounds complete. If it says incomplete, it waits for a fallback silence of `stop_secs` before ending the turn.

- Baseline behavior with defaults in scenarios A, B and C (what went wrong).
- Values I changed, what I changed them to, and why.
- How interruptions behave and how fast the agent stops speaking.

## Latency

p50 and p95 of user-stopped to first audio, plus average time in each stage (STT, LLM first token, TTS first audio), from `latency.jsonl`. Include how each timestamp was captured or approximated.

## What's weak and what I'd do next

Known design limits so far:
- The follow-up schedule is fixed: each question gets exactly two candidate turns. If a user's answer is split into two turns by a long pause, the schedule shifts by one.
- If the user asks for a repeat or clarification, it still counts as an answer turn.

## AI tools

(Claude)- Claude was used for scaffolding the Pipecat pipeline, designing the interview flow controller, and debugging environment problems.

## Time spent

6  hours.
