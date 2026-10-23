"""Latency + transcript logging. Nothing here changes the conversation.

LatencyObserver watches frames flow through the pipeline and writes logs/latency.jsonl.
attach_transcript() writes logs/transcript.txt from the context aggregators.
All *_ms values are milliseconds since the session started (the first frame we see).
"""

import json
import time
from pathlib import Path

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMTextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

NS_PER_MS = 1_000_000


class LatencyObserver(BaseObserver):
    def __init__(self, path: str = "logs/latency.jsonl"):
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("")
        self._t0 = None  # pipeline-clock time of the first frame, in ns
        self._turns = []
        self._bot_speaking = False
        self._reset_pending()

    def _reset_pending(self):
        self._user_stopped = None
        self._stt_final = None
        self._llm_first = None

    def _ms(self, ts_ns: int) -> int:
        return round((ts_ns - self._t0) / NS_PER_MS)

    def _write(self):
        lines = [json.dumps(t) for t in self._turns]
        self._path.write_text("\n".join(lines) + "\n")

    async def on_push_frame(self, data: FramePushed):
        if not data.first_push:
            return  # every processor re-pushes a frame; count each frame once
        if self._t0 is None:
            self._t0 = data.timestamp
        frame, ts = data.frame, data.timestamp

        if isinstance(frame, VADUserStartedSpeakingFrame):
            if self._bot_speaking and self._turns:
                self._turns[-1]["interrupted"] = True  # user talked over the agent
                self._write()
            self._reset_pending()  # new speech: forget any earlier pause

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # VAD confirms silence only after stop_secs, so the user really went quiet earlier.
            self._user_stopped = self._ms(ts) - round(frame.stop_secs * 1000)

        elif isinstance(frame, TranscriptionFrame):
            self._stt_final = self._ms(ts)  # keep the latest final transcript

        elif isinstance(frame, LLMTextFrame):
            if self._llm_first is None:
                self._llm_first = self._ms(ts)

        elif isinstance(frame, BotStartedSpeakingFrame):
            if not self._bot_speaking:
                self._bot_speaking = True
                if self._user_stopped is not None:  # skip the opening greeting
                    self._turns.append({
                        "turn": len(self._turns) + 1,
                        "user_stopped_ms": self._user_stopped,
                        "stt_final_ms": self._stt_final,
                        "llm_first_token_ms": self._llm_first,
                        "tts_first_audio_ms": self._ms(ts),
                        "interrupted": False,
                    })
                    self._write()
                    self._reset_pending()

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False


def attach_transcript(user_agg, assistant_agg, path: str = "logs/transcript.txt"):
    """Write who said what, in order. Interrupted agent turns are marked."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    start = time.monotonic()

    def add(who: str, text: str):
        stamp = f"[{time.monotonic() - start:6.1f}s]"
        with p.open("a") as f:
            f.write(f"{stamp} {who}: {text}\n")

    @user_agg.event_handler("on_user_turn_stopped")
    async def _user(aggregator, strategy, message):
        if message.content:
            add("USER", message.content)

    @assistant_agg.event_handler("on_assistant_turn_stopped")
    async def _bot(aggregator, message):
        if message.content:
            suffix = "  [INTERRUPTED by user]" if message.interrupted else ""
            add("MAYA", message.content + suffix)
