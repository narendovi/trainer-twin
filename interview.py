"""Interview flow controller.

The LLM never decides which question comes next. This processor reads the conversation so far,
works out where the interview is, and attaches ONE private instruction to a COPY of the context
for that single LLM call. The saved history stays clean.

Where we are is decided from what Maya actually SAID, not from how many times the candidate spoke:
- a main question counts as asked only if its full text appears in one of Maya's messages
- a follow-up counts as given only if Maya's message contains a question mark
Interrupted or half-spoken replies therefore never move the interview forward by accident.
"""

import re

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import prompts

NUM_QUESTIONS = len(prompts.QUESTIONS)
FEEDBACK_MIN_CHARS = 60  # a spoken reply this long after the last answer counts as feedback


def _norm(text: str) -> str:
    """Lowercase and drop punctuation so 'you\'re' matches 'youre' and spacing differences."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower().replace("'", ""))


def _text(message) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else ""


def plan(messages) -> str:
    """Return the private instruction for the next LLM call."""
    msgs = [m for m in messages if isinstance(m, dict) and m.get("role") in ("user", "assistant")]

    # Which main question did Maya most recently ask IN FULL, and where?
    asked, anchor = -1, -1
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        for k, question in enumerate(prompts.QUESTIONS):
            if _norm(question) in _norm(_text(m)) and k >= asked:
                asked, anchor = k, i

    if asked < 0:  # no main question has been fully spoken yet
        if any(m["role"] == "assistant" for m in msgs):
            return prompts.REASK.format(q=1, question=prompts.QUESTIONS[0])
        return prompts.OPENING.format(question=prompts.QUESTIONS[0])

    after = msgs[anchor + 1 :]  # everything since that question
    if not any(m["role"] == "user" for m in after):
        return prompts.REASK.format(q=asked + 1, question=prompts.QUESTIONS[asked])

    # Has Maya already put a follow-up (a spoken question) to this answer?
    follow = next((j for j, m in enumerate(after) if m["role"] == "assistant" and "?" in _text(m)), None)
    if follow is None:
        return prompts.FOLLOW_UP.format(q=asked + 1)

    if asked < NUM_QUESTIONS - 1:
        return prompts.NEXT_QUESTION.format(
            q=asked + 1, next_q=asked + 2, question=prompts.QUESTIONS[asked + 1]
        )

    # Last question, follow-up answered: give feedback once, then wrap up.
    answer = next((j for j in range(follow + 1, len(after)) if after[j]["role"] == "user"), None)
    if answer is not None and any(
        m["role"] == "assistant" and len(_text(m)) >= FEEDBACK_MIN_CHARS for m in after[answer + 1 :]
    ):
        return prompts.WRAP_UP
    return prompts.FEEDBACK


class InterviewController(FrameProcessor):
    """Sits between the user aggregator and the LLM."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            messages = list(frame.context.messages)
            directive = plan(messages)
            messages.append({"role": "user", "content": "[DIRECTIVE] " + directive})
            frame = LLMContextFrame(
                context=LLMContext(messages=messages),
                speculation=frame.speculation,
            )

        await self.push_frame(frame, direction)
