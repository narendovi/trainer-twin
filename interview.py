"""Interview flow controller.

The LLM never decides which question comes next. This processor does, by counting how many
times the candidate has spoken, and attaches a private instruction to a COPY of the context
for that single LLM call. The saved conversation history stays clean.
"""

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import prompts

NUM_QUESTIONS = len(prompts.QUESTIONS)


def directive_for(user_turns: int) -> str:
    """Pick the instruction for the LLM call, given how many times the candidate has spoken.

    user_turns = 0 is the session start. After that each question uses two candidate turns:
    odd turn = first answer (we follow up), even turn = answer to the follow-up (we move on).
    """
    if user_turns == 0:
        return prompts.OPENING.format(question=prompts.QUESTIONS[0])
    if user_turns > 2 * NUM_QUESTIONS:
        return prompts.WRAP_UP

    q = (user_turns + 1) // 2  # main question the candidate is on, 1-based
    if user_turns % 2 == 1:
        return prompts.FOLLOW_UP.format(q=q)

    # Even turn: they answered the follow-up for question q.
    if q < NUM_QUESTIONS:
        return prompts.NEXT_QUESTION.format(
            q=q, next_q=q + 1, question=prompts.QUESTIONS[q]
        )
    return prompts.FEEDBACK


def count_user_turns(messages) -> int:
    """Number of real candidate turns in the history (the first 'ready' message is turn 0)."""
    users = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "user")
    return max(0, users - 1)


class InterviewController(FrameProcessor):
    """Sits between the user aggregator and the LLM."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            messages = list(frame.context.messages)
            turns = count_user_turns(messages)
            directive = directive_for(turns)
            messages.append({"role": "system", "content": directive})
            frame = LLMContextFrame(
                context=LLMContext(messages=messages),
                speculation=frame.speculation,
            )

        await self.push_frame(frame, direction)
