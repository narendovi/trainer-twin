"""Interview flow controller.

The LLM never decides which question comes next. This code does.

It tracks what Maya has actually DELIVERED. A turn of hers counts only if the user did not
interrupt it: a half-spoken question was never really asked, so we do not move on.
For each LLM call we attach ONE private instruction to a COPY of the context; the saved
conversation history stays clean.
"""

import re

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import prompts

NUM_QUESTIONS = len(prompts.QUESTIONS)
FEEDBACK_MIN_CHARS = 60  # an uninterrupted reply this long after the last answer is the feedback


def _norm(text: str) -> str:
    """Lowercase and drop punctuation so wording differences like quotes do not matter."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower().replace("'", ""))


class InterviewState:
    """Where the interview is, based on what Maya finished saying."""

    def __init__(self):
        self.spoke = False           # Maya finished at least one turn
        self.question = -1           # index of the last main question fully delivered
        self.followup_done = False   # a follow-up was delivered for that question
        self.feedback_done = False   # closing feedback was delivered

    def on_assistant_turn(self, text: str, interrupted: bool):
        if interrupted or not text:
            return  # a cut-off turn was not delivered, so nothing advances
        self.spoke = True
        spoken = _norm(text)
        for k, question in enumerate(prompts.QUESTIONS):
            if _norm(question) in spoken and k >= self.question:
                self.question, self.followup_done = k, False
                return
        if self.question < 0:
            return
        if self.question == NUM_QUESTIONS - 1 and self.followup_done:
            if len(text) >= FEEDBACK_MIN_CHARS:
                self.feedback_done = True
        elif "?" in text:
            self.followup_done = True


def plan(state: InterviewState) -> str:
    """Return the private instruction for the next LLM call."""
    if state.question < 0:
        if state.spoke:  # she spoke but never finished question 1
            return prompts.REASK.format(q=1, question=prompts.QUESTIONS[0])
        return prompts.OPENING.format(question=prompts.QUESTIONS[0])
    if not state.followup_done:
        return prompts.FOLLOW_UP.format(q=state.question + 1)
    if state.question < NUM_QUESTIONS - 1:
        return prompts.NEXT_QUESTION.format(
            q=state.question + 1,
            next_q=state.question + 2,
            question=prompts.QUESTIONS[state.question + 1],
        )
    return prompts.WRAP_UP if state.feedback_done else prompts.FEEDBACK


REPEAT_WORDS = re.compile(
    r"\b(repeat|say that again|say it again|come again|once more|pardon|"
    r"(didn't|did not|couldn't|could not) (catch|hear|get)|what was the question|rephrase)\b"
)
MAX_REPEAT_REQUEST_WORDS = 12  # a long answer that merely contains 'repeat' is still an answer


def is_repeat_request(messages) -> bool:
    """True if the candidate's latest words are a short request to repeat or clarify."""
    parts = []
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get("role") != "user":
            break
        parts.append(m.get("content") if isinstance(m.get("content"), str) else "")
    heard = " ".join(reversed(parts)).lower().replace("\u2019", "'")
    return len(heard.split()) <= MAX_REPEAT_REQUEST_WORDS and bool(REPEAT_WORDS.search(heard))


def attach_state(assistant_agg) -> InterviewState:
    """Create the state and keep it updated from the assistant aggregator's turn events."""
    state = InterviewState()

    @assistant_agg.event_handler("on_assistant_turn_stopped")
    async def _update(aggregator, message):
        state.on_assistant_turn(message.content or "", bool(message.interrupted))

    return state


class InterviewController(FrameProcessor):
    """Sits between the user aggregator and the LLM."""

    def __init__(self, state: InterviewState):
        super().__init__()
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            messages = list(frame.context.messages)
            directive = prompts.REPEAT if is_repeat_request(messages) else plan(self._state)
            messages.append({"role": "user", "content": "[DIRECTIVE] " + directive})
            frame = LLMContextFrame(
                context=LLMContext(messages=messages),
                speculation=frame.speculation,
            )

        await self.push_frame(frame, direction)
