import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings
from pipecat.services.deepgram.tts import DeepgramTTSService, DeepgramTTSSettings
from pipecat.services.groq.llm import GroqLLMService, GroqLLMSettings
from pipecat.transports.base_transport import TransportParams

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies


from interview import InterviewController, attach_state
from prompts import SYSTEM_PROMPT

from latency import LatencyObserver, attach_transcript

load_dotenv(override=True)

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)

    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramSTTSettings(model="nova-3", language="en-IN",keyterm=["LangChain", "Groq", "OMDb", "Streamlit", "MovieMind", "Pipecat"],)
    )
    llm = GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMSettings(
            model="qwen/qwen3.8-27b",
            reasoning_effort="none",
            system_instruction=SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=300,
            extra={"extra_body": {"reasoning_format": "hidden"}},
        ),
    )
    tts = DeepgramTTSService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramTTSSettings(voice="aura-2-thalia-en"),
    )

    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
         user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.6)),
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(
                            params=SmartTurnParams(stop_secs=2.0)
                        )
                    )
                ]
            ),
            empty_user_turn=None,
        ),)
    attach_transcript(user_agg, assistant_agg, "logs/transcript.txt")
    state = attach_state(assistant_agg)

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        InterviewController(state),
        llm,
        tts,
        transport.output(),
        assistant_agg,
    ])

    task = PipelineTask(
           pipeline,
           params=PipelineParams(enable_metrics=True),
           observers=[LatencyObserver("logs/latency.jsonl")],
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info("Client connected")
        context.add_message({"role": "user", "content": "Hi, I'm ready to start."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    from pipecat.pipeline.runner import PipelineRunner
    await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
