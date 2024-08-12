from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.frames.frames import (LLMFullResponseEndFrame, 
                                   LLMFullResponseStartFrame,
                                   LLMMessagesFrame,
                                   TextFrame)

from pipecat.services.ai_services import LLMService
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
import google.generativeai as genai
from google.generativeai import GenerativeModel

from loguru import logger


class GeminiLLMContext:
    def __init__(self):
        self.messages = []
    
    @staticmethod
    def from_messages(messages: list[dict]):
        context = GeminiLLMContext()
        for message in messages:
            context.add_message({
                "content": message["content"],
                "role": message["role"],
                "name": message["name"] if "name" in message else message["role"]
            })
        return context
    
      
    def add_message(self, message: dict) -> "GeminiLLMContext":
        self.messages.append(message)
    
    def get_messages(self) -> list[dict]:
        return self.messages
    
class BaseGeminiLLMService(LLMService):
    def __init__(self, model: str, api_key: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = self._get_model(model=model)
        self._create_client(api_key=api_key)
    
    def _get_model(self, model) -> GenerativeModel:
        return genai.GenerativeModel(model)
    
    def _create_client(self, api_key: str) -> None:
        genai.configure(api_key=api_key)
    
    async def _get_chat_completions(self, context: GeminiLLMContext):
        logger.info(context.messages)
        response = self._model.generate_content(context.messages[-1].get("content"), stream=True)
        return response 
        ...
    
    async def _stream_chat_completions(self, context: GeminiLLMContext):
        # messages = context.get_messages()
        
        try:
            chunks = await self._get_chat_completions(context)
        except Exception as e:
            logger.error(f"{self} exception: {e}")
        
        return chunks
        ...
    
    async def _process_context(self, context: GeminiLLMContext):
        
        chunk_stream = await self._stream_chat_completions(context)
        
        for chunk in chunk_stream:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(TextFrame(chunk.text))
            await self.push_frame(LLMFullResponseEndFrame())
        ...
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # logger.log(f"Frame going through llm {frame}")
        context=None
        if isinstance(frame, LLMMessagesFrame):
            # logger.debug("Creating Context")
            context = GeminiLLMContext.from_messages(frame.messages)
        else:
            logger.debug(f"here is a frame going through Gemini {frame}")
        
        if context:
            await self.push_frame(LLMFullResponseStartFrame())
            await self._process_context(context)
            await self.push_frame(LLMFullResponseEndFrame())


class GeminiLLMService(BaseGeminiLLMService):
    def __init__(self, model: str, api_key: str, **kwargs) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)