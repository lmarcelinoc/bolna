import asyncio
import copy
import websockets
import base64
import json
import aiohttp
import os
import traceback
from collections import deque

from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet, pcm_to_wav_bytes, resample


logger = configure_logger(__name__)


class ElevenlabsSynthesizer(BaseSynthesizer):
    def __init__(self, voice, voice_id, model="eleven_multilingual_v1", audio_format="mp3", sampling_rate="16000",
                 stream=False, buffer_size=400, temperature=0.9, similarity_boost=0.5, synthesier_key=None,
                 caching=True, **kwargs):
        super().__init__(stream)
        self.api_key = os.environ["ELEVENLABS_API_KEY"] if synthesier_key is None else synthesier_key
        self.voice = voice_id
        self.use_turbo = kwargs.get("use_turbo", False)
        self.model = "eleven_turbo_v2" if self.use_turbo else "eleven_multilingual_v2"
        logger.info(f"Using turbo or not {self.model}")
        self.stream = False  # Issue with elevenlabs streaming that we need to always send the text quickly
        self.websocket_connection = None
        self.connection_open = False
        self.sampling_rate = sampling_rate
        self.audio_format = "mp3"
        self.use_mulaw = kwargs.get("use_mulaw", False)
        self.ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice}/stream-input?model_id={self.model}&optimize_streaming_latency=2&output_format={self.get_format(self.audio_format, self.sampling_rate)}"
        self.api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice}?optimize_streaming_latency=2&output_format="
        self.first_chunk_generated = False
        self.last_text_sent = False
        self.text_queue = deque()
        self.meta_info = None
        self.temperature = 0.8
        self.similarity_boost = similarity_boost
        self.caching = caching
        if self.caching:
            self.cache = InmemoryScalarCache()
        self.synthesized_characters = 0
        self.previous_request_ids = []

    def get_format(self, format, sampling_rate):
        if self.use_mulaw:
            return "ulaw_8000"
        return f"mp3_44100_128"

    def get_engine(self):
        return self.model

    async def sender(self, text, end_of_llm_stream=False):  # sends text to websocket
        if self.websocket_connection is None or not self.websocket_connection.open:
            await self.open_connection()

        if text:
            logger.info(f"Sending message {text}")
            input_message = {
                "text": f"{text} ",
                "try_trigger_generation": True,
                "flush": True
            }
            await self.websocket_connection.send(json.dumps(input_message))
            if end_of_llm_stream:
                self.last_text_sent = True

    async def receiver(self):
        while True:
            if self.websocket_connection is None or not self.websocket_connection.open:
                await self.open_connection()
            try:
                response = await self.websocket_connection.recv()
                data = json.loads(response)
                logger.info("response for isFinal: {}".format(data.get('isFinal', False)))
                if "audio" in data and data["audio"]:
                    chunk = base64.b64decode(data["audio"])
                    if len(chunk) % 2 == 1:
                        chunk += b'\x00'
                    yield chunk
                    if "isFinal" in data and data["isFinal"]:
                        yield b'\x00'
                else:
                    logger.info("No audio data in the response")
            except websockets.exceptions.ConnectionClosed:
                break

    async def __send_payload(self, payload, format=None):
        headers = {
            'xi-api-key': self.api_key
        }
        url = f"{self.api_url}{self.get_format(self.audio_format, self.sampling_rate)}" if format is None else f"{self.api_url}{format}"
        async with aiohttp.ClientSession() as session:
            if payload:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        logger.error(f"Error: {response.status} - {await response.text()}")
            else:
                logger.info("Payload was null")

    async def synthesize(self, text):
        return await self.__generate_http(text, format="mp3_44100_128")

    async def __generate_http(self, text, format=None):
        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": self.temperature,
                "similarity_boost": self.similarity_boost,
                "optimize_streaming_latency": 3
            }
        }
        return await self.__send_payload(payload, format=format)

    def get_synthesized_characters(self):
        return self.synthesized_characters

    async def generate(self):
        try:
            if self.stream:
                async for message in self.receiver():
                    if self.text_queue:
                        self.meta_info = self.text_queue.popleft()
                    audio = convert_audio_to_wav(message, source_format="mp3") if not self.use_mulaw else message
                    self.meta_info['format'] = 'wav' if not self.use_mulaw else 'mulaw'
                    yield create_ws_data_packet(audio, self.meta_info)
                    if self.last_text_sent:
                        self.first_chunk_generated = False
                    if message == b'\x00':
                        self.meta_info["end_of_synthesizer_stream"] = True
                        yield create_ws_data_packet(resample(message, int(self.sampling_rate)), self.meta_info)
                        self.first_chunk_generated = False
            else:
                while True:
                    message = await self.internal_queue.get()
                    meta_info, text = message.get("meta_info"), message.get("data")
                    if self.caching and self.cache.get(text):
                        audio = self.cache.get(text)
                        meta_info['is_cached'] = True
                    else:
                        self.synthesized_characters += len(text)
                        audio = await self.__generate_http(text)
                        if self.caching:
                            self.cache.set(text, audio)
                        meta_info['is_cached'] = False
                    meta_info['text'] = text
                    meta_info['format'] = 'wav' if not self.use_mulaw else 'mulaw'
                    audio = convert_audio_to_wav(audio, source_format="mp3") if not self.use_mulaw else audio
                    yield create_ws_data_packet(audio, meta_info)
                    if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                        meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in eleven labs generate {e}")

    async def open_connection(self):
        if self.websocket_connection is None or not self.websocket_connection.open:
            self.websocket_connection = await websockets.connect(self.ws_url)
            self.connection_open = True
            logger.info("Connected to the server")

    async def close_connection(self):
        if self.websocket_connection and self.websocket_connection.open:
            await self.websocket_connection.close()
            self.connection_open = False
            logger.info("Closed the WebSocket connection")

    async def push(self, message):
        if self.stream:
            meta_info, text = message.get("meta_info"), message.get("data")
            self.meta_info = copy.deepcopy(meta_info)
            self.text_queue.append(meta_info)
            await self.sender(text, "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"])
        else:
            self.internal_queue.put_nowait(message)
