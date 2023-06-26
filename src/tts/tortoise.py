import os
import sys
import io
from time import time
from datetime import datetime
import boto3

from typing import List, Tuple, Union, Optional
import torch
import torchaudio

from tortoise.api import TextToSpeech, MODELS_DIR
from tortoise.utils.audio import load_audio, load_voices
from tortoise.utils.text import split_and_recombine_text

s3 = boto3.client("s3")


class TortoiseModal:
    def __init__(self) -> None:
        from tortoise.api import MODELS_DIR, TextToSpeech
        from tortoise.utils.audio import load_audio, load_voices

        self.load_voices = load_voices
        self.load_audio = load_audio
        self.tts: TextToSpeech = TextToSpeech(models_dir=MODELS_DIR)
        self.tts.get_random_conditioning_latents()

    async def get_voice_latents_from_url(self, url: str) -> torch.Tensor:
        # load the voice from an s3 bucket, url is the s3 url
        # use tts.get_latents_from_audio to get the latents
        # download the file from s3
        file_name = url.split("/")[-1]
        s3.download_file("my-bucket", url, file_name)

        # load the file
        audio = self.load_audio(file_name)
        latents = self.tts.get_latents_from_audio(audio)
        return latents

    def process_synthesis_result(self, result: torch.Tensor) -> io.BytesIO:
        import pydub
        import torchaudio

        with tempfile.NamedTemporaryFile() as converted_wav_tmp:
            torchaudio.save(
                converted_wav_tmp.name + ".wav",
                result,
                24000,
            )
            wav = io.BytesIO()
            _ = pydub.AudioSegment.from_file(
                converted_wav_tmp.name + ".wav", format="wav"
            ).export(wav, format="wav")

        return wav

    def get_voices():
        voices = sorted(os.listdir("../tortoise-tts/tortoise/voices"))
        return dict(zip(voices, voices))

    def get_audios() -> List[str]:
        voices = []
        paths = sorted(
            ["static/voices/" + voice + "/" for voice in os.listdir("static/voices/")]
        )

        for path in paths:
            file = path + os.listdir(path)[0]
            voices.append(
                {
                    "file": "/" + file,
                    "format": file.split(".")[-1],
                    "voice": path.split("/")[-2],
                }
            )

        return voices

    def get_quality() -> dict:
        values = ["ultra_fast", "fast", "standard", "high_quality"]
        return dict(zip(values, values))

    def get_candidates() -> List[int]:
        return list(range(1, 6))

    def generate_tts(
        self, voice="mol", text="Hello world", preset="fast", candidates=1
    ) -> None:
        output_path = "results/"
        model_dir = MODELS_DIR
        seed =  int(time.time())  
        produce_debug_state = False
        cvvp_amount = 0.0
        os.makedirs(output_path, exist_ok=True)

        # strip special characters, new lines, etc.
        text = text.strip()
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        text = text.replace("\t", " ")
        text = text.replace("  ", " ")

        # Preparing the input text
        text = " ".join(text.splitlines())

        if len(text) > 1000:
            text = split_and_recombine_text(text, 1000)

        # Loading the voice
        voice = voice
        voice = load_voices(model_dir)[voice]

        # Synthesizing the voice
        start = time()
        result = self.tts.tts_with_preset(text, voice, preset, candidates, seed)
        end = time()
        print(f"Synthesis took {end - start:.2f} seconds")

        # Processing the result
        wav = self.process_synthesis_result(result)

        # Saving the result
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        file_name = f"{output_path}/{timestamp}.wav"
        with open(file_name, "wb") as f:
            f.write(wav.read())

        return file_name
