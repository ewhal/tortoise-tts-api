from sanic import Sanic
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import List, Tuple, Union
from sanic.response import Response, json
from sanic.request import Request
from sqlalchemy import create_engine, Column, Integer, String, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tts.tortoise import TortoiseModal


import os
import sys
from time import time
from datetime import datetime
import torch


Base = declarative_base()


class Voice(Base):
    __tablename__ = "voices"
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    file_path = Column(String(255))
    format = Column(String(255))
    voice = Column(String(255))
    tensor_data = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.utcnow)

    def set_tensor(self, tensor):
        self.tensor_data = torch.save(tensor)

    def get_tensor(self):
        return torch.load(self.tensor_data)


class TTSRequest(Base):
    __tablename__ = "tts_requests"
    id = Column(Integer, primary_key=True)
    voice_name = Column(String(255))
    text = Column(String(255))
    preset = Column(String(255))
    candidates = Column(String(255))
    file_path = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)


@app.route("/create_voice", methods=["POST"])
async def create_voice(request: Request):
    # Get the name parameter
    name = request.form.get("name")

    # Get the uploaded file
    file = request.files.get("file")
    file_name = file.name
    file_path = f"{name}/{file_name}"
    # Save the file to S3
    request.app.s3.upload_fileobj(file.body, "my-bucket", file_path)
    latents = app.tts.get_voice_latents_from_url(file_path)

    voice = Voice(
        name=name,
        file_path=file_path,
        format=file_name.split(".")[-1],
        voice=request.form.get("voice"),
    )
    voice.set_tensor(latents)
    async with request.app.session() as session:
        session.add(voice)

    return json({"message": "Voice created successfully"})


@app.route("/generate_tts", methods=["POST"])
async def generate_tts(request: Request):
    # Parse the JSON payload
    payload = json.loads(request.body)

    # Get the voice name parameter
    voice_name = payload.get("voice_name")

    # Query the voice table for the voice's latents
    async with request.app.session() as session:
        voice = await session.query(Voice).filter_by(name=voice_name).first()
    latents = voice.get_tensor()

    # Call the TTS generator with the latents and other parameters
    # ... code to generate TTS ...
    wav = request.app.tts.generate_tts(latents, payload.get("text"))

    # Save the TTS to S3
    output_path = "results/"
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    file_name = f"{voice_name}_{timestamp}.wav"
    file_path = f"{output_path}/{file_name}"
    request.app.s3.upload_file(wav, "my-bucket", file_path)

    # Create a new TTS request object
    tts_request = TTSRequest(
        voice_name=voice_name,
        text=payload.get("text"),
        preset=payload.get("preset"),
        candidates=payload.get("candidates"),
        file_path=file_path,
    )

    # Save the TTS request object to the database
    async with request.app.session() as session:
        session.add(tts_request)
        await session.commit()

    # Return the file URL
    return json({"file_url": f"https://my-bucket.s3.amazonaws.com/{file_path}"})


def setup_app():
    # Set up SQLAlchemy
    app: Sanic = Sanic(__name__)
    engine = create_engine("sqlite:///voices.db")
    Session = sessionmaker(bind=engine)
    app.engine = engine
    app.session = Session
    app.tts: TortoiseModal = TortoiseModal()
    app.s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    )


if __name__ == "__main__":
    app = setup_app()
    app.run(host="0.0.0.0", port=8000)
