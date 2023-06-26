from sqlalchemy import Column, Integer, String, ForeignKey, relationship, String
from sqlalchemy import Column, Integer, String, LargeBinary
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Voice(Base):
    __tablename__ = "voices"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    sample_file = Column(String, nullable=False)
    conditioning_latents_file = Column(String, nullable=False)


class SynthesizedSpeech(Base):
    __tablename__ = "synthesized_speeches"
    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)
    synthesized_file = Column(String, nullable=False)
    voice_id = Column(Integer, ForeignKey("voices.id"))
    voice = relationship("Voice", foreign_keys=voice_id)
