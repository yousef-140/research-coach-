from enum import Enum
from pydantic import BaseModel


class SourceType(str, Enum):
    ACADEMIC = "academic"
    BLOG = "blog"
    NEWS = "news"


class Source(BaseModel):
    title: str
    url: str
    content: str
    source_type: SourceType

class SourceList(BaseModel):
    sources: list[Source]


class SourceEvaluation(BaseModel):
    chunk_id: str
    score: float
    reason: str


class CriticOutput(BaseModel):
    evaluations: list[SourceEvaluation]

class Subscription(BaseModel):
    topic: str
    frequency: str
