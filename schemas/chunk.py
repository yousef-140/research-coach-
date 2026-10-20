from pydantic import BaseModel

class Chunk(BaseModel):
    chunk_id: str
    text: str
    url: str
    title: str
    topic: str