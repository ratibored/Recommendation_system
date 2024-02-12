from pydantic import BaseModel
from typing import List

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        from_attributes = True


class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]

    class Config:
        from_attributes = True