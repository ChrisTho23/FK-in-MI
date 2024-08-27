from typing import List

from pydantic import BaseModel, ConfigDict


class FKDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: List[str]
    answer: List[str]