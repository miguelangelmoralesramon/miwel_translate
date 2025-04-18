from pydantic import BaseModel, Field
from typing import Optional

class TranslationInput(BaseModel):
    sentence: str = Field(
        ..., 
        description="English sentence to translate", 
        min_length=1,
        example="hello world"
    )
    
class TranslationOutput(BaseModel):
    translation: str = Field(
        ..., 
        description="Translated Spanish sentence"
    )
    original: Optional[str] = Field(
        None, 
        description="Original English sentence"
    )