import os
from pydantic import field_validator
from pydantic_settings import BaseSettings
from typing import Optional, List, Union

class Settings(BaseSettings):
    PROJECT_NAME: str = "Psychometric Service"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "unsafe_default")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_HOURS: int = 1
    
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")

    OPENAI_MODEL_NAME: str = "gpt-4o"
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash" 
    DEEPSEEK_MODEL_NAME: str = "deepseek-chat"

    FRONTENT_ALLOWED_ORIGINS: Union[List[str], str] = []

    BASE_URL: str = "http://localhost:8000"

    @field_validator("FRONTENT_ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """
        Parses a comma-separated string OR a JSON-formatted list.
        """
        if isinstance(v, str):
            if v.strip().startswith("["):
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        
        return v or []

    class Config:
        env_file = ".env"

settings = Settings()