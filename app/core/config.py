import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Psychometric Service"

    # Where the service listens. PORT is also what docker-compose publishes and
    # what the container's healthcheck asks, so one line in .env moves all three.
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "5010"))
    SECRET_KEY: str = os.getenv("SECRET_KEY")
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

    

    BASE_URL: str = "http://localhost:5010"

    

    class Config:
        env_file = ".env"

settings = Settings()