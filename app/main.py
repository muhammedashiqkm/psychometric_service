from fastapi import FastAPI
from app.api import auth, analysis
from fastapi.middleware.cors import CORSMiddleware
from app.core.logging_config import configure_logging, app_logger
from app.core.config import settings

configure_logging()

app = FastAPI(title="Psychometric Analysis Service")

app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(analysis.router, prefix="/psychometric", tags=["Analysis"])

if settings.FRONTENT_ALLOWED_ORIGINS:
  app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.FRONTENT_ALLOWED_ORIGINS], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    app_logger.info("System Startup: Psychometric Service is now running.")

@app.get("/")
def health_check():
    app_logger.debug("Health check endpoint accessed.")
    return {"status": "running"}