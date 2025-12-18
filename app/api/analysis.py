from fastapi import APIRouter, Depends
from app.models.schemas import PsychometricAnalysisResponse, PsychometricRequest
from app.services.llm_service import PsychometricLLMService
from app.core.security import get_current_user
from app.core.logging_config import app_logger

router = APIRouter()

@router.post("/generate", response_model=PsychometricAnalysisResponse)
async def generate_report(
    payload: PsychometricRequest, 
    current_user: str = Depends(get_current_user)
):
    app_logger.info(f"User '{current_user}' requested report generation using {payload.model}")
    service = PsychometricLLMService()
    return await service.analyze_test_performance(payload.data, model_provider=payload.model)