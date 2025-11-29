import json
import re
from typing import List
from openai import AsyncOpenAI
import google.generativeai as genai
from fastapi import HTTPException

from app.core.config import settings
from app.models.schemas import PsychometricItem, PsychometricAnalysisResponse
from app.core.logging_config import app_logger, error_logger

# Initialize Clients
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
deepseek_client = AsyncOpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

def extract_json_from_text(text: str) -> dict:
    """Robustly extract JSON from LLM response, handling Markdown blocks."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        raise ValueError("Could not extract JSON from response")

async def analyze_data(
    data_list: List[PsychometricItem], 
    model_provider: str = "gemini"
) -> PsychometricAnalysisResponse:
    
    if not data_list:
        raise HTTPException(status_code=400, detail="No data provided in request.")

    first_item = data_list[0]
    category = first_item.category
    instance_id = first_item.instance_id

    context_parts = []
    
    for item in data_list:
        section = item.section_name or "General"
        
        student_resp = "No Answer Provided"
        perf_tag = ""

        if item.student_text_answer:
            student_resp = f"Text Answer: '{item.student_text_answer}'"
        
        elif item.student_selected_id is not None:
            student_resp = f"Selected Option ID: {item.student_selected_id}"
            
            if item.correct_option_id is not None:
                is_correct = (item.student_selected_id == item.correct_option_id)
                perf_tag = "[CORRECT]" if is_correct else "[INCORRECT]"

        correct_resp = item.correct_option_text or item.solution or "N/A"

        entry = (
            f"Section: {section}\n"
            f"Question: {item.question}\n"
            f"Student Answer: {student_resp} {perf_tag}\n"
            f"Correct Answer: {correct_resp}"
        )
        context_parts.append(entry)

    full_context_str = "\n---\n".join(context_parts)

    prompt = f"""
    You are a Psychometric Analyst. Analyze the student's performance for the category: "{category}".
    
    DATA:
    {full_context_str}

    TASK:
    1. 'description': Define what "{category}" measures (2 sentences).
    2. 'representation': Consolidate summary (2-3 sentences) of performance across sections.

    OUTPUT JSON STRICTLY:
    {{ "category": "{category}", "description": "...", "representation": "..." }}
    """

    try:
        response_text = ""
        
        if model_provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("Gemini API Key missing")
            response = await gemini_model.generate_content_async(
                prompt, generation_config={"response_mime_type": "application/json"}
            )
            response_text = response.text
            
        elif model_provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API Key missing")
            response = await openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content
            
        elif model_provider == "deepseek":
            if not settings.DEEPSEEK_API_KEY:
                raise ValueError("DeepSeek API Key missing")
            response = await deepseek_client.chat.completions.create(
                model=settings.DEEPSEEK_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content

        data = extract_json_from_text(response_text)
        
        return PsychometricAnalysisResponse(
            category=data.get("category", category),
            description=data.get("description", "Description unavailable"),
            representation=data.get("representation", "Representation unavailable"),
            instance_id=instance_id
        )

    except Exception as e:
        error_logger.error(f"LLM Error [{category}]: {str(e)}", exc_info=True)
    
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "LLM_PROCESSING_ERROR",
                "message": "Analysis generation failed.",
            }
        )