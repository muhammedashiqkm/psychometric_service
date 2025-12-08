import asyncio
import json
from typing import List, Dict, Any
from openai import AsyncOpenAI
import google.generativeai as genai
from fastapi import HTTPException, status

from app.core.config import settings
from app.models.schemas import (
    PsychometricItem,
    PsychometricAnalysisResponse,
    PsychometricSections,
)
from app.core.logging_config import app_logger, error_logger


openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
deepseek_client = AsyncOpenAI(
    api_key=settings.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1"
)

if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)


async def _get_llm_response(prompt: str, model_provider: str) -> str:
    """
    Centralized helper to call the appropriate LLM provider.
    Errors here will now propagate up to be caught as HTTPExceptions.
    """
    try:
        if model_provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("Gemini API Key missing")
            response = await gemini_model.generate_content_async(
                prompt, generation_config={"response_mime_type": "application/json"}
            )
            return response.text

        elif model_provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API Key missing")
            response = await openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        elif model_provider == "deepseek":
            if not settings.DEEPSEEK_API_KEY:
                raise ValueError("DeepSeek API Key missing")
            response = await deepseek_client.chat.completions.create(
                model=settings.DEEPSEEK_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        raise ValueError(f"Unknown model provider: {model_provider}")
    
    except Exception as e:
        error_logger.error(f"LLM Provider Error ({model_provider}): {str(e)}")
        raise e


async def _grade_descriptive_answer(
    item: PsychometricItem, 
    max_marks: float, 
    model_provider: str
) -> float:
    """
    Grades a single descriptive answer.
    Now propagates errors instead of returning 0.0 on failure.
    """
    difficulty_txt = item.question_difficulty_level or "Standard"
    
    scoring_prompt = f"""
    You are an expert examiner. Evaluate the student's answer based on SEMANTIC MEANING, CONTENT RELEVANCE, and DIFFICULTY LEVEL.

    CONTEXT:
    - Difficulty Level: {difficulty_txt}
    - Max Marks: {max_marks}

    GUIDELINES:
    1. Focus on the CONCEPT and MEANING. Do NOT require word-for-word matching.
    2. Consider the DIFFICULTY LEVEL:
        - If "Easy": Expect precision.
        - If "Hard": Award partial marks for demonstrating core understanding even if imperfect.
    3. Be fair but accurate.

    QUESTION: {item.question}
    MODEL SOLUTION: {item.solution or "N/A"}
    STUDENT ANSWER: {item.student_text_answer}

    TASK: Give a numeric score between 0 and {max_marks}.
    OUTPUT JSON STRICTLY: {{ "score": <number> }}
    """

    score_txt = await _get_llm_response(scoring_prompt, model_provider)
    
    try:
        score_clean = score_txt.replace("```json", "").replace("```", "").strip()
        score_json = json.loads(score_clean)
        raw_score = float(score_json.get("score", 0.0))
    except (json.JSONDecodeError, ValueError) as e:
         raise ValueError(f"LLM returned invalid JSON format for grading: {e}")

    if raw_score < 0: return 0.0
    if raw_score > max_marks: return max_marks
    return raw_score


async def _analyze_single_section(
    section_name: str, 
    context_str: str, 
    model_provider: str
) -> Dict[str, str]:
    """
    Generates the description and representation for a SINGLE section.
    Now propagates errors instead of returning fallback strings.
    """
    prompt = f"""
    You are a Psychometric Analyst. Analyze the student's performance for the SECTION: "{section_name}".
    
    DATA FOR THIS SECTION:
    {context_str}

    TASK:
    1. 'description': Define what the section "{section_name}" generally measures (1-2 sentences).
    2. 'representation': Write a summary (2-3 sentences) of the student's specific performance in this section.
       - Highlight strengths or weaknesses based on the provided answers.

    OUTPUT JSON STRICTLY:
    {{ 
      "description": "...", 
      "representation": "..." 
    }}
    """
    
    response_text = await _get_llm_response(prompt, model_provider)
    
    try:
        clean_json = response_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON for section '{section_name}': {e}")

    return {
        "section": section_name,
        "description": data.get("description", "Description unavailable."),
        "representation": data.get("representation", "Representation unavailable.")
    }


async def analyze_data(
    data_list: List[PsychometricItem],
    model_provider: str = "gemini",
) -> PsychometricAnalysisResponse:

    if not data_list:
        raise HTTPException(status_code=400, detail="No data provided in request.")

    first_item = data_list[0]
    category = first_item.category
    test_name = first_item.test_name
    instance_id = first_item.instance_id

    
    descriptive_tasks = []
    descriptive_indices = []

    for index, item in enumerate(data_list):
        q_max = float(item.question_mark or 0.0)
        if q_max < 0: q_max = 0.0
        
        is_descriptive = bool(item.solution and item.solution.strip())
        has_text_answer = bool(item.student_text_answer and item.student_text_answer.strip())

        if is_descriptive and q_max > 0 and has_text_answer:
            descriptive_indices.append(index)
            descriptive_tasks.append(
                _grade_descriptive_answer(item, q_max, model_provider)
            )

    grading_results = []
    if descriptive_tasks:
        try:
            grading_results = await asyncio.gather(*descriptive_tasks)
        except Exception as e:
            error_logger.error(f"Grading Phase Failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"LLM Service Error during grading: {str(e)}"
            )

    grading_map = dict(zip(descriptive_indices, grading_results))

    
    sections_map: Dict[str, Dict[str, Any]] = {}

    for index, item in enumerate(data_list):
        section_name = item.section_name or "General"

        if section_name not in sections_map:
            sections_map[section_name] = {
                "total_obtained": 0.0,
                "total_max": 0.0,
                "entries": []
            }

        q_max = float(item.question_mark or 0.0)
        if q_max < 0: q_max = 0.0
        
        q_obtained = 0.0
        student_response_display = ""
        performance_tag = ""
        correct_answer_display = item.correct_option_text if item.correct_option_text else item.solution or "N/A"
        is_descriptive = bool(item.solution and item.solution.strip())
        
        has_mcq_ids = (
            item.student_selected_id is not None
             and item.correct_option_id is not None
            and item.correct_option_id != 0
        )

        has_scaled_option = bool(
            item.student_selected_option and item.student_selected_option.strip()
        )

        if is_descriptive:
            q_obtained = grading_map.get(index, 0.0)
            student_response_display = f"Text Answer: '{item.student_text_answer}'"

        elif has_mcq_ids:
            mcq_max = float(item.correct_option_score or item.question_mark or 0.0)
            if mcq_max < 0: mcq_max = 0.0
            q_max = mcq_max
            if item.student_selected_id == item.correct_option_id:
                q_obtained = q_max
            is_correct = item.student_selected_id == item.correct_option_id
            performance_tag = "[CORRECT]" if is_correct else "[INCORRECT]"
            student_response_display = f"Selected ID: {item.student_selected_id} ('{item.student_selected_option or ''}')"

        elif has_scaled_option:
            if item.student_selected_option_score is not None:
                q_obtained = float(item.student_selected_option_score)
                if q_obtained < 0: q_obtained = 0.0
                if q_max > 0 and q_obtained > q_max: q_obtained = q_max
            score_disp = item.student_selected_option_score if item.student_selected_option_score is not None else "N/A"
            student_response_display = f"Selected: '{item.student_selected_option}' (Score: {score_disp})"
        
        else:
            student_response_display = "No Answer Provided"

        sections_map[section_name]["total_obtained"] += q_obtained
        sections_map[section_name]["total_max"] += q_max

        entry_str = (
            f"Question: {item.question}\n"
            f"Student Answer: {student_response_display} {performance_tag}\n"
            f"Correct Answer: {correct_answer_display}\n"
            f"Difficulty: {item.question_difficulty_level or 'N/A'}\n"
            f"Points Earned: {q_obtained} out of {q_max}"
        )
        sections_map[section_name]["entries"].append(entry_str)

    
    analysis_tasks = []
    
    for sec_name, data in sections_map.items():
        context_str = "\n---\n".join(data["entries"])
        analysis_tasks.append(
            _analyze_single_section(sec_name, context_str, model_provider)
        )

    try:
        analysis_results = await asyncio.gather(*analysis_tasks)
    except Exception as e:
        error_logger.error(f"Analysis Phase Failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM Service Error during section analysis: {str(e)}"
        )

    
    final_sections_list = []
    result_lookup = {res["section"]: res for res in analysis_results}

    for sec_name, data in sections_map.items():
        llm_res = result_lookup.get(sec_name, {})
        
        t_obt = data["total_obtained"]
        t_max = data["total_max"]
        score_text = f"{round(t_obt, 2)}/{round(t_max, 2)}" if t_max > 0 else "0/0"

        final_sections_list.append(
            PsychometricSections(
                section=sec_name,
                description=llm_res.get("description", "N/A"),
                representation=llm_res.get("representation", "N/A"),
                section_score=score_text
            )
        )

    return PsychometricAnalysisResponse(
        sections=final_sections_list,
        category=category,
        instance_id=instance_id,
        test_name=test_name
    )