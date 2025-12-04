import json
from typing import List
from openai import AsyncOpenAI
import google.generativeai as genai
from fastapi import HTTPException

from app.core.config import settings
from app.models.schemas import PsychometricItem, PsychometricAnalysisResponse
from app.core.logging_config import app_logger, error_logger

openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
deepseek_client = AsyncOpenAI(
    api_key=settings.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1"
)

if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)


async def analyze_data(
    data_list: List[PsychometricItem],
    model_provider: str = "gemini",
) -> PsychometricAnalysisResponse:

    if not data_list:
        raise HTTPException(status_code=400, detail="No data provided in request.")

    first_item = data_list[0]
    category = first_item.category
    instance_id = first_item.instance_id

    context_parts: List[str] = []

    total_obtained = 0.0
    total_max = 0.0

    for item in data_list:
        section = item.section_name or "General"

        is_descriptive = bool(item.solution and item.solution.strip())
        has_scaled_option = bool(
            item.student_selected_option and item.student_selected_option.strip()
        )
        has_mcq_ids = (
            item.student_selected_id is not None
             and item.correct_option_id is not None
            and item.correct_option_id != 0
        )

        q_max = 0.0
        q_obtained = 0.0

        if is_descriptive:
            q_max = float(item.question_mark or 0.0)
            if q_max < 0:
                q_max = 0.0

            if (
                q_max > 0
                and item.student_text_answer
                and item.student_text_answer.strip()
            ):
                
                scoring_prompt = f"""
                 You are a strict examiner.

                QUESTION:
                 {item.question}

                MODEL SOLUTION (IDEAL ANSWER):
                 {item.solution or "N/A"}

                STUDENT ANSWER:
                 {item.student_text_answer  or "N/A" }

                TASK:
                  - Give a numeric score between 0 and {q_max}.
                  - Consider correctness, completeness, and relevance.
                  - Be consistent and not overly generous.

                OUTPUT JSON STRICTLY:
                  {{ "score": <number between 0 and {q_max}> }}
                               
                """

                score_response_content = ""
                try:
                    if model_provider == "gemini":
                        if not settings.GEMINI_API_KEY:
                            raise ValueError("Gemini API Key missing")
                        response = await gemini_model.generate_content_async(
                            scoring_prompt,
                            generation_config={
                                "response_mime_type": "application/json"
                            },
                        )
                        score_response_content = response.text

                    elif model_provider == "openai":
                        if not settings.OPENAI_API_KEY:
                            raise ValueError("OpenAI API Key missing")
                        response = await openai_client.chat.completions.create(
                            model=settings.OPENAI_MODEL_NAME,
                            messages=[{"role": "user", "content": scoring_prompt}],
                            response_format={"type": "json_object"},
                        )
                        score_response_content = response.choices[0].message.content

                    elif model_provider == "deepseek":
                        if not settings.DEEPSEEK_API_KEY:
                            raise ValueError("DeepSeek API Key missing")
                        response = await deepseek_client.chat.completions.create(
                            model=settings.DEEPSEEK_MODEL_NAME,
                            messages=[{"role": "user", "content": scoring_prompt}],
                            response_format={"type": "json_object"},
                        )
                        score_response_content = response.choices[0].message.content

                    score_clean = (
                        score_response_content.replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )
                    score_data = json.loads(score_clean)
                    raw_score = float(score_data.get("score", 0.0))
                except Exception as e:
                    error_logger.error(
                        f"Descriptive scoring failed for question_id={item.question_id}: {e}",
                        exc_info=True,
                    )
                    raw_score = 0.0

                if raw_score < 0:
                    q_obtained = 0.0
                elif raw_score > q_max:
                    q_obtained = q_max
                else:
                    q_obtained = raw_score
            else:
                q_obtained = 0.0

        elif has_scaled_option:
            q_max = float(item.question_mark or 0.0)
            if q_max < 0:
                q_max = 0.0

            if item.student_selected_option_score is not None:
                q_obtained = float(item.student_selected_option_score or 0.0)
                if q_obtained < 0:
                    q_obtained = 0.0
                if q_max > 0 and q_obtained > q_max:
                    q_obtained = q_max
            else:
                q_obtained = 0.0

        elif has_mcq_ids:
            q_max = float(item.correct_option_score or item.question_mark or 0.0)
            if q_max < 0:
                q_max = 0.0

            if item.student_selected_id == item.correct_option_id:
                q_obtained = q_max
            else:
                q_obtained = 0.0

        else:
            q_max = float(item.question_mark or 0.0)
            if q_max < 0:
                q_max = 0.0
            q_obtained = 0.0

        total_obtained += q_obtained
        total_max += q_max

        student_response_display = ""
        performance_tag = ""

        if is_descriptive and item.student_text_answer:
            student_response_display = f"Text Answer: '{item.student_text_answer}'"

        elif has_mcq_ids:
            student_response_display = f"Selected Option ID: {item.student_selected_id}"
            if item.correct_option_id is not None and item.correct_option_id != 0:
                is_correct = item.student_selected_id == item.correct_option_id
                performance_tag = "[CORRECT]" if is_correct else "[INCORRECT]"

        elif has_scaled_option:
            score_display = (
                item.student_selected_option_score
                if item.student_selected_option_score is not None
                else "N/A"
            )
            student_response_display = (
                f"Selected Option: '{item.student_selected_option}' "
                f"(Score: {score_display})"
            )
        else:
            student_response_display = "No Answer Provided"

        correct_answer_display = (
            item.correct_option_text if item.correct_option_text else item.solution
        )
        if not correct_answer_display:
            correct_answer_display = "N/A"

        entry = (
            f"Section: {section}\n"
            f"Question: {item.question}\n"
            f"Student Answer: {student_response_display} {performance_tag}\n"
            f"Correct Answer/Solution: {correct_answer_display}\n"
            f"Marks: {q_obtained} / {q_max}"
        )
        context_parts.append(entry)

    full_context_str = "\n---\n".join(context_parts)

    if total_max > 0:
        category_score_text = (
            f"{round(total_obtained, 2)}/{round(total_max, 2)}"
        )
    else:
        category_score_text = "0/0"

    prompt = f"""
    You are a Psychometric Analyst. Analyze the student's performance for the category: "{category}".
    The data below includes questions from multiple sections (Objective and Descriptive).

    DATA:
    {full_context_str}

    TASK:
    1. 'description': Define what the category "{category}" measures in general (2 sentences).
    2. 'representation': Write a SINGLE, consolidated summary (2-3 sentences) of the student's performance across ALL sections. 
       - Evaluate their accuracy in objective questions.
       - Evaluate the quality of their text answers in descriptive questions.
       - Highlight any difference in performance between sections (e.g., Section A vs Section B) if noticeable.
       - ALSO mention the total score clearly in this summary (example: "The student scored 56 out of 60.")

    OUTPUT JSON STRICTLY:
    {{ "description": "...", "representation": "..." }}
    """

    response_content = ""

    try:
        if model_provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("Gemini API Key missing")
            response = await gemini_model.generate_content_async(
                prompt, generation_config={"response_mime_type": "application/json"}
            )
            response_content = response.text

        elif model_provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API Key missing")
            response = await openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            response_content = response.choices[0].message.content

        elif model_provider == "deepseek":
            if not settings.DEEPSEEK_API_KEY:
                raise ValueError("DeepSeek API Key missing")
            response = await deepseek_client.chat.completions.create(
                model=settings.DEEPSEEK_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            response_content = response.choices[0].message.content

        clean_json = response_content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)

        return PsychometricAnalysisResponse(
            category=category,
            description=data.get("description", "Description unavailable"),
            representation=data.get("representation", "Representation unavailable"),
            instance_id=instance_id,
            category_score=category_score_text,
        )

    except Exception as e:
        error_logger.error(f"LLM Generation Failed for {category}: {e}", exc_info=True)

        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "LLM_PROCESSING_ERROR",
                "message": f"Failed to generate analysis using {model_provider}.",
                "technical_details": str(e),
            },
        )
