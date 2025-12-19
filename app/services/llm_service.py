import asyncio
import json
import random
import logging
from typing import List, Dict, Any, Optional

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


try:
    openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    deepseek_client = AsyncOpenAI(
        api_key=settings.DEEPSEEK_API_KEY, 
        base_url="https://api.deepseek.com/v1"
    )
    
    if settings.GEMINI_API_KEY:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
    else:
        gemini_model = None
        app_logger.warning("Gemini API Key not found. Gemini provider will fail.")

except Exception as e:
    error_logger.critical(f"Failed to initialize LLM clients: {e}")
    raise e


class PromptTemplates:
    INTEREST_GRADING = """
You are acting as a professional career counselor.
Your task is to evaluate a student's response to a psychometric question.

CONTEXT:
- Test Name: {test_name}
- Section Name: {section_name}
- Dimension/Trait Measured: {trait_desc}
- INPUT TYPE: {input_type_desc}

EVALUATION RULES:
{grading_instruction}

SCORING SCALE (0–{max_marks}):
- Low Score: Indicates weak or negative alignment with the trait.
- Medium Score: Neutral or moderate alignment.
- High Score: Strong positive alignment with the trait.

STUDENT RESPONSE:
"{answer_content}"

IMPORTANT RULES:
- Analyze the phrasing of the question carefully to detect if it is a "negative" question.
- Do NOT judge desirability.
- Do NOT infer personality beyond the response itself.

OUTPUT REQUIREMENTS:
- Return ONLY valid JSON.
- No explanations.

OUTPUT JSON FORMAT:
{{
  "score": <numeric value between 0 and {max_marks}>
}}
"""

    ACADEMIC_GRADING = """
You are acting as a certified psychometric examiner and academic evaluator.
Your responsibility is to assess the quality of a student's response to an academic or cognitive question.

EVALUATION OBJECTIVE:
Determine how well the student's answer demonstrates understanding of the underlying concept or skill being tested.

QUESTION CONTEXT:
- Test Name: {test_name}
- Section Name: {section_name}
- Difficulty Level: {difficulty_level}

EXPECTED KNOWLEDGE / MODEL SOLUTION:
{model_solution}

QUESTION PRESENTED TO STUDENT:
"{question}"

STUDENT RESPONSE:
"{answer_content}"

SCORING CONSTRAINTS:
- The maximum possible score is {max_marks}.
- Focus on conceptual correctness and logical reasoning.
- Ignore spelling errors.

OUTPUT REQUIREMENTS:
- Return ONLY a valid JSON object.
- No explanations.

OUTPUT JSON FORMAT:
{{
  "score": <numeric value between 0 and {max_marks}>
}}
"""

    SECTION_ANALYSIS = """
You are acting as a psychometric analyst preparing a formal assessment report.
Your task is to interpret the student’s overall performance within ONE specific section of a test.

SECTION NAME:
"{section_name}"

SECTION DATA CONTEXT:
{context_str}

INTERPRETATION OBJECTIVE:
Provide a concise, high-level interpretation of the student’s performance in this section.

INTERPRETATION GUIDELINES:
- Focus on overall strength or weakness.
- Identify consistency or hesitation patterns.
- Use neutral, student-friendly language.
- Do NOT mention specific questions.
- Do NOT include scores or numbers.

LENGTH CONSTRAINT:
- Limit to 1–2 complete sentences.

OUTPUT REQUIREMENTS:
- Return ONLY valid JSON.

OUTPUT JSON FORMAT:
{{
  "interpretation": "<section-level interpretation>"
}}
"""

    TEST_SUMMARY = """
You are acting as a senior psychometric analyst.
Your task is to produce a high-level narrative summary of a psychometric test based on
section-level performance patterns only.

TEST INFORMATION:
- Test Name: {test_name}
- Test Category: {category}

SECTION PERFORMANCE PROFILE:
{section_performance_profile}

OBJECTIVES:
1. DESCRIPTION: One sentence explaining what this test measures.
2. REPRESENTATION: 2-3 sentences summarizing the student’s overall performance (strengths, balance, consistency).

STRICT RULES:
- Do NOT include scores or percentages.
- Do NOT mention individual questions.
- Do NOT give advice.
- Maintain a neutral tone.

OUTPUT JSON FORMAT:
{{
  "description": "...",
  "representation": "..."
}}
"""


class PsychometricLLMService:
    def __init__(self):
        self.sem = asyncio.Semaphore(20)

    async def _safe_llm_call(self, prompt: str, model_provider: str, max_retries: int = 3) -> str:
        """
        Executes an LLM call with Rate Limiting and Retries.
        Raises 503 immediately if all retries fail.
        """
        async with self.sem:
            attempt = 0
            
            while attempt < max_retries:
                try:
                    if model_provider == "gemini":
                        if not gemini_model:
                            raise ValueError("Gemini API Key missing")
                        response = await gemini_model.generate_content_async(
                            prompt, 
                            generation_config={"response_mime_type": "application/json"}
                        )
                        return response.text

                    elif model_provider == "openai":
                        response = await openai_client.chat.completions.create(
                            model=settings.OPENAI_MODEL_NAME,
                            messages=[{"role": "user", "content": prompt}],
                            response_format={"type": "json_object"},
                        )
                        return response.choices[0].message.content

                    elif model_provider == "deepseek":
                        response = await deepseek_client.chat.completions.create(
                            model=settings.DEEPSEEK_MODEL_NAME,
                            messages=[{"role": "user", "content": prompt}],
                            response_format={"type": "json_object"},
                        )
                        return response.choices[0].message.content

                    else:
                        raise ValueError(f"Unknown model provider: {model_provider}")

                except Exception as e:
                    is_retryable = False
                    if "429" in str(e) or "503" in str(e) or "500" in str(e):
                        is_retryable = True
                    
                    if is_retryable and attempt < max_retries - 1:
                        sleep_time = (2 ** attempt) + random.uniform(0, 1)
                        app_logger.warning(f"LLM Error ({model_provider}): {e}. Retrying in {sleep_time:.2f}s...")
                        await asyncio.sleep(sleep_time)
                        attempt += 1
                    else:
                        error_msg = f"LLM Service Unavailable ({model_provider}): {str(e)}"
                        error_logger.error(error_msg)
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                            detail=error_msg
                        )

    async def _grade_answer(
        self, 
        item: PsychometricItem, 
        max_marks: float, 
        model_provider: str,
        override_answer_text: Optional[str] = None,
        is_interest_check: bool = False
    ) -> float:
        
        answer_content = override_answer_text if override_answer_text else item.student_text_answer
        if not answer_content:
            return 0.0

        test_name = item.test_name or "Psychometric Assessment"
        section_name = item.section_name or "General"

        if item.student_text_answer and item.student_text_answer.strip():
             input_type_desc = "Free Text Explanation"
             grading_instruction = "Grade based on depth, engagement, and personal elaboration."
        else:
             input_type_desc = "Selected Short Option"
             grading_instruction = """Determine if the option indicates the PRESENCE or ABSENCE of the trait.
             - HIGH SCORE: The option confirms the trait exists strongly.
               (e.g., 'Strongly Agree' to a positive question OR 'Strongly Disagree' to a negative question).
             - LOW SCORE: The option indicates the trait is missing or weak.
               (e.g., 'Strongly Disagree' to a positive question OR 'Strongly Agree' to a negative question).
             - MID SCORE: Neutral, 'Maybe', or ambivalent answers."""

        if is_interest_check:
            trait_desc = item.question or "Interest in this topic"
            final_prompt = PromptTemplates.INTEREST_GRADING.format(
                test_name=test_name,
                section_name=section_name,
                trait_desc=trait_desc,
                input_type_desc=input_type_desc,
                grading_instruction=grading_instruction,
                max_marks=max_marks,
                answer_content=answer_content
            )
        else:
            difficulty_level = item.question_difficulty_level or "Standard"
            model_solution = item.solution or "Evaluate based on logical consistency and accuracy."
            final_prompt = PromptTemplates.ACADEMIC_GRADING.format(
                test_name=test_name,
                section_name=section_name,
                difficulty_level=difficulty_level,
                model_solution=model_solution,
                question=item.question,
                answer_content=answer_content,
                max_marks=max_marks
            )

        try:
            score_txt = await self._safe_llm_call(final_prompt, model_provider)
            score_clean = score_txt.replace("```json", "").replace("```", "").strip()
            score_json = json.loads(score_clean)
            raw_score = float(score_json.get("score", 0.0))
            
            if raw_score < 0: return 0.0
            if raw_score > max_marks: return max_marks
            return raw_score
            
        except (json.JSONDecodeError, ValueError):
             return 0.0

    async def _analyze_single_section(
        self, section_name: str, context_str: str, model_provider: str
    ) -> Dict[str, str]:
        
        prompt = PromptTemplates.SECTION_ANALYSIS.format(
            section_name=section_name,
            context_str=context_str
        )
        
        response_text = await self._safe_llm_call(prompt, model_provider)
        
        try:
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            return {
                "section": section_name,
                "interpretation": data.get("interpretation", "Interpretation unavailable.")
            }
        except json.JSONDecodeError as e:
            error_logger.error(f"Section analysis JSON failed for '{section_name}': {e}")
            return {
                "section": section_name,
                "interpretation": "Analysis unavailable due to processing error."
            }

    async def _generate_test_summary(
        self, test_name: str, category: str, instance_id: int, 
        section_performance_profile: str, model_provider: str
    ) -> Dict[str, str]:
        
        prompt = PromptTemplates.TEST_SUMMARY.format(
            test_name=test_name,
            category=category,
            section_performance_profile=section_performance_profile
        )
        
        response_text = await self._safe_llm_call(prompt, model_provider)
        
        try:
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except json.JSONDecodeError as e:
            error_logger.error(f"Test summary LLM JSON failed: {e}")
            return {}

    async def analyze_test_performance(
        self,
        data_list: List[PsychometricItem],
        model_provider: str = "gemini",
    ) -> PsychometricAnalysisResponse:

        if not data_list:
            raise HTTPException(status_code=400, detail="No data provided in request.")

        first_item = data_list[0]
        category = first_item.category
        test_name = first_item.test_name
        instance_id = first_item.instance_id

        academic_tasks = []
        academic_indices = []
        interest_tasks = []
        interest_indices = []
        interest_item_indices = set()

        for index, item in enumerate(data_list):
            db_mark = float(item.question_mark or 0.0)
            
            has_text_answer = bool(item.student_text_answer and item.student_text_answer.strip())
            has_option_text = bool(item.student_selected_option and item.student_selected_option.strip())
            
            if db_mark <= 0:
                interest_item_indices.add(index)
                q_max = 5.0 
            else:
                q_max = db_mark

            is_standard_descriptive = (bool(item.solution) and db_mark > 0 and has_text_answer)
            is_interest_grading = (index in interest_item_indices and (has_text_answer or has_option_text))

            if is_standard_descriptive:
                academic_indices.append(index)
                academic_tasks.append(
                    self._grade_answer(item, q_max, model_provider, is_interest_check=False)
                )

            elif is_interest_grading:
                text_to_grade = item.student_text_answer if has_text_answer else item.student_selected_option
                interest_indices.append(index)
                interest_tasks.append(
                    self._grade_answer(
                        item, 
                        q_max, 
                        model_provider, 
                        override_answer_text=text_to_grade, 
                        is_interest_check=True
                    )
                )

        academic_results = []
        if academic_tasks:
            academic_results = await asyncio.gather(*academic_tasks)

        interest_results = []
        if interest_tasks:
            interest_results = await asyncio.gather(*interest_tasks)

        academic_score_map = dict(zip(academic_indices, academic_results))
        interest_score_map = dict(zip(interest_indices, interest_results))

        sections_map: Dict[str, Dict[str, Any]] = {}

        for index, item in enumerate(data_list):
            section_name = item.section_name or "General"

            if section_name not in sections_map:
                sections_map[section_name] = {
                    "total_obtained": 0.0,
                    "total_max": 0.0,
                    "entries": []
                }

            if index in interest_item_indices:
                q_max = 5.0
            else:
                q_max = float(item.question_mark or 0.0)
                if q_max < 0: q_max = 0.0
            
            q_obtained = 0.0
            student_response_display = ""
            performance_tag = ""
            correct_answer_display = item.correct_option_text if item.correct_option_text else item.solution or "N/A"
            
            is_graded_interest = index in interest_score_map
            is_graded_academic = index in academic_score_map
            
            has_mcq_ids = (item.student_selected_id is not None and item.correct_option_id is not None and item.correct_option_id != 0)
            has_scaled_option = bool(item.student_selected_option and item.student_selected_option.strip())

            if is_graded_interest:
                q_obtained = interest_score_map.get(index, 0.0)
                used_text = item.student_text_answer or item.student_selected_option
                student_response_display = f"Interest Response: '{used_text}' (AI Graded)"
                correct_answer_display = "Evaluated on Interest/Engagement"

            elif index in interest_item_indices:
                 q_obtained = 0.0
                 student_response_display = "No Answer Provided (Skipped)"
                 correct_answer_display = "Evaluated on Interest/Engagement"

            elif is_graded_academic:
                q_obtained = academic_score_map.get(index, 0.0)
                student_response_display = f"Text Answer: '{item.student_text_answer}'"

            elif has_mcq_ids:
                mcq_max = float(item.correct_option_score or item.question_mark or 0.0)
                if mcq_max < 0: mcq_max = 0.0
                q_max = mcq_max 
                if item.student_selected_id == item.correct_option_id:
                    q_obtained = q_max
                is_correct = item.student_selected_id == item.correct_option_id
                performance_tag = "[CORRECT]" if is_correct else "[INCORRECT]"
                student_response_display = f"Selected ID: {item.student_selected_id}"

            elif has_scaled_option:
                if item.student_selected_option_score is not None:
                    q_obtained = float(item.student_selected_option_score)
                    if q_obtained < 0: q_obtained = 0.0
                    if q_max > 0 and q_obtained > q_max: q_obtained = q_max
                score_disp = item.student_selected_option_score
                student_response_display = f"Selected: '{item.student_selected_option}' (Score: {score_disp})"
            
            else:
                student_response_display = "No Answer Provided"

            sections_map[section_name]["total_obtained"] += q_obtained
            sections_map[section_name]["total_max"] += q_max

            entry_str = (
                f"Question: {item.question}\n"
                f"Student Answer: {student_response_display} {performance_tag}\n"
                f"Correct Answer: {correct_answer_display}\n"
                f"Points Earned: {q_obtained} out of {q_max}"
            )
            sections_map[section_name]["entries"].append(entry_str)

        analysis_tasks = []
        for sec_name, data in sections_map.items():
            context_str = "\n---\n".join(data["entries"])
            analysis_tasks.append(
                self._analyze_single_section(sec_name, context_str, model_provider)
            )

        analysis_results = []
        if analysis_tasks:
            analysis_results = await asyncio.gather(*analysis_tasks)

        final_sections_list = []
        result_lookup = {res["section"]: res for res in analysis_results}
        profile_lines = []

        for sec_name, data in sections_map.items():
            llm_res = result_lookup.get(sec_name, {})
            interpretation_text = llm_res.get("interpretation", "Analysis pending.")
            t_obt = data["total_obtained"]
            t_max = data["total_max"]
            score_text = f"{round(t_obt, 2)}/{round(t_max, 2)}" if t_max > 0 else "0/0"

            final_sections_list.append(
                PsychometricSections(
                    section=sec_name,
                    interpretation=interpretation_text,
                    section_score=score_text
                )
            )
            profile_lines.append(f"- Section '{sec_name}': {interpretation_text}")

        section_performance_profile_str = "\n".join(profile_lines)

        test_summary_data = await self._generate_test_summary(
            test_name, category, instance_id, section_performance_profile_str, model_provider
        )

        response = PsychometricAnalysisResponse(
            sections=final_sections_list,
            category=category,
            description=test_summary_data.get("description", "N/A"),
            representation=test_summary_data.get("representation", "N/A"),
            instance_id=instance_id,
            test_name=test_name
        )

        return response