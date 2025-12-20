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
    
    BATCH_ACADEMIC = """
You are an expert academic examiner.
Your task is to grade a batch of student answers by comparing them to the official Model Solution.

CONTEXT:
- Test Name: {test_name}
- Section Name: {section_name}

GRADING RULES:
1. **Comparison:** Compare the "Student Answer" against the "Model Solution".
2. **Concept Match:** If the meaning matches (even if phrased differently), award full marks.
3. **Incorrect:** If the answer is factually wrong or irrelevant, award 0.
4. **Partial:** Award partial marks ONLY if the answer is partially correct and the max score allows it.

INPUT DATA (JSON):
{items_json}

OUTPUT FORMAT (Strict JSON List):
[
  {{ "id": 101, "score": 1.0 }},
  {{ "id": 102, "score": 0.0 }}
]
"""

    BATCH_INTEREST = """
You are a professional psychometric evaluator.
Your task is to score a batch of student responses to determine how strongly they possess the specific Trait defined by the Section Name.

CONTEXT:
- Test Name: {test_name}
- **TRAIT BEING MEASURED:** {section_name}

SCORING RULES (0 to Max Mark):
1. **Identify Question Polarity:**
   - **Positive Question:** Directly supports the trait (e.g., "I love leading teams" for Leadership).
   - **Negative Question:** Opposes the trait (e.g., "I shy away from responsibility" for Leadership).

2. **Score based on TRAIT POSSESSION (Not just agreement):**
   - **High Score (Max):** The answer proves the student **HAS** the trait.
     (Examples: Agreeing with a Positive question OR Disagreeing with a Negative question).
   - **Low Score (0):** The answer proves the student **LACKS** the trait.
     (Examples: Disagreeing with a Positive question OR Agreeing with a Negative question).
   - **Mid Score (~50%):** Answer is neutral, hesitant, or "Sometimes".

3. **Context Awareness:**
   - If a student answers logically/analytically in a "Creative" section, score LOW for Creativity.

INPUT DATA (JSON):
{items_json}

OUTPUT FORMAT (Strict JSON List):
[
  {{ "id": 201, "score": 5.0 }},
  {{ "id": 202, "score": 0.0 }}
]
"""

    SECTION_ANALYSIS = """
You are acting as a senior psychometric analyst writing a formal report.
Interpret the student’s performance for the section: "{section_name}".

DATA CONTEXT:
{context_str}

WRITING GUIDELINES:
1. **Voice:** Use strictly **Third-Person** language (e.g., "The student", "The candidate", "They"). **DO NOT use "You".**
2. **Insightful Analysis:** - If scores are HIGH: Explain what strengths the student demonstrated.
   - If scores are LOW: Explain *what preference they showed instead* (e.g., "The student prioritized analytical facts over creative expression" rather than just "The student lacks creativity").
3. **Tone:** Professional, objective, and constructive.
4. **Length:** Limit to 1–2 concise, impactful sentences.

OUTPUT JSON FORMAT:
{{
  "interpretation": "<interpretation text>"
}}
"""

    TEST_SUMMARY = """
You are a senior psychometric analyst.
Produce a high-level summary of the test based on the section profiles provided.

TEST INFORMATION:
- Test: {test_name}
- Category: {category}

SECTION PROFILES:
{section_performance_profile}

OBJECTIVES:
1. **Description:** 1-2 sentence explaining what this specific test measures.
2. **Representation:** 2-3 sentences summarizing the student’s overall profile. Use **Third-Person** voice (e.g., "The student exhibits..."). Highlight their dominant traits and any notable gaps.

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
        Executes an LLM call with built-in Rate Limiting (Semaphore) and Retries.
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
                    is_retryable = "429" in str(e) or "503" in str(e) or "500" in str(e)
                    if is_retryable and attempt < max_retries - 1:
                        sleep_time = (2 ** attempt) + random.uniform(0, 1)
                        app_logger.warning(f"LLM Retry ({attempt+1}) for {model_provider}: {e}")
                        await asyncio.sleep(sleep_time)
                        attempt += 1
                    else:
                        error_logger.error(f"LLM Fatal Error: {str(e)}")
                        return "{}"

    async def _process_batch(
        self, 
        batch_type: str, 
        section_name: str, 
        test_name: str, 
        batch_items: List[Dict[str, Any]], 
        model_provider: str
    ) -> Dict[int, float]:
        
        if not batch_items:
            return {}

        items_payload = json.dumps(batch_items, indent=2)
        
        template = PromptTemplates.BATCH_ACADEMIC if batch_type == "academic" else PromptTemplates.BATCH_INTEREST

        prompt = template.format(
            test_name=test_name,
            section_name=section_name,
            items_json=items_payload
        )

        try:
            response_text = await self._safe_llm_call(prompt, model_provider)
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            
            try:
                data = json.loads(clean_json)
            except json.JSONDecodeError:
                return {}

            if isinstance(data, dict):
                for key in ["scores", "results", "data", "grading"]:
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
            
            if not isinstance(data, list):
                return {}

            score_map = {}
            for res in data:
                idx = res.get("id")
                score = res.get("score")
                if idx is not None and score is not None:
                    try:
                        score_map[idx] = float(score)
                    except (ValueError, TypeError):
                        score_map[idx] = 0.0
            
            return score_map

        except Exception as e:
            error_logger.error(f"Batch ({batch_type}) failed for {section_name}: {e}")
            return {}

    async def _analyze_single_section(
        self, section_name: str, context_str: str, model_provider: str
    ) -> Dict[str, str]:
        
        prompt = PromptTemplates.SECTION_ANALYSIS.format(
            section_name=section_name,
            context_str=context_str
        )
        try:
            response_text = await self._safe_llm_call(prompt, model_provider)
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            return {
                "section": section_name,
                "interpretation": data.get("interpretation", "Analysis unavailable.")
            }
        except Exception:
            return {"section": section_name, "interpretation": "Analysis unavailable."}

    async def _generate_test_summary(
        self, test_name: str, category: str, instance_id: int, 
        section_performance_profile: str, model_provider: str
    ) -> Dict[str, str]:
        
        prompt = PromptTemplates.TEST_SUMMARY.format(
            test_name=test_name,
            category=category,
            section_performance_profile=section_performance_profile
        )
        try:
            response_text = await self._safe_llm_call(prompt, model_provider)
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception:
            return {}

    async def analyze_test_performance(
        self,
        data_list: List[PsychometricItem],
        model_provider: str = "gemini",
    ) -> PsychometricAnalysisResponse:

        if not data_list:
            raise HTTPException(status_code=400, detail="No data provided.")

        first_item = data_list[0]
        category = first_item.category
        test_name = first_item.test_name
        instance_id = first_item.instance_id


        section_buckets: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        
        final_results: Dict[int, Dict[str, Any]] = {}

        for index, item in enumerate(data_list):
            
            try:
                raw_mark = float(item.question_mark) if item.question_mark else 0.0
                q_max = raw_mark if raw_mark > 0 else 5.0
            except (ValueError, TypeError):
                q_max = 5.0

            sec_name = item.section_name or "General"
            if sec_name not in section_buckets:
                section_buckets[sec_name] = {"academic": [], "interest": []}
                
            has_text = bool(item.student_text_answer and item.student_text_answer.strip())
            has_solution = bool(item.solution and item.solution.strip())
            has_sid = (item.student_selected_id is not None and item.student_selected_id != 0)
            has_cid = (item.correct_option_id is not None and item.correct_option_id != 0)
            has_option_text = bool(item.student_selected_option and item.student_selected_option.strip())

            if has_solution and has_text:
                section_buckets[sec_name]["academic"].append({
                    "id": index,
                    "question": item.question,
                    "student_answer": item.student_text_answer,
                    "model_solution": item.solution,
                    "max_marks": q_max
                })

                final_results[index] = {"max": q_max, "obtained": 0.0, "status": "Academic", "disp": item.student_text_answer}

            elif has_cid and has_sid:
                score = q_max if item.correct_option_id == item.student_selected_id else 0.0
                status_txt = "MCQ Correct" if score > 0 else "MCQ Incorrect"
                final_results[index] = {"max": q_max, "obtained": score, "status": status_txt, "disp": f"Option ID {item.student_selected_id}"}

            elif has_text or has_option_text:
                ans_text = item.student_text_answer if has_text else item.student_selected_option
                sub_type = "Text" if has_text else "Option"
                
                section_buckets[sec_name]["interest"].append({
                    "id": index,
                    "question": item.question,
                    "student_answer": ans_text,
                    "max_marks": q_max
                })
                final_results[index] = {"max": q_max, "obtained": 0.0, "status": f"Interest ({sub_type})", "disp": ans_text}

            else:
                final_results[index] = {"max": q_max, "obtained": 0.0, "status": "Unanswered", "disp": "No Answer"}

        batch_tasks = []
        
        for sec_name, buckets in section_buckets.items():
            if buckets["academic"]:
                batch_tasks.append(
                    self._process_batch("academic", sec_name, test_name, buckets["academic"], model_provider)
                )
            if buckets["interest"]:
                batch_tasks.append(
                    self._process_batch("interest", sec_name, test_name, buckets["interest"], model_provider)
                )

        if batch_tasks:
            batch_results_list = await asyncio.gather(*batch_tasks)
            
            for result_map in batch_results_list:
                for idx, score in result_map.items():
                    if idx in final_results:
                        max_allowed = final_results[idx]["max"]
                        final_results[idx]["obtained"] = min(score, max_allowed)

        sections_map: Dict[str, Dict[str, Any]] = {}

        for index, item in enumerate(data_list):
            sec_name = item.section_name or "General"
            if sec_name not in sections_map:
                sections_map[sec_name] = {"total_obtained": 0.0, "total_max": 0.0, "entries": []}

            res = final_results[index]
            
            sections_map[sec_name]["total_obtained"] += res["obtained"]
            sections_map[sec_name]["total_max"] += res["max"]
            
            entry_str = (
                f"Q: {item.question}\n"
                f"Ans: {res['disp']}\n"
                f"Score: {res['obtained']}/{res['max']}"
            )
            sections_map[sec_name]["entries"].append(entry_str)

        analysis_tasks = []
        for sec_name, data in sections_map.items():
            context_str = "\n---\n".join(data["entries"])
            analysis_tasks.append(
                self._analyze_single_section(sec_name, context_str, model_provider)
            )

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

            final_sections_list.append(PsychometricSections(
                section=sec_name,
                interpretation=interpretation_text,
                section_score=score_text
            ))
            profile_lines.append(f"- Section '{sec_name}': {interpretation_text}")

        test_summary = await self._generate_test_summary(
            test_name, category, instance_id, "\n".join(profile_lines), model_provider
        )

        return PsychometricAnalysisResponse(
            sections=final_sections_list,
            category=category,
            description=test_summary.get("description", "N/A"),
            representation=test_summary.get("representation", "N/A"),
            instance_id=instance_id,
            test_name=test_name
        )