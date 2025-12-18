import asyncio
import json
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

class PsychometricLLMService:
    def __init__(self):
        """
        Initialize LLM clients based on configuration.
        """
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        self.deepseek_client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY, 
            base_url="https://api.deepseek.com/v1"
        )
        
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
        else:
            self.gemini_model = None

    async def _get_llm_response(self, prompt: str, model_provider: str) -> str:
        """
        Routes the prompt to the requested LLM provider (Gemini, OpenAI, or DeepSeek).
        """
        try:
            if model_provider == "gemini":
                if not self.gemini_model:
                    raise ValueError("Gemini API Key missing")
                
                response = await self.gemini_model.generate_content_async(
                    prompt, 
                    generation_config={"response_mime_type": "application/json"}
                )
                return response.text

            elif model_provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model=settings.OPENAI_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content

            elif model_provider == "deepseek":
                response = await self.deepseek_client.chat.completions.create(
                    model=settings.DEEPSEEK_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content

            raise ValueError(f"Unknown model provider: {model_provider}")
        
        except Exception as e:
            error_logger.error(f"LLM Provider Error ({model_provider}): {str(e)}")
            raise e

    async def _grade_answer(
        self, 
        item: PsychometricItem, 
        max_marks: float, 
        model_provider: str,
        override_answer_text: Optional[str] = None,
        is_interest_check: bool = False
    ) -> float:
        """
        Grades a student answer using an LLM.
        
        Args:
            item: The question item.
            max_marks: The maximum score allowed.
            model_provider: 'gemini', 'openai', etc.
            override_answer_text: If provided, grades this text instead of student_text_answer.
            is_interest_check: If True, uses the 'Career Counselor' prompt to gauge interest.
        """
        
        answer_content = override_answer_text if override_answer_text else item.student_text_answer
        if not answer_content:
            return 0.0

        if is_interest_check:

            system_role = "You are a Career Counselor and Psychometric Analyst."
            
            scoring_task = f"""
            TASK: Evaluate the STUDENT'S LEVEL OF INTEREST and ENGAGEMENT shown in the answer.
            
            SCORING GUIDE (0 to {max_marks}):
            - Low Score (0-{max_marks*0.3}): Disinterest, one-word answers, avoidance, or negative sentiment.
            - Mid Score ({max_marks*0.4}-{max_marks*0.7}): Neutral, standard compliance, factual but emotionless.
            - High Score ({max_marks*0.8}-{max_marks}): High enthusiasm, personal elaboration, specific examples, or strong positive sentiment.
            
            QUESTION CONTEXT: This question measures the student's interest in: "{item.category or 'this topic'}".
            """
        else:
            system_role = "You are an Expert Examiner."
            difficulty_txt = item.question_difficulty_level or "Standard"
            solution_context = item.solution or "Evaluate based on logical consistency and accuracy."
            
            scoring_task = f"""
            TASK: Evaluate the ACCURACY and SEMANTIC MEANING of the answer compared to the solution.
            
            CONTEXT:
            - Difficulty: {difficulty_txt}
            - Model Solution: {solution_context}
            
            GUIDELINES:
            - Focus on concept mastery, not word-for-word matching.
            - Award partial marks for correct reasoning.
            """

        final_prompt = f"""
        {system_role}
        
        {scoring_task}

        INPUT DATA:
        - Question: "{item.question}"
        - Student Answer: "{answer_content}"
        - Max Marks: {max_marks}

        OUTPUT:
        Return a JSON object with a single key "score" (numeric).
        OUTPUT JSON STRICTLY: {{ "score": <number> }}
        """

        score_txt = await self._get_llm_response(final_prompt, model_provider)
        
        try:
            score_clean = score_txt.replace("```json", "").replace("```", "").strip()
            score_json = json.loads(score_clean)
            raw_score = float(score_json.get("score", 0.0))
        except (json.JSONDecodeError, ValueError) as e:
             error_logger.warning(f"Grading JSON parse failed for QID {item.question_id}: {e}")
             return 0.0

        if raw_score < 0: return 0.0
        if raw_score > max_marks: return max_marks
        return raw_score

    async def _analyze_single_section(
        self, section_name: str, context_str: str, model_provider: str
    ) -> Dict[str, str]:
        """
        Generates the interpretation for a SINGLE section.
        """
        prompt = f"""
        You are a Psychometric Analyst. Analyze the student's performance for the SECTION: "{section_name}".
        
        DATA FOR THIS SECTION:
        {context_str}

        TASK:
        1. 'interpretation': Write a short (1-2 sentence) interpretation of the student's specific performance in this section.
           - Highlight strengths or weaknesses based on the provided answers.
           - Keep it concise.

        OUTPUT JSON STRICTLY:
        {{ 
          "interpretation": "..."
        }}
        """
        
        try:
            response_text = await self._get_llm_response(prompt, model_provider)
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            return {
                "section": section_name,
                "interpretation": data.get("interpretation", "Interpretation unavailable.")
            }
        except Exception as e:
            error_logger.error(f"Section analysis JSON failed for '{section_name}': {e}")
            return {
                "section": section_name,
                "interpretation": "Analysis unavailable due to processing error."
            }

    async def _generate_test_summary(
        self, test_name: str, category: str, instance_id: int, 
        sections_map: Dict, model_provider: str
    ) -> Dict[str, str]:
        """
        Generates the overall description and representation for the entire test.
        """
        sections_summary_json = json.dumps({
            k: f'{round(v["total_obtained"],2)}/{round(v["total_max"],2)}' 
            for k, v in sections_map.items()
        })
        
        overall_score = f"{sum(v['total_obtained'] for v in sections_map.values())}/{sum(v['total_max'] for v in sections_map.values())}"

        prompt = f"""
        You are a Psychometric Analyst. Provide a test-level analysis for the test "{test_name}".

        CONTEXT:
        - Category: {category}
        - Instance ID: {instance_id}
        - Sections Summary: {sections_summary_json}
        - Overall Score: {overall_score}

        TASK:
        1. 'description': One-sentence explanation of what this TEST measures overall.
        2. 'representation': 1-2 sentence summary of how the student performed overall based on section performances.

        OUTPUT JSON STRICTLY:
        {{
          "description": "...",
          "representation": "..."
        }}
        """
        
        try:
            response_text = await self._get_llm_response(prompt, model_provider)
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            error_logger.error(f"Test summary LLM call failed: {e}")
            return {}

    async def analyze_test_performance(
        self,
        data_list: List[PsychometricItem],
        model_provider: str = "gemini",
    ) -> PsychometricAnalysisResponse:
        """
        Main orchestrator:
        1. Identifies questions (separating Academic vs Interest).
        2. Grades them in parallel streams.
        3. Aggregates scores.
        4. Analyzes sections and test overall.
        """

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
        
        forced_max_marks_map = {} 

        for index, item in enumerate(data_list):
            q_max = float(item.question_mark or 0.0)
            if q_max < 0: q_max = 0.0
            
            has_text_answer = bool(item.student_text_answer and item.student_text_answer.strip())
            has_option_text = bool(item.student_selected_option and item.student_selected_option.strip())
            
            is_standard_descriptive = (bool(item.solution) and q_max > 0 and has_text_answer)

            is_open_eval = (q_max <= 0) and (has_text_answer or has_option_text)

            if is_standard_descriptive:
                academic_indices.append(index)
                academic_tasks.append(
                    self._grade_answer(
                        item, 
                        q_max, 
                        model_provider, 
                        is_interest_check=False 
                    )
                )

            elif is_open_eval:
                forced_max = 5.0
                forced_max_marks_map[index] = forced_max
                
                text_to_grade = item.student_text_answer if has_text_answer else item.student_selected_option
                
                interest_indices.append(index)
                interest_tasks.append(
                    self._grade_answer(
                        item, 
                        forced_max, 
                        model_provider, 
                        override_answer_text=text_to_grade,
                        is_interest_check=True
                    )
                )

        academic_results = []
        if academic_tasks:
            try:
                academic_results = await asyncio.gather(*academic_tasks)
            except Exception as e:
                error_logger.error(f"Academic Grading Failed: {e}", exc_info=True)
                raise HTTPException(status_code=502, detail=f"LLM Error (Academic): {str(e)}")

        interest_results = []
        if interest_tasks:
            try:
                interest_results = await asyncio.gather(*interest_tasks)
            except Exception as e:
                error_logger.error(f"Interest Grading Failed: {e}", exc_info=True)
                raise HTTPException(status_code=502, detail=f"LLM Error (Interest): {str(e)}")

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

            if index in forced_max_marks_map:
                q_max = forced_max_marks_map[index]
            else:
                q_max = float(item.question_mark or 0.0)
                if q_max < 0: q_max = 0.0
            
            q_obtained = 0.0
            student_response_display = ""
            performance_tag = ""
            correct_answer_display = item.correct_option_text if item.correct_option_text else item.solution or "N/A"
            
            is_interest_item = index in interest_score_map
            is_academic_item = index in academic_score_map
            
            has_mcq_ids = (
                item.student_selected_id is not None
                 and item.correct_option_id is not None
                and item.correct_option_id != 0
            )

            has_scaled_option = bool(
                item.student_selected_option and item.student_selected_option.strip()
            )

            
            if is_interest_item:
                q_obtained = interest_score_map.get(index, 0.0)
                
                used_text = item.student_text_answer or item.student_selected_option
                student_response_display = f"Interest Response: '{used_text}' (AI Graded)"
                correct_answer_display = "Evaluated on Interest/Engagement"

            elif is_academic_item:
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
                self._analyze_single_section(sec_name, context_str, model_provider)
            )

        analysis_results = []
        if analysis_tasks:
            try:
                analysis_results = await asyncio.gather(*analysis_tasks)
            except Exception as e:
                error_logger.error(f"Analysis Phase Failed: {e}", exc_info=True)
             
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
                    interpretation=llm_res.get("interpretation", "Analysis pending or unavailable."),
                    section_score=score_text
                )
            )

        test_summary_data = await self._generate_test_summary(
            test_name, category, instance_id, sections_map, model_provider
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