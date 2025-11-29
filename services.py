import os
import json
from typing import List
from dotenv import load_dotenv

from app.core.config import settings

# AI Clients
import google.generativeai as genai
from openai import OpenAI

from models import PsychometricItem, PsychometricAnalysisResponse

load_dotenv()

# --- Config ---
# Ensure you have GEMINI_API_KEY or OPENAI_API_KEY in your .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel()

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

async def analyze_psychometric_data(data_list: List[PsychometricItem]) -> PsychometricAnalysisResponse:
    if not data_list:
        raise ValueError("No data provided")

    # 1. Extract Metadata from the first item (assuming all are from same category/instance)
    first_item = data_list[0]
    category = first_item.PsychometricTestCategory
    instance_id = first_item.PsychometricTestInstancesID

    # 2. Build Context for the AI
    # We format the questions and answers so the AI can understand student performance.
    questions_context = []
    
    for item in data_list:
        # Determine what the student answered
        student_ans = "No Answer"
        if item.StudentTextAnswer and item.StudentTextAnswer.strip():
            student_ans = f"Text Answer: '{item.StudentTextAnswer}'"
        elif item.StudentSelectedOptionID is not None:
            # We compare IDs to see if they were correct
            is_correct = (item.StudentSelectedOptionID == item.CorrectOptionID)
            student_ans = f"Selected Option ID: {item.StudentSelectedOptionID} (Correct: {is_correct})"
        
        # Add to context list
        questions_context.append(
            f"Question: {item.Question}\n"
            f"Student Answer: {student_ans}\n"
            f"Correct Answer/Solution: {item.CorrectOptionText or item.Solution}\n"
            "---"
        )

    full_context_str = "\n".join(questions_context)

    # 3. Construct Prompt
    prompt = f"""
    You are an expert Psychometric Analyst. Analyze the following student responses for the category: "{category}".

    **Data:**
    {full_context_str}

    **Task:**
    1. **Description**: Define what the category "{category}" generally measures (1 sentence).
    2. **Representation**: Write a short, personalized summary (2-3 sentences) of how this specific student performed based on the provided questions and their answers. Did they get them right? Did they struggle?

    **Output strictly JSON format:**
    {{
        "category": "{category}",
        "description": "string",
        "representation": "string"
    }}
    """

    # 4. Call AI (Prefer Gemini, fallback to OpenAI)
    try:
        if GEMINI_API_KEY:
            response = gemini_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            result_text = response.text
        elif OPENAI_API_KEY:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            result_text = response.choices[0].message.content
        else:
            raise Exception("No API Key found")

        # 5. Parse AI Result
        ai_data = json.loads(result_text)

        return PsychometricAnalysisResponse(
            category=ai_data.get("category", category),
            description=ai_data.get("description", "Analysis unavailable"),
            Representation=ai_data.get("representation", "Analysis unavailable"),
            instance_id=instance_id
        )

    except Exception as e:
        print(f"AI Error: {e}")
        # Fallback response
        return PsychometricAnalysisResponse(
            category=category,
            description=f"Analysis of {category}",
            Representation="Could not generate analysis due to server error.",
            instance_id=instance_id
        )