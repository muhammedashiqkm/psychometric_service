from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

# Helper to convert snake_case to CamelCase automatically
def to_camel(string: str) -> str:
    return ''.join(word.capitalize() for word in string.split('_'))

class PsychometricItem(BaseModel):
    # API sends "TestName", Python uses "test_name"
    test_name: Optional[str] = Field(None, alias="TestName")
    category: str = Field(..., alias="PsychometricTestCategory")
    section_id: Optional[int] = Field(None, alias="PsychometricSectionID")
    section_name: Optional[str] = Field(None, alias="SectionName")
    question_id: int = Field(..., alias="PsychometricQuestionID")
    question: str = Field(..., alias="Question")
    solution: Optional[str] = Field(None, alias="Solution")
    correct_option_id: Optional[int] = Field(None, alias="CorrectOptionID")
    correct_option_label: Optional[str] = Field(None, alias="CorrectOptionLabel")
    correct_option_text: Optional[str] = Field(None, alias="CorrectOptionText")
    correct_option_score: Optional[float] = Field(None, alias="CorrectOptionScoreValue")
    response_id: Optional[int] = Field(None, alias="PsychometricTestResponseID")
    student_selected_id: Optional[int] = Field(None, alias="StudentSelectedOptionID")
    student_text_answer: Optional[str] = Field(None, alias="StudentTextAnswer")
    instance_id: int = Field(..., alias="PsychometricTestInstancesID")

    model_config = ConfigDict(
        populate_by_name=True, 
        alias_generator=None   
    )

class PsychometricRequest(BaseModel):
    model: Literal["gemini", "openai", "deepseek"] = "gemini"
    data: List[PsychometricItem]

class PsychometricAnalysisResponse(BaseModel):
    category: str
    description: str
    representation: str = Field(..., alias="Representation") 
    instance_id: int