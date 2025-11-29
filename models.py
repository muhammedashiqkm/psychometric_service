from typing import List, Optional, Any
from pydantic import BaseModel, Field

class PsychometricItem(BaseModel):
    TestName: Optional[str] = None
    PsychometricTestCategory: str = Field(..., description="The category used for grouping")
    PsychometricSectionID: Optional[int] = None
    SectionName: Optional[str] = None
    PsychometricQuestionID: int
    Question: str
    Solution: Optional[str] = None
    CorrectOptionID: Optional[int] = None
    CorrectOptionLabel: Optional[str] = None
    CorrectOptionText: Optional[str] = None
    CorrectOptionScoreValue: Optional[float] = None
    PsychometricTestResponseID: Optional[int] = None
    StudentSelectedOptionID: Optional[int] = None
    StudentTextAnswer: Optional[str] = None
    PsychometricTestInstancesID: int


class PsychometricAnalysisResponse(BaseModel):
    category: str
    description: str
    representation: str 
    instance_id: int

    class Config:
        populate_by_name = True