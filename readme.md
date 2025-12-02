📂 Project Structure

psychometric_service/
├── app/
│   ├── api/
│   │   ├── analysis.py       # Analysis generation endpoints
│   │   └── auth.py           # Authentication endpoints
│   ├── core/
│   │   ├── config.py         # Environment configuration (Pydantic)
│   │   ├── logging_config.py # Logging setup
│   │   └── security.py       # JWT & Hashing logic
│   ├── models/
│   │   └── schemas.py        # Pydantic data models (Request/Response)
│   ├── services/
│   │   └── llm_service.py    # AI integration logic
│   └── main.py               # Application entry point
├── logs/                     # Log files (mounted via volume)
├── .env                      # Environment variables
├── docker-compose.yaml       # Container orchestration
├── Dockerfile                # Image definition
└── requirements.txt          # Python dependencies


🛠️ Configuration (.env)

Create a .env file in the root directory.

# --- Security ---
SECRET_KEY=your_super_secret_production_key_change_this_immediately
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_HOURS=24
ADMIN_USERNAME=admin
ADMIN_PASSWORD=secure_password

# --- LLM API Keys (Leave empty if not using a specific provider) ---
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=AIza...
DEEPSEEK_API_KEY=sk-..

FRONTENT_ALLOWED_ORIGINS=

# --- Model Configurations (Optional overrides) ---
OPENAI_MODEL_NAME=gpt-4o
GEMINI_MODEL_NAME=gemini-1.5-flash
DEEPSEEK_MODEL_NAME=deepseek-chat


🐳 Docker Installation (Recommended)

Build and Run:

docker-compose up --build -d


Verify Status:
The API will be available at http://localhost:8000.
Health check: GET http://localhost:8000/

View Logs:
Logs are persisted in the local ./logs folder.

💻 Local Installation

Prerequisites: Python 3.10+

Install Dependencies:

pip install -r requirements.txt


Run Application:

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload


🔗 API Documentation

1. Authentication

Endpoint: POST /auth/login
Content-Type: application/json

Request Body:

{
  "username": "admin",
  "password": "secure_password"
}


Response:

{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer"
}


2. Generate Psychometric Analysis

Endpoint: POST /psychometric/generate
Headers:

Authorization: Bearer <your_access_token>

Content-Type: application/json

Request Body:

Field

Type

Description

model

String

gemini, openai, or deepseek

data

List

List of question objects

{
  "model": "gemini",
  "data": [
    {
      "TestName": "Aptitude Test 1",
      "PsychometricTestCategory": "Logical Reasoning",
      "PsychometricSectionID": 1,
      "SectionName": "Section A",
      "PsychometricQuestionID": 101,
      "Question": "If A > B and B > C, is A > C?",
      "Solution": "Yes, transitivity.",
      "CorrectOptionID": 1,
      "CorrectOptionText": "Yes",
      "PsychometricTestInstancesID": 55,
      "StudentSelectedOptionID": 2,
      "StudentTextAnswer": null
    }
  ]
}


Response:

{
  "category": "Logical Reasoning",
  "description": "Logical Reasoning measures the ability to analyze patterns...",
  "Representation": "The student struggled with transitive properties in Section A...",
  "instance_id": 55
}

