

## 📂 Project Structure

```

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

````

---

## 🛠️ Environment Configuration (`.env`)

Create a `.env` file in the root directory.

```env
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
````

---


### Build & Run

```bash
docker-compose up --build -d
```


## 🔗 API Documentation

### 1️⃣ Authentication

**Endpoint**

```
POST /auth/login
```

**Headers**

```
Content-Type: application/json
```

**Request Body**

```json
{
  "username": "admin",
  "password": "secure_password"
}
```

**Response**

```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer"
}
```

---

### 2️⃣ Generate Psychometric Analysis

**Endpoint**

```
POST /psychometric/generate
```

**Headers**

```
Authorization: Bearer <your_access_token>
Content-Type: application/json
```

**Request Body Format**

| Field | Type   | Description                       |
| ----- | ------ | --------------------------------- |
| model | String | `gemini`, `openai`, or `deepseek` |
| data  | List   | List of question objects          |

**Sample Request**

```json
{
  "model": "gemini",
  "data": [
    {
      "TestName": "Final Test",
      "PsychometricTestCategory": "General Skills",
      "PsychometricSectionID": 1,
      "SectionName": "MCQ",
      "PsychometricQuestionID": 1,
      "Question": "5 + 5 = ?",
      "Solution": "",
      "CorrectOptionID": 11,
      "QuestionMark": 4,
      "CorrectOptionLabel": "A",
      "CorrectOptionText": "10",
      "CorrectOptionScoreValue": 4,
      "PsychometricTestResponseID": 1,
      "StudentSelectedOptionID": 11,
      "StudentSelectedOption": "",
      "StudentSelectedOptionScore": 0,
      "StudentTextAnswer": "",
      "PsychometricTestInstancesID": 500
    } 
  ]
}
```

**Sample Response**

```json
{
    "category": "General Skills",
    "description": "General Skills encompass fundamental cognitive and practical abilities essential for learning and daily tasks. This category typically measures basic academic competencies, problem-solving aptitude, and self-awareness regarding personal attributes.",
    "representation": "The student demonstrated mixed performance in objective questions, achieving 50% accuracy by correctly answering basic addition and subtraction but struggling with multiplication and division. Their responses in descriptive questions were generally good, indicating a solid understanding of concepts, though some answers were not fully comprehensive. Overall, the student scored 35 out of 50, showing better consistency in descriptive and self-assessment questions than in the objective arithmetic tasks.",
    "instance_id": 500,
    "category_score": "35.0 out of 50.0"
}
