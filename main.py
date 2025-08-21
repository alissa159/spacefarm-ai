from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot_model import ChatbotModel # 위에서 만든 모델 임포트
from fastapi.middleware.cors import CORSMiddleware  # CORS 설정

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    # 실제 배포 시에는 프론트엔드 도메인을 여기에 추가해야 합니다.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 챗봇 모델 로드 (애플리케이션 시작 시 한 번만 로드)
chatbot_instance = ChatbotModel() # LLM 통합 모델 로드

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_with_bot(request: QueryRequest):
    try:
        # chatbot_model.py의 get_answer 함수가 이제 LLM을 사용
        response_text = chatbot_instance.get_answer(request.query)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Chatbot service is running!"}
