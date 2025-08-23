# main.py
import os
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chatbot_model import ChatbotModel

# 환경변수
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERVICE_KEY = os.getenv("SERVICE_KEY")  # Spring이 호출할 때 사용할 키 (예: "my-secret-key")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수에 추가하세요.")

app = FastAPI(title="Chatbot Service")

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

# 싱글턴 모델 인스턴스 로드
chatbot_instance = ChatbotModel()


# 서비스키 검증 의존성
def validate_service_key(x_service_key: str | None = Header(None)):
    """
    SERVICE_KEY가 설정되어 있으면 요청 헤더 X-Service-Key와 비교합니다.
    일치하지 않으면 401 반환.
    """
    if SERVICE_KEY:
        if not x_service_key or x_service_key != SERVICE_KEY:
            raise HTTPException(status_code=401, detail="Invalid service key")

@app.post("/chat")
async def chat_with_bot(request: QueryRequest, _=Depends(validate_service_key)):
    """
    Spring이 이 엔드포인트를 호출할 때 반드시 헤더 `X-Service-Key: <SERVICE_KEY>`를 포함해야 합니다.
    """
    try:
        response_text = chatbot_instance.get_answer(request.query, prefer_faq_direct=True)
        return {"response": response_text}
    except Exception as e:
        # 내부 에러는 500으로 반환
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/faq")
def list_faq(_=Depends(validate_service_key)):
    """
    FAQ 목록(질문/답변) 반환 — 프론트가 버튼을 만들고 싶을 때 사용 가능.
    Spring에서 호출할 때도 service key 필요.
    """
    items = chatbot_instance.faq_df[["질문", "답변"]].to_dict(orient="records")
    return {"faqs": items}

@app.post("/reindex")
def reindex(_=Depends(validate_service_key)):
    """
    FAQ 업데이트 후 재인덱스용 엔드포인트 (관리용).
    """
    try:
        chatbot_instance.rebuild_index()
        return {"status": "ok", "message": "Reindexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Chatbot service is running!"}