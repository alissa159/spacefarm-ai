# chatbot_model.py  (OpenAI Embeddings + FAISS 사용)
import os
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = "text-embedding-3-small"  # 비용/속도 고려해 소형 모델 추천
LLM_MODEL = "gpt-3.5-turbo"

def get_openai_embeddings(texts: list[str]) -> np.ndarray:
    # OpenAI Embeddings 호출 (배치 지원)
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    embeddings = [d.embedding for d in resp.data]
    arr = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(arr)  # 정규화하면 inner product를 cosine 유사도로 사용 가능
    return arr

class ChatbotModel:
    def __init__(self, faq_path="faq_data.csv", faiss_index_path="faq_faiss.index"):
        self.faq_path = faq_path
        self.faiss_index_path = faiss_index_path

        # FAQ CSV 로드 (컬럼명: '질문', '답변'으로 가정)
        self.faq_df = pd.read_csv(self.faq_path)
        if "질문" not in self.faq_df.columns or "답변" not in self.faq_df.columns:
            raise RuntimeError("FAQ CSV에 '질문' 및 '답변' 컬럼이 필요합니다.")

        if os.path.exists(self.faiss_index_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(self.faiss_index_path)
        else:
            print("Creating FAISS index with OpenAI embeddings (this will call OpenAI embeddings API)...")
            texts = self.faq_df['질문'].tolist()
            emb = get_openai_embeddings(texts)
            dim = emb.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(emb)
            faiss.write_index(self.index, self.faiss_index_path)
            print("FAISS index created and saved.")

    def search(self, query: str, top_k: int = 3, score_threshold: float = 0.75):
        q_emb = get_openai_embeddings([query])
        D, I = self.index.search(q_emb, top_k)  # D: similarity (since normalized & IP)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append({
                "score": float(score),
                "index": int(idx),
                "question": self.faq_df.loc[int(idx), "질문"],
                "answer": self.faq_df.loc[int(idx), "답변"]
            })
        # 필터링 후 반환
        return [r for r in results if r["score"] >= score_threshold]

    def get_answer(self, query: str):
        # 1) RAG 검색
        hits = self.search(query, top_k=3, score_threshold=0.72)
        rag_context = ""
        if hits:
            # 상위 결과들을 컨텍스트로 합침
            rag_context = "\n\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in hits])

        # 2) LLM 메시지 구성
        system_prompt = (
            "당신은 텃밭 대여 웹사이트의 친절한 챗봇 새싹이입니다. "
            "사용자의 질문에 대해 명확하고 도움이 되는 답변을 제공하세요. "
            "만약 제공된 지식(context) 내에서 답변을 찾을 수 없다면, "
            "'죄송합니다만, 해당 질문에 대한 정확한 답변을 찾을 수 없습니다. 다른 질문을 해주시거나 고객센터로 문의해주세요.' 라고 답변해주세요."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if rag_context:
            messages.append({"role": "system", "content": f"참고지식:\n{rag_context}"})
        messages.append({"role": "user", "content": query})

        # 3) OpenAI Chat 호출
        resp = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        return resp.choices[0].message.content
