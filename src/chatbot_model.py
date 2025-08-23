# chatbot_model.py  (OpenAI Embeddings + FAISS 사용, 재인덱스 및 직출 로직 포함)
import os
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = "text-embedding-3-small"  # 비용/속도 고려
LLM_MODEL = "gpt-3.5-turbo"
DEFAULT_FAISS_PATH = "faq_faiss.index"

def get_openai_embeddings(texts: list[str]) -> np.ndarray:
    # 배치로 임베딩 호출
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    embeddings = [d.embedding for d in resp.data]
    arr = np.array(embeddings, dtype=np.float32)
    # L2 정규화: inner product를 cosine 유사도로 사용 가능
    faiss.normalize_L2(arr)
    return arr

class ChatbotModel:
    def __init__(self, faq_path="faq_data.csv", faiss_index_path=DEFAULT_FAISS_PATH):
        self.faq_path = faq_path
        self.faiss_index_path = faiss_index_path

        # CSV 읽기 (BOM 방지)
        self.faq_df = pd.read_csv(self.faq_path, encoding='utf-8-sig')
        if "질문" not in self.faq_df.columns or "답변" not in self.faq_df.columns:
            raise RuntimeError("FAQ CSV에 '질문' 및 '답변' 컬럼이 필요합니다.")

        # 인덱스 로드 또는 생성
        if os.path.exists(self.faiss_index_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(self.faiss_index_path)
        else:
            print("Creating FAISS index (this will call OpenAI embeddings API)...")
            self.rebuild_index()

    def rebuild_index(self):
        texts = self.faq_df['질문'].astype(str).tolist()
        if len(texts) == 0:
            # 빈 인덱스 생성(차원은 1로 임시 설정) — 실제로는 FAQ가 필요함
            self.index = faiss.IndexFlatIP(1)
            faiss.write_index(self.index, self.faiss_index_path)
            return

        emb = get_openai_embeddings(texts)
        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)
        faiss.write_index(self.index, self.faiss_index_path)
        print("FAISS index created and saved.")

    def search(self, query: str, top_k: int = 3, score_threshold: float = 0.0):
        q_emb = get_openai_embeddings([query])
        D, I = self.index.search(q_emb, top_k)  # D: similarity (IP), I: indices
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append({
                "score": float(score),
                "index": int(idx),
                "question": str(self.faq_df.loc[int(idx), "질문"]),
                "answer": str(self.faq_df.loc[int(idx), "답변"])
            })
        return [r for r in results if r["score"] >= score_threshold]

    def get_answer(self, query: str, prefer_faq_direct: bool = True):
        """
        - prefer_faq_direct=True: exact 또는 매우 높은 유사도인 경우 CSV의 원문 답변을 바로 반환.
        - 그렇지 않으면 RAG 컨텍스트(있으면 포함)로 LLM을 호출하여 자연스러운 답변 생성.
        """
        # 1) 검색
        hits = self.search(query, top_k=3, score_threshold=0.0)  # 우선 상위 결과들 로드
        rag_context = ""
        if hits:
            rag_context = "\n\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in hits])

        # 1a) 바로 반환 조건: 정확 매칭 또는 매우 높은 유사도
        if prefer_faq_direct and hits:
            top = hits[0]
            # 정확 문자열 매칭 또는 아주 높은 유사도(예: 0.98)
            if query.strip() == top["question"].strip() or top["score"] >= 0.98:
                return top["answer"]

        # 2) 메시지 구성: hits 유무에 따라 프롬프트 분기
        if hits and any(h["score"] >= 0.72 for h in hits):
            system_prompt = (
                "당신은 텃밭 대여 웹사이트의 친절한 챗봇 새싹이입니다. "
                "아래의 참고지식을 우선 활용하여 사용자 질문에 답변하십시오. "
                "참고지식에 명확한 답이 있으면 그 내용을 중심으로 답변하고, "
                "추가 설명이 필요하면 친절히 덧붙이세요."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"참고지식:\n{rag_context}"},
                {"role": "user", "content": query},
            ]
        else:
            system_prompt = (
                "당신은 텃밭 대여 웹사이트의 친절한 챗봇 새싹이입니다. "
                "사용자 질문에 대해 가능한 한 정확하고 실용적으로 답변하세요. "
                "정보가 불확실하면 '확인 필요'를 명시하고, 추가 확인 방법(예: 고객센터 연락)을 제시하세요."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

        # 3) LLM 호출
        resp = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        # API 응답 형식에 따라 안전하게 접근
        content = ""
        try:
            content = resp.choices[0].message.content
        except Exception:
            # 예외 시 raw 응답을 문자열로 반환
            content = str(resp)

        return content
