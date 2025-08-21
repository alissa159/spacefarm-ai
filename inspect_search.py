from chatbot_model import ChatbotModel

m = ChatbotModel()   # 이미 색인이 있으면 로드만 함
q = "대여 방법을 알려줘"
hits = m.search(q, top_k=3, score_threshold=0.0)  # threshold 0으로 해서 모든 결과 확인
for i, h in enumerate(hits):
    print(f"#{i+1} score={h['score']:.4f} index={h['index']}")
    print(" Q:", h['question'])
    print(" A:", h['answer'])
    print("-" * 40)
