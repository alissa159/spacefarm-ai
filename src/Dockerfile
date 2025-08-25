FROM python:3.11-slim

# 시스템 패키지: faiss-cpu 등 빌드에 필요한 도구들 일부 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 먼저 requirements 설치 (캐시 효율)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# 앱 소스 복사
COPY . /app

# 데이터(FAISS 인덱스 등)를 마운트할 경로 (fly volumes와 매칭)
ENV DATA_DIR=/data
RUN mkdir -p $DATA_DIR

# 포트 (fly는 내부_port로 바꿀 수 있음)
ENV PORT=8080

# production: gunicorn + uvicorn worker
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120"]
