from pathlib import Path

# 프로젝트 루트 디렉토리
ROOT_DIR = Path(__file__).parent.parent.parent

# 데이터 디렉토리
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# RAG 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250
TOP_K_RESULTS = 5