import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


# =========================================
# API 配置（环境相关，从环境变量读取）
# =========================================
VIDEO_API_BASE_URL = os.getenv("VIDEO_API_BASE_URL",)
API_HOST = os.getenv("API_HOST",)
API_PORT = int(os.getenv("API_PORT",))

# =========================================
# 视频帧配置
# =========================================
MAX_FRAMES = 12  # API 默认返回帧数
CANONICAL_MAX_FRAMES = 24  # 缓存时提取的帧数（推荐）
API_MAX_FRAMES_LIMIT = 32  # API 最大帧数限制

FRAME_EXT = "jpg"
FRAME_QUALITY = 2  # 1-31 (smaller = better quality)
FRAME_MIME = "image/jpeg"

# =========================================
# 路径配置
# =========================================
ROOT_DIR = Path(".").resolve()
SOURCE_DIR = ROOT_DIR / "server" / "data"
REAL_DIR = SOURCE_DIR / "Real"
FAKE_DIR = SOURCE_DIR / "Fake"

CACHE_DIR = ROOT_DIR / "cache"
EVID_DIR = CACHE_DIR / "evidence"
PROGRESS_DIR = CACHE_DIR / "progress"

# =========================================
# 并发控制配置
# =========================================
LOCK_WAIT_SEC = 8.0
LOCK_POLL_SEC = 0.25

# =========================================
# 批量处理配置
# =========================================
BATCH_DELAY_SEC = 0.1
# =========================================
# LLM 配置（隐私相关，从环境变量读取）
# =========================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER",)
OPENAI_MODEL = os.getenv("OPENAI_MODEL",)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY",)
