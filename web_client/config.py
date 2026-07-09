"""
web_client/config.py — 全局配置：路径、参考图目录、流式推理分辨率等常量。
"""
import os
from pathlib import Path

from omegaconf import OmegaConf

# ── 路径 ────────────────────────────────────────────────────────────────────
WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent

# ── 调试模式 ─────────────────────────────────────────────────────────────────
# 仅前端调试：不加载模型，直接启动 Web。ABOTWORLD_DEBUG_FRONTEND=1 开启
DEBUG_FRONTEND_ONLY = os.environ.get(
    "ABOTWORLD_DEBUG_FRONTEND", ""
).strip().lower() in ("1", "true", "yes")

# ── 参考图 ───────────────────────────────────────────────────────────────────
DEFAULT_REF_IMAGE = str(
    WEB_DIR / "datasets/images/example.png"
)
# 首帧图 + Prompt 的唯一数据源（含 default_prompt）；见 scene_presets.yaml 头部说明
SCENE_PRESETS_PATH = WEB_DIR / "scene_presets.yaml"
# 仅当 scene_presets.yaml 缺失或 default_prompt 为空时的兜底文案
PRESETS_FALLBACK_PROMPT = "请编辑 web_client/scene_presets.yaml：设置 default_prompt 与 groups。"

# ── 流式推理 ─────────────────────────────────────────────────────────────────
# STREAM_HEIGHT = 480
# STREAM_WIDTH = 832
STREAM_HEIGHT = 704
STREAM_WIDTH = 1280


# ── 模型推理参数 ─────────────────────────────────────────────────────────────
LATENT_CHANNELS = 48
LATENT_HEIGHT = STREAM_HEIGHT // 16   # 60
LATENT_WIDTH = STREAM_WIDTH // 16     # 104

# ── 运行时阈值 ───────────────────────────────────────────────────────────────
LOW_MEMORY_THRESHOLD_GB = 40         # 低于此 VRAM (GB) 启用动态内存交换
FRAME_QUEUE_SIZE = 2                 # 帧队列最大深度
QUEUE_POLL_TIMEOUT = 0.5             # 队列轮询超时 (秒)
WORKER_JOIN_TIMEOUT = 10             # Worker 线程退出等待 (秒)
SHUTDOWN_GRACE_SECONDS = 3           # 优雅关机等待 (秒)
MAX_BLOCKS = 600                      # 最大生成块数（视频长度限制）
FRAMES_PER_BLOCK = 12                # 每 block 解码后的标准帧数
FIRST_BLOCK_FRAMES = 9              # 首 block 解码帧数（含参考图，少于标准帧数）

# ── 视频输出 ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "outputs"  # 视频输出目录
VIDEO_FPS = 12                         # 视频帧率
VIDEO_CODEC = "libx264"                # 视频编码器
VIDEO_QUALITY = 8                      # 视频质量 (1-10, 越小质量越高)
MAX_VIDEO_HISTORY = 10                 # Gallery 最多显示的视频数量
AUTO_DELETE_OLD_VIDEOS = True          # 超出限制时自动删除磁盘上的旧视频

# ── 键盘 ─────────────────────────────────────────────────────────────────────
KEY_ORDER = ["W", "A", "S", "D", "I", "J", "K", "L"]
CONFLICT_GROUPS = [
    ("W", "S"),  # 前/后
    ("A", "D"),  # 左/右
    ("I", "K"),  # 上/下
    ("J", "L"),  # 左/右
]

# ── 服务器 ───────────────────────────────────────────────────────────────────
SERVER_NAME = "0.0.0.0"
SERVER_PORT = 2233
# SSL 已禁用，使用 HTTP

# ── 推理后端（与 configs/*.yaml 对齐，UI 不再暴露开关）────────────────────────
_merged_ui_cfg = OmegaConf.merge(
    OmegaConf.load(PROJECT_ROOT / "configs/default_config.yaml"),
    OmegaConf.load(PROJECT_ROOT / "configs/long_forcing_dmd.yaml"),
)
# Unified VAE type: wan2.2 | taew2_2 | mg_lightvae | mg_lightvae_v2
_vae_type = str(getattr(_merged_ui_cfg, "vae_type", "")).strip().lower()
if not _vae_type:
    _vae_type = "taew2_2"
VAE_TYPE = _vae_type
USE_FP8_GEMM = bool(getattr(_merged_ui_cfg, "use_fp8_gemm", False))
