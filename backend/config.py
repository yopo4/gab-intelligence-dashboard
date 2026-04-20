"""
Centralized configuration — reads from environment variables or .env file.
All hardcoded values that previously lived in main.py are managed here.
"""
import os

# ── Admin & security ──────────────────────────────────────────────────────────
ADMIN_TOKEN: str = os.getenv("GAB_ADMIN_TOKEN", "dev-token-change-in-prod")

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_TTL: int        = int(os.getenv("GAB_CACHE_TTL", "300"))       # 5 min
MONITORING_TTL: int   = int(os.getenv("GAB_MONITORING_TTL", "25"))   # 25 s

# ── CORS ──────────────────────────────────────────────────────────────────────
_raw = os.getenv("GAB_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
CORS_ORIGINS: list[str] = [o.strip() for o in _raw.split(",") if o.strip()]

# ── Monitoring ring-buffer ────────────────────────────────────────────────────
MONITORING_HISTORY_SIZE: int = int(os.getenv("GAB_HISTORY_SIZE", "48"))  # ~24 h @ 30 min

# ── Pagination defaults ───────────────────────────────────────────────────────
MONITORING_DEFAULT_LIMIT: int = 50
MONITORING_MAX_LIMIT: int     = 200
