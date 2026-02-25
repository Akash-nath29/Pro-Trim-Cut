from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.routes import router
from app.core.config import settings
from app.core.logging import setup_logging, log_device_info, get_logger

logger = get_logger("main")

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    setup_logging(debug=settings.DEBUG)

    app = FastAPI(
        title="Pro Auto-Trim",
        description="AI-powered video editor that trims raw footage into clean, professional videos",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, tags=["Auto-Trim"])

    @app.get("/")
    async def serve_ui():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.on_event("startup")
    async def startup():
        logger.info("=" * 60)
        logger.info("  Pro Auto-Trim — AI Video Editor")
        logger.info("=" * 60)
        log_device_info()
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📂 Storage: {settings.storage_dir}")
        logger.info(f"🎙️  Whisper: {settings.WHISPER_MODEL_SIZE}")
        logger.info(f"🧠 LLM: {settings.LLM_MODEL}")
        logger.info("=" * 60)

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "device": settings.resolved_device,
            "whisper_model": settings.WHISPER_MODEL_SIZE,
            "llm_model": settings.LLM_MODEL,
        }

    return app


app = create_app()
