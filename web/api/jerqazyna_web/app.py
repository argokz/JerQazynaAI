"""Приложение FastAPI веб-карты.

Локально: python -m uvicorn jerqazyna_web.app:create_app --factory --app-dir web/api
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from jerqazyna_web.routes import router
from jerqazyna_web.service import MapService
from jerqazyna_web.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Данные проверяются здесь, до приёма запросов: несогласованный экспорт не стартует."""
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="JerQazyna: веб-карта",
        description="Слои задачи, их источники и лицензии, карточка происхождения ячейки.",
        version="0.1.0",
    )
    app.state.service = MapService(settings)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins,
            allow_methods=["GET"], allow_credentials=False,
        )
    app.include_router(router)
    return app
