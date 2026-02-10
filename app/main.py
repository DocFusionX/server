from fastapi import FastAPI
from .core.config import settings
from .core.logging import setup_logging

from .api.v1.routes_health import router as health_router

def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)

    app.include_router(health_router, prefix=settings.api_prefix, tags=["health"])

    return app

setup_logging()
app = create_app()
