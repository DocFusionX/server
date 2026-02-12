from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .core.logging import setup_logging

from .api.v1.routes_health import router as health_router
from .api.v1.routes_rag import router as rag_router

def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix=settings.api_prefix, tags=["health"])
    app.include_router(rag_router, prefix=f"{settings.api_prefix}/rag", tags=["rag"])

    return app

setup_logging()
app = create_app()
