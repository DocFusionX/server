from fastapi import FastAPI

def create_app() -> FastAPI:
    app = FastAPI(title="DocFusionX Server")

    from .api.v1.routes_health import router as health_router

    app.include_router(health_router, prefix="/api/v1", tags=["health"])

    return app

app = create_app()
