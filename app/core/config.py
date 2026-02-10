from pydantic import BaseModel

class Settings(BaseModel):
    app_name: str = "DocFusionX Server"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"

settings = Settings()
