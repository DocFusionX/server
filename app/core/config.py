from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "DocFusionX Server"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    mistral_api_key: str = ""
    chroma_db_path: str = "./chroma_db"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
