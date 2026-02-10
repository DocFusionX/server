from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "DocFusionX Server"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"
    mistral_max_tokens: int = 64000
    chroma_db_path: str = "./chroma_db"
    upload_dir: str = "./data/uploads"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
