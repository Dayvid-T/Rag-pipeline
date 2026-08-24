"""
Central place for reading configuration/secrets from environment variables.

Phase: 2 (Build Core Retrieval Logic, Locally)
Why it exists: keeps secrets and settings out of application code, so the
same code behaves correctly locally, in Docker, and in App Runner without
being edited.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "rag-pipeline-qa")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
