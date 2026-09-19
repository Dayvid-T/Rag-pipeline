"""
Central place for reading configuration/secrets from environment variables.

Phase: 2 (Build Core Retrieval Logic, Locally)
Why it exists: keeps secrets and settings out of application code, so the
same code behaves correctly locally, in Docker, and in App Runner without
being edited.
"""

import logging
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "rag-pipeline-qa")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    static_dir: str = os.getenv("STATIC_DIR", "../frontend/dist")


settings = Settings()

# Without this, the root logger sits at WARNING with no handler, so every
# logger.info() call - including the guardrails audit log - is silently
# dropped rather than reaching stdout / the container's log driver.
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
