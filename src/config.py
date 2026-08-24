"""Application configuration."""

import os

from dotenv import load_dotenv


load_dotenv()

APP_NAME = "Agentic RAG Document Assistant"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SUPPORTED_EXTENSIONS = {"pdf", "txt", "csv", "xlsx"}
