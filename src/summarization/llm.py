# src/summarization/llm.py
from __future__ import annotations
import os
from enum import Enum
from typing import Optional

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None  # type: ignore
try:
    from langchain_anthropic import ChatAnthropic
except Exception:
    ChatAnthropic = None  # type: ignore
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None  # type: ignore
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore
try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None  # type: ignore


class ModelProvider(str, Enum):
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    GROQ = "GROQ"
    GOOGLE = "GOOGLE"
    OLLAMA = "OLLAMA"


def _env_get(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if not v:
        raise ValueError(f"Missing environment variable: {name}")
    return v


def get_model(model_name: str,
              model_provider: ModelProvider,
              temperature: float = 0.2,
              max_tokens: Optional[int] = None):
    if model_provider == ModelProvider.OPENAI:
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is not installed.")
        return ChatOpenAI(
            model=model_name,
            api_key=_env_get("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if model_provider == ModelProvider.ANTHROPIC:
        if ChatAnthropic is None:
            raise ImportError("langchain-anthropic is not installed.")
        return ChatAnthropic(
            model=model_name,
            api_key=_env_get("ANTHROPIC_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if model_provider == ModelProvider.GROQ:
        if ChatGroq is None:
            raise ImportError("langchain-groq is not installed.")
        return ChatGroq(
            model=model_name,
            api_key=_env_get("GROQ_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if model_provider == ModelProvider.GOOGLE:
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=_env_get("GOOGLE_API_KEY"),
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    if model_provider == ModelProvider.OLLAMA:
        if ChatOllama is None:
            raise ImportError("langchain-ollama is not installed.")
        host = os.getenv("OLLAMA_HOST", "localhost")
        base_url = os.getenv("OLLAMA_BASE_URL", f"http://{host}:11434")
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported provider: {model_provider}")
