#!/usr/bin/env python3
"""
Quick test script to check connectivity with OpenAI GPT.

Usage:
  python scripts/test_openai.py

Requires:
  - .env with OPENAI_API_KEY set
  - pip install langchain-openai python-dotenv
"""

import os
from dotenv import load_dotenv

# load .env
load_dotenv()

from langchain_openai import ChatOpenAI

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found. Please set it in your .env")
        return

    try:
        chat = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.0)
        resp = chat.invoke("Say hello in one short sentence.")
        print("✅ Connected successfully!")
        print("Model reply:", resp.content if hasattr(resp, "content") else resp)
    except Exception as e:
        print("Error connecting to OpenAI:", e)

if __name__ == "__main__":
    main()
