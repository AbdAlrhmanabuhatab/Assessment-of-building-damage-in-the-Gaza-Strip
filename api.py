"""
Root Entry Point for FastAPI Web Service.
Delegates to the modular clean architecture in src.api.server.
"""
import os
import sys

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from config.settings import config
from src.api.server import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host=config.HOST, port=config.PORT, reload=config.DEBUG)
