"""
server/app.py - OpenEnv multi-mode deployment entry point.
Delegates to root app.py server.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import run_server

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    run_server(port=port)
