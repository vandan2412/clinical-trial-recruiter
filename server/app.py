"""
server/app.py - OpenEnv multi-mode deployment server entry point.
Required by openenv validate. Must have main() and if __name__ == '__main__'.
"""

import os
import sys
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _json_response(handler: BaseHTTPRequestHandler, status: int, data: Any) -> None:
    body = json.dumps(data, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", 0))
    if length > 0:
        try:
            return json.loads(handler.rfile.read(length))
        except Exception:
            return {}
    return {}


# Lazy-load env to avoid import errors at module level
_ENV = None
_initialized = False


def _get_env():
    global _ENV
    if _ENV is None:
        from src.env import ClinicalTrialRecruiterEnv
        _ENV = ClinicalTrialRecruiterEnv(seed=42)
    return _ENV


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/health", "/"):
            _json_response(self, 200, {"status": "ok", "env": "ClinicalTrialRecruiter"})
        elif self.path == "/state":
            try:
                _json_response(self, 200, _get_env().state().model_dump())
            except Exception as e:
                _json_response(self, 500, {"error": str(e)})
        elif self.path == "/tasks":
            from src.tasks import list_tasks
            _json_response(self, 200, {"tasks": list_tasks()})
        else:
            _json_response(self, 404, {"error": "Not found"})

    def do_POST(self):
        global _initialized
        if self.path == "/reset":
            try:
                body = _read_body(self)
                task = body.get("task", "easy_single_criterion")
                seed = body.get("seed", 42)
                obs = _get_env().reset(task=task, seed=seed)
                _initialized = True
                _json_response(self, 200, obs.model_dump())
            except Exception as e:
                _json_response(self, 500, {"error": str(e)})
        elif self.path == "/step":
            if not _initialized:
                _json_response(self, 400, {"error": "Call /reset first."})
                return
            try:
                body = _read_body(self)
                result = _get_env().step(body.get("action", "screen_eligible"))
                _json_response(self, 200, result.model_dump())
            except Exception as e:
                _json_response(self, 500, {"error": str(e)})
        else:
            _json_response(self, 404, {"error": "Not found"})


def main():
    """Main entry point — required by openenv validate and [project.scripts]."""
    port = int(os.getenv("PORT", "7860"))
    host = "0.0.0.0"
    server = HTTPServer((host, port), Handler)
    print(f"ClinicalTrialRecruiter running on http://{host}:{port}", flush=True)
    print("Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /health")
    server.serve_forever()


if __name__ == "__main__":
    main()
