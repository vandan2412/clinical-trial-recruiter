"""
app.py - Hugging Face Spaces endpoint for ClinicalTrialRecruiter.
Exposes REST API compatible with `openenv validate`.

Endpoints:
  POST /reset                     - Reset environment, return initial observation
  POST /step                      - Execute action, return (obs, reward, done, info)
  GET  /state                     - Return full environment state
  GET  /tasks                     - List available tasks
  GET  /health                    - Health check
"""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

from src.env import ClinicalTrialRecruiterEnv
from src.tasks import list_tasks

# Global environment instance (stateful across calls)
ENV = ClinicalTrialRecruiterEnv(seed=42)
_initialized = False


def _json_response(handler: BaseHTTPRequestHandler, status: int, data: Any) -> None:
    body = json.dumps(data, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", 0))
    if content_length > 0:
        raw = handler.rfile.read(content_length)
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        global ENV

        if self.path in ("/health", "/"):
            _json_response(self, 200, {"status": "ok", "env": "ClinicalTrialRecruiter"})

        elif self.path == "/state":
            try:
                state = ENV.state()
                _json_response(self, 200, state.dict())
            except Exception as e:
                _json_response(self, 500, {"error": str(e)})

        elif self.path == "/tasks":
            _json_response(self, 200, {"tasks": list_tasks()})

        else:
            _json_response(self, 404, {"error": "Not found"})

    def do_POST(self):
        global ENV, _initialized

        if self.path == "/reset":
            try:
                body = _read_body(self)
                task = body.get("task", "easy_single_criterion")
                seed = body.get("seed", 42)
                obs = ENV.reset(task=task, seed=seed)
                _initialized = True
                _json_response(self, 200, obs.dict())
            except Exception as e:
                _json_response(self, 500, {"error": str(e)})

        elif self.path == "/step":
            if not _initialized:
                _json_response(self, 400, {"error": "Call /reset first."})
                return
            try:
                body = _read_body(self)
                action = body.get("action", "screen_eligible")
                result = ENV.step(action)
                _json_response(self, 200, result.dict())
            except Exception as e:
                _json_response(self, 500, {"error": str(e)})

        else:
            _json_response(self, 404, {"error": "Not found"})


def run_server(host: str = "0.0.0.0", port: int = 7860) -> None:
    server = HTTPServer((host, port), RequestHandler)
    print(f"ClinicalTrialRecruiter running on http://{host}:{port}", flush=True)
    print("Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /health")
    server.serve_forever()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    run_server(port=port)
