from __future__ import annotations

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import json
import mimetypes
import os
import threading
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def configure_llm_defaults() -> None:
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        return
    os.environ.setdefault("LLM_API_KEY", groq_key)
    os.environ.setdefault("LLM_BASE_URL", os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    os.environ.setdefault("LLM_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))


load_env_file(ROOT / ".env")
configure_llm_defaults()

from indexer import KnowledgeIndex

DATASET_DIR = Path(os.getenv("VERTIV_DATASET_DIR", ROOT / "Vertiv")).resolve()
CHROMA_PATH = Path(os.getenv("VERTIV_CHROMA_PATH", ROOT / "data" / "chroma")).resolve()
HOST = os.getenv("VERTIV_HOST", "127.0.0.1")
PORT = int(os.getenv("VERTIV_PORT", "8000"))

index = KnowledgeIndex(DATASET_DIR, CHROMA_PATH)


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


class VertivHandler(SimpleHTTPRequestHandler):
    server_version = "VertivKnowledge/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, value: object, status: int = 200) -> None:
        payload = json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("Request is too large")
        raw = self.rfile.read(length)
        return json.loads(raw or b"{}")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json(index.health())
            return
        if parsed.path == "/api/stats":
            self.send_json(index.stats())
            return
        if parsed.path == "/api/index/status":
            self.send_json(index.index_status())
            return
        if parsed.path == "/api/search":
            params = urllib.parse.parse_qs(parsed.query)
            query = params.get("q", [""])[0].strip()
            category = params.get("category", [""])[0].strip()
            try:
                limit = min(max(int(params.get("limit", ["12"])[0]), 1), 50)
                self.send_json({"results": index.search(query, limit, category)})
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/file":
            params = urllib.parse.parse_qs(parsed.query)
            relative_path = params.get("path", [""])[0]
            self.serve_dataset_file(relative_path)
            return
        if parsed.path.startswith("/api/"):
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        if parsed.path not in ("/", "/index.html") and not (STATIC_DIR / parsed.path.lstrip("/")).is_file():
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            payload = self.read_json()
            if parsed.path == "/api/chat":
                question = str(payload.get("question", "")).strip()
                if not question:
                    raise ValueError("Please enter a question")
                if len(question) > 4_000:
                    raise ValueError("Question is too long")
                history = payload.get("history", [])
                category = str(payload.get("category", "")).strip()
                self.send_json(index.answer(question, history, category))
                return
            if parsed.path == "/api/index/start":
                started = index.start_indexing(force=bool(payload.get("force", False)))
                self.send_json({"started": started, **index.index_status()}, 202 if started else 200)
                return
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            print(f"Request error: {exc!r}")
            self.send_json({"error": "The request could not be completed"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def serve_dataset_file(self, relative_path: str) -> None:
        candidate = (DATASET_DIR / relative_path).resolve()
        try:
            candidate.relative_to(DATASET_DIR)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        size = candidate.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        disposition = "inline" if content_type == "application/pdf" or content_type.startswith("image/") else "attachment"
        safe_name = candidate.name.replace('"', "")
        self.send_header("Content-Disposition", f'{disposition}; filename="{safe_name}"')
        self.end_headers()
        with candidate.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                self.wfile.write(chunk)


def main() -> None:
    if not DATASET_DIR.exists():
        raise SystemExit(f"Dataset folder not found: {DATASET_DIR}")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    if os.getenv("VERTIV_AUTO_INDEX", "1") == "1":
        threading.Timer(0.8, index.start_indexing).start()
    server = ThreadingHTTPServer((HOST, PORT), VertivHandler)
    print(f"Vertiv Knowledge is running at http://{HOST}:{PORT}")
    print(f"Dataset: {DATASET_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
