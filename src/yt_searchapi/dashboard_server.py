"""Small local HTTP layer for the live DuckDB dashboard."""

from __future__ import annotations

import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from yt_searchapi.analysis import DuckDBAnalytics


class _DashboardHandler(BaseHTTPRequestHandler):
    server_version = "yt-crawl-dashboard/1"

    @property
    def dashboard_server(self) -> _DashboardServer:
        return self.server  # type: ignore[return-value]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._json(
                    {
                        "ok": True,
                        "runId": self.dashboard_server.analytics.run_dir.name,
                        "fingerprint": self.dashboard_server.analytics.fingerprint,
                    }
                )
                return
            if parsed.path == "/api/snapshot":
                self._json(self.dashboard_server.analytics.snapshot())
                return
            if parsed.path == "/api/videos":
                self._json(
                    self.dashboard_server.analytics.videos(parse_qs(parsed.query))
                )
                return
            if parsed.path.startswith("/api/videos/"):
                video_id = unquote(parsed.path.removeprefix("/api/videos/"))
                detail = self.dashboard_server.analytics.video_detail(video_id)
                if detail is None:
                    self._json({"error": "video not found"}, status=404)
                else:
                    self._json(detail)
                return
            self._asset(parsed.path)
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._headers("application/json")
        self.end_headers()

    def _asset(self, request_path: str) -> None:
        web_dir = self.dashboard_server.web_dir
        if not web_dir.is_dir():
            self._json(
                {
                    "error": (
                        f"dashboard assets not found at {web_dir}; "
                        "build the dashboard app in dashboard-app"
                    )
                },
                status=503,
            )
            return
        relative = Path(unquote(request_path.lstrip("/")))
        candidate = (web_dir / relative).resolve()
        if web_dir not in candidate.parents and candidate != web_dir:
            self._json({"error": "invalid asset path"}, status=400)
            return
        if not candidate.is_file():
            candidate = web_dir / "index.html"
        if not candidate.is_file():
            self._json({"error": "dashboard index not found"}, status=503)
            return
        content = candidate.read_bytes()
        self.send_response(200)
        self._headers(
            mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _json(self, value: object, *, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._headers("application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _headers(self, content_type: str) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class _DashboardServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        analytics: DuckDBAnalytics,
        web_dir: Path,
    ) -> None:
        super().__init__(address, _DashboardHandler)
        self.analytics = analytics
        self.web_dir = web_dir


def serve_dashboard(
    run_dir: str | Path,
    *,
    web_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    """Serve a live dashboard backed by the run's append-only JSONL files."""

    analytics = DuckDBAnalytics(run_dir)
    server = _DashboardServer(
        (host, port), analytics, Path(web_dir).expanduser().resolve()
    )
    url = f"http://{host}:{server.server_port}/"
    print(f"Live dashboard: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        analytics.close()


__all__ = ["serve_dashboard"]
