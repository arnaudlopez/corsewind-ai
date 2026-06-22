#!/usr/bin/env python3
"""Serve Wind2D static files with gzip compression for local payload testing."""

from __future__ import annotations

import argparse
import gzip
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
COMPRESSIBLE_SUFFIXES = {".html", ".js", ".css", ".json", ".csv", ".txt", ".md"}
# Some minimal base images don't ship a .webp mime mapping; register it so tiles are served as
# image/webp rather than application/octet-stream.
mimetypes.add_type("image/webp", ".webp")


class GzipStaticHandler(SimpleHTTPRequestHandler):
    server_version = "CorseWindGzipHTTP/0.1"

    def __init__(self, *args, directory: str | None = None, **kwargs) -> None:
        super().__init__(*args, directory=directory or str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Vary", "Accept-Encoding")
        # Pre-baked tiles are immutable for a given run; the client versions their URL by the run
        # (?v=<runTimeUtc>), so we can cache hard. This eliminates the per-tile revalidation
        # round-trips (304s) on pan-back/reload that made serving the PNG tiles feel slow.
        if self._is_immutable_tile():
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        super().end_headers()

    def _is_immutable_tile(self) -> bool:
        path = urlsplit(self.path).path.lower()
        return path.endswith((".png", ".webp")) and "/visualizations/wind2d/" in path

    def do_GET(self) -> None:
        self._send_maybe_compressed(head_only=False)

    def do_HEAD(self) -> None:
        self._send_maybe_compressed(head_only=True)

    def _send_maybe_compressed(self, head_only: bool) -> None:
        path = self._local_path()
        if not path or not path.exists() or not path.is_file():
            return super().do_HEAD() if head_only else super().do_GET()
        if not self._client_accepts_gzip() or path.suffix.lower() not in COMPRESSIBLE_SUFFIXES:
            return super().do_HEAD() if head_only else super().do_GET()

        data = path.read_bytes()
        compressed = gzip.compress(data, compresslevel=6)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(compressed)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not head_only:
            self.wfile.write(compressed)

    def _client_accepts_gzip(self) -> bool:
        accepted = self.headers.get("Accept-Encoding", "")
        return "gzip" in {item.strip().split(";", 1)[0] for item in accepted.split(",")}

    def _local_path(self) -> Path | None:
        raw_path = unquote(urlsplit(self.path).path)
        if raw_path.endswith("/"):
            raw_path += "index.html"
        candidate = (ROOT / raw_path.lstrip("/")).resolve()
        try:
            candidate.relative_to(ROOT)
        except ValueError:
            return None
        return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    address = (args.host, args.port)
    httpd = ThreadingHTTPServer(address, GzipStaticHandler)
    print(f"Serving Wind2D with gzip at http://{args.host}:{args.port}/visualizations/wind2d/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
