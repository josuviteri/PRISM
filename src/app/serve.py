"""Run a local web interface for exploring downloaded PRISM diffusion checkpoints.

Example:
    python -m src.app.serve
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

import torch

from src.app.diffusion_service import DiffusionExplorer, discover_checkpoints


STATIC_DIR = Path(__file__).with_name("static")


class ExplorerServer(ThreadingHTTPServer):
    """HTTP server holding the inference service and its generation lock."""

    def __init__(self, address: tuple[str, int], explorer: DiffusionExplorer):
        super().__init__(address, ExplorerHandler)
        self.explorer = explorer
        self.generation_lock = Lock()


class ExplorerHandler(BaseHTTPRequestHandler):
    """Serve the explorer page and its small JSON API."""

    server: ExplorerServer

    def do_GET(self) -> None:
        """Return the page or the checkpoint selector data."""
        if self.path == "/":
            self._static("index.html", "text/html; charset=utf-8")
        elif self.path == "/static/app.css":
            self._static("app.css", "text/css; charset=utf-8")
        elif self.path == "/static/app.js":
            self._static("app.js", "application/javascript; charset=utf-8")
        elif self.path == "/api/models":
            models = [
                {
                    "id": item.identifier,
                    "dataset": item.dataset,
                    "timesteps": item.timesteps,
                    "num_classes": item.num_classes,
                    "conditional": item.num_classes > 1,
                    "labels": _class_labels(item.dataset, item.num_classes),
                    "p_uncond": item.p_uncond,
                    "guidance_scale": item.guidance_scale,
                }
                for item in self.server.explorer.checkpoints.values()
            ]
            self._json(HTTPStatus.OK, models)
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        """Generate a pair of image grids for validated user controls."""
        if self.path != "/api/generate":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            with self.server.generation_lock:
                noise, samples, frames, info, used_steps = self.server.explorer.generate(
                    str(payload["checkpoint"]),
                    str(payload["method"]),
                    int(payload["steps"]),
                    int(payload["samples"]),
                    int(payload["seed"]),
                    int(payload["label"]) if payload.get("label") is not None else None,
                    float(payload["guidance_scale"]) if payload.get("guidance_scale") is not None else None,
                )
            self._json(HTTPStatus.OK, {
                "noise": _data_url(noise), "samples": _data_url(samples), "dataset": info.dataset,
                "method": payload["method"], "steps": used_steps, "seed": payload["seed"],
                "label": payload.get("label"), "guidance_scale": payload.get("guidance_scale"),
                "trajectory": [
                    {"iteration": iteration, "timestep": timestep, "image": _data_url(image)}
                    for iteration, timestep, image in frames
                ],
            })
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except RuntimeError as error:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"sampling failed: {error}"})

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the local server output focused on startup and sampling errors."""
        del format, args

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        self._send(status, json.dumps(payload), "application/json")

    def _send(self, status: HTTPStatus, body: str, content_type: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _static(self, filename: str, content_type: str) -> None:
        """Serve one of the application-owned static resources."""
        self._send(HTTPStatus.OK, (STATIC_DIR / filename).read_text(encoding="utf-8"), content_type)


def _data_url(image) -> str:
    """Encode a Pillow image as a data URL for the browser response."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _class_labels(dataset: str, num_classes: int) -> list[dict[str, int | str]]:
    """Return stable browser labels for a checkpoint's class indices."""
    names = (
        [str(index) for index in range(num_classes)]
        if dataset == "mnist"
        else ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    )
    return [{"value": index, "name": name} for index, name in enumerate(names[:num_classes])]


def main(argv: Sequence[str] | None = None) -> int:
    """Start the local PRISM diffusion explorer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    checkpoints = discover_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        parser.error(f"no valid PRISM checkpoints found in {args.checkpoint_dir}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    server = ExplorerServer((args.host, args.port), DiffusionExplorer(checkpoints, device))
    print(f"PRISM explorer: http://{args.host}:{args.port} ({device}); Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nExplorer stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
