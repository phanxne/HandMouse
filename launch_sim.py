#!/usr/bin/env python3
"""Serve the 3D hand sandbox on localhost and open it in the default browser."""
import functools
import http.server
import threading
import urllib.request
import webbrowser
from pathlib import Path

PORT = 8765
ROOT = Path(__file__).parent / "sim"


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".task": "application/octet-stream"}

    def log_message(self, *a):
        pass


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


def ensure_model():
    model = ROOT / "hand_landmarker.task"
    if not model.exists():
        root_copy = ROOT.parent / "hand_landmarker.task"
        if root_copy.exists():
            model.write_bytes(root_copy.read_bytes())
        else:
            print("Downloading hand model ...")
            urllib.request.urlretrieve(MODEL_URL, model)


def main():
    ensure_model()
    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", PORT), functools.partial(Handler, directory=str(ROOT)))
    url = f"http://localhost:{PORT}/"
    print(f"Hand Sandbox running at {url}  (Ctrl-C to stop)")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
