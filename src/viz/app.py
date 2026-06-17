"""
viz.app -- Lightweight HTTP server for the text-similarity visualisation tool.

Serves ``app.html`` and provides JSON API endpoints for alignment computation.
Uses only Python stdlib (http.server) — no extra dependencies.

Run standalone::

    python -m viz.app
    # then open http://localhost:7421 in your browser
"""

from __future__ import annotations

import json
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional

# Ensure src/ is on sys.path so ``core`` is importable
_SRC = Path(__file__).resolve().parent.parent  # viz -> src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_PORT = 7421
_HTML_FILE = Path(__file__).resolve().parent / "app.html"


# ====================================================================== #
#  Alignment computation                                                 #
# ====================================================================== #

def _split_for_viz(text: str) -> list[str]:
    """Split text into sentences for visualisation, preserving ``\\n``.

    Newlines are accumulated and prepended to the next real sentence
    instead of being emitted as standalone tokens.  This avoids inflating
    the alignment matrix dimensions with spurious ``\\n\\n`` rows/columns.
    """
    import re
    if not text or not text.strip():
        return []
    pattern = r'[^。！？.!?…\n]+[。！？.!?…]+["\u201d\u2019\']*|\n+'
    parts = re.findall(pattern, text)
    result: list[str] = []
    newline_buf = ""
    for p in parts:
        if p.startswith("\n"):
            newline_buf += p          # accumulate newlines
        else:
            s = p.strip()
            if s:
                if newline_buf:
                    s = newline_buf + s   # prepend accumulated newlines to sentence
                    newline_buf = ""
                result.append(s)
    return result if result else [text.strip()]


def compute_alignment(text_a: str, text_b: str) -> dict[str, Any]:
    """Run sentence-level alignment + richdiff + heatmap computation.

    Returns a dict with ``segments`` (for rich diff) and ``heatmap`` (for matrix).
    """
    from core.sglalign import Aligner
    from core.sglsim import hybrid_similarity
    from core.sgldiff import richdiff

    # 1. Split into sentences (preserving \n for paragraph reconstruction)
    sents_a = _split_for_viz(text_a)
    sents_b = _split_for_viz(text_b)
    if not sents_a:
        sents_a = [text_a.strip()] if text_a.strip() else [""]
    if not sents_b:
        sents_b = [text_b.strip()] if text_b.strip() else [""]

    # 2. Sentence-level NW alignment with full matrices (for heatmap)
    aligner = Aligner(gap_open=-1.5, gap_extend=-0.2, mm_th=0.2, linear=False)
    alignment_result = aligner.alignfb(sents_a, sents_b, hybrid_similarity)

    # 3. richdiff on ALL aligned pairs (block merging across sentences)
    diff_result = richdiff(alignment_result.pairs, similarity_func=hybrid_similarity)

    # 4. Build segments (matching TSX AlignmentSegment interface)
    segments: list[dict[str, Any]] = []
    for block in diff_result.blocks:
        chars = [[cd.char_src, cd.char_tgt, cd.diff_type] for cd in block.chars]
        segments.append({
            "type": block.block_type,
            "orig": block.source_text,
            "rew": block.target_text,
            "chars": chars,
        })

    # 5. Build heatmap data
    n, m = len(sents_a), len(sents_b)
    sim_matrix = [
        [hybrid_similarity(sents_a[i], sents_b[j]) for j in range(m)]
        for i in range(n)
    ]
    dp_matrix = alignment_result.dp_matrix or []
    fb_matrix = alignment_result.fb_matrix or []
    path = [[r, c] for r, c, _ in (alignment_result.backtrace_path or [])]

    return {
        "segments": segments,
        "heatmap": {
            "sim_matrix": sim_matrix,
            "dp_matrix": dp_matrix,
            "fb_matrix": fb_matrix,
            "path": path,
            "total_score": alignment_result.score,
            "seq1": sents_a,
            "seq2": sents_b,
        },
    }


# ====================================================================== #
#  HTTP server                                                            #
# ====================================================================== #

class Handler(BaseHTTPRequestHandler):
    """Request handler: serves app.html + JSON API."""

    def log_message(self, format, *args):
        # Suppress default logging (keep terminal clean)
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- GET ----

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/app.html":
            try:
                html = _HTML_FILE.read_text(encoding="utf-8")
                self._send_html(html)
            except FileNotFoundError:
                self._send_json({"success": False, "error": "app.html not found"}, 404)
        elif self.path == "/api/corpus":
            self._serve_corpus()
        elif self.path == "/api/example":
            self._serve_example()
        else:
            self._send_json({"success": False, "error": "Not found"}, 404)

    # ---- POST ----

    def do_POST(self) -> None:
        if self.path == "/api/align":
            self._handle_align()
        else:
            self._send_json({"success": False, "error": "Not found"}, 404)

    # ---- API handlers ----

    def _serve_corpus(self) -> None:
        module_dir = Path(__file__).resolve().parent
        project_root = module_dir.parent.parent  # viz -> src -> root
        candidates = [
            (project_root / "corpus" / "text1.txt",
             project_root / "corpus" / "text2.txt"),
        ]
        for fa, fb in candidates:
            if fa.is_file() and fb.is_file():
                try:
                    self._send_json({
                        "success": True,
                        "text_a": fa.read_text(encoding="utf-8"),
                        "text_b": fb.read_text(encoding="utf-8"),
                    })
                    return
                except Exception as e:
                    self._send_json({"success": False, "error": str(e)})
                    return
        self._send_json({"success": False, "error": "Corpus not found"})

    def _serve_example(self) -> None:
        module_dir = Path(__file__).resolve().parent
        project_root = module_dir.parent.parent
        corpus_a = (project_root / "corpus" / "text1.txt").read_text(encoding="utf-8")
        corpus_b = (project_root / "corpus" / "text2.txt").read_text(encoding="utf-8")
        self._send_json({"success": True, "text_a": corpus_a, "text_b": corpus_b})

    def _handle_align(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8"))
            text1 = req.get("text1", "")
            text2 = req.get("text2", "")
            if not text1.strip() or not text2.strip():
                self._send_json({"success": False, "error": "请输入要对齐的两段文本"})
                return
            data = compute_alignment(text1, text2)
            self._send_json({"success": True, "data": data})
        except Exception as e:
            self._send_json({"success": False, "error": str(e)})


# ====================================================================== #
#  Entry point                                                            #
# ====================================================================== #

def main() -> None:
    server = HTTPServer(("127.0.0.1", _PORT), Handler)
    url = f"http://127.0.0.1:{_PORT}/"
    print(f"SGL Visualisation Server running at {url}")
    print("Press Ctrl+C to stop.\n")
    # Open browser after a short delay
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
