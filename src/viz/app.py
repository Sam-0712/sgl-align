"""
viz.app -- PyQt6 Diff Viewer with inline char-level rich diff.

Provides a ``DiffViewer`` QMainWindow with a unified inline diff view
matching the TSX reference design: character-level alignment rendered as
colour-coded inline spans, grouped into paragraphs.

Can be run standalone:

    python -m viz.app
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from typing import Optional

_SRC = Path(__file__).resolve().parent.parent  # viz -> src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QApplication, QHBoxLayout, QLabel, QMainWindow,
        QSplitter, QStatusBar, QTextEdit, QVBoxLayout, QWidget,
    )
    _HAS_PYQT6 = True
except ImportError:
    _HAS_PYQT6 = False

# ------- colour palette -------------------------------------------------

_COLOURS = {
    "equal":   {"bg": "#f8f9fa", "text": "#212529", "label": "#6c757d"},
    "modify":  {"bg": "#fff3cd", "text": "#856404", "label": "#ffc107"},
    "delete":  {"bg": "#f8d7da", "text": "#721c24", "label": "#dc3545"},
    "insert":  {"bg": "#d4edda", "text": "#155724", "label": "#28a745"},
    "move":    {"bg": "#e2d9f3", "text": "#432874", "label": "#6f42c1"},
}

# Inline char-level styles (matching TSX reference)
_CHAR_STYLE = {
    "equal":  "color:#374151;",
    "delete": "color:#b45309;background:rgba(254,243,199,0.4);text-decoration:line-through;",
    "insert": "color:#047857;background:rgba(209,250,229,0.5);font-weight:500;border-bottom:1px solid rgba(52,211,153,0.5);",
    "replace": "color:#b45309;background:rgba(254,243,199,0.4);text-decoration:line-through;",  # src part
    "replace_ins": "color:#047857;background:rgba(209,250,229,0.5);font-weight:500;",  # tgt part
}

# ------- helpers ----------------------------------------------------------

def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(text, quote=False)


def _build_inline_html(chars: list) -> str:
    """Build inline HTML from a flat list of [c_src, c_tgt, diff_type] triples.

    This mirrors the TSX ``renderRichDiff`` approach: each char triple is
    rendered as one or two <span> elements according to its diff_type.
    Paragraph breaks (``\\n\\n``) become ``<br><br>``.
    """
    parts: list[str] = []
    i = 0
    n = len(chars)

    while i < n:
        c_src, c_tgt, d_type = chars[i]

        # Paragraph break detection (\\n\\n)
        if c_src == "\n" and i + 1 < n and chars[i + 1][0] == "\n":
            parts.append("<br><br>")
            i += 2
            continue
        if c_tgt == "\n" and i + 1 < n and chars[i + 1][1] == "\n":
            parts.append("<br><br>")
            i += 2
            continue
        # Single newline -> <br>
        if c_src == "\n" or c_tgt == "\n":
            parts.append("<br>")
            i += 1
            continue

        if d_type == "equal":
            ch = c_src or c_tgt or ""
            parts.append(f'<span style="{_CHAR_STYLE["equal"]}">{_esc(ch)}</span>')
            i += 1

        elif d_type == "delete":
            parts.append(f'<span style="{_CHAR_STYLE["delete"]}">{_esc(c_src or "")}</span>')
            i += 1

        elif d_type == "insert":
            parts.append(f'<span style="{_CHAR_STYLE["insert"]}">{_esc(c_tgt or "")}</span>')
            i += 1

        elif d_type == "replace":
            # Show deleted char then inserted char (TSX-style)
            if c_src:
                parts.append(f'<span style="{_CHAR_STYLE["replace"]}">{_esc(c_src)}</span>')
            if c_tgt:
                parts.append(f'<span style="{_CHAR_STYLE["replace_ins"]}">{_esc(c_tgt)}</span>')
            i += 1

        else:
            # fallback: show whatever is present
            ch = c_src or c_tgt or ""
            parts.append(_esc(ch))
            i += 1

    return "".join(parts)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{
    font-family: "Microsoft YaHei", "Consolas", sans-serif;
    font-size: 15px;
    line-height: 2.0;
    margin: 24px 32px;
    color: #374151;
    text-align: justify;
    text-indent: 2em;
}}
</style></head>
<body>{body}</body></html>"""


def _wrap_html(body: str) -> str:
    return _HTML_TEMPLATE.format(body=body)


if _HAS_PYQT6:

    class DiffViewer(QMainWindow):
        """SGL Diff Viewer — unified inline char-level diff.

        Auto-loads ``corpus/text1.txt`` and ``corpus/text2.txt`` on startup,
        runs sentence-level alignment + character-level richdiff, and renders
        a single unified view with colour-coded inline changes.
        """

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)

            self.setWindowTitle("SGL Diff Viewer")
            self.resize(1100, 800)

            self._text_a = ""
            self._text_b = ""

            self._setup_ui()
            self._setup_status_bar()
            self._auto_load_corpus()

        # ========== UI Setup ==========

        def _setup_ui(self) -> None:
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(4)

            # ---- top bar: file label + legend + stats ----
            top_bar = QHBoxLayout()
            top_bar.setSpacing(8)

            self.info_label = QLabel("Loading corpus ...")
            self.info_label.setStyleSheet(
                "font-size:12px; color:#495057; background:#e9ecef;"
                "padding:3px 10px; border-radius:4px;"
            )
            top_bar.addWidget(self.info_label)

            # Legend pills
            for bt, label in [("equal", "相同"), ("modify", "修改"),
                              ("delete", "删除"), ("insert", "插入"),
                              ("move", "移动")]:
                c = _COLOURS[bt]
                lbl = QLabel(label)
                lbl.setStyleSheet(
                    f"background:{c['bg']}; color:{c['text']};"
                    f"padding:1px 8px; border-radius:3px; font-size:11px; font-weight:bold;"
                )
                top_bar.addWidget(lbl)

            top_bar.addStretch()

            self.stats_label = QLabel("")
            self.stats_label.setStyleSheet(
                "font-size:11px; color:#6c757d; padding:3px 6px;"
            )
            top_bar.addWidget(self.stats_label)

            layout.addLayout(top_bar)

            # ---- main diff view (unified) ----
            self.diff_view = QTextEdit()
            self.diff_view.setReadOnly(True)
            self.diff_view.setFont(QFont("Microsoft YaHei", 12))

            layout.addWidget(self.diff_view, stretch=1)  # stretch=1: take all remaining space

        def _setup_status_bar(self) -> None:
            self.status = QStatusBar()
            self.setStatusBar(self.status)
            self.status_label = QLabel("Ready")
            self.status.addPermanentWidget(self.status_label)

        # ========== Auto-load corpus ==========

        def _auto_load_corpus(self) -> None:
            module_dir = Path(__file__).resolve().parent
            project_root = module_dir.parent.parent  # viz -> src -> root

            candidates = [
                (project_root / "corpus" / "text1.txt",
                 project_root / "corpus" / "text2.txt"),
                (Path("corpus") / "text1.txt",
                 Path("corpus") / "text2.txt"),
            ]

            for fa, fb in candidates:
                if fa.is_file() and fb.is_file():
                    self.load_files(str(fa), str(fb))
                    return

            self.info_label.setText("No corpus found. Use load_files() to begin.")

        # ========== Public API ==========

        def load_files(self, file_a: str, file_b: str) -> None:
            """Load two text files, compute sentence-level + char-level diff, and render."""
            try:
                with open(file_a, encoding="utf-8") as f:
                    text_a = f.read()
                with open(file_b, encoding="utf-8") as f:
                    text_b = f.read()
            except FileNotFoundError as e:
                self.status.showMessage(f"File not found: {e}", 5000)
                return
            except IOError as e:
                self.status.showMessage(f"IOError: {e}", 5000)
                return

            self._text_a = text_a
            self._text_b = text_b
            self._file_a = file_a
            self._file_b = file_b

            self._compute_and_render(text_a, text_b)

            fa_name = Path(file_a).name
            fb_name = Path(file_b).name
            self.info_label.setText(f"{fa_name} ↔ {fb_name}")

        def _compute_and_render(self, text_a: str, text_b: str) -> None:
            """Paragraph-level alignment → direct char-diff on each paragraph pair.

            The text is split into paragraphs by ``\\n\\n``, paragraphs are aligned
            globally via NW, then each matched paragraph pair is character-diffed
            directly (no sentence-level intermediate step).  This avoids sentence-
            count mismatches that break the alignment when the target text merges or
            splits sentences differently from the source.

            Paragraph breaks from the original text are preserved in the output.
            """
            try:
                from core.sglalign import Aligner
                from core.sglsim import hybrid_similarity
                from core.sgldiff import CharLevelAligner
            except ImportError as e:
                self._fallback_render(text_a, text_b, str(e))
                return

            # 1. Split into paragraphs (preserve \n\n boundaries)
            paras_a = [p.strip() for p in text_a.split("\n\n") if p.strip()]
            paras_b = [p.strip() for p in text_b.split("\n\n") if p.strip()]

            if not paras_a and not paras_b:
                self.diff_view.setPlainText("(empty texts)")
                return
            if not paras_a:
                paras_a = [text_a.strip()] if text_a.strip() else [""]
            if not paras_b:
                paras_b = [text_b.strip()] if text_b.strip() else [""]

            ca = CharLevelAligner()
            all_chars: list[list] = []
            modify_count = delete_count = insert_count = 0

            # 2. Align paragraphs globally (NW on the paragraph list)
            para_aligner = Aligner(gap_open=-2.0, gap_extend=-0.5, mm_th=0.2, linear=True)
            para_result = para_aligner.align(paras_a, paras_b, hybrid_similarity)

            for ppair in para_result.pairs:
                # Paragraph break between paragraphs
                if all_chars:
                    all_chars.append(["\n", "\n", "equal"])

                if ppair.state == "match" and ppair.source and ppair.target:
                    # ---- Aligned paragraph pair: direct character-level diff ----
                    cdiffs = ca.align_chars(ppair.source, ppair.target)
                    for cd in cdiffs:
                        dt = cd.diff_type
                        if dt == "equal":
                            all_chars.append([cd.char_src or "", cd.char_tgt or "", "equal"])
                        elif dt == "delete":
                            all_chars.append([cd.char_src or "", None, "delete"])
                        elif dt == "insert":
                            all_chars.append([None, cd.char_tgt or "", "insert"])
                    modify_count += 1

                elif ppair.state == "delete" and ppair.source:
                    for ch in ppair.source:
                        all_chars.append([ch, None, "delete"])
                    delete_count += 1

                elif ppair.state == "insert" and ppair.target:
                    for ch in ppair.target:
                        all_chars.append([None, ch, "insert"])
                    insert_count += 1

            # 5. Build inline HTML
            inline_html = _build_inline_html(all_chars)
            full_html = _wrap_html(inline_html)

            self.diff_view.setHtml(full_html)

            # 5. Stats
            stats_parts = []
            if modify_count: stats_parts.append(f"修改 {modify_count}")
            if delete_count: stats_parts.append(f"删除 {delete_count}")
            if insert_count: stats_parts.append(f"插入 {insert_count}")
            self.stats_label.setText("  |  ".join(stats_parts) if stats_parts else "无差异")
            self.status_label.setText(
                f"Text A: {len(paras_a)} paragraphs, {len(text_a)} chars  |  "
                f"Text B: {len(paras_b)} paragraphs, {len(text_b)} chars  |  "
                f"{len(all_chars)} chars diffed"
            )
            self.status.showMessage("Diff ready", 3000)

        def _fallback_render(self, text_a: str, text_b: str, reason: str = "") -> None:
            """Fallback: plain text when dependencies are missing."""
            self.diff_view.setPlainText(
                f"=== Source ===\n{text_a}\n\n=== Target ===\n{text_b}"
            )
            self.status.showMessage(
                f"Dependencies not available ({reason}). Showing plain text.", 10000
            )


else:
    class DiffViewer:  # type: ignore[no-redef]
        """Stub — requires PyQt6."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyQt6 is required for the visualization module. "
                "Install with: pip install PyQt6"
            )


def main() -> None:
    """Run the DiffViewer as a standalone application."""
    if not _HAS_PYQT6:
        print("PyQt6 is not installed. Install with: pip install PyQt6")
        sys.exit(1)

    app = QApplication(sys.argv)
    viewer = DiffViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
