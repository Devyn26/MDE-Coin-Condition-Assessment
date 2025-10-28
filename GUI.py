# Updated for F25-06 coin assessment team
# Updated by: Luke Graham
# Date: 2025-10-02

import sys  # system-level utilities (argv, exit)
import os  # filesystem ops
import time  # timestamps for "new PDF" detection
import shutil  # copy file for "Save As"
from pathlib import Path  # filesystem path handling

import pipe
import time

from PyQt5.QtCore import (  # core Qt classes/signals/threads/settings
    pyqtSlot, Qt, QObject, QThread, pyqtSignal, QPoint, QSize, QSettings, QUrl
)
from PyQt5.QtGui import QPixmap, QTextDocument, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLineEdit,
    QLabel,
    QFileDialog,
    QTextEdit,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QSpacerItem,
    QProgressBar,
    QCheckBox,
    QDialog,
    QToolBar,
    QAction,
    QScrollArea,
    QMessageBox,
)
from PyQt5.QtPrintSupport import QPrinter

# ---------- Optional PDF backends ----------
_HAVE_QTPDF = False
try:
    # Check if QtPdf is available (native PDF viewer)
    from PyQt5.QtPdf import QPdfDocument
    from PyQt5.QtPdfWidgets import QPdfView
    _HAVE_QTPDF = hasattr(QPdfView, "setDocument")
except Exception:
    _HAVE_QTPDF = False

_HAVE_FITZ = False
if not _HAVE_QTPDF:
    try:
        # PyMuPDF fallback: render PDF pages to images
        import fitz  # PyMuPDF
        _HAVE_FITZ = True
    except Exception:
        _HAVE_FITZ = False

# WebEngine (Chromium) fallback for PDF viewing if neither of the above is present
_HAVE_WEBENGINE = False
if not _HAVE_QTPDF and not _HAVE_FITZ:
    try:
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        _HAVE_WEBENGINE = True
    except Exception:
        _HAVE_WEBENGINE = False

# ---------------- Local modules (keep relative imports for package) ----------------
try:
    from . import patternMatching
    from . import WheatStalkGrader
    from . import MorganGrader
except Exception:
    patternMatching = None
    WheatStalkGrader = None
    MorganGrader = None

THIS_DIR = Path(__file__).resolve().parent  # ...\Coin Assesment\LincolnCent
REPO_ROOT = THIS_DIR.parent                # ...\Coin Assesment
ABS_TEST_PDF = os.path.join("MorganSilverDollar/Morgan_Dollar_main", "test.pdf")

# ---------- Known/expected report paths (used by GradeWorker first) ----------
REPORT_DEFAULT_PATHS = [
    str(ABS_TEST_PDF),  # \Coin Assesment\MorganSilverDollar\Morgan_Dollar_main\test.pdf
    "Reports/coin_report.pdf",
    "Reports/latest.pdf",
    "Reports/test.pdf",
]


# ---------------- Image label (blank by default; aspect-preserving when set) ----------------
class ImageLabel(QLabel):
    """
    Starts BLANK (no default coin image). Shows a subtle placeholder frame + text
    until set_image_path() points at a valid image. Rescales with aspect ratio.
    """

    # Create an empty image label with a placeholder message.
    def __init__(self, placeholder_text: str = "No image loaded", parent=None):
        super().__init__(parent)
        self._orig_pixmap = QPixmap()      # no default pixmap
        self._have_image = False
        self.placeholder_text = placeholder_text
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(240, 240)
        self.apply_theme(dark=False)

    # Update placeholder border style to match dark/light theme.
    def apply_theme(self, dark: bool):
        border = "#3d3d3d" if dark else "#caaea2"
        self.setStyleSheet(f"QLabel {{ border: 2px dashed {border}; border-radius: 8px; }}")
        self.update()

    # Load an image from disk; if invalid, revert to placeholder.
    def set_image_path(self, path: str):
        pm = QPixmap(path)
        self._have_image = not pm.isNull()
        self._orig_pixmap = pm if self._have_image else QPixmap()
        if self._have_image:
            self._apply_scaled_pixmap()
        else:
            self.clear()  # back to placeholder
        self.update()

    # Explicitly clear the image and show the placeholder.
    def clear_image(self):
        self._orig_pixmap = QPixmap()
        self._have_image = False
        self.clear()
        self.update()

    # Keep the image correctly scaled when the widget resizes.
    def resizeEvent(self, event):
        if self._have_image:
            self._apply_scaled_pixmap()
        super().resizeEvent(event)

    # Scale and set the pixmap while preserving aspect ratio.
    def _apply_scaled_pixmap(self):
        if not self._have_image:
            return
        target = self.size()
        if not target.isValid() or self._orig_pixmap.isNull():
            return
        scaled = self._orig_pixmap.scaled(
            target.width(), target.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        super().setPixmap(scaled)

    # Draw the placeholder text when no image is present.
    def paintEvent(self, event):
        super().paintEvent(event)
        if self._have_image:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.palette().color(self.foregroundRole()))
        pen.setStyle(Qt.SolidLine)
        painter.setPen(pen)
        painter.drawText(self.rect(), Qt.AlignCenter, self.placeholder_text)
        painter.end()


# ---------------- Report Viewer ----------------
class ReportViewer(QDialog):
    """
    A themed dialog that previews a PDF report with scrolling/zoom and 'Save As…'.

    Backends (in priority order):
      1) QtPdf (native) if available (PyQt5/6 guarded)
      2) PyMuPDF fallback (renders pages to images)
      3) PyQtWebEngine (Chromium PDF viewer) fallback
    """

    # Build the report viewer UI and choose the best available PDF backend.
    def __init__(self, parent=None, dark=False):
        super().__init__(parent)
        self.setWindowTitle("Grading Report")
        self.resize(900, 700)
        self._pdf_path = None
        self._dark = dark

        # Toolbar
        self.toolbar = QToolBar(self)
        self.act_zoom_in = QAction("Zoom +", self)
        self.act_zoom_out = QAction("Zoom -", self)
        self.act_fit_width = QAction("Fit Width", self)
        self.act_actual = QAction("Actual Size", self)
        self.act_save = QAction("Save As…", self)
        self.toolbar.addAction(self.act_zoom_in)
        self.toolbar.addAction(self.act_zoom_out)
        self.toolbar.addAction(self.act_fit_width)
        self.toolbar.addAction(self.act_actual)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_save)

        lay = QVBoxLayout(self)
        lay.addWidget(self.toolbar)

        # Backends
        self._qtpdf_doc = None
        self._qtpdf_view = None

        self._scroll = None
        self._container = None
        self._v = None
        self._page_labels = []  # PyMuPDF widgets
        self._scale = 1.0

        self._web = None
        self._web_zoom = 1.0

        if _HAVE_QTPDF:
            self._qtpdf_doc = QPdfDocument(self)
            self._qtpdf_view = QPdfView(self)
            if hasattr(self._qtpdf_view, "setPageMode") and hasattr(QPdfView, "PageMode"):
                try:
                    self._qtpdf_view.setPageMode(QPdfView.PageMode.MultiPage)
                except Exception:
                    pass
            if hasattr(self._qtpdf_view, "setZoomMode") and hasattr(QPdfView, "ZoomMode"):
                try:
                    self._qtpdf_view.setZoomMode(QPdfView.ZoomMode.Custom)
                except Exception:
                    pass
            lay.addWidget(self._qtpdf_view, 1)
            self.act_zoom_in.triggered.connect(self._qtpdf_zoom_in)
            self.act_zoom_out.triggered.connect(self._qtpdf_zoom_out)
            self.act_fit_width.triggered.connect(self._qtpdf_fit_width)
            self.act_actual.triggered.connect(self._qtpdf_actual)

        elif _HAVE_FITZ:
            # Scrollable image-based viewer
            self._scroll = QScrollArea(self)
            self._scroll.setWidgetResizable(True)
            self._container = QWidget(self)
            self._v = QVBoxLayout(self._container)
            self._v.setContentsMargins(0, 0, 0, 0)
            self._v.setSpacing(12)
            self._scroll.setWidget(self._container)
            lay.addWidget(self._scroll, 1)

            self.act_zoom_in.triggered.connect(self._img_zoom_in)
            self.act_zoom_out.triggered.connect(self._img_zoom_out)
            self.act_fit_width.triggered.connect(self._img_fit_width)
            self.act_actual.triggered.connect(self._img_actual)

        elif _HAVE_WEBENGINE:
            # Chromium viewer
            self._web = QWebEngineView(self)
            lay.addWidget(self._web, 1)
            self.act_zoom_in.triggered.connect(self._web_zoom_in)
            self.act_zoom_out.triggered.connect(self._web_zoom_out)
            self.act_fit_width.triggered.connect(self._web_fit_width)
            self.act_actual.triggered.connect(self._web_actual)

        self.act_save.triggered.connect(self._save_as)
        self._apply_theme(dark)

    # Apply dark/light styles to the dialog and toolbar.
    def _apply_theme(self, dark: bool):
        if dark:
            self.setStyleSheet("""
                QDialog { background:#1f1f1f; color:#f0f0f0; }
                QToolBar { background:#2b2b2b; border:none; }
            """)
        else:
            self.setStyleSheet("""
                QDialog { background:#f1c6b6; color:#1e1a18; }
                QToolBar { background:#f6e5df; border:none; }
            """)

    # Open and display a PDF file using the selected backend.
    def load_pdf(self, pdf_path: str):
        self._pdf_path = pdf_path
        if not pdf_path or not os.path.exists(pdf_path):
            QMessageBox.warning(self, "Report", "Report file not found.")
            return

        if _HAVE_QTPDF:
            try:
                status = self._qtpdf_doc.load(pdf_path)
            except Exception as e:
                QMessageBox.warning(self, "Report", f"Could not load PDF (QtPdf error: {e}).")
                return
            try:
                ok = (status == QPdfDocument.NoError)
            except Exception:
                ok = True
            if not ok:
                QMessageBox.warning(self, "Report", "Could not load PDF (QtPdf error).")
                return
            self._qtpdf_view.setDocument(self._qtpdf_doc)
            self._qtpdf_actual()
            return

        if _HAVE_FITZ:
            self._render_all_pages_as_images()
            return

        if _HAVE_WEBENGINE:
            self._web.setUrl(QUrl.fromLocalFile(os.path.abspath(pdf_path)))
            self._web_actual()
            return

        QMessageBox.information(
            self, "Report",
            "PDF preview backend not available.\n\nInstall one of:\n"
            "• PyMuPDF (pip install pymupdf)\n"
            "• PyQtWebEngine (pip install PyQtWebEngine)\n"
            "• QtPdf wheels compatible with PyQt version."
        )

    # ----- QtPdf actions -----

    # Set the current zoom factor (clamped) for the QtPdf view.
    def _qtpdf_set_zoom_factor(self, factor: float):
        if hasattr(self._qtpdf_view, "setZoomFactor"):
            try:
                self._qtpdf_view.setZoomFactor(max(0.05, min(factor, 20.0)))
            except Exception:
                pass

    # Read the current zoom factor from the QtPdf view.
    def _qtpdf_get_zoom_factor(self) -> float:
        if hasattr(self._qtpdf_view, "zoomFactor"):
            try:
                return float(self._qtpdf_view.zoomFactor())
            except Exception:
                return 1.0
        return 1.0

    # Switch to "fit to width" mode where available; otherwise approximate.
    def _qtpdf_set_zoom_mode_fit_width(self):
        if hasattr(self._qtpdf_view, "setZoomMode") and hasattr(QPdfView, "ZoomMode"):
            try:
                self._qtpdf_view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
            except Exception:
                self._qtpdf_set_zoom_factor(self._qtpdf_get_zoom_factor())

    # Switch to custom zoom mode (free zoom control).
    def _qtpdf_set_zoom_mode_custom(self):
        if hasattr(self._qtpdf_view, "setZoomMode") and hasattr(QPdfView, "ZoomMode"):
            try:
                self._qtpdf_view.setZoomMode(QPdfView.ZoomMode.Custom)
            except Exception:
                pass

    # Increase zoom level in the QtPdf view.
    def _qtpdf_zoom_in(self):
        self._qtpdf_set_zoom_mode_custom()
        self._qtpdf_set_zoom_factor(self._qtpdf_get_zoom_factor() * 1.25)

    # Decrease zoom level in the QtPdf view.
    def _qtpdf_zoom_out(self):
        self._qtpdf_set_zoom_mode_custom()
        self._qtpdf_set_zoom_factor(self._qtpdf_get_zoom_factor() / 1.25)

    # Fit the current page(s) to the available width.
    def _qtpdf_fit_width(self):
        self._qtpdf_set_zoom_mode_fit_width()

    # Restore 100% zoom.
    def _qtpdf_actual(self):
        self._qtpdf_set_zoom_mode_custom()
        self._qtpdf_set_zoom_factor(1.0)

    # ----- PyMuPDF (image) backend -----

    # Render all PDF pages to QLabels as images (initial population).
    def _render_all_pages_as_images(self):
        for i in reversed(range(self._v.count())):
            item = self._v.itemAt(i)
            w = item.widget() if item else None
            if w:
                w.setParent(None)
        self._page_labels.clear()

        doc = fitz.open(self._pdf_path)
        try:
            for page in doc:
                m = fitz.Matrix(self._scale, self._scale)
                pix = page.get_pixmap(matrix=m, alpha=False)
                from PyQt5.QtGui import QImage
                qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy()
                lbl = QLabel(self._container)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setPixmap(QPixmap.fromImage(qimg))
                self._v.addWidget(lbl)
                self._page_labels.append(lbl)
        finally:
            doc.close()
        self._v.addStretch(1)

    # Zoom in for the image backend and re-render pages.
    def _img_zoom_in(self):
        self._scale = min(10.0, self._scale * 1.25)
        self._rerender_pages()

    # Zoom out for the image backend and re-render pages.
    def _img_zoom_out(self):
        self._scale = max(0.1, self._scale / 1.25)
        self._rerender_pages()

    # Approximate "fit to width" by computing a scale for the first page width.
    def _img_fit_width(self):
        if not (self._scroll and self._page_labels and self._pdf_path):
            return
        doc = fitz.open(self._pdf_path)
        try:
            base_w = doc[0].rect.width
        finally:
            doc.close()
        view_w = max(1, self._scroll.viewport().width() - 24)
        if base_w > 0:
            self._scale = max(0.1, view_w / base_w)
            self._rerender_pages()

    # Reset to 100% scale for the image backend.
    def _img_actual(self):
        self._scale = 1.0
        self._rerender_pages()

    # Re-render current pages at self._scale; update existing labels.
    def _rerender_pages(self):
        if not (_HAVE_FITZ and self._pdf_path):
            return
        doc = fitz.open(self._pdf_path)
        try:
            for i, page in enumerate(doc):
                m = fitz.Matrix(self._scale, self._scale)
                pix = page.get_pixmap(matrix=m, alpha=False)
                from PyQt5.QtGui import QImage
                qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy()
                if i < len(self._page_labels):
                    self._page_labels[i].setPixmap(QPixmap.fromImage(qimg))
                else:
                    lbl = QLabel(self._container)
                    lbl.setAlignment(Qt.AlignCenter)
                    lbl.setPixmap(QPixmap.fromImage(qimg))
                    self._v.insertWidget(i, lbl)
                    self._page_labels.append(lbl)
        finally:
            doc.close()

    # ----- WebEngine backend -----

    # Set zoom factor for the Chromium viewer (clamped).
    def _web_set_zoom(self, z):
        if not self._web:
            return
        self._web_zoom = max(0.25, min(5.0, z))
        try:
            self._web.setZoomFactor(self._web_zoom)
        except Exception:
            pass

    # Zoom in for the Chromium viewer.
    def _web_zoom_in(self):
        self._web_set_zoom(self._web_zoom * 1.25)

    # Zoom out for the Chromium viewer.
    def _web_zoom_out(self):
        self._web_set_zoom(self._web_zoom / 1.25)

    # Best-effort "fit width" (Chromium usually does this automatically).
    def _web_fit_width(self):
        self._web_set_zoom(1.0)

    # Reset to 100% zoom in the Chromium viewer.
    def _web_actual(self):
        self._web_set_zoom(1.0)

    # Prompt the user for a destination and copy the current PDF there.
    def _save_as(self):
        if not self._pdf_path or not os.path.exists(self._pdf_path):
            QMessageBox.warning(self, "Save As", "No report loaded.")
            return
        start_dir = str(Path(self._pdf_path).resolve().parent)
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Report As",
            os.path.join(start_dir, Path(self._pdf_path).name),
            "PDF Files (*.pdf)"
        )
        if not fn:
            return
        try:
            shutil.copy(self._pdf_path, fn)
        except Exception as e:
            QMessageBox.critical(self, "Save As", f"Failed to save: {e}")
        else:
            QMessageBox.information(self, "Save As", "Report saved.")


# ---------------- Worker to run grading off the UI thread ----------------
class GradeWorker(QObject):
    finished = pyqtSignal(dict)       # emits {"fft": str, "fm": str, "report_path": str}
    failed = pyqtSignal(str)

    # Store the selected image paths and start time for report detection.
    def __init__(self, obverse_path: str, reverse_path: str):
        super().__init__()
        self.obverse_path = obverse_path
        self.reverse_path = reverse_path
        self._t0 = time.time()  # mark start time to detect newly created PDFs

    # Try to find the most likely report PDF created since grading started.
    def _find_new_report_pdf(self) -> str:
        """
        Prefer the known default path, then fall back to the newest .pdf
        created after grading started in likely directories.
        """
        for p in REPORT_DEFAULT_PATHS:
            try:
                rp = Path(p)
                if rp.exists() and rp.stat().st_mtime >= self._t0 - 5.0:
                    return str(rp.resolve())
            except Exception:
                pass

        candidates_dirs = {Path(os.getcwd())}
        if self.reverse_path:
            candidates_dirs.add(Path(self.reverse_path).resolve().parent)
        reports_dir = Path(os.getcwd()) / "Reports"
        if reports_dir.exists():
            candidates_dirs.add(reports_dir)

        newest_path = ""
        newest_mtime = 0.0
        for d in candidates_dirs:
            try:
                for p in d.glob("*.pdf"):
                    mt = p.stat().st_mtime
                    if mt >= self._t0 - 5.0 and mt > newest_mtime:
                        newest_mtime = mt
                        newest_path = str(p.resolve())
            except Exception:
                continue
        return newest_path

    # Execute grading using the available graders and emit results (off UI thread).
    def run(self):
        try:
            is_msd = False
            try:
                if self.reverse_path and patternMatching:
                    is_msd = patternMatching.imgIsMSD(self.reverse_path)
            except Exception:
                is_msd = False

            # FFT-like grading (Morgan or WheatStalk)
            if is_msd and self.reverse_path and MorganGrader:
                blw, brw, ulw, urw, oss = MorganGrader.gradeMorganSilverDollar(self.reverse_path)
                avg = (blw + brw + ulw + urw) / 4.0
                if avg < 26.194:
                    oss_string = "<3 | Possible Image Error."
                elif avg > 87.487:
                    oss_string = "70 | Above Standard Image Range."
                else:
                    oss_string = str(round(oss, 3))
                fft_text = (
                    f"Lower Left Wing: {round(blw,3)}\n"
                    f"Lower Right Wing: {round(brw,3)}\n"
                    f"Upper Left Wing: {round(ulw,3)}\n"
                    f"Upper Right Wing: {round(urw,3)}\n"
                    f"Estimated Sheldon Scale Grade: {oss_string}"
                )
            elif self.reverse_path and WheatStalkGrader:
                lsg, rsg, oss = WheatStalkGrader.gradeWheatStalkPenny(self.reverse_path)
                avg = (lsg + rsg) / 2.0
                if avg < 46.211:
                    oss_string = "<8 | Possible Image Error."
                elif avg > 118.711:
                    oss_string = ">68 | Above Standard Image Range."
                else:
                    oss_string = str(round(oss, 3))
                fft_text = (
                    f"Left Stalk Grade: {round(lsg,3)}\n"
                    f"Right Stalk Grade: {round(rsg,3)}\n"
                    f"Estimated Sheldon Scale Grade: {oss_string}"
                )
            else:
                fft_text = "FFT grading unavailable: required module not found."

            # Feature-match grading (reverse if Morgan; otherwise obverse)
            if is_msd and self.reverse_path and patternMatching:
                fm = round(patternMatching.gradeCoin(self.reverse_path, True, False))
                fm_text = str(fm)
            elif self.obverse_path and patternMatching:
                fm = round(patternMatching.gradeCoin(self.obverse_path, False, False))
                fm_text = str(fm)
            else:
                fm_text = "Feature Match unavailable: required module not found."

            report_path = self._find_new_report_pdf()
            self.finished.emit({"fft": fft_text, "fm": fm_text, "report_path": report_path})
        except Exception as e:
            self.failed.emit(str(e))


# ---------------- Main App ----------------
class App(QWidget):
    ORG = "F25-06-Team"
    APP = "CoinGraderGUI"

    # Public signals for integrators
    imagesSelected = pyqtSignal(str, str)  # (obverse_path, reverse_path)
    reportOpened = pyqtSignal(str)         # emitted when a report is opened via this UI

    # Build the main window, wire up controls, and restore saved settings.
    def __init__(self):
        super().__init__()
        self.title = "Automated Coin Grader"

        # state
        self.obverse_path = ""
        self.reverse_path = ""
        self.fft_results = "----"
        self.fm_results = "----"
        self.dark_mode = False
        self.last_report_path = ""  # remember last found report

        self.settings = QSettings(self.ORG, self.APP)

        self._build_ui()
        self._restore_settings()

    # Return the stylesheet used in light mode (colors + spacing).
    def _light_stylesheet(self) -> str:
        return """
            QWidget { background:#f1c6b6; color:#1e1a18; }
            QPushButton { background:#a8937e; border-radius:6px; padding:8px 14px; color:#1e1a18; }
            QPushButton:disabled { background:#c9bbaf; color:#6e625a; }
            QLineEdit { background:#f6e5df; padding:6px; color:#1e1a18; }
            QTextEdit { background:#f6e5df; padding:10px; color:#1e1a18; }
            QLabel#title { font-size:22px; font-weight:bold; }
            QLabel.section { font-size:18px; font-weight:bold; }
            QLabel#instructions { background:#f6e5df; border:1px solid #caaea2; padding:10px; }
        """

    # Return the stylesheet used in dark mode.
    def _dark_stylesheet(self) -> str:
        return """
            QWidget { background:#1f1f1f; color:#f0f0f0; }
            QPushButton { background:#3a3a3a; border-radius:6px; padding:8px 14px; color:#f0f0f0; }
            QPushButton:disabled { background:#2a2a2a; color:#777; }
            QLineEdit { background:#2b2b2b; padding:6px; color:#f0f0f0; border:1px solid #3d3d3d; }
            QTextEdit { background:#2b2b2b; padding:10px; color:#f0f0f0; border:1px solid #3d3d3d; }
            QLabel#title { font-size:22px; font-weight:bold; }
            QLabel.section { font-size:18px; font-weight:bold; }
            QLabel#instructions { background:#2b2b2b; border:1px solid #3d3d3d; padding:10px; }
        """

    # Apply the active theme and propagate to child widgets that have theme hooks.
    def _apply_theme(self):
        self.setStyleSheet(self._dark_stylesheet() if self.dark_mode else self._light_stylesheet())
        self.chk_dark.setChecked(self.dark_mode)
        self.lbl_obverse.apply_theme(self.dark_mode)
        self.lbl_reverse.apply_theme(self.dark_mode)

    # Construct the widget tree, lay out the UI, and connect signals.
    def _build_ui(self):
        self.setWindowTitle(self.title)
        self.resize(1200, 800)
        self.setMinimumSize(900, 600)

        root = QGridLayout(self)
        root.setHorizontalSpacing(16)
        root.setVerticalSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        # Top bar
        top_row = QHBoxLayout()
        instructions = QLabel(self)
        instructions.setObjectName("instructions")
        instructions.setWordWrap(True)
        instructions.setText(
            "To use this software, load the obverse and the reverse of the coin you want to characterize. "
            "Then click Grade. FFT and Feature Match estimate wear on a 0–70 scale (0 = most worn, 70 = least)."
        )
        instructions.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_row.addWidget(instructions, 10)

        self.chk_dark = QCheckBox("Dark mode")
        self.chk_dark.stateChanged.connect(self._toggle_dark_mode)
        top_row.addWidget(self.chk_dark, 0, Qt.AlignRight)

        root.addLayout(top_row, 0, 0, 1, 3)

        # Left image (BLANK at start)
        self.lbl_obverse = ImageLabel("No obverse selected")
        root.addWidget(self.lbl_obverse, 1, 0)

        # Center column
        center_col = QVBoxLayout()

        self.result_box = QTextEdit(self)
        self.result_box.setReadOnly(True)
        self.result_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        center_col.addWidget(self.result_box, 1)

        busy_row = QHBoxLayout()
        busy_row.addStretch(1)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setFixedWidth(220)
        self.busy.setVisible(False)
        busy_row.addWidget(self.busy)
        busy_row.addStretch(1)
        center_col.addLayout(busy_row)

        root.addLayout(center_col, 1, 1)

        # Right image (BLANK at start)
        self.lbl_reverse = ImageLabel("No reverse selected")
        root.addWidget(self.lbl_reverse, 1, 2)

        # File pickers row
        left_controls = QHBoxLayout()
        self.btn_load_obv = QPushButton("Load Obverse Of Coin")
        self.btn_load_obv.clicked.connect(self.on_obverse_click)
        self.txt_obv = QLineEdit(self)
        self.txt_obv.setPlaceholderText("Obverse filepath")
        self.txt_obv.setReadOnly(True)
        left_controls.addWidget(self.btn_load_obv, 2)
        left_controls.addWidget(self.txt_obv, 3)
        root.addLayout(left_controls, 2, 0)

        root.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum), 2, 1)

        right_controls = QHBoxLayout()
        self.btn_load_rev = QPushButton("Load Reverse Of Coin")
        self.btn_load_rev.clicked.connect(self.on_reverse_click)
        self.txt_rev = QLineEdit(self)
        self.txt_rev.setPlaceholderText("Reverse filepath")
        self.txt_rev.setReadOnly(True)
        right_controls.addWidget(self.btn_load_rev, 2)
        right_controls.addWidget(self.txt_rev, 3)
        root.addLayout(right_controls, 2, 2)

        # Bottom bar: Reset (left) + Grade (center) + Open Report (right)
        bottom_bar = QHBoxLayout()
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.on_reset_click)
        bottom_bar.addWidget(self.btn_reset, 0, Qt.AlignLeft)

        bottom_bar.addStretch(1)

        self.btn_grade = QPushButton("Grade")
        self.btn_grade.setMinimumWidth(240)
        self.btn_grade.clicked.connect(self.on_grade_click)
        self.btn_grade.setEnabled(False)
        bottom_bar.addWidget(self.btn_grade, 0, Qt.AlignCenter)

        bottom_bar.addStretch(1)

        self.btn_open_report = QPushButton("Open Report")
        self.btn_open_report.setEnabled(False)
        self.btn_open_report.clicked.connect(self.on_open_report_click)
        bottom_bar.addWidget(self.btn_open_report, 0, Qt.AlignRight)

        root.addLayout(bottom_bar, 3, 0, 1, 3)

        # Grow
        root.setRowStretch(1, 1)
        root.setColumnStretch(0, 1)
        root.setColumnStretch(1, 1)
        root.setColumnStretch(2, 1)

        self._apply_theme()
        self._update_status_panel()
        self.show()

    # Restore persisted UI settings (theme, size, position).
    def _restore_settings(self):
        self.dark_mode = bool(self.settings.value("dark_mode", False, type=bool))
        self._apply_theme()
        size = self.settings.value("win_size")
        pos = self.settings.value("win_pos")
        try:
            if isinstance(size, QSize) and size.isValid():
                self.resize(size)
            if isinstance(pos, QPoint):
                self.move(pos)
        except Exception:
            pass

    # Persist UI settings on close.
    def _save_settings(self):
        self.settings.setValue("dark_mode", self.dark_mode)
        self.settings.setValue("win_size", self.size())
        self.settings.setValue("win_pos", self.pos())

    # True if both obverse and reverse images are selected.
    def _both_selected(self) -> bool:
        return bool(self.obverse_path) and bool(self.reverse_path)

    # Compose the status/instructions text based on current selection state.
    def _status_text(self) -> str:
        if not self.obverse_path and not self.reverse_path:
            return (
                "Next steps:\n\n"
                "• Select obverse image (left).\n"
                "• Select reverse image (right).\n"
                "• Then click Grade."
            )
        if not self.obverse_path:
            return (
                "Next steps:\n\n"
                "• Select obverse image (left).\n"
                "• Then select reverse image and click Grade."
            )
        if not self.reverse_path:
            return (
                "Next steps:\n\n"
                "• Select reverse image (right).\n"
                "• Then click Grade."
            )
        return (
            "Both images selected.\n\n"
            "Click Grade to compute: \n"
            "• FFT-based wear estimate\n"
            "• Feature-Match grade"
        )

    # Refresh the right-hand summary panel and enable/disable buttons.
    def _update_status_panel(self):
        self.result_box.setText(self._status_text())
        self.btn_grade.setEnabled(self._both_selected() and not self.busy.isVisible())
        self.btn_open_report.setEnabled(bool(self.last_report_path and os.path.exists(self.last_report_path)))

    # Toggle dark mode and re-apply styles.
    def _toggle_dark_mode(self, state: int):
        self.dark_mode = (state == Qt.Checked)
        self._apply_theme()

    # ---- Public API for integrators ----

    # Return current user selections (obverse, reverse).
    def get_selected_images(self) -> tuple:
        """Return currently selected (obverse_path, reverse_path)."""
        return self.obverse_path, self.reverse_path

    # Store a generated report path and optionally open the viewer.
    def set_report_path(self, pdf_path: str, auto_open: bool = False):
        """
        External grading code calls this after exporting the PDF.
        - Stores path
        - Enables 'Open Report'
        - Optionally opens the viewer (default OFF)
        - Emits reportOpened(pdf_path) when opened via this UI
        """
        if not pdf_path or not os.path.exists(pdf_path):
            QMessageBox.warning(self, "Report", "Provided report path does not exist.")
            return
        self.last_report_path = pdf_path
        self.result_box.append(f"\nReport ready: {self.last_report_path}")
        self._update_status_panel()
        if auto_open:
            self._show_report(self.last_report_path)

    # Open the given report path, or the last stored one.
    def open_report(self, pdf_path: str = ""):
        """Helper for external code to explicitly open a report path or the stored one."""
        path = pdf_path or self.last_report_path
        if not path or not os.path.exists(path):
            QMessageBox.information(self, "Report", "No report available.")
            return
        self._show_report(path)

    # ---------- file dialogs ----------

    # Best-effort start directory for a file dialog (obverse or reverse).
    def _start_dir(self, which: str) -> str:
        key = "last_dir_obv" if which == "obv" else "last_dir_rev"
        start = self.settings.value(key, "", type=str)
        if start and Path(start).exists():
            return start
        return ""

    # After a file is chosen, remember its directory for the next dialog.
    def _store_dir(self, which: str, filepath: str):
        try:
            p = Path(filepath).resolve().parent
            key = "last_dir_obv" if which == "obv" else "last_dir_rev"
            self.settings.setValue(key, str(p))
        except Exception:
            pass

    # Prompt the user to select an obverse image; update UI state.
    @pyqtSlot()
    def on_obverse_click(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Obverse Image", self._start_dir("obv"),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        if not path:
            return
        self.obverse_path = path
        self.txt_obv.setText(path)
        self.lbl_obverse.set_image_path(path)
        self._store_dir("obv", path)
        self._update_status_panel()

    # Prompt the user to select a reverse image; update UI state.
    @pyqtSlot()
    def on_reverse_click(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reverse Image", self._start_dir("rev"),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        if not path:
            return
        self.reverse_path = path
        self.txt_rev.setText(path)
        self.lbl_reverse.set_image_path(path)
        self._store_dir("rev", path)
        self._update_status_panel()

    # Clear selections and results; return UI to initial (blank) state.
    @pyqtSlot()
    def on_reset_click(self):
        self.obverse_path = ""
        self.reverse_path = ""
        self.txt_obv.clear()
        self.txt_rev.clear()
        self.lbl_obverse.clear_image()  # back to placeholder (blank)
        self.lbl_reverse.clear_image()  # back to placeholder (blank)
        self.fft_results = "----"
        self.fm_results = "----"
        self.busy.setVisible(False)
        self.last_report_path = ""
        self._update_status_panel()

    # Open the most recent report, if available.
    @pyqtSlot()
    def on_open_report_click(self):
        if not (self.last_report_path and os.path.exists(self.last_report_path)):
            QMessageBox.information(self, "Report", "No report available.")
            return
        self._show_report(self.last_report_path)

    # Create and execute the modal ReportViewer dialog for the given PDF path.
    def _show_report(self, pdf_path: str):
        dlg = ReportViewer(self, dark=self.dark_mode)
        dlg.load_pdf(pdf_path)
        dlg.exec_()
        self.reportOpened.emit(pdf_path)

    # Spawn a background worker to grade the selected images.
    @pyqtSlot()
    def on_grade_click(self):
        pipe.runPre(self.obverse_path, self.reverse_path)
        self._on_grade_finished()

    # Receive results from the background worker and update the UI.
    @pyqtSlot()
    def _on_grade_finished(self):
        self.set_report_path(ABS_TEST_PDF, auto_open=False)
        self.open_report()

    # Display a worker error and return UI to idle state.
    @pyqtSlot(str)
    def _on_grade_failed(self, message: str):
        self.result_box.setText(f"An error occurred during grading:\n{message}")
        self.busy.setVisible(False)
        self._update_status_panel()

    # Persist settings when the window is closed.
    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


# ---------------- Demo/Test Helpers ----------------

# Ensure the parent directory of 'path' exists.
def _ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# Create a minimal, valid PDF for demo/testing when the real test.pdf is absent.
def create_demo_pdf(path: str):
    """Create a tiny, valid PDF using Qt (fallback if the requested test.pdf is missing)."""
    _ensure_dir(path)
    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(path)

    doc = QTextDocument()
    doc.setHtml(
        "<h1>Coin Report (Demo)</h1>"
        "<p>This is a fallback demo PDF generated by the test harness.</p>"
        "<ul><li>Shows how the review UI opens</li>"
        "<li>Zoom and Fit Width work</li>"
        "<li>Use 'Save As…' to export</li></ul>"
    )
    doc.print_(printer)
    return path


# Launch the app (callable from external scripts).
def runLWCCode():
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

def runGUI():
    # Standalone entry point: create the app and optional demo wiring.
    app = QApplication(sys.argv)
    ui = App()

    sys.exit(app.exec_())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = App()

    sys.exit(app.exec_())