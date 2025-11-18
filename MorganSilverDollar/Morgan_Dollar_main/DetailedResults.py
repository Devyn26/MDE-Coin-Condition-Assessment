# =============================================================================
# DetailedResults.py  (MorganSilverDollar/Morgan_Dollar_main/)
# -----------------------------------------------------------------------------
# Purpose : Generate a multi-page PDF report with coin images, condition masks,
#           and a concise condition summary (with Confidence) for the coin
#           assessment project.
# Author  : Luke Graham
# Updated : 2025-10-02
# Notes   : Page 1 - originals and condition edges
#           Page 2 - condition summary ONLY (shows "Confidence:")
#           Pages 3-4 - overlays (Flat/High-Sig, then Low-Sig/Rim)
#           Text is ASCII-only to avoid FPDF Latin-1 issues.
# =============================================================================

from fpdf import FPDF
import os
import numpy as np
from PIL import Image
import tempfile
from typing import Iterable, Optional, Tuple

import COIN_identifier

# --- Lazy import of Grader so no other files need changes --------------------
_MDGrader = None
try:
    # Package-relative import when running as part of the module
    from .MorganGrader import Grader as _MDGrader  # type: ignore
except Exception:
    try:
        # Fallback: flat import when running this file outside the package
        from MorganGrader import Grader as _MDGrader  # type: ignore
    except Exception:
        _MDGrader = None  # If unavailable, we skip auto-confidence gracefully

PDF_WIDTH = 210.0   # A4 width (mm)
PDF_HEIGHT = 297.0  # A4 height (mm)


class PDF(FPDF):
    """Minimal PDF generator for current functionality (no brightness/table)."""

    # Initialize report object and default state
    def __init__(self, coin_name: str = "Morgan Silver Dollar"):
        """Set up metadata, placeholders for images/masks/scores, and layout constants."""
        super().__init__()
        self.coin_name = coin_name
        self.set_title(f"{self.coin_name} - Detailed Results")
        self.set_author("F25-06 Coin Assessment Team")

        # Originals (PIL.Image, ndarray RGB, or file path)
        self.ogObverse = None
        self.ogReverse = None

        # Condition edges (optional)
        self.conditionObverse = None
        self.conditionReverse = None

        # Condition-related masks (binary; 0 or >0)
        self.flatObverse = None
        self.flatReverse = None
        self.condMasks = {
            "highSigObverse": None,
            "highSigReverse": None,
            "lowSigObverse": None,
            "lowSigReverse": None,
            "rimObverse": None,
            "rimReverse": None,
        }
        self.LWCMaks = {
            
            }

        # Scores / meta
        self.conditionScore: Optional[float] = None
        self.conditionDistribution: Optional[Iterable[float]] = None
        self.conditionConfidence: Optional[float] = None  # 0..100 (%)

        # Optional Hough circle detections
        self.hough_center_obv: Optional[Tuple[int, int]] = None
        self.hough_radius_obv: Optional[int] = None
        self.hough_center_rev: Optional[Tuple[int, int]] = None
        self.hough_radius_rev: Optional[int] = None

        # Labels and small notes
        self.coin_year: Optional[str] = None
        self.coin_mint: Optional[str] = None
        self.front_label: str = "Obverse"
        self.back_label: str = "Reverse"
        self.obv_note: Optional[str] = None
        self.rev_note: Optional[str] = None

        # Layout helpers
        self._tmp_files = []
        self._img_w = 80.0
        self._img_h = 80.0
        self._col1_x = 20.0
        self._col2_x = 110.0

    # ----------------------------- utilities -----------------------------

    # Convert ndarray/path/PIL to PIL.Image
    def _to_pil(self, img):
        """Normalize ndarray/path/PIL to PIL.Image, returning None if conversion fails."""
        if img is None:
            return None
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, np.ndarray):
            a = img
            if a.dtype != np.uint8:
                a = np.clip(a, 0, 255).astype(np.uint8)
            if a.ndim == 2:
                return Image.fromarray(a, mode="L")
            if a.ndim == 3 and a.shape[2] == 3:
                return Image.fromarray(a, mode="RGB")
            if a.ndim == 3 and a.shape[2] == 4:
                return Image.fromarray(a, mode="RGBA")
            return Image.fromarray(a)
        if isinstance(img, str) and os.path.exists(img):
            try:
                return Image.open(img)
            except Exception:
                return None
        return None

    # Save a PIL image to a temp PNG for FPDF
    def _save_for_fpdf(self, pil_img: Image.Image) -> str:
        """Persist a PIL image to a temporary PNG on disk and return its path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpf:
            pil_img.save(tmpf.name, "PNG")
            p = tmpf.name
        self._tmp_files.append(p)
        return p

    # Draw a framed image or a placeholder if missing
    def _frame_and_image(self, x, y, w, h, img_like, alt_txt="N/A"):
        """Draw a rectangular frame and place an image inside (or centered placeholder text)."""
        self.rect(x, y, w, h, 'D')
        pil = self._to_pil(img_like)
        if pil is None:
            self.set_xy(x, y + h / 2 - 3)
            self.set_font('Arial', 'I', 10)
            self.cell(w=w, h=6, align='C', txt=alt_txt, border=0)
            return
        path = self._save_for_fpdf(pil)
        self.image(path, x=x, y=y, w=w, h=h)

    # Overlay a binary mask onto a dimmed base coin image
    def _overlay_mask(
        self,
        base_img,
        mask_img,
        color=(255, 0, 0),
        alpha=0.40,
        base_dim=0.55,
        base_dim_color=(255, 255, 255),
    ):
        """Return the base coin image dimmed and overlaid with color wherever mask>0."""
        base = self._to_pil(base_img)
        mask = self._to_pil(mask_img)
        if base is None or mask is None:
            return None

        base_rgb = base.convert("RGB")
        if mask.size != base_rgb.size:
            mask = mask.resize(base_rgb.size, Image.NEAREST)

        b = np.array(base_rgb, dtype=np.float32)
        if base_dim > 0:
            tgt = np.array(base_dim_color, dtype=np.float32)
            b = b * (1.0 - base_dim) + tgt * base_dim

        m = np.array(mask.convert("L"), dtype=np.uint8)
        m = (m > 0).astype(np.float32)
        if m.max() == 0:
            return Image.fromarray(np.clip(b, 0, 255).astype(np.uint8), "RGB")

        overlay = np.zeros_like(b) + np.array(color, dtype=np.float32)
        a = (alpha * m)[..., None]
        out = b * (1.0 - a) + overlay * a
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")

    # Write a centered caption under an image slot
    def _caption(self, center_x, y, text, width=80.0, size=10, style='I'):
        """Draw a simple centered one-line caption under an image frame."""
        self.set_xy(center_x - width / 2, y)
        self.set_font('Arial', style, size)
        self.cell(w=width, h=6, align='C', txt=text, border=0)

    # Compute basic empirical percentile against a population
    def _percentile(self, value: Optional[float], population: Optional[Iterable[float]]) -> Optional[float]:
        """Compute a simple empirical percentile (<= value) against a supplied population."""
        if value is None or population is None:
            return None
        try:
            arr = np.asarray(list(population), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return None
            pct = 100.0 * (np.sum(arr <= value) / arr.size)
            return float(np.round(pct, 2))
        except Exception:
            return None

    # Format confidence as an int 0..100 if present
    def _fmt_conf(self) -> Optional[int]:
        """Clamp and round the confidence field to an integer percentage if present."""
        if not isinstance(self.conditionConfidence, (int, float)):
            return None
        return int(round(max(0.0, min(100.0, float(self.conditionConfidence)))))

    # Auto-calc confidence via Grader if not already set
    def _auto_confidence(self):
        """If confidence is missing but we have a condition score, compute it via Grader."""
        if self.conditionConfidence is not None:
            return
        if not isinstance(self.conditionScore, (int, float)):
            return
        if _MDGrader is None:
            return
        try:
            grader = _MDGrader()
            conf_val = grader.CalculateConfidence(self.conditionScore)
            if isinstance(conf_val, (int, float)) and np.isfinite(conf_val):
                self.conditionConfidence = float(max(0.0, min(100.0, conf_val)))
        except Exception:
            # Fail silently; confidence will simply not appear in the report.
            pass

    # ====================== Page 1: images only ==========================

    # Compose Page 1 layout and content
    def genPageOneImagesOnly(self):
        """Render Page 1: original obverse/reverse and condition edge maps."""
        self.set_xy(0.0, 5.0)
        self.set_font('Arial', 'B', 18)
        header = f"{self.coin_name} - Detailed Results"
        extra = " ".join([x for x in [self.coin_year, self.coin_mint] if x])
        if extra:
            header += f" ({extra})"
        self.cell(w=PDF_WIDTH, h=20.0, align='C', txt=header, border=0)

        top_y = 25.0
        if self.ogObverse is not None:
            self._frame_and_image(self._col1_x, top_y, self._img_w, self._img_h, self.ogObverse,
                                  alt_txt=f"No {self.front_label}")
            cap = f"{self.coin_name} - {self.front_label}"
            if self.obv_note:
                cap += f" ({self.obv_note})"
            self._caption(self._col1_x + self._img_w / 2, top_y + self._img_h + 5.0, cap)
        if self.ogReverse is not None:
            self._frame_and_image(self._col2_x, top_y, self._img_w, self._img_h, self.ogReverse,
                                  alt_txt=f"No {self.back_label}")
            cap = f"{self.coin_name} - {self.back_label}"
            if self.rev_note:
                cap += f" ({self.rev_note})"
            self._caption(self._col2_x + self._img_w / 2, top_y + self._img_h + 5.0, cap)

        bottom_y = 130.0
        if self.conditionObverse is not None:
            self._frame_and_image(self._col1_x, bottom_y, self._img_w, self._img_h, self.conditionObverse,
                                  alt_txt=f"No condition/{self.front_label.lower()}")
            self._caption(self._col1_x + self._img_w / 2, bottom_y + self._img_h + 5.0,
                          f"Condition ({self.front_label})")
        if self.conditionReverse is not None:
            self._frame_and_image(self._col2_x, bottom_y, self._img_w, self._img_h, self.conditionReverse,
                                  alt_txt=f"No condition/{self.back_label.lower()}")
            self._caption(self._col2_x + self._img_w / 2, bottom_y + self._img_h + 5.0,
                          f"Condition ({self.back_label})")

    # ================== Page 2: condition ONLY ===========================

    # Compose Page 2 text summary
    def genTextPageTwo_Scores(self):
        """Render Page 2: condition summary line (with Confidence) plus brief details."""
        if(self.coin_name == "Morgan Silver Dollar"):
            # Attempt to fill confidence from MorganGrader if not already provided
            self._auto_confidence()

        self.set_xy(20.0, 15.0)
        self.set_font('Arial', 'B', 16)
        self.cell(w=PDF_WIDTH - 40.0, h=10.0, align='L', txt="Summary: Condition", border=0)

        # Condition line + Confidence + Percentile
        parts = []
        if isinstance(self.conditionScore, (int, float)):
            parts.append(f"Condition: {self.conditionScore:.1f}/70.0")
        else:
            parts.append("Condition: N/A")

        if(self.coin_name == "Morgan Silver Dollar"):
            conf_int = self._fmt_conf()
            if conf_int is not None:
                parts.append(f"Confidence: {conf_int}%")

        cond_pct = self._percentile(self.conditionScore, self.conditionDistribution)
        if cond_pct is not None:
            parts.append(f"Percentile vs coins: {cond_pct}%")

        line = "  |  ".join(parts)

        self.set_xy(20.0, 30.0)
        self.set_font('Arial', '', 12)
        self.multi_cell(w=PDF_WIDTH - 40.0, h=6.0, align='L', txt=line, border=0)

        # Hough readout (if available)
        hough_bits = []
        if self.hough_center_obv and self.hough_radius_obv:
            hough_bits.append(f"{self.front_label}: center={self.hough_center_obv}, radius={self.hough_radius_obv}")
        if self.hough_center_rev and self.hough_radius_rev:
            hough_bits.append(f"{self.back_label}: center={self.hough_center_rev}, radius={self.hough_radius_rev}")
        if hough_bits:
            self.set_xy(20.0, 46.0)
            self.set_font('Arial', '', 10)
            self.multi_cell(w=PDF_WIDTH - 40.0, h=4.8, align='L',
                            txt="Detected coin circle -> " + "  |  ".join(hough_bits), border=0)

        # Description of how to read the rest of the report
        desc = (
            "Condition uses edge density under coin specific masks. "
            "Overlays highlight detected problem regions (flat, high significance, "
            "low significance, rim)."
        )
        self.set_xy(20.0, 60.0)
        self.set_font('Arial', '', 10)
        self.multi_cell(w=PDF_WIDTH - 40.0, h=4.8, align='L', txt=desc, border=0)

    # Reserved hook for Page 2 images (currently unused)
    def genImagesPageTwo_Scores(self):
        """Reserved for Page 2 images (none currently, kept for API compatibility)."""
        return  # no-op

    # ================== Page 3: Flat and High Significance ===============

    # Compose Page 3 titles (no coverage line)
    def genTextPageThree(self):
        """Render Page 3 titles; coverage line intentionally removed per spec."""
        if(self.coin_name == "Morgan Silver Dollar"):
            self.set_xy(20.0, 15.0)
            self.set_font('Arial', 'B', 14)
            self.cell(w=PDF_WIDTH, h=10.0, align='L', txt="Flat Regions (Overlays)", border=0)

        # Coverage text intentionally removed.

            self.set_xy(20.0, 135.0)
            self.set_font('Arial', 'B', 14)
            self.cell(w=PDF_WIDTH, h=10.0, align='L', txt="High Significance (Overlays)", border=0)

        if(self.coin_name == "Lincoln Wheat Cent"):
            self.set_xy(20.0, 15.0)
            self.set_font('Arial', 'B', 14)
            self.cell(w=PDF_WIDTH, h=10.0, align='L', txt="Regions Obverse (Overlays)", border=0)

    # Render all Page 3 overlay images
    def genImagesPageThree(self):
        """Render Page 3 images: flat overlays (top) and high-significance overlays (bottom)."""
        if(self.coin_name == "Morgan Silver Dollar"):
            flat_obv_overlay = self._overlay_mask(self.ogObverse, self.flatObverse,
                                                  color=(0, 128, 255), alpha=0.40, base_dim=0.55)
            flat_rev_overlay = self._overlay_mask(self.ogReverse, self.flatReverse,
                                                  color=(0, 128, 255), alpha=0.40, base_dim=0.55)

            self._frame_and_image(self._col1_x, 30.0, self._img_w, self._img_h, flat_obv_overlay,
                                  alt_txt=f"No flat/{self.front_label.lower()}")
            self._caption(self._col1_x + self._img_w / 2, 30.0 + self._img_h + 5.0, f"Flat Regions ({self.front_label})")

            self._frame_and_image(self._col2_x, 30.0, self._img_w, self._img_h, flat_rev_overlay,
                                  alt_txt=f"No flat/{self.back_label.lower()}")
            self._caption(self._col2_x + self._img_w / 2, 30.0 + self._img_h + 5.0, f"Flat Regions ({self.back_label})")

            high_obv_overlay = self._overlay_mask(self.ogObverse, self.condMasks["highSigObverse"],
                                                  color=(255, 64, 0), alpha=0.40, base_dim=0.55)
            high_rev_overlay = self._overlay_mask(self.ogReverse, self.condMasks["highSigReverse"],
                                                  color=(255, 64, 0), alpha=0.40, base_dim=0.55)

            self._frame_and_image(self._col1_x, 150.0, self._img_w, self._img_h, high_obv_overlay,
                                  alt_txt=f"No high-sig/{self.front_label.lower()}")
            self._caption(self._col1_x + self._img_w / 2, 150.0 + self._img_h + 5.0, f"High Significance ({self.front_label})")

            self._frame_and_image(self._col2_x, 150.0, self._img_w, self._img_h, high_rev_overlay,
                                  alt_txt=f"No high-sig/{self.back_label.lower()}")
            self._caption(self._col2_x + self._img_w / 2, 150.0 + self._img_h + 5.0, f"High Significance ({self.back_label})")

    # ================= Page 4: Low Significance and Rim ===================

    # Compose Page 4 titles
    def genTextPageFour(self):
        """Render Page 4 titles for low-significance and rim overlays."""
        self.set_xy(20.0, 15.0)
        self.set_font('Arial', 'B', 14)
        self.cell(w=PDF_WIDTH, h=10.0, align='L', txt="Low Significance (Overlays)", border=0)

        self.set_xy(20.0, 135.0)
        self.set_font('Arial', 'B', 14)
        self.cell(w=PDF_WIDTH, h=10.0, align='L', txt="Rim (Overlays)", border=0)

    # Render all Page 4 overlay images
    def genImagesPageFour(self):
        """Render Page 4 images: low-significance overlays (top) and rim overlays (bottom)."""
        low_obv_overlay = self._overlay_mask(self.ogObverse, self.condMasks["lowSigObverse"],
                                             color=(255, 200, 0), alpha=0.40, base_dim=0.55)
        low_rev_overlay = self._overlay_mask(self.ogReverse, self.condMasks["lowSigReverse"],
                                             color=(255, 200, 0), alpha=0.40, base_dim=0.55)

        self._frame_and_image(self._col1_x, 30.0, self._img_w, self._img_h, low_obv_overlay,
                              alt_txt=f"No low-sig/{self.front_label.lower()}")
        self._caption(self._col1_x + self._img_w / 2, 30.0 + self._img_h + 5.0, f"Low Significance ({self.front_label})")

        self._frame_and_image(self._col2_x, 30.0, self._img_w, self._img_h, low_rev_overlay,
                              alt_txt=f"No low-sig/{self.back_label.lower()}")
        self._caption(self._col2_x + self._img_w / 2, 30.0 + self._img_h + 5.0, f"Low Significance ({self.back_label})")

        rim_obv_overlay = self._overlay_mask(self.ogObverse, self.condMasks["rimObverse"],
                                             color=(0, 200, 0), alpha=0.40, base_dim=0.55)
        rim_rev_overlay = self._overlay_mask(self.ogReverse, self.condMasks["rimReverse"],
                                             color=(0, 200, 0), alpha=0.40, base_dim=0.55)

        self._frame_and_image(self._col1_x, 150.0, self._img_w, self._img_h, rim_obv_overlay,
                              alt_txt=f"No rim/{self.front_label.lower()}")
        self._caption(self._col1_x + self._img_w / 2, 150.0 + self._img_h + 5.0, f"Rim ({self.front_label})")

        self._frame_and_image(self._col2_x, 150.0, self._img_w, self._img_h, rim_rev_overlay,
                              alt_txt=f"No rim/{self.back_label.lower()}")
        self._caption(self._col2_x + self._img_w / 2, 150.0 + self._img_h + 5.0, f"Rim ({self.back_label})")

    # ========================= Build and Cleanup ==========================

    # Assemble all pages and write the PDF
    def build_report(self, output_path: str = 'MorganSilverDollar/Morgan_Dollar_main/test.pdf'):
        """Assemble all pages and write the PDF to disk, cleaning temp files afterwards."""

        self.add_page()                     # Page 1
        self.genPageOneImagesOnly()

        self.add_page()                     # Page 2
        self.genTextPageTwo_Scores()
        self.genImagesPageTwo_Scores()      # reserved no-op

        self.add_page()                     # Page 3
        self.genTextPageThree()
        self.genImagesPageThree()

        self.add_page()                     # Page 4
        self.genTextPageFour()
        self.genImagesPageFour()

        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        self.output(output_path)

        # cleanup temp files
        for p in self._tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass
        self._tmp_files.clear()


# Simple wrapper to preserve existing call sites
def generateTemplate(pdf: PDF):
    """Compatibility wrapper so existing callers can build the report without changes."""
    pdf.build_report('MorganSilverDollar/Morgan_Dollar_main/test.pdf')
