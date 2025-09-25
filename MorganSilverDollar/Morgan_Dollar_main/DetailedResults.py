'''
Updated for F25-06 coin assessment team
Updated by: Luke Graham
Date: 9/18/2025

Notes:
- This version focuses on CONDITION / MASK PAGES.
- Brightness, Toning, and Colors sections are intentionally commented out for now.
- Mask pages now show OVERLAYS (original image + transparent color where mask = bad region).
'''

from fpdf import FPDF
import cv2
import os
from PIL import Image
import numpy as np
import tempfile

# A4 orientation Standard (mm)
PDF_WIDTH = 210.0
PDF_HEIGHT = 297.0

# Text formatting helper (kept to match your style; consider margins instead of tabs later)
STUPID_INDENT = "\t\t\t\t\t\t\t"


class PDF(FPDF):
    def __init__(self, coin_name: str = "Morgan Silver Dollar"):
        super().__init__()
        # --- Metadata ---
        self.coin_name = coin_name
        self.set_title(f"{self.coin_name} - Detailed Results")
        self.set_author("F25-06 Coin Assessment Team")

        # --- Inputs (images) ---
        # Originals (PIL Image or np.ndarray acceptable)
        self.ogObverse = None
        self.ogReverse = None

        # Condition edge/score (np.ndarray or PIL Image)
        self.conditionObverse = None
        self.conditionReverse = None

        # Flat regions (binary mask images)
        self.flatObverse = None
        self.flatReverse = None

        # Condition masks dict: (binary masks, 255 where TRUE)
        self.condMasks = {
            "highSigObverse": None,
            "highSigReverse": None,
            "lowSigObverse": None,
            "lowSigReverse": None,
            "rimObverse": None,
            "rimReverse": None
        }

        # --- Scores ---
        self.conditionScore = None

        # --- (Commented out: not used right now) ---
        # self.toningObverse = None
        # self.toningReverse = None
        # self.obToningCovImgDR = None
        # self.reToningCovImgDR = None
        # self.brillianceObverse = None
        # self.brillianceReverse = None
        # self.brillianceScore = None
        # self.toningScore = None
        # self.histBrillliance = None

        # Track temp files for cleanup
        self._tmp_files = []

        # --- Common layout values ---
        self._img_w = 80.0
        self._img_h = 80.0
        self._col1_x = 20.0
        self._col2_x = 110.0

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _to_pil(self, img):
        """Normalize an image-like object to a PIL.Image or return None."""
        if img is None:
            return None
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, np.ndarray):
            # Assume array is RGB/Gray already (caller can convert BGR->RGB earlier)
            try:
                return Image.fromarray(img)
            except Exception:
                return None
        if isinstance(img, str):
            if os.path.exists(img):
                try:
                    return Image.open(img)
                except Exception:
                    return None
        return None

    def _save_for_fpdf(self, pil_img: Image.Image) -> str:
        """Save PIL image to a temp PNG and return path for FPDF.image()."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpf:
            pil_img.save(tmpf.name, "PNG")
            path = tmpf.name
        self._tmp_files.append(path)
        return path

    def _frame_and_image(self, x, y, w, h, img_like, alt_txt="N/A"):
        """Draw border and place image if available; otherwise label N/A."""
        self.rect(x, y, w, h, 'D')
        pil = self._to_pil(img_like)
        if pil is None:
            self.set_xy(x, y + h / 2 - 3)
            self.set_font('Arial', 'I', 10)
            self.cell(w=w, h=6, align='C', txt=alt_txt, border=0)
            return
        path = self._save_for_fpdf(pil)
        self.image(path, x=x, y=y, w=w, h=h)

    def _overlay_mask(self, base_img, mask_img, color=(255, 0, 0), alpha=0.35):
        """
        Create overlay: original image with semi-transparent color painted
        where mask_img > 0. Returns PIL.Image (RGB).
        - base_img: PIL or ndarray (RGB)
        - mask_img: PIL or ndarray (binary or grayscale; >0 is "bad")
        - color: RGB tuple for highlight
        - alpha: transparency 0..1
        """
        base = self._to_pil(base_img)
        mask = self._to_pil(mask_img)
        if base is None or mask is None:
            return None

        # Ensure same size
        if base.size != mask.size:
            mask = mask.resize(base.size, Image.NEAREST)

        base_rgb = base.convert("RGB")
        mask_gray = mask.convert("L")  # use luminance; >0 = bad

        # Create a solid color image and composite where mask is on
        color_layer = Image.new("RGB", base_rgb.size, color)
        # Convert mask to an alpha mask scaled by 'alpha'
        # Normalize: any nonzero pixel -> alpha*255
        m = np.array(mask_gray, dtype=np.uint8)
        m = (m > 0).astype(np.uint8) * int(255 * alpha)
        alpha_mask = Image.fromarray(m, mode="L")

        # Composite: base overlaid with color_layer at masked regions
        overlay = Image.composite(color_layer, base_rgb, alpha_mask)  # color where mask>0 else base
        # Now blend to keep base visible; we already used partial alpha via mask,
        # but a small blend can soften the result:
        blended = Image.blend(base_rgb, overlay, alpha=alpha)
        return blended

    def _caption(self, center_x, y, text, width=80.0, size=10):
        self.set_xy(center_x - width / 2, y)
        self.set_font('Arial', 'I', size)
        self.cell(w=width, h=6, align='C', txt=text, border=0)

    # ----------------------------------------------------------------------
    # Page 1 - Detailed Results + Condition
    # ----------------------------------------------------------------------
    def genTextPageOne(self):
        # Title
        self.set_xy(0.0, 5.0)
        self.set_font('Arial', 'B', 18)
        self.cell(w=PDF_WIDTH, h=20.0, align='C', txt=f"{self.coin_name} - Detailed Results", border=0)

        # Sub-Title: Condition
        self.set_xy(20.0, 125.0)
        self.set_font('Arial', 'B', 14)
        self.cell(w=PDF_WIDTH, h=10.0, align='L', txt="Condition Analysis", border=0)

        # Score
        coin_grade = "Condition Score: " + (str(self.conditionScore) if self.conditionScore is not None else "N/A") + "/70.0"
        self.set_xy(20.0, 140.0)
        self.set_font('Arial', '', 12)
        self.multi_cell(w=PDF_WIDTH - 40.0, h=5.0, align='L', txt=coin_grade, border=0)

        # Description
        desc = (STUPID_INDENT +
                "The condition of a coin is evaluated by the frequency of white pixels (Intensity: 255)\n"
                "found within an edge filtered image. This evaluation is applied multiple times with\n"
                "different specialized coin masks.")
        self.set_xy(20.0, 150.0)
        self.set_font('Arial', '', 11)
        self.multi_cell(w=PDF_WIDTH - 40.0, h=5.0, align='L', txt=desc, border=0)

    def genImagesPageOne(self):
        # Originals (top row)
        self._frame_and_image(self._col1_x, 25.0, self._img_w, self._img_h, self.ogObverse, alt_txt="No Obverse")
        self._caption(self._col1_x + self._img_w / 2, 25.0 + self._img_h + 5.0, f"{self.coin_name} - Obverse")

        self._frame_and_image(self._col2_x, 25.0, self._img_w, self._img_h, self.ogReverse, alt_txt="No Reverse")
        self._caption(self._col2_x + self._img_w / 2, 25.0 + self._img_h + 5.0, f"{self.coin_name} - Reverse")

        # Condition edges (bottom row) – raw edge images (no overlay by design)
        self._frame_and_image(self._col1_x, 170.0, self._img_w, self._img_h, self.conditionObverse, alt_txt="No cond/obv")
        self._caption(self._col1_x + self._img_w / 2, 170.0 + self._img_h + 5.0, "Condition (Obverse)")

        self._frame_and_image(self._col2_x, 170.0, self._img_w, self._img_h, self.conditionReverse, alt_txt="No cond/rev")
        self._caption(self._col2_x + self._img_w / 2, 170.0 + self._img_h + 5.0, "Condition (Reverse)")

    # ----------------------------------------------------------------------
    # Page 2 - Flat & High Significance (OVERLAYS)
    # ----------------------------------------------------------------------
    def genTextPageTwo(self):
        # Flat Regions
        self.set_xy(20.0, 15.0)
        self.set_font('Arial', 'B', 14)
        self.cell(w=PDF_WIDTH, h=10.0, align='L', txt="Flat Regions (Overlays)", border=0)

        # High Significance
        self.set_xy(20.0, 135.0)
        self.set_font('Arial', 'B', 14)
        self.cell(w=PDF_WIDTH, h=10.0, align='L', txt="High Significance (Overlays)", border=0)

    def genImagesPageTwo(self):
        # FLAT overlays (blue)
        flat_obv_overlay = self._overlay_mask(self.ogObverse, self.flatObverse, color=(0, 128, 255), alpha=0.40)
        flat_rev_overlay = self._overlay_mask(self.ogReverse, self.flatReverse, color=(0, 128, 255), alpha=0.40)

        self._frame_and_image(self._col1_x, 30.0, self._img_w, self._img_h, flat_obv_overlay, alt_txt="No flat/obv")
        self._caption(self._col1_x + self._img_w / 2, 30.0 + self._img_h + 5.0, "Flat Regions (Obverse)")

        self._frame_and_image(self._col2_x, 30.0, self._img_w, self._img_h, flat_rev_overlay, alt_txt="No flat/rev")
        self._caption(self._col2_x + self._img_w / 2, 30.0 + self._img_h + 5.0, "Flat Regions (Reverse)")

        # HIGH SIGNIFICANCE overlays (red/orange)
        high_obv_overlay = self._overlay_mask(self.ogObverse, self.condMasks["highSigObverse"], color=(255, 64, 0), alpha=0.40)
        high_rev_overlay = self._overlay_mask(self.ogReverse, self.condMasks["highSigReverse"], color=(255, 64, 0), alpha=0.40)

        self._frame_and_image(self._col1_x, 150.0, self._img_w, self._img_h, high_obv_overlay, alt_txt="No high-sig/obv")
        self._caption(self._col1_x + self._img_w / 2, 150.0 + self._img_h + 5.0, "High Significance (Obverse)")

        self._frame_and_image(self._col2_x, 150.0, self._img_w, self._img_h, high_rev_overlay, alt_txt="No high-sig/rev")
        self._caption(self._col2_x + self._img_w / 2, 150.0 + self._img_h + 5.0, "High Significance (Reverse)")

    # ----------------------------------------------------------------------
    # Page 3 - Low Significance & Rim (OVERLAYS)
    # ----------------------------------------------------------------------
    def genTextPageThree(self):
        # Low Significance
        self.set_xy(20.0, 15.0)
        self.set_font('Arial', 'B', 14)
        self.cell(w=PDF_WIDTH, h=10.0, align='L', txt="Low Significance (Overlays)", border=0)

        # Rim
        self.set_xy(20.0, 135.0)
        self.set_font('Arial', 'B', 14)
        self.cell(w=PDF_WIDTH, h=10.0, align='L', txt="Rim (Overlays)", border=0)

    def genImagesPageThree(self):
        # LOW SIGNIFICANCE overlays (yellow)
        low_obv_overlay = self._overlay_mask(self.ogObverse, self.condMasks["lowSigObverse"], color=(255, 200, 0), alpha=0.40)
        low_rev_overlay = self._overlay_mask(self.ogReverse, self.condMasks["lowSigReverse"], color=(255, 200, 0), alpha=0.40)

        self._frame_and_image(self._col1_x, 30.0, self._img_w, self._img_h, low_obv_overlay, alt_txt="No low-sig/obv")
        self._caption(self._col1_x + self._img_w / 2, 30.0 + self._img_h + 5.0, "Low Significance (Obverse)")

        self._frame_and_image(self._col2_x, 30.0, self._img_w, self._img_h, low_rev_overlay, alt_txt="No low-sig/rev")
        self._caption(self._col2_x + self._img_w / 2, 30.0 + self._img_h + 5.0, "Low Significance (Reverse)")

        # RIM overlays (green)
        rim_obv_overlay = self._overlay_mask(self.ogObverse, self.condMasks["rimObverse"], color=(0, 200, 0), alpha=0.40)
        rim_rev_overlay = self._overlay_mask(self.ogReverse, self.condMasks["rimReverse"], color=(0, 200, 0), alpha=0.40)

        self._frame_and_image(self._col1_x, 150.0, self._img_w, self._img_h, rim_obv_overlay, alt_txt="No rim/obv")
        self._caption(self._col1_x + self._img_w / 2, 150.0 + self._img_h + 5.0, "Rim (Obverse)")

        self._frame_and_image(self._col2_x, 150.0, self._img_w, self._img_h, rim_rev_overlay, alt_txt="No rim/rev")
        self._caption(self._col2_x + self._img_w / 2, 150.0 + self._img_h + 5.0, "Rim (Reverse)")

    # ----------------------------------------------------------------------
    # Page 4/5 (COMMENTED OUT UNTIL NEEDED)
    # ----------------------------------------------------------------------
    # def genTextPageFour(self):
    #     # Brightness & Histogram (commented out for now)
    #     pass
    #
    # def genImagesPageFour(self):
    #     # Placeholder for brightness/toning coverage images if revived later
    #     pass
    #
    # def genTextPageFive(self):
    #     # Colors (commented out for now)
    #     pass

    # ----------------------------------------------------------------------
    # Build & Cleanup
    # ----------------------------------------------------------------------
    def build_report(self, output_path: str = 'MorganSilverDollar/Morgan_Dollar_main/test.pdf'):
        # Page 1
        self.add_page()
        self.genTextPageOne()
        self.genImagesPageOne()

        # Page 2 (Flat & High Sig overlays)
        self.add_page()
        self.genTextPageTwo()
        self.genImagesPageTwo()

        # Page 3 (Low Sig & Rim overlays)
        self.add_page()
        self.genTextPageThree()
        self.genImagesPageThree()

        # (Commented out future pages)
        # self.add_page()
        # self.genTextPageFour()
        # self.genImagesPageFour()
        #
        # self.add_page()
        # self.genTextPageFive()

        # Output
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        self.output(output_path)

        # Cleanup temp files
        for p in self._tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass
        self._tmp_files.clear()


def generateTemplate(pdf: PDF):
    """
    Kept for compatibility with your existing call site.
    Builds the same 3-page (condition/masks) report and writes default output.
    """
    pdf.build_report('MorganSilverDollar/Morgan_Dollar_main/test.pdf')


if __name__ == "__main__":
    # --- Demo wiring (replace with your pipeline data) ---
    testPDF = PDF(coin_name="Morgan Silver Dollar")

    # Load originals (convert BGR->RGB for PIL)
    oPath = os.path.abspath('ScrapedImages/obverse') + '\\'
    rPath = os.path.abspath('ScrapedImages/reverse') + '\\'

    oImg_bgr = cv2.imread(oPath + "Morgan 1881-S NGC MS65 2363354 obverse.jpg")
    rImg_bgr = cv2.imread(rPath + "Morgan 1881-S NGC MS65 2363354 reverse.jpg")

    if oImg_bgr is None or rImg_bgr is None:
        raise FileNotFoundError("Check the ScrapedImages paths and filenames.")

    oImg_rgb = cv2.cvtColor(oImg_bgr, cv2.COLOR_BGR2RGB)
    rImg_rgb = cv2.cvtColor(rImg_bgr, cv2.COLOR_BGR2RGB)

    testPDF.ogObverse = Image.fromarray(oImg_rgb)
    testPDF.ogReverse = Image.fromarray(rImg_rgb)

    # Optional: supply condition edges if you have them (uint8 arrays)
    # testPDF.conditionObverse = <np.uint8 array 0..255>
    # testPDF.conditionReverse = <np.uint8 array 0..255>

    # Supply masks (binary 0/255) to demonstrate overlays; placeholders here:
    # For a quick visual test, make fake circular masks the size of the originals:
    H, W, _ = oImg_rgb.shape
    yy, xx = np.ogrid[:H, :W]
    center = (H // 2, W // 2)
    rad1 = min(H, W) // 4
    rad2 = min(H, W) // 3

    circle1 = ((yy - center[0]) ** 2 + (xx - center[1]) ** 2) <= (rad1 ** 2)
    circle2 = ((yy - center[0]) ** 2 + (xx - center[1]) ** 2) <= (rad2 ** 2)

    # Fake masks for demo (replace with your real masks)
    testPDF.flatObverse = (circle1.astype(np.uint8) * 255)
    testPDF.flatReverse = (circle2.astype(np.uint8) * 255)

    testPDF.condMasks["highSigObverse"] = (np.fliplr(circle1).astype(np.uint8) * 255)
    testPDF.condMasks["highSigReverse"] = (np.flipud(circle2).astype(np.uint8) * 255)

    testPDF.condMasks["lowSigObverse"] = (np.logical_xor(circle1, circle2).astype(np.uint8) * 255)
    testPDF.condMasks["lowSigReverse"] = (np.logical_and(circle1, ~circle2).astype(np.uint8) * 255)

    testPDF.condMasks["rimObverse"] = (((yy - center[0]) ** 2 + (xx - center[1]) ** 2) >= (rad2 ** 2)).astype(np.uint8) * 255
    testPDF.condMasks["rimReverse"] = np.zeros((H, W), dtype=np.uint8)  # empty rim for demo

    # Optional: condition score
    testPDF.conditionScore = 65.0

    # Build the 3-page report focusing on masks
    generateTemplate(testPDF)
