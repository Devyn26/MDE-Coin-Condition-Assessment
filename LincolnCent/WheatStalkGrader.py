'''
Updated for F25-06 coin assessment team
Updated by: Eric Morley
Date: 4/25/2025
'''

import numpy as np
import cv2
from scipy.signal import find_peaks, peak_prominences
#from ImageAdjuster import houghCenters, coinFlattener

def houghCenters(img):
    gray = cv2.medianBlur(img, 5)
    
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30)
    circles = np.reshape(circles, [circles.shape[1], circles.shape[2]])

    return circles

def coinFlattener(img):
    circles = houghCenters(img)

    flatendImg = cv2.linearPolar(img, [circles[0][0], circles[0][1]], circles[0][2], 0)

    return flatendImg


#220 coin dataset ran at 6:53pm 20251111_185251
#20251111_183153.csv
def gradeWheatStalkPenny(image_name):

    import time
    import cv2
    import numpy as np
    from scipy.signal import find_peaks

    t0_all = time.time()

    # cubic mapping 
    trend = [0.00014162340303767857, -0.05219235501283228,
             6.332637857746728, -187.17387181557189]


    # need help gates
    L_NEED_HELP_MIN_SCORE   = 80.0
    L_NEED_HELP_MAX_PEAKS   = 3
    L_NEED_HELP_RATIO_LOW   = 0.55

    R_NEED_HELP_MIN_SCORE   = 30.0
    R_NEED_HELP_MAX_PEAKS   = 2
    R_NEED_HELP_RATIO_HIGH  = 3.0

    # adoption guard
    L_ADOPT_DELTA_ABS       = 25.0
    L_ADOPT_DELTA_REL       = 0.12

    # asymmetry bounds
    LR_RATIO_BASE_SOFT_MAX  = 3.0
    LR_RATIO_MAX_AFTER      = 3.5

    # search offsets
    Y_FULL  = np.array([-0.03,-0.02,-0.01,0.0,0.01,0.02,0.03])
    X_FULL  = np.array([-0.01,0.0,0.01])
    Y_MICRO = np.array([-0.01,0.0,0.01])
    X_MICRO = np.array([0.0])


    img = image_name
    if img is None or img.size == 0:
        raise ValueError(f"Could not read image: {image_name}")

    t_flat = time.time()
    flat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = coinFlattener(flat) 
    if flat is None or flat.size == 0:
        raise ValueError("coinFlattener returned empty result")
    if flat.dtype != np.uint8:
        flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    H, W = flat.shape[:2]
    print(f"[DEBUG] [Grade] Start gradeWheatStalkPenny | shape(flat)=({H}, {W}) | flatten: {time.time()-t_flat:.3f}s")

  
    YBANDS = [(0.473, 0.527), (0.845, 0.891)] 
    XWIN   = (0.655, 0.773)

    MIN_H = max(8, H // 200)
    MIN_W = max(16, W // 100)

    def _crop_fraction(img_u8, yrange, xrange):
        y0 = int(round(yrange[0] * H)); y1 = int(round(yrange[1] * H))
        x0 = int(round(xrange[0] * W)); x1 = int(round(xrange[1] * W))
        y0 = max(0, min(y0, H - 1)); y1 = max(0, min(y1, H))
        x0 = max(0, min(x0, W - 1)); x1 = max(0, min(x1, W))
        if (y1 - y0) < MIN_H:
            pad = (MIN_H - (y1 - y0)) // 2 + 1
            y0 = max(0, y0 - pad); y1 = min(H, y1 + pad)
        if (x1 - x0) < MIN_W:
            pad = (MIN_W - (x1 - x0)) // 2 + 1
            x0 = max(0, x0 - pad); x1 = min(W, x1 + pad)
        if y1 <= y0 or x1 <= x0:
            cy, cx = H // 2, W // 2
            y0, y1 = max(0, cy - MIN_H//2), min(H, cy + MIN_H//2)
            x0, x1 = max(0, cx - MIN_W//2), min(W, cx + MIN_W//2)
        return img_u8[y0:y1, x0:x1], (y0, y1, x0, x1)


    def _fft_band_and_peaks(repaired, peak_prom=4.5, peak_dist=3.0):
        f = np.fft.fft(repaired, axis=-1)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-9)
        avg = mag.sum(axis=0) / max(1, mag.shape[0])
        band = np.array(avg[40:90], dtype=float)
        if band.size < 3:
            return band, np.array([]), avg
        peaks, _ = find_peaks(
            band,
            height=(float(np.max(avg)) - 110.0),
            prominence=peak_prom,
            distance=peak_dist
        )
        return band, peaks, avg

    def _score_from_band(band, peaks, single_peak_weight=0.5):
        if band.size < 3:
            return 0.0
        overall_min = float(np.min(band))
        band_max    = float(np.max(band))

        if peaks.size <= 0:
            return 0.0
        if peaks.size == 1:
            p = int(peaks[0])
            val = float(band[p])
            if val >= band_max:
                return 0.0
            return single_peak_weight * max(0.0, val - overall_min)

        s = 0.0
        for p in peaks:
            val = float(band[p])
            if val >= band_max:  
                continue
            s += (val - overall_min)
        return float(s)

    def _score_fixed200(crop_u8, allow_single_peak=False, peak_prom=4.5, peak_dist=3.0):
        if crop_u8.dtype != np.uint8:
            crop_u8 = cv2.normalize(crop_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(crop_u8, 200, 255, cv2.THRESH_BINARY)
        cov = mask.mean() / 255.0 if crop_u8.size else 0.0
        repaired = cv2.inpaint(crop_u8, mask.astype(np.uint8), 1.0, cv2.INPAINT_TELEA)
        band, peaks, _ = _fft_band_and_peaks(repaired, peak_prom, peak_dist)
        s = _score_from_band(band, peaks, single_peak_weight=0.5 if allow_single_peak else 0.0)
        return s, band, peaks, cov, 200

    def _score_adaptive(crop_u8, allow_single_peak=False, peak_prom=4.5, peak_dist=3.0):
        if crop_u8.dtype != np.uint8:
            crop_u8 = cv2.normalize(crop_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        p97 = float(np.percentile(crop_u8, 97.0)) if crop_u8.size else 255.0
        thr = int(max(180.0, min(245.0, p97)))
        _, mask = cv2.threshold(crop_u8, thr, 255, cv2.THRESH_BINARY)
        cov = mask.mean() / 255.0
        if cov > 0.35: 
            thr2 = min(245, thr + 10)
            _, mask = cv2.threshold(crop_u8, thr2, 255, cv2.THRESH_BINARY)
            if (mask.mean() / 255.0) > 0.50:
                mask[:] = 0; thr = -1
        repaired = cv2.inpaint(crop_u8, mask.astype(np.uint8), 1.0, cv2.INPAINT_TELEA)
        band, peaks, _ = _fft_band_and_peaks(repaired, peak_prom, peak_dist)
        s = _score_from_band(band, peaks, single_peak_weight=0.5 if allow_single_peak else 0.0)
        return s, band, peaks, cov, thr

    cropL, (y0L, y1L, x0L, x1L) = _crop_fraction(flat, YBANDS[0], XWIN)
    cropR, (y0R, y1R, x0R, x1R) = _crop_fraction(flat, YBANDS[1], XWIN)

    sL_base, bandL_base, peaksL_base, covL_base, thrL_base = _score_fixed200(cropL)
    sR_base, bandR_base, peaksR_base, covR_base, thrR_base = _score_adaptive(cropR)

    L_peaks_base = int(peaksL_base.size)
    R_peaks_base = int(peaksR_base.size)

    print(f"[DEBUG] [LEFT]  base window={(y0L,y1L,x0L,x1L)} | crop={cropL.shape} | base score={sL_base:.3f} | band_len={len(bandL_base)} | peaks={L_peaks_base} | cov={covL_base:.4f}")
    print(f"[DEBUG] [RIGHT] base window={(y0R,y1R,x0R,x1R)} | crop={cropR.shape} | base score={sR_base:.3f} | band_len={len(bandR_base)} | peaks={R_peaks_base} | cov={covR_base:.4f} | thr={thrR_base}")

    left_needs_help = (
        (sL_base < L_NEED_HELP_MIN_SCORE) or
        (L_peaks_base <= L_NEED_HELP_MAX_PEAKS) or
        (sR_base > 0 and (sL_base / max(1e-9, sR_base)) < L_NEED_HELP_RATIO_LOW)
    )
    right_needs_help = (
        (sR_base < R_NEED_HELP_MIN_SCORE) or
        (R_peaks_base <= R_NEED_HELP_MAX_PEAKS) or
        (sR_base == 0.0) or
        (sR_base > 0 and (sL_base / sR_base) > R_NEED_HELP_RATIO_HIGH)
    )

    y_off_L = (Y_FULL if left_needs_help else Y_MICRO)
    x_off_L = (X_FULL if left_needs_help else X_MICRO)
    y_off_R = Y_FULL
    x_off_R = X_FULL

    ratio_base = (sL_base / sR_base) if sR_base > 0 else float('inf')
    print(f"[DEBUG] [LEFT]  search={'full' if left_needs_help else 'micro'} | y_offsets={[int(v*H) for v in y_off_L]} | x_offsets={[int(v*W) for v in x_off_L]}")
    print(f"[DEBUG] [RIGHT] need_help={right_needs_help} | y_offsets={[int(v*H) for v in y_off_R]} | x_offsets={[int(v*W) for v in x_off_R]} | L/R={ratio_base if np.isfinite(ratio_base) else 'inf'}")

    def _search_best(y0, y1, x0, x1, y_frac, x_frac, scorer, tag, **sc_kwargs):
        tried, best, best_dbg = 0, None, None
        for dy in (y_frac * H).astype(int):
            yy0 = max(0, min(H - 1, y0 + dy)); yy1 = max(0, min(H, y1 + dy))
            if yy1 <= yy0: continue
            for dx in (x_frac * W).astype(int):
                xx0 = max(0, min(W - 1, x0 + dx)); xx1 = max(0, min(W, x1 + dx))
                if xx1 <= xx0: continue
                tried += 1
                crop = flat[yy0:yy1, xx0:xx1]
                if crop.shape[0] < 5 or crop.shape[1] < 5: continue
                s, band, peaks, cov, thr = scorer(crop, **sc_kwargs)
                if (best is None) or (s > best):
                    best = s
                    best_dbg = dict(score=s, band_len=len(band), peaks=int(peaks.size),
                                    cov=cov, thr=thr, mode=f"{tag}_raw",
                                    y0=yy0, y1=yy1, x0=xx0, x1=xx1,
                                    crop_size=crop.shape, base=False)
        return tried, best, best_dbg


    left_best_score = sL_base
    left_dbg = dict(score=sL_base, band_len=len(bandL_base), peaks=L_peaks_base, cov=covL_base,
                    thr=200, mode="fixed200_raw", y0=y0L, y1=y1L, x0=x0L, x1=x1L,
                    crop_size=cropL.shape, base=True)

    triedL, sL_try, dbgL_try = _search_best(y0L, y1L, x0L, x1L, y_off_L, x_off_L,
                                            _score_fixed200, "fixed200")
    print(f"[DEBUG] [LEFT]  tried={triedL} | candidate={dbgL_try}")

    if sL_try is not None:
        adopt_abs = (sL_try - sL_base) >= L_ADOPT_DELTA_ABS
        adopt_rel = (sL_try - sL_base) / max(1e-9, sL_base) >= L_ADOPT_DELTA_REL if sL_base > 0 else True
        proj_ratio = (sL_try / sR_base) if sR_base > 0 else float('inf')
        asym_guard = not (np.isfinite(proj_ratio) and (ratio_base <= LR_RATIO_BASE_SOFT_MAX) and (proj_ratio > LR_RATIO_MAX_AFTER))
        if adopt_abs and adopt_rel and asym_guard:
            left_best_score = float(sL_try)
            left_dbg = dbgL_try
        else:
            print(f"[DEBUG] [LEFT]  adopt? abs={adopt_abs} rel={adopt_rel} asym_guard={asym_guard} -> kept BASE")


    right_best_score = sR_base
    right_dbg = dict(score=sR_base, band_len=len(bandR_base), peaks=R_peaks_base, cov=covR_base,
                     thr=thrR_base, mode="adaptive_raw", y0=y0R, y1=y1R, x0=x0R, x1=x1R,
                     crop_size=cropR.shape, base=True)
    if right_needs_help:
        triedR, sR_try, dbgR_try = _search_best(y0R, y1R, x0R, x1R, y_off_R, x_off_R,
                                                _score_adaptive, "adaptive", allow_single_peak=False)
        print(f"[DEBUG] [RIGHT] tried={triedR} | candidate={dbgR_try}")
        if (sR_try is not None) and (sR_try > right_best_score):
            right_best_score = float(sR_try)
            right_dbg = dbgR_try
    else:
        print(f"[DEBUG] [RIGHT] kept BASE (no help needed)")

    feature_left  = float(left_best_score)
    feature_right = float(right_best_score)

    print(f"[DEBUG] [LEFT]  chosen={left_dbg}")
    print(f"[DEBUG] [RIGHT] chosen={right_dbg}")
    print("Left Wheat Stalk Rating: "  + str(round(feature_left,  3)))
    print("Right Wheat Stalk Rating: " + str(round(feature_right, 3)))

    # soft balance 
    def _soft_balance(L, R):
        ratio = (L / R) if R > 0 else float('inf')
        if np.isfinite(ratio) and ratio > 3.0:
            target = 2.5 * R
            newL = max(0.0, min(L, (0.6*L + 0.4*target)))  # ≤ ~40% pull
            print(f"[DEBUG] [MAP] imbalance={ratio:.2f} -> soft-balance L {L:.1f}->{newL:.1f} | x={(newL+R)/2:.1f}")
            return newL, R
        elif np.isfinite(ratio) and ratio < (1/3.0):
            target = 2.5 * L
            newR = max(0.0, min(R, (0.6*R + 0.4*target)))
            print(f"[DEBUG] [MAP] imbalance={ratio:.2f} -> soft-balance R {R:.1f}->{newR:.1f} | x={(L+newR)/2:.1f}")
            return L, newR
        elif not np.isfinite(ratio) and L > 0 and R == 0:
            print(f"[DEBUG] [MAP] imbalance=inf -> soft-balance L {L:.1f}->{0.0} | x=0.0")
            return 0.0, R
        return L, R

    # base map
    L0, R0 = _soft_balance(feature_left, feature_right)
    x0 = (L0 + R0) / 2.0
    raw0 = x0*x0*x0*trend[0] + x0*x0*trend[1] + x0*trend[2] + trend[3]


    def _cov_damp(cov, lo=0.01, hi=0.06, min_fac=0.40):
        if cov <= lo: return min_fac
        if cov >= hi: return 1.0
        t = (cov - lo) / (hi - lo)
        return min_fac + t*(1.0 - min_fac)

    def _soft_knee(s, A=120.0):
        return A * (1.0 - np.exp(-s / A))

    def _delta_abs_needed(base):
        return float(min(25.0, max(5.0, 0.30*base + 5.0)))

    def _right_salvage_floor(crop_u8):
        gx = cv2.Sobel(crop_u8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(crop_u8, cv2.CV_32F, 0, 1, ksize=3)
        g = np.median(np.sqrt(gx*gx + gy*gy))
        f = np.fft.fft(np.float32(crop_u8), axis=-1)
        fshift = np.fft.fftshift(f)
        mag = np.log(np.abs(fshift) + 1e-9).sum(axis=0)
        band = mag[40:90]
        bb = float(np.mean(band)) if band.size else 0.0
        floor = max(0.0, min(60.0, 8.0*g + 0.6*bb - 10.0))
        return floor

    def _estimate_val_saturation(crop_u8):
        rgb = cv2.cvtColor(crop_u8, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        return float(np.median(hsv[...,2])), float(np.median(hsv[...,1]))

    mitigated_L, mitigated_R = L0, R0
    raw_under, raw_over = None, None

    # -------- UNDERFLOW (raw0 < 55) --------
    if raw0 < 55.0:
        print(f"[MITIGATE-U] raw {raw0:.1f} < 55 → underflow mitigation")

        allow_singleL = (left_dbg.get("peaks", 2) <= 1) or (mitigated_L == 0.0)
        allow_singleR = (right_dbg.get("peaks", 2) <= 1) or (mitigated_R == 0.0)

        if allow_singleL:
            sL_u1, _, _, _, _ = _score_fixed200(
                flat[left_dbg["y0"]:left_dbg["y1"], left_dbg["x0"]:left_dbg["x1"]],
                allow_single_peak=True, peak_prom=4.0, peak_dist=2.0
            )
            need = _delta_abs_needed(mitigated_L)
            if (sL_u1 - mitigated_L) >= need:
                print(f"[MITIGATE-U][L] single-peak adopt {mitigated_L:.1f}→{sL_u1:.1f} (Δneed={need:.1f})")
                mitigated_L = float(sL_u1)

        if allow_singleR:
            sR_u1, _, _, _, _ = _score_adaptive(
                flat[right_dbg["y0"]:right_dbg["y1"], right_dbg["x0"]:right_dbg["x1"]],
                allow_single_peak=True, peak_prom=4.0, peak_dist=2.0
            )
            needR = _delta_abs_needed(mitigated_R)
            if (sR_u1 - mitigated_R) >= needR:
                print(f"[MITIGATE-U][R] single-peak adopt {mitigated_R:.1f}→{sR_u1:.1f} (Δneed={needR:.1f})")
                mitigated_R = float(sR_u1)

        if mitigated_R <= 0.0:
            floor = _right_salvage_floor(cropR.astype(np.uint8))
            if floor > mitigated_R:
                print(f"[MITIGATE-U][R] salvage floor {mitigated_R:.1f}→{floor:.1f}")
                mitigated_R = float(floor)

        ratio_now = (mitigated_L / mitigated_R) if mitigated_R > 0 else float('inf')
        if (not np.isfinite(ratio_now)) or (ratio_now > 8.0):
            yy = np.array([-0.05,-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.04,0.05])
            xx = np.array([-0.02,-0.01,0.0,0.01,0.02])
            triedRX, sR_tryX, dbgRX = _search_best(y0R, y1R, x0R, x1R, yy, xx, _score_adaptive, "adaptive_broad",
                                                   allow_single_peak=True, peak_prom=3.5, peak_dist=2.0)
            triedRF, sR_tryF, dbgRF = _search_best(y0R, y1R, x0R, x1R, yy, xx, _score_fixed200, "fixed200_RIGHT",
                                                   allow_single_peak=True, peak_prom=3.8, peak_dist=2.0)
            cand = [(sR_tryX, dbgRX), (sR_tryF, dbgRF)]
            best = max([c for c in cand if c[0] is not None], key=lambda t: t[0], default=(None, None))
            if best[0] is not None and best[0] > mitigated_R:
                print(f"[MITIGATE-U][R] broaden adopt {mitigated_R:.1f}→{best[0]:.1f} | dbg={best[1]}")
                mitigated_R = float(best[0])

        V_med, S_med = _estimate_val_saturation(cropR.astype(np.uint8))
        if (right_dbg.get("thr", 200) == -1) or (V_med >= 220 and S_med <= 70):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cropR_clahe = clahe.apply(cropR.astype(np.uint8))
            sR_g, _, _, _, _ = _score_adaptive(cropR_clahe, allow_single_peak=True, peak_prom=3.8, peak_dist=2.0)
            if sR_g > mitigated_R:
                print(f"[MITIGATE-U][R] glare rescue {mitigated_R:.1f}→{sR_g:.1f} (V≈{V_med:.0f},S≈{S_med:.0f})")
                mitigated_R = float(sR_g)

        L1, R1 = _soft_balance(mitigated_L, mitigated_R)
        x_floor = 0.35 * max(L1, R1) if max(L1, R1) > 0 else 0.0
        x1 = max((L1 + R1) / 2.0, x_floor)
        raw_under = x1*x1*x1*trend[0] + x1*x1*trend[1] + x1*trend[2] + trend[3]
        print(f"[MITIGATE-U] recompute: L={L1:.1f} R={R1:.1f} | x={x1:.1f} | raw={raw_under:.1f}")


    elif raw0 > 70.0:
        print(f"[MITIGATE-O] raw {raw0:.1f} > 70 → overflow mitigation")


        L_hi = (feature_left  > feature_right + 100.0)
        R_hi = (feature_right > feature_left  + 100.0)
        L_suspect = L_hi and ((left_dbg.get("cov",0.0) < 0.01) or (left_dbg.get("peaks",0) >= 7))
        R_suspect = R_hi and ((right_dbg.get("cov",0.0) < 0.01) or (right_dbg.get("peaks",0) >= 7))

        def _damp_side(s, cov): return _soft_knee(s * _cov_damp(cov, lo=0.01, hi=0.06, min_fac=0.40), A=120.0)

        L2, R2 = feature_left, feature_right
        if L_suspect and not R_suspect:
            L2 = _damp_side(feature_left, left_dbg.get("cov",0.05))
            print(f"[MITIGATE-O] asymmetric: damp LEFT {feature_left:.1f}→{L2:.1f} (cov={left_dbg.get('cov',0):.3f})")
        elif R_suspect and not L_suspect:
            R2 = _damp_side(feature_right, right_dbg.get("cov",0.05))
            print(f"[MITIGATE-O] asymmetric: damp RIGHT {feature_right:.1f}→{R2:.1f} (cov={right_dbg.get('cov',0):.3f})")
        else:
            L2 = _damp_side(feature_left, left_dbg.get("cov",0.05))
            R2 = _damp_side(feature_right, right_dbg.get("cov",0.05))
            print(f"[MITIGATE-O] cov-damp mild: L {feature_left:.1f}→{L2:.1f}, R {feature_right:.1f}→{R2:.1f}")

        ratio = (L2 / R2) if R2 > 0 else float('inf')
        if np.isfinite(ratio) and ratio > 3.2:
            target = 2.7 * R2
            newL = max(0.0, 0.65*L2 + 0.35*target)
            print(f"[MITIGATE-O] ratio cap: L {L2:.1f}→{newL:.1f} (R={R2:.1f}, ratio={ratio:.2f})")
            L2 = newL
        elif np.isfinite(ratio) and ratio < (1.0/3.2):
            target = 2.7 * L2
            newR = max(0.0, 0.65*R2 + 0.35*target)
            print(f"[MITIGATE-O] ratio cap: R {R2:.1f}→{newR:.1f} (L={L2:.1f}, ratio={ratio:.2f})")
            R2 = newR

        Lb, Rb = _soft_balance(L2, R2)
        x1 = (Lb + Rb) / 2.0
        raw_over = x1*x1*x1*trend[0] + x1*x1*trend[1] + x1*trend[2] + trend[3]
        print(f"[MITIGATE-O] recompute: L={Lb:.1f} R={Rb:.1f} | x={x1:.1f} | raw={raw_over:.1f}")


    a_star = min(70.0, max(55.0, raw0))  

    peaks_sum = int(left_dbg.get("peaks", 0)) + int(right_dbg.get("peaks", 0))
    covL = float(left_dbg.get("cov", 0.0))
    covR = float(right_dbg.get("cov", 0.0))
    ratio_LR = (feature_left / feature_right) if feature_right > 0 else float('inf')
    in_ratio = (1/6 <= ratio_LR <= 6)

    if (56.0 <= raw0 <= 69.0 and
        peaks_sum >= 7 and
        0.005 <= covL <= 0.12 and
        0.005 <= covR <= 0.12 and
        in_ratio):
        sheldon = round(raw0, 1)
        print(f"[TRUST] base trusted: raw0={raw0:.1f} peaks={peaks_sum} covL={covL:.3f} covR={covR:.3f} ratio={ratio_LR:.2f}")
        print(f"Estimated Wheat Stalk Sheldon Scale Grade: {round(sheldon)}")
        print(f"[DEBUG] [Grade] Total time: {time.time()-t0_all:.3f}s | x={(L0+R0)/2.0:.3f} | raw={raw0:.2f}")
        return [float(L0), float(R0), float(sheldon)]


    dampL_cov = _cov_damp(left_dbg.get("cov", 0.05), lo=0.01, hi=0.06, min_fac=0.40)
    dampR_cov = _cov_damp(right_dbg.get("cov", 0.05), lo=0.01, hi=0.06, min_fac=0.40)
    L_cov = _soft_knee(L0 * dampL_cov, A=120.0)
    R_cov = _soft_knee(R0 * dampR_cov, A=120.0)
    Lc, Rc = _soft_balance(L_cov, R_cov)
    xc = (Lc + Rc) / 2.0
    raw_cov = xc*xc*xc*trend[0] + xc*xc*trend[1] + xc*trend[2] + trend[3]


    cands = [("base", float(raw0))]
    if raw_under is not None:
        cands.append(("under", float(raw_under)))
    if raw_over is not None and (raw0 > 70.0) and ((raw0 - a_star) > 3.0):
        cands.append(("over", float(raw_over)))
    else:
        if raw_over is not None:
            print(f"[ARB] overflow candidate gated out (raw0={raw0:.1f}, a*={a_star:.1f})")
    cands.append(("cov", float(raw_cov)))



    def _loss(g):
        lb = max(0.0, 55.0 - g)
        ub = max(0.0, g - 70.0)
        return abs(g - a_star) + 2.0*(lb + ub) + 0.2*abs(g - raw0)

    label_best, g_best = min(cands, key=lambda kv: _loss(kv[1]))
    print(f"[ARB] anchor={a_star:.1f} base={raw0:.1f} cand={{" + ", ".join([f"{k}:{v:.1f}" for k,v in cands]) + f"}} -> pick={label_best} {g_best:.1f}")


    def _soft_clip(g):
        if g < 55.0: return 55.0 - (55.0 - g) * 0.25
        if g > 70.0: return 70.0 + (g - 70.0) * 0.25
        return g

    g_final = _soft_clip(g_best)

    if g_final < 55.0: g_final = 55.0
    if g_final > 70.0: g_final = 70.0

    sheldon = round(g_final, 1)
    print(f"Estimated Wheat Stalk Sheldon Scale Grade: {round(sheldon)}")
    print(f"[DEBUG] [Grade] Total time: {time.time()-t0_all:.3f}s | x={(L0+R0)/2.0:.3f} | raw={sheldon:.2f}")
    return [float(L0), float(R0), float(sheldon)]