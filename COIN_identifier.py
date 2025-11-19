import cv2
import numpy as np
import os


def identify_coin_type(coin_image, face_type):
    """
    Test both LWC and MSD masks and return the one with better correlation.
    """
    print(f"Testing both LWC and MSD masks for {face_type}...")
    
    # test LWC mask
    print("Testing LWC mask...")
    from LWC_rotation import find_optimal_rotation, load_lwc_reference_mask
    lwc_mask = load_lwc_reference_mask(face_type)
    lwc_score = 0.0
    lwc_angle = 0.0
    lwc_scores = []
    lwc_angles = []
    if lwc_mask is not None:
        lwc_angle, lwc_score, lwc_scores, lwc_angles = find_optimal_rotation(coin_image, lwc_mask, face_type, angle_range=360, step=1.0)
        print(f"LWC {face_type}: angle={lwc_angle:.1f} deg, correlation={lwc_score:.4f}")
    else:
        print("LWC mask not available")
    
    # test MSD mask
    print("Testing MSD mask...")
    from MSD_rotation import find_optimal_rotation, load_msd_reference_mask
    msd_mask = load_msd_reference_mask(face_type)
    msd_score = 0.0
    msd_angle = 0.0
    msd_scores = []
    msd_angles = []
    if msd_mask is not None:
        msd_angle, msd_score, msd_scores, msd_angles = find_optimal_rotation(coin_image, msd_mask, face_type, angle_range=360, step=1.0)
        print(f"MSD {face_type}: angle={msd_angle:.1f} deg, correlation={msd_score:.4f}")
    else:
        print("MSD mask not available")
    
    # Raw score (for display only) and curvature (steepness) for identification
    print("Raw correlation scores:")
    print(f"LWC: {lwc_score:.4f} (angle: {lwc_angle:.1f} deg)")
    print(f"MSD: {msd_score:.4f} (angle: {msd_angle:.1f} deg)")

    # Steepness metric near best_angle: avg |slope| * (mean - min); pick stronger side
    def compute_steepness_score(angles_list, scores_list, best_angle, delta=5.0):
        import numpy as _np
        if angles_list is None or scores_list is None or len(angles_list) < 2:
            return 0.0, 0.0, 0.0
        paired = sorted(zip(_np.asarray(angles_list, float), _np.asarray(scores_list, float)), key=lambda t: t[0])
        ang = _np.array([a for a, _ in paired], dtype=float)
        sc = _np.array([b for _, b in paired], dtype=float)
        ang_ext = _np.concatenate([ang - 360.0, ang, ang + 360.0])
        sc_ext = _np.concatenate([sc, sc, sc])

        left_mask = (ang_ext >= best_angle - float(delta)) & (ang_ext <= best_angle)
        right_mask = (ang_ext >= best_angle) & (ang_ext <= best_angle + float(delta))

        def side_score(window_angles, window_scores):
            if window_angles.size < 2:
                return 0.0, 0.0, 0.0
            order = _np.argsort(window_angles)
            wa = window_angles[order]
            ws = window_scores[order]
            diffs = _np.diff(ws) / _np.maximum(_np.diff(wa), 1e-9)
            mean_abs_slope = float(_np.mean(_np.abs(diffs))) if diffs.size else 0.0
            window_mean = float(_np.mean(ws))
            window_min = float(_np.min(ws))
            depth = window_mean - window_min
            return float(mean_abs_slope * depth), mean_abs_slope, depth

        left_score, left_slope, left_depth = side_score(ang_ext[left_mask], sc_ext[left_mask])
        right_score, right_slope, right_depth = side_score(ang_ext[right_mask], sc_ext[right_mask])

        return (right_score, right_slope, right_depth) if right_score >= left_score else (left_score, left_slope, left_depth)

    lwc_score_metric, lwc_slope, lwc_depth = compute_steepness_score(lwc_angles, lwc_scores, lwc_angle)
    msd_score_metric, msd_slope, msd_depth = compute_steepness_score(msd_angles, msd_scores, msd_angle)
    print(f"Steepness score (slope*depth) in +/- 5 deg: LWC={lwc_score_metric:.6f} (slope={lwc_slope:.6f}, depth={lwc_depth:.6f}), "
          f"MSD={msd_score_metric:.6f} (slope={msd_slope:.6f}, depth={msd_depth:.6f})")

    # Decide by steepness score (higher wins)
    if lwc_score_metric > msd_score_metric + 1e-12:
        coin_type = "LWC"
        best_score = lwc_score
        best_angle = lwc_angle
        print("Identification by steepness score: LWC wins")
    elif msd_score_metric > lwc_score_metric + 1e-12:
        coin_type = "MSD"
        best_score = msd_score
        best_angle = msd_angle
        print("Identification by steepness score: MSD wins")
    else:
        # perfect tie: fall back to higher raw correlation score
        if lwc_score > msd_score:
            coin_type = "LWC"; best_score = lwc_score; best_angle = lwc_angle
            print("Tie on steepness score; raw score tie-breaker -> LWC")
        else:
            coin_type = "MSD"; best_score = msd_score; best_angle = msd_angle
            print("Tie on steepness score; raw score tie-breaker -> MSD")
    
    # Create visualization only if not suppressed via env var
    '''
    if os.environ.get('COIN_ID_VIS', '1') == '1':
        create_identification_visualization(coin_image, face_type, lwc_score, msd_score, 
                                         lwc_angle, msd_angle, lwc_score_metric, msd_score_metric,
                                         coin_type, 0, 0, 0, 0,
                                         lwc_scores, msd_scores, lwc_angles, msd_angles)
        '''
    
    return coin_type, best_score, best_angle


def identify_coin_type_metrics(coin_image, face_type):
    """
    Same as identify_coin_type, but also returns per-mask steepness scores used
    for the decision (lwc_score_metric, msd_score_metric). No visualization.
    Returns: (coin_type, best_score, best_angle, lwc_score_metric, msd_score_metric)
    """
    print(f"Testing both LWC and MSD masks for {face_type}...")

    # LWC
    from LWC_rotation import find_optimal_rotation as _find_lwc, load_lwc_reference_mask as _load_lwc
    lwc_mask = _load_lwc(face_type)
    lwc_score = 0.0; lwc_angle = 0.0; lwc_scores = []; lwc_angles = []
    if lwc_mask is not None:
        lwc_angle, lwc_score, lwc_scores, lwc_angles = _find_lwc(coin_image, lwc_mask, face_type, angle_range=360, step=1.0)
        print(f"LWC {face_type}: angle={lwc_angle:.1f} deg, correlation={lwc_score:.4f}")
    else:
        print("LWC mask not available")

    # MSD
    from MSD_rotation import find_optimal_rotation as _find_msd, load_msd_reference_mask as _load_msd
    msd_mask = _load_msd(face_type)
    msd_score = 0.0; msd_angle = 0.0; msd_scores = []; msd_angles = []
    if msd_mask is not None:
        msd_angle, msd_score, msd_scores, msd_angles = _find_msd(coin_image, msd_mask, face_type, angle_range=360, step=1.0)
        print(f"MSD {face_type}: angle={msd_angle:.1f} deg, correlation={msd_score:.4f}")
    else:
        print("MSD mask not available")

    # compute steepness score
    def _compute_steepness_score(angles_list, scores_list, best_angle, delta=5.0):
        import numpy as _np
        if angles_list is None or scores_list is None or len(angles_list) < 2:
            return 0.0
        arr = sorted(zip(_np.asarray(angles_list, float), _np.asarray(scores_list, float)), key=lambda t: t[0])
        xs = _np.array([a for a, _ in arr], dtype=float)
        ys = _np.array([b for _, b in arr], dtype=float)
        xs_ext = _np.concatenate([xs - 360.0, xs, xs + 360.0])
        ys_ext = _np.concatenate([ys, ys, ys])
        left_mask = (xs_ext >= best_angle - float(delta)) & (xs_ext <= best_angle)
        right_mask = (xs_ext >= best_angle) & (xs_ext <= best_angle + float(delta))

        def _side_score(xw, yw):
            if xw.size < 2:
                return 0.0
            order = _np.argsort(xw)
            xw = xw[order]; yw = yw[order]
            slopes = []
            for i in range(1, len(xw)):
                dx = xw[i] - xw[i - 1]
                if dx == 0:
                    continue
                slopes.append(abs((yw[i] - yw[i - 1]) / dx))
            mean_abs_slope = float(_np.mean(slopes)) if len(slopes) else 0.0
            window_mean = float(_np.mean(yw))
            window_min = float(_np.min(yw))
            depth = window_mean - window_min
            return float(mean_abs_slope * depth)

        left_score = _side_score(xs_ext[left_mask], ys_ext[left_mask])
        right_score = _side_score(xs_ext[right_mask], ys_ext[right_mask])
        return float(max(left_score, right_score))

    lwc_score_metric = _compute_steepness_score(lwc_angles, lwc_scores, lwc_angle)
    msd_score_metric = _compute_steepness_score(msd_angles, msd_scores, msd_angle)

    if lwc_score_metric > msd_score_metric:
        coin_type = "LWC"; best_score = lwc_score; best_angle = lwc_angle
    elif msd_score_metric > lwc_score_metric:
        coin_type = "MSD"; best_score = msd_score; best_angle = msd_angle
    else:
        if lwc_score > msd_score:
            coin_type = "LWC"; best_score = lwc_score; best_angle = lwc_angle
        else:
            coin_type = "MSD"; best_score = msd_score; best_angle = msd_angle

    return coin_type, best_score, best_angle, lwc_score_metric, msd_score_metric


def identify_and_rotate_coin_pair(obverse_image, reverse_image):
    """
    Identify both coins and apply appropriate rotation.
    
    Args:
        obverse_image: 1000x1000 extracted obverse coin
        reverse_image: 1000x1000 extracted reverse coin
    
    Returns:
        tuple: (rotated_obverse, rotated_reverse, obverse_type, reverse_type)
    """
    print("=== COIN IDENTIFICATION AND ROTATION ===")
    
    # identify obverse coin type and get optimal angle
    obverse_type, obverse_score, obverse_angle = identify_coin_type(obverse_image, "obverse")
    
    # identify reverse coin type and get optimal angle
    reverse_type, reverse_score, reverse_angle = identify_coin_type(reverse_image, "reverse")
    
    print(f"\nIdentification Results:")
    print(f"Obverse: {obverse_type} (score: {obverse_score:.4f}, angle: {obverse_angle:.1f} deg)")
    print(f"Reverse: {reverse_type} (score: {reverse_score:.4f}, angle: {reverse_angle:.1f} deg)")
    
    # apply rotation using the angles already found during identification
    if obverse_type == "LWC":
        print(f"\nApplying LWC rotation to obverse: {obverse_angle:.1f} deg")
        from LWC_rotation import apply_rotation_normalization
        if abs(obverse_angle) > 0.5:
            rotated_obverse = apply_rotation_normalization(obverse_image, obverse_angle)
        else:
            rotated_obverse = obverse_image
    elif obverse_type == "MSD":
        print(f"\nApplying MSD rotation to obverse: {obverse_angle:.1f} deg")
        from MSD_rotation import apply_rotation_normalization
        if abs(obverse_angle) > 0.5:
            rotated_obverse = apply_rotation_normalization(obverse_image, obverse_angle)
        else:
            rotated_obverse = obverse_image
    else:
        print(f"Unknown obverse type: {obverse_type}, using original")
        rotated_obverse = obverse_image
    
    if reverse_type == "LWC":
        print(f"Applying LWC rotation to reverse: {reverse_angle:.1f} deg")
        from LWC_rotation import apply_rotation_normalization
        if abs(reverse_angle) > 0.5:
            rotated_reverse = apply_rotation_normalization(reverse_image, reverse_angle)
        else:
            rotated_reverse = reverse_image
    elif reverse_type == "MSD":
        print(f"Applying MSD rotation to reverse: {reverse_angle:.1f} deg")
        from MSD_rotation import apply_rotation_normalization
        if abs(reverse_angle) > 0.5:
            rotated_reverse = apply_rotation_normalization(reverse_image, reverse_angle)
        else:
            rotated_reverse = reverse_image
    else:
        print(f"Unknown reverse type: {reverse_type}, using original")
        rotated_reverse = reverse_image
    
    return rotated_obverse, rotated_reverse, obverse_type, reverse_type


def identify_coin_type_from_pair(obverse_image, reverse_image):
    """
    Identify coin type from both obverse and reverse images.
    Only returns a coin type if both faces agree.
    
    Args:
        obverse_image: The processed obverse coin image (1000x1000)
        reverse_image: The processed reverse coin image (1000x1000)
    
    Returns:
        str: "LWC", "MSD", or "UNKNOWN" (if faces don't agree)
    """
    print("=== COIN TYPE IDENTIFICATION ===")
    
    # identify obverse coin type
    obverse_type, obverse_score, obverse_angle = identify_coin_type(obverse_image, "obverse")
    
    # identify reverse coin type  
    reverse_type, reverse_score, reverse_angle = identify_coin_type(reverse_image, "reverse")
    
    print(f"\nIdentification Results:")
    print(f"Obverse: {obverse_type} (score: {obverse_score:.4f}, angle: {obverse_angle:.1f} deg)")
    print(f"Reverse: {reverse_type} (score: {reverse_score:.4f}, angle: {reverse_angle:.1f} deg)")
    
    # check if both faces agree on coin type
    if obverse_type == reverse_type:
        print(f"\nBoth faces agree: {obverse_type}")
        return obverse_type
    else:
        print(f"\nFaces disagree: obverse={obverse_type}, reverse={reverse_type}")
        return "UNKNOWN"


# AI written visualization method, for demos, will be removed later
def create_identification_visualization(coin_image, face_type, lwc_score, msd_score, 
                                     lwc_angle, msd_angle, lwc_curv, msd_curv,
                                     coin_type, lwc_mean, lwc_std, msd_mean, msd_std,
                                     lwc_scores, msd_scores, lwc_angles, msd_angles):

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Coin Identification - {face_type.title()}  |  Winner: {coin_type}', fontsize=16, fontweight='bold')

    # original image
    coin_rgb = cv2.cvtColor(coin_image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(coin_rgb)
    axes[0].set_title(f'Original {face_type.title()} (1000x1000)')
    axes[0].axis('off')

    # sort series helper
    def _sort_series(angles_list, scores_list):
        try:
            arr = sorted(zip(angles_list, scores_list), key=lambda x: x[0])
            if len(arr) == 0:
                return [], []
            a, s = zip(*arr)
            return list(a), list(s)
        except Exception:
            return angles_list, scores_list

    lwc_a, lwc_s = _sort_series(lwc_angles, lwc_scores)
    msd_a, msd_s = _sort_series(msd_angles, msd_scores)

    # rotation curves and windows
    axes[1].plot(lwc_a, lwc_s, 'b-', linewidth=2, label='LWC Scores')
    axes[1].plot(msd_a, msd_s, 'r-', linewidth=2, label='MSD Scores')
    axes[1].axvline(lwc_angle, color='blue', linestyle='--', linewidth=1.5, label=f'LWC best {lwc_angle:.1f} deg')
    axes[1].axvline(msd_angle, color='red', linestyle='--', linewidth=1.5, label=f'MSD best {msd_angle:.1f} deg')
    # highlight +/- 5 deg regions
    axes[1].axvspan(lwc_angle - 5.0, lwc_angle + 5.0, color='blue', alpha=0.12)
    axes[1].axvspan(msd_angle - 5.0, msd_angle + 5.0, color='red', alpha=0.12)
    axes[1].set_title('Correlation Score vs. Rotation Angle')
    axes[1].set_xlabel('Rotation Angle (degrees)')
    axes[1].set_ylabel('Correlation Score')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    if len(lwc_a) > 0 or len(msd_a) > 0:
        try:
            xmin = min([min(lwc_a) if lwc_a else 0, min(msd_a) if msd_a else 0])
            xmax = max([max(lwc_a) if lwc_a else 0, max(msd_a) if msd_a else 0])
            pad_x = 0.05 * (xmax - xmin if xmax != xmin else 1.0)
            axes[1].set_xlim(xmin - pad_x, xmax + pad_x)
        except ValueError:
            pass
    axes[1].set_ylim(-1.0, 1.0)

    text_str = (
        f"Curvature magnitude mean (+/- 5 deg):\n"
        f"  LWC: {lwc_curv:.6f}\n"
        f"  MSD: {msd_curv:.6f}\n\n"
        f"Winner: {coin_type}"
    )
    axes[1].text(0.02, 0.98, text_str, transform=axes[1].transAxes, va='top', ha='left',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

    # Print surrounding slope magnitudes to console only
    def _print_slope_magnitudes(angles_list, scores_list, label, best_angle, delta=5.0):
        arr = sorted(zip(np.asarray(angles_list, float), np.asarray(scores_list, float)), key=lambda t: t[0])
        if len(arr) < 2:
            print(f"{label}: insufficient points for slope magnitudes")
            return
        xs = np.array([a for a, _ in arr], dtype=float)
        ys = np.array([b for _, b in arr], dtype=float)
        xs_ext = np.concatenate([xs - 360.0, xs, xs + 360.0])
        ys_ext = np.concatenate([ys, ys, ys])
        window = (xs_ext >= best_angle - float(delta)) & (xs_ext <= best_angle + float(delta))
        xs_w, ys_w = xs_ext[window], ys_ext[window]
        mags = []
        for i in range(1, len(xs_w)):
            dx = xs_w[i] - xs_w[i - 1]
            if dx == 0:
                continue
            mags.append(abs((ys_w[i] - ys_w[i - 1]) / dx))
        if len(mags) == 0:
            print(f"{label}: no slope magnitudes in window")
        else:
            print(f"{label} slope magnitude (mean +/- std over window): {np.mean(mags):.6f} +/- {np.std(mags):.6f}")

    _print_slope_magnitudes(lwc_angles, lwc_scores, 'LWC', lwc_angle)
    _print_slope_magnitudes(msd_angles, msd_scores, 'MSD', msd_angle)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # load extracted coins from COIN_preprocessing
    obverse_path = "COIN_Proc_ob.jpg"
    reverse_path = "COIN_Proc_rev.jpg"
    
    if not os.path.exists(obverse_path) or not os.path.exists(reverse_path):
        print("Error: COIN_Proc_ob.jpg and COIN_Proc_rev.jpg not found.")
        print("Please run COIN_preprocessing.py first to extract the coins.")
        exit(1)
    
    # load extracted coins
    obverse_image = cv2.imread(obverse_path)
    reverse_image = cv2.imread(reverse_path)
    
    if obverse_image is None or reverse_image is None:
        print("Error: Could not load extracted coin images.")
        exit(1)
    
    # identify and rotate
    rotated_obverse, rotated_reverse, obverse_type, reverse_type = identify_and_rotate_coin_pair(
        obverse_image, reverse_image)
    
    # save final results
    cv2.imwrite('COIN_Final_ob.jpg', rotated_obverse)
    cv2.imwrite('COIN_Final_rev.jpg', rotated_reverse)
    
    print(f"\nFinal Results:")
    print(f"Obverse: {obverse_type} -> saved to COIN_Final_ob.jpg")
    print(f"Reverse: {reverse_type} -> saved to COIN_Final_rev.jpg")