import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def crop_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h > w:
        offset = (h - w) // 2
        return img[offset:offset + w, :]
    elif w > h:
        offset = (w - h) // 2
        return img[:, offset:offset + h]
    return img

def detect_coin(img: np.ndarray,
                         min_rad: float = 0.6,
                         max_rad: float = 0.9,
                         center_deviation: float = 0.03,
                         dp: float = 1.2,
                         min_dist: float = 50,
                         param1: float = 100,
                         param2: float = 30) -> tuple:
    sq = crop_to_square(img)
    h, w = sq.shape[:2]
    cx, cy = w // 2, h // 2

    min_r = int((min_rad * w) / 2)
    max_r = int((max_rad * w) / 2)

    gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_r,
        maxRadius=max_r
    )

    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            if abs(x - cx) <= center_deviation * w and abs(y - cy) <= center_deviation * h:
                return (x, y, r)

    return None

def draw_coin_detection(img: np.ndarray,
                             detection: tuple,
                             min_rad: float = 0.6,
                             max_rad: float = 0.9) -> None:
    sq = crop_to_square(img)
    h, w = sq.shape[:2]
    cx, cy = w // 2, h // 2
    min_r = (min_rad * w) / 2
    max_r = (max_rad * w) / 2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
    ax.set_title("Anomaly Detection")
    ax.axis("off")

    # Draw min and max radius circles
    ax.add_patch(Circle((cx, cy), min_r, edgecolor='blue', fill=False, linewidth=2, label='Min Radius'))
    ax.add_patch(Circle((cx, cy), max_r, edgecolor='green', fill=False, linewidth=2, label='Max Radius'))

    # Draw detected coin
    if detection:
        x, y, r = detection
        ax.add_patch(Circle((x, y), r, edgecolor='red', fill=False, linewidth=2, label='Detected Coin'))
        ax.plot(x, y, 'ro', markersize=5)
        # Get average brightness in the coin.
        brightness = get_coin_brightness(img, x, y, r)
        print(f"Average brightness inside coin: {brightness:.2f}")

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def get_coin_brightness(img: np.ndarray, x: int, y: int, r: int) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create circular mask
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, thickness=-1)

    # Compute mean brightness inside the mask
    mean_val = cv2.mean(gray, mask=mask)[0]
    return mean_val

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print("Error loading image.")
        sys.exit(1)

    detection = detect_coin(img)
    if detection:
        print(f"Coin detected at (x={detection[0]}, y={detection[1]}) with radius={detection[2]}")
    else:
        print("No coin detected.")

    draw_coin_detection(img, detection)
