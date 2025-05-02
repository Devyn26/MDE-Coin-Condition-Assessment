"""
Prepare_coin_image.py

Preprocesses coin image by formatting an inputted coin image into the style of a database coin image.
Resizes image to be 1000x1000 then uses hough transform to detect circle of coin and use that detection to center and resize coin in image.
Removes background by changing everything outside detected circle to white.

Author: Eric Morley
Date: 5/2/2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def trim_to_square_and_resize(image):
    """
    Trims an image to a square shape by removing equal amounts from the top and bottom,
    and then resizes it to 1000x1000 pixels.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The processed square and resized image.
    """
    height, width = image.shape[:2]

    if height == width:
        trimmed_image = image
    elif height > width:
        diff = height - width
        trim_amount = diff // 2
        trimmed_image = image[trim_amount:height - trim_amount, :]
    else:  # width > height
        diff = width - height
        trim_amount = diff // 2
        trimmed_image = image[:, trim_amount:width - trim_amount]

    resized_image = cv2.resize(trimmed_image, (1000, 1000))
    return resized_image

def process(image_path, hough_param2=50, hough_minRadius=370, hough_maxRadius=380):
    """
    Processes the image to be the database format.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Image formatted in database style.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    processed_image = trim_to_square_and_resize(image)
    output = processed_image.copy()
    mask = np.zeros(processed_image.shape[:2], dtype=np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.3, minDist=100,
                                param1=80, param2=hough_param2, minRadius=hough_minRadius, maxRadius=hough_maxRadius)

    # Data for Lincoln_test_rev
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.3, minDist=100,
    #                            param1=80, param2=55, minRadius=370, maxRadius=380)

    # Data for MSD_rev
    #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.3, minDist=100,
    #                           param1=80, param2=50, minRadius=310, maxRadius=330)

    debug_image = processed_image.copy()
    output = processed_image.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # The radius of the coin images we're matching

        # Get the largest circle by radius
        largest_circle = max(circles, key=lambda c: c[2])
        x, y, r = largest_circle

        # Draw the detected circle and center
        cv2.circle(debug_image, (x, y), r, (0, 0, 255), 3)   # Red outline
        cv2.circle(debug_image, (x, y), 2, (0, 255, 0), 3)   # Green center dot

        radius_x, radius_y = 950, 950
        target_radius = 475
        diameter = 2 * r
        image_h, image_w = processed_image.shape[:2]

        # Crop square region around the circle
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(image_w, x + r)
        y2 = min(image_h, y + r)
        coin_crop = processed_image[y1:y2, x1:x2]

        # Resize the cropped coin region to have radius desired (i.e.,radius 465 : size 930×930)
        scale = (2 * target_radius) / (x2 - x1)
        coin_resized = cv2.resize(coin_crop, (radius_x, radius_y), interpolation=cv2.INTER_LINEAR)

        # Center it in a 1000×1000 white canvas
        canvas = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        top_left_x = (1000 - radius_x) // 2
        top_left_y = (1000 - radius_y) // 2
        canvas[top_left_y:top_left_y + radius_y, top_left_x:top_left_x + radius_x] = coin_resized

        output = canvas

        # Mask out everything outside the radius of the circle
        mask = np.zeros((1000, 1000), dtype=np.uint8)
        cv2.circle(mask, (500, 500), target_radius, 255, -1)

        # Set pixels outside the circle to white
        output[mask == 0] = [255, 255, 255]

        print(f"Largest circle: center=({x}, {y}), radius={r}")
    else:
        print("No circles were detected.")

    debug_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
    plt.imshow(debug_rgb)
    plt.title("Detected Largest Circle")
    plt.axis("off")
    plt.show()

    return output

if __name__ == '__main__':
    image_file_path = "/test_images/MSDob.jpg"
    cv2.imwrite('MSD_Proc_ob.jpg', process(image_file_path))
    #image_file_path = "/test_images/NeW_MSD_REV.jpg"
    #cv2.imwrite('MSD_Proc_rev.jpg', process(image_file_path))

    #image_file_path = "/test_images/MSDob.jpg"
    #cv2.imwrite('MSD_Proc_ob.jpg', process(image_file_path))
    #image_file_path = "/test_images/NeW_MSD_REV.jpg"
    #cv2.imwrite('MSD_Proc_rev.jpg', process(image_file_path))
    