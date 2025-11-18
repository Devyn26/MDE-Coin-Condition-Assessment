import cv2

def rotate_image(image, mask):
    # Load template and input image
    template = cv2.imread(mask, 0)  # grayscale template
    #image = cv2.imread(coin_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize template to match the input image if needed
    template = cv2.resize(template, (gray.shape[1], gray.shape[0]))

    best_score = -1
    best_angle = 0

    # Try multiple angles
    for angle in range(0, 360, 2):
        M = cv2.getRotationMatrix2D((gray.shape[1] // 2, gray.shape[0] // 2), angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
    
        res = cv2.matchTemplate(rotated, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
    
        if score > best_score:
            best_score = score
            best_angle = angle

    # Rotate image using best angle
    final_rotation = cv2.getRotationMatrix2D((gray.shape[1] // 2, gray.shape[0] // 2), best_angle, 1.0)
    rotated_final = cv2.warpAffine(image, final_rotation, (image.shape[1], image.shape[0]))
    # Save image to disk
    cv2.imwrite('LincolnCent/rotated_upright_coin.jpg', rotated_final)

    return rotated_final

if __name__ == '__main__':
    rotate_image('LincolnCent/coin_test.jpg', 'LincolnCent/upright_mask.jpg')
