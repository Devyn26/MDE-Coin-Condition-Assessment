import cv2

# Load an upright coin image
image = cv2.imread('upright_coin.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary threshold to create a mask
_, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Save the mask
cv2.imwrite('upright_mask.jpg', mask)