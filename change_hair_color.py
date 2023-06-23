import numpy as np
def change_hair_color(image, mask, color, alpha):
    masking_area = np.where(mask == 255)
    org_colors = image[masking_area]
    color = np.array(color)

    soft_color = np.round(alpha * color + (1 - alpha) * org_colors).astype(np.uint8)
    image[masking_area] = soft_color
    return image





















#contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image, contours, -1, color, 0)
