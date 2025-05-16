import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import cv2
import numpy as np
import matplotlib.pyplot as plt

def variance_of_laplacian(image):
    """计算拉普拉斯方差以评估清晰度"""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sliding_window_blur_detection(image, window_size=16, threshold=128, color_threshold=50):
    """滑动窗口检测模糊区域，并移除颜色差距过大的点"""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for y in range(0, h - window_size, window_size):
        for x in range(0, w - window_size, window_size):
            window = image[y:y+window_size, x:x+window_size]
            gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            variance = variance_of_laplacian(gray)
            
            if variance < threshold:
                mask[y:y+window_size, x:x+window_size] = 255  # 标记模糊区域

                # 计算窗口的平均颜色
                avg_color = np.mean(window, axis=(0, 1))

                # 找出颜色差距过大的点
                color_diff = np.linalg.norm(window - avg_color, axis=2)
                outliers = color_diff > color_threshold
                
                # 在 mask 中去除这些点
                mask[y:y+window_size, x:x+window_size][outliers] = 0

    return mask

def extract_blur_regions(mask, min_area=10*16*16):
    """拆分 mask，获取每个独立的模糊区域"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masks = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        # print(area, min_area)
        single_mask = np.zeros_like(mask)
        cv2.drawContours(single_mask, [contour], -1, 255, thickness=cv2.FILLED)
        masks.append(single_mask)
    # print(len(masks))
    return masks

def overlay_mask(image, mask, alpha=0.5):
    """将 mask 叠加到原图上"""
    overlay = image.copy()
    overlay[mask == 255] = [0, 0, 255]  # 红色标记模糊区域
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def expand_blur_mask_by_color(image, mask, dilation_iterations=3, color_threshold=128):
    """
    1. 先计算原模糊区域内的平均颜色
    2. 对原mask做膨胀扩展区域
    3. 在扩展区域内计算每个像素与平均颜色的欧式距离，小于阈值则认为颜色相近
    """
    # 计算模糊区域内像素的平均颜色（确保模糊区域非空）
    blurred_pixels = image[mask == 255]
    if blurred_pixels.size == 0:
        return mask
    mean_color = np.mean(blurred_pixels, axis=0)

    # 膨胀原mask以扩展区域
    kernel = np.ones((5,5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
    
    # 对扩展区域内的每个像素，计算与平均颜色的欧式距离
    diff = np.linalg.norm(image - mean_color, axis=2)
    
    # 创建新的 mask：仅保留扩展区域中颜色与平均颜色接近的像素
    expanded_mask = np.zeros_like(mask)
    expanded_mask[(dilated_mask == 255) & (diff < color_threshold)] = 255
    expanded_mask = cv2.bitwise_or(expanded_mask, mask)
    return expanded_mask



def detect_damage_mask_lap(image, dilation_iter=3, min_area=50):
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    # 计算模糊区域 mask
    mask = sliding_window_blur_detection(image)

    # 拆分 mask，得到多个独立的模糊区域
    masks = extract_blur_regions(mask)

    exp_masks = []

    for m in masks:
        for i in range(5):
            m = expand_blur_mask_by_color(image, m)
            exp_masks.append(m)

    masks = exp_masks

    # 合并所有的模糊区域
    mask[:] = 0
    for m in masks:
        mask = cv2.bitwise_or(mask, m)


    kernel = np.ones((3, 3), np.uint8)  # 5x5 的膨胀核
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    # 创建带 Alpha 通道的图像
    alpha_channel = cv2.bitwise_not(mask)  # 反转 Alpha 通道
    rgba_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba_mask[:, :, 0:3] = 0  # 白色区域（R, G, B 全部为 255）
    rgba_mask[:, :, 3] = alpha_channel  # 反转后的 Alpha 通道

    return rgba_mask
