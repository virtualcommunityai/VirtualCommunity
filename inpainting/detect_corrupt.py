import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def detect_damage_mask(image, dilation_iter=3, min_area=50):
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[-1] != 1 else image
    
    # 设定黑色区域的阈值（假设损坏区域接近黑色）
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    
    # 过滤掉极小区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # 膨胀操作，使 mask 区域向外扩张
    kernel = np.ones((3, 3), np.uint8)  # 5x5 的膨胀核
    filtered_mask = cv2.dilate(filtered_mask, kernel, iterations=dilation_iter)
    
    # 创建带 Alpha 通道的图像
    alpha_channel = cv2.bitwise_not(filtered_mask)  # 反转 Alpha 通道
    rgba_mask = np.zeros((filtered_mask.shape[0], filtered_mask.shape[1], 4), dtype=np.uint8)
    rgba_mask[:, :, 0:3] = 0  # 白色区域（R, G, B 全部为 255）
    rgba_mask[:, :, 3] = alpha_channel  # 反转后的 Alpha 通道
    
    # import pdb;pdb.set_trace()

    return rgba_mask