import cv2
import glob
import numpy as np

def convert_to_grayscale(image):
    """
    转换为灰度图
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=(11, 11)):
    """
    高斯滤波
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

# 流程整合
def preprocess_image(image):
    """
    图像预处理：灰度化 -> 高斯滤波
    """
    # 1: 转换为灰度图
    grayscale_image = convert_to_grayscale(image)

    # 2: 高斯滤波
    blurred_image = apply_gaussian_blur(grayscale_image)

    return grayscale_image,blurred_image

# 示例调用
if __name__ == "__main__":
    # 读取图像
    image_path = "D:\\QianDUan\\HouDuan\\111\zhixian\\test_images\\solidWhiteCurve.jpg"  # 替换为你的图像路径
    image = cv2.imread(image_path)

    if image is None:
        print("图像加载失败，请检查路径！")
    else:
        grayscale, blurred = preprocess_image(image)

        # 显示各步骤结果
        cv2.imshow("Original Image", image)
        cv2.imshow("Grayscale Image", grayscale)
        cv2.imshow("Gaussian Blurred Image", blurred)

        cv2.waitKey(0)
        cv2.destroyAllWindows()