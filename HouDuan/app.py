from flask import Flask, request, jsonify
from skimage.measure import moments_central, moments_hu, regionprops
from ultralytics import YOLO
import os
import base64
from PIL import Image
import io
from flask_cors import CORS
from skimage.measure import label
from flask import Flask, request, jsonify
import cv2
import numpy as np
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
# 从 straight_lines.py 导入需要的函数
from straight_lines import processImage, color_filter, roi, grayscale, canny, hough_lines, weighted_img

app = Flask(__name__)
CORS(app)

# 设置环境变量，避免可能的库重复加载问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 指定使用的设备，这里使用的是第一个CUDA设备
device = "cuda:0"

# 加载模型，这里使用的是训练好的best.pt模型
model = YOLO('best.pt').to(device)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=(11, 11)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def preprocess_image(image):
    grayscale_image = convert_to_grayscale(image)
    blurred_image = apply_gaussian_blur(grayscale_image)
    return blurred_image

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    image_data = data['image']

    # 将Base64字符串解码为图像
    image_decoded = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_decoded))

    # 将PIL图像转换为OpenCV格式
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 使用自定义的 processImage 函数处理图像
    processed_image_cv = processImage(image_cv)

    # 使用YOLO模型进行推理
    results = model.predict(source=processed_image_cv, save=True, save_dir='runs/detect/predict', show=False)

    # 读取并编码处理后的图像
    with open('runs/detect/predict/image0.jpg', "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # 返回Base64编码的图像
    return jsonify({'image': encoded_string})

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data['image']

    # 将Base64字符串解码为图像
    image_decoded = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_decoded))

    # 将PIL图像转换为OpenCV格式
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Preprocess the image
    processed_img = preprocess_image(image_cv)

    # 将处理后的图像转换为PIL格式，并编码为Base64字符串
    processed_img_pil = Image.fromarray(processed_img)
    img_bytes = io.BytesIO()
    processed_img_pil.save(img_bytes, format='JPEG')
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    # 返回Base64编码的图像
    return jsonify({'image': img_base64})

# 3

def extract_color_histogram(image):
    # 转换为灰度图像
    gray_image = image.convert('L')
    # 计算颜色直方图
    hist = gray_image.histogram()
    return hist

def extract_texture_features(image):
    # 转换为灰度图像
    gray_image = np.array(image.convert('L'), dtype=np.uint8)
    # 计算灰度共生矩阵
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
    # 提取纹理特征
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    ASM = graycoprops(glcm, 'ASM')[0, 0]
    return {
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'ASM': ASM
    }

def extract_shape_features(image):
    # 转换为二值图像
    bw_image = np.array(image.convert('1'), dtype=bool)  # 使用 Python 内置的 bool 类型
    # 标记连通组件
    label_image = label(bw_image)
    # 提取区域属性
    regions = regionprops(label_image)
    if not regions:
        return None
    # 使用第一个区域（假设只有一个对象）
    region = regions[0]
    # 计算中心矩
    mu = moments_central(region.image.astype(np.uint8))  # 确保数据类型为 uint8
    # 计算Hu矩
    hu_moments = moments_hu(mu)
    return hu_moments.tolist()

@app.route('/extract-features', methods=['POST'])
def extract_features():
    data = request.get_json()
    image_data = data.get('image')

    if not image_data:
        return jsonify({'error': 'No image data provided.'}), 400

    try:
        # 将 Base64 字符串解码为二进制数据
        image_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_data))

        # 调用特征提取函数
        color_histogram = extract_color_histogram(image)
        texture_features = extract_texture_features(image)
        shape_features = extract_shape_features(image)

        # 将特征组合成一个字典
        features = {
            'color_histogram': color_histogram,
            'texture_features': texture_features,
            'shape_features': shape_features
        }

        # 返回特征字典
        return jsonify(features)

    except Exception as e:
        # 如果发生异常，返回错误信息
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)