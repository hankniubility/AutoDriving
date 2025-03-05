import os

from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

# 设置环境变量，避免可能的库重复加载问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 指定使用的设备，这里使用的是第一个CUDA设备
device = "cuda:0"


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=(11, 11)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def preprocess_image(image):
    grayscale_image = convert_to_grayscale(image)
    blurred_image = apply_gaussian_blur(grayscale_image)
    return blurred_image

@app.route('/process_image', methods=['POS'])
def process_image():
    data = request.get_json()
    image_data = data['image']

    # 将Base64字符串解码为图像
    image_decoded = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_decoded))

    # 将PIL图像转换为OpenCV格式
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Preprocess the image
    processed_img = preprocess_image(image_cv)

    # 将处理后的图像转换为PIL格式，并编码为Base64字符串
    processed_img_pil = Image.fromarray(processed_img)
    img_bytes = BytesIO()
    processed_img_pil.save(img_bytes, format='JPEG')
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    # 返回Base64编码的图像
    return jsonify({'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)