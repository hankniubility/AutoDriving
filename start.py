if __name__ == '__main__':
    import os
    from ultralytics import YOLO

    # 设置环境变量，避免可能的库重复加载问题
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # 指定使用的设备，这里使用的是第一个CUDA设备
    device = "cuda:0"

    # 加载模型，这里使用的是训练好的best.pt模型
    model = YOLO('HouDuan/best.pt').to(device)

    # 指定要检测的图像路径
    img_path = 'R-C.jpg'  # 替换为你的图像路径
    save_dir='result.jpg'

    # 使用模型进行推理
    results = model.predict(source=img_path, save=True, save_dir=save_dir, show=True, project='runs/detect')

    # results是一个包含检测结果的列表，每个元素是一个字典
    # 你可以遍历results来获取每个检测到的对象的详细信息
    for result in results:
        boxes = result.boxes  # 检测到的边界框
        for box in boxes:
            # 获取边界框的坐标、类别ID、置信度等信息
            x1, y1, x2, y2 = box.xyxy[0]  # 边界框的坐标
            cls_id = box.cls[0].item()  # 类别ID
            confidence = box.conf[0].item()  # 置信度
            print(f"Detected object at ({x1}, {y1}) to ({x2}, {y2}) with class ID {cls_id} and confidence {confidence}")

