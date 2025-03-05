import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip,ImageSequenceClip
import matplotlib.pyplot as plt

# 全局变量初始化
left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

# 图像预处理函数
def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = np.copy(img)
    # 转换到 HLS 色彩空间
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)  # 注意BGR色彩空间
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel 边缘检测
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # 对x方向取梯度
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # 阈值处理
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

# 透视变换
def perspective_warp(img, dst_size=(1280, 720),
                     src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
                     dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

# 反透视变换
def inv_perspective_warp(img, dst_size=(1280, 720),
                         src=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
                         dst=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

# 获取直方图
def get_hist(img):
    return np.sum(img[img.shape[0] // 2:, :], axis=0)

# 滑动窗口拟合车道线
def sliding_window(img, nwindows=9, margin=100, minpix=1, draw_windows=True):
    global left_a, left_b, left_c, right_a, right_b, right_c
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255

    histogram = get_hist(img)
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int32(img.shape[0] / nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if draw_windows:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (100, 255, 255), 3)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (100, 255, 255), 3)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 将曲线拟合参数保存到历史记录中，并通过加权平均来平滑
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-5:])  # 使用最近5帧数据进行平滑
    left_fit_[1] = np.mean(left_b[-5:])
    left_fit_[2] = np.mean(left_c[-5:])
    right_fit_[0] = np.mean(right_a[-5:])
    right_fit_[1] = np.mean(right_b[-5:])
    right_fit_[2] = np.mean(right_c[-5:])

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit_[0] * ploty**2 + left_fit_[1] * ploty + left_fit_[2]
    right_fitx = right_fit_[0] * ploty**2 + right_fit_[1] * ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty


# 绘制车道线
def draw_lanes(original_img, warped_img, left_fit, right_fit):
    ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    left_fitx = left_fit
    right_fitx = right_fit

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = inv_perspective_warp(color_warp, dst_size=(original_img.shape[1], original_img.shape[0]))
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    return result

# 图像处理流程函数
def process_image(img):
    processed_img = pipeline(img)
    warped = perspective_warp(processed_img)
    out_img, curves, lanes, ploty = sliding_window(warped)
    result = draw_lanes(img, warped, curves[0], curves[1])
    return result

# 测试代码
img = cv2.imread('test_images\\test4.jpg')
result = process_image(img)

# 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[1].set_title("Final Lane Detection")
plt.show()


output1 = 'test_videos_output\\1.mp4'
output2 = 'test_videos_output\\2.mp4'
#output3 = 'test_videos_output\\3.mp4'

# 自动创建输出目录（如果不存在）
if not os.path.exists('test_videos_output'):
    os.makedirs('test_videos_output')

# 打开视频文件
clip1 = VideoFileClip("test_videos\\challenge.mp4")
clip2 = VideoFileClip("test_videos\\challenge_video.mp4")
#clip3 = VideoFileClip("test_videos\\harder_challenge_video.mp4")

def process_clip(clip, output_path):
    frames = []
    for t in np.arange(0, clip.duration, 1 / clip.fps):  # 按照帧率遍历每个时间点
        try:
            frame = clip.get_frame(t)
            processed_frame = process_image(frame)  # 对当前帧进行处理
            frames.append(processed_frame)
        except Exception as e:
            print(f"Warning: Could not read frame at time {t}. Error: {e}")
            frames.append(frames[-1])  # 使用上一帧代替当前帧

    processed_clip = ImageSequenceClip(frames, fps=clip.fps)
    processed_clip.write_videofile(output_path, audio=False)


# 处理所有视频
process_clip(clip1, output1)
process_clip(clip2, output2)
#process_clip(clip3, output3)
