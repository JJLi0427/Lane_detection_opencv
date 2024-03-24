import cv2
import argparse

# 创建解析器并添加参数
parser = argparse.ArgumentParser(description='Canny edge detection on a video.')
parser.add_argument('video_path', type=str, help='Path to the video file.')

# 解析参数
args = parser.parse_args()

cv2.namedWindow('edge_detection')
cv2.createTrackbar('minThreshold', 'edge_detection', 50, 1000, lambda x: x)
cv2.createTrackbar('maxThreshold', 'edge_detection', 100, 1000, lambda x: x)

video_path = args.video_path  # 从命令行参数获取视频路径
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
cv2.createTrackbar('framePosition', 'edge_detection', 0, total_frames, lambda x: x)  # 添加滑动条 'framePosition'

while True:  # 逐帧显示视频
    frame_pos = cv2.getTrackbarPos('framePosition', 'edge_detection')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    minThreshold = cv2.getTrackbarPos('minThreshold', 'edge_detection')  # 获取滑动条的值
    maxThreshold = cv2.getTrackbarPos('maxThreshold', 'edge_detection')
    edges = cv2.Canny(img, minThreshold, maxThreshold)
    cv2.imshow('edge_detection', edges)  # 显示canny边缘检测效果

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()