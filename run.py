import cv2 
import argparse
import numpy as np
import sys
from visualization import show_lane 

selecting_roi = False 
selected_pts = [] 
left_line_prev = None
right_line_prev = None
color_frame = None 

def click_event(event, x, y, flags, param):# 鼠标点击函数
    global selecting_roi, selected_pts 
    if event == cv2.EVENT_LBUTTONDOWN: 
        if len(selected_pts) < 4: 
            selected_pts.append((x, y)) 
            cv2.circle(color_frame, (x, y), 2, (0, 0, 255), -1) 
            if len(selected_pts) == 4: 
                selecting_roi = False 

def select_roi(frame): # 选取ROI区域
    '''顺时针选取类梯形区域'''
    global color_frame, selecting_roi, selected_pts 
    color_frame = frame.copy() 
    selected_pts = [] 
    selecting_roi = True 
    cv2.namedWindow("Select ROI") 
    cv2.setMouseCallback("Select ROI", click_event) 

    while selecting_roi: 
        cv2.imshow("Select ROI", color_frame) 
        cv2.waitKey(1) 
        if cv2.waitKey(1000) == ord(' '): # 按空格键退出
            sys.exit() # 退出程序

    cv2.destroyAllWindows() 

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Lane detection on a video.')
    parser.add_argument('video_path', help='Path to the video file.')
    args = parser.parse_args()

    CAPTURE = cv2.VideoCapture(args.video_path)
    '''选取视频'''
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 

    ret, frame = CAPTURE.read()
    select_roi(frame) 
    left_line_prev = np.array(
        [
            (selected_pts[0][0], selected_pts[0][1]), 
            (selected_pts[1][0], selected_pts[1][1])
        ]
    )
    right_line_prev = np.array(
        [
            (selected_pts[3][0], selected_pts[3][1]), 
            (selected_pts[2][0], selected_pts[2][1])
        ]
    )

    while CAPTURE.isOpened(): # 逐帧处理视频
        _, frame = CAPTURE.read() 
        origin = np.copy(frame) 
        frame = show_lane(
            left_line_prev, 
            right_line_prev, 
            frame, 
            selected_pts
        ) 
        output = np.concatenate((origin, frame), axis=1) 
        cv2.imshow('video', output) 
        if cv2.waitKey(50) == ord(' '): 
            cv2.destroyAllWindows() 
            break