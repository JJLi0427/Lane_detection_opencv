import cv2 
import numpy as np 
from sklearn.linear_model import RANSACRegressor 
import sys
import argparse

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

def get_edge_img(color_img, gaussian_ksize=5, gaussian_sigmax=1, canny_threshold1=330, canny_threshold2=380): # 彩色图转灰度+高斯模糊+canny变换, 实现边缘检测 
    ''' 阈值设置: tunnel样例350-500, Alan样例330-400'''
    gaussian = cv2.GaussianBlur(color_img, (gaussian_ksize, gaussian_ksize), gaussian_sigmax) 
    gray_img = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY) 
    edges_img = cv2.Canny(gray_img, canny_threshold1, canny_threshold2) 
    return edges_img 

def roi_mask(gray_img, pts): # 掩膜, 选定ROI区域  
    mask = np.zeros_like(gray_img) 
    mask = cv2.fillPoly(mask, pts=[np.array(pts)], color=255) 
    img_mask = cv2.bitwise_and(gray_img, mask) 
    return img_mask 

def get_lines(edge_img, left_line_prev, right_line_prev):# 检测直线部分
    def calculate_slope(line):# 计算斜率
        x_1, y_1, x_2, y_2 = line[0] 
        return (y_2 - y_1) / (x_2 - x_1) 

    def reject_abnormal_lines(lines, threshold=0.3):# 去除异常直线
        slopes = [calculate_slope(line) for line in lines] 
        while len(lines) > 0: # 除斜率异常的直线
            mean = np.mean(slopes) 
            diff = [abs(s - mean) for s in slopes] 
            idx = np.argmax(diff) 
            if diff[idx] > threshold: 
                slopes.pop(idx) 
                lines.pop(idx) 
            else: 
                break 
        return lines 

    def ransac_fit(lines): # RANSAC拟合直线
        x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines]) # ravel()将多维数组降为一维
        y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines]) 
        X = x_coords.reshape(-1, 1) 
        y = y_coords.reshape(-1, 1) 
        ransac = RANSACRegressor(residual_threshold=10, max_trials=200)
        '''RANSAC, 参数可微调'''
        ransac.fit(X, y) 
        inlier_mask = ransac.inlier_mask_ 
        outlier_mask = np.logical_not(inlier_mask) 
        x_inliers = x_coords[inlier_mask] 
        y_inliers = y_coords[inlier_mask] 
        poly = np.polyfit(x_inliers, y_inliers, deg=1) 
        point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords))) # polyval()计算多项式的值
        point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords))) 
        return np.array([point_min, point_max], dtype=int) 

    lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 15, minLineLength=35, maxLineGap=20)# 霍夫变换检测直线
    '''Hough, 参数可调'''
    if lines is None:# 如果没有检测到直线, 则使用上一帧的直线
        left_line = left_line_prev
        right_line = right_line_prev
    else:# 如果检测到直线, 则使用RANSAC拟合直线
        left_lines = [line for line in lines if calculate_slope(line) > 0]
        right_lines = [line for line in lines if calculate_slope(line) < 0]
        left_lines = reject_abnormal_lines(left_lines)
        right_lines = reject_abnormal_lines(right_lines)

        if len(left_lines) == 0:
            left_line = left_line_prev
        else:
            left_line = ransac_fit(left_lines)

        if len(right_lines) == 0:
            right_line = right_line_prev
        else:
            right_line = ransac_fit(right_lines)

    return left_line, right_line

def draw_lines(img, lines):
    left_line, right_line = lines
    cv2.line(img, tuple(left_line[0]), tuple(left_line[1]), color=(0, 0, 255), thickness=3)# 用红色线表示检测到的车道线   
    cv2.line(img, tuple(right_line[0]), tuple(right_line[1]), color=(0, 0, 255), thickness=3) 

    y_coords = [left_line[0][1], left_line[1][1], right_line[0][1], right_line[1][1]]
    y_min, y_max = min(y_coords), max(y_coords)
    slope_left = (left_line[1][1] - left_line[0][1]) / (left_line[1][0] - left_line[0][0])
    intercept_left = left_line[0][1] - slope_left * left_line[0][0]
    slope_right = (right_line[1][1] - right_line[0][1]) / (right_line[1][0] - right_line[0][0])
    intercept_right = right_line[0][1] - slope_right * right_line[0][0]# 根据最大和最小纵坐标计算新的端点横坐标

    if slope_left != 0:
        x_min_left = (y_min - intercept_left) / slope_left
        x_max_left = (y_max - intercept_left) / slope_left
    else:
        x_min_left = x_max_left = 0
    if slope_right != 0:
        x_min_right = (y_min - intercept_right) / slope_right
        x_max_right = (y_max - intercept_right) / slope_right
    else:
        x_min_right = x_max_right = 0

    cv2.line(img, (int(x_min_left), y_min), (int(x_max_left), y_max), color=(0, 255, 0), thickness=1)# 重新计算绘制延长的且高度平行的绿色线
    cv2.line(img, (int(x_min_right), y_min), (int(x_max_right), y_max), color=(0, 255, 0), thickness=1)

def show_lane(color_img):# 显示车道线
    edge_img = get_edge_img(color_img)
    mask_gray_img = roi_mask(edge_img, selected_pts)

    global left_line_prev, right_line_prev# 获取左右车道线
    lines = get_lines(mask_gray_img, left_line_prev, right_line_prev)
    left_line_prev = lines[0]
    right_line_prev = lines[1]

    draw_lines(color_img, lines)
    return color_img

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Lane detection on a video.')
    parser.add_argument('video_path', help='Path to the video file.')
    args = parser.parse_args()

    CAPTURE = cv2.VideoCapture(args.video_path)
    '''选取视频'''
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 

    ret, frame = CAPTURE.read()
    select_roi(frame) 
    left_line_prev = np.array([(selected_pts[0][0], selected_pts[0][1]), (selected_pts[1][0], selected_pts[1][1])])
    right_line_prev = np.array([(selected_pts[3][0], selected_pts[3][1]), (selected_pts[2][0], selected_pts[2][1])])

    while CAPTURE.isOpened(): # 逐帧处理视频
        _, frame = CAPTURE.read() 
        origin = np.copy(frame) 
        frame = show_lane(frame) 
        output = np.concatenate((origin, frame), axis=1) 
        cv2.imshow('video', output) 
        if cv2.waitKey(50) == ord(' '): 
            cv2.destroyAllWindows() 
            break