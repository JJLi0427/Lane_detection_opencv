import cv2 
import numpy as np 
from sklearn.linear_model import RANSACRegressor 

def get_edge_img(
    color_img, 
    gaussian_ksize=5, 
    gaussian_sigmax=1, 
    canny_threshold1=330, 
    canny_threshold2=380
): 
    # 彩色图转灰度+高斯模糊+canny变换, 实现边缘检测 
    ''' 阈值设置: tunnel样例350-500, Alan样例330-400'''
    gaussian = cv2.GaussianBlur(
        color_img, 
        (gaussian_ksize, gaussian_ksize), 
        gaussian_sigmax
    ) 
    gray_img = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY) 
    edges_img = cv2.Canny(gray_img, canny_threshold1, canny_threshold2) 
    return edges_img 

def roi_mask(gray_img, pts): 
    # 掩膜, 选定ROI区域  
    mask = np.zeros_like(gray_img) 
    mask = cv2.fillPoly(mask, pts=[np.array(pts)], color=255) 
    img_mask = cv2.bitwise_and(gray_img, mask) 
    return img_mask 

def get_lines(edge_img, left_line_prev, right_line_prev):
    # 检测直线部分
    def calculate_slope(line):
        # 计算斜率
        x_1, y_1, x_2, y_2 = line[0] 
        change_x = x_2 - x_1
        change_y = y_2 - y_1
        return change_y / change_x if change_x != 0 else 0

    def reject_abnormal_lines(lines, threshold=0.3):
        # 去除异常直线
        slopes = [calculate_slope(line) for line in lines] 
        while len(lines) > 0: 
            # 除斜率异常的直线
            mean = np.mean(slopes) 
            diff = [abs(s - mean) for s in slopes] 
            idx = np.argmax(diff) 
            if diff[idx] > threshold: 
                slopes.pop(idx) 
                lines.pop(idx) 
            else: 
                break 
        return lines 

    def ransac_fit(lines): 
        # RANSAC拟合直线
        # ravel()将多维数组降为一维
        x_coords = np.ravel(
            [
                [line[0][0], line[0][2]] for line in lines
            ]
        ) 
        y_coords = np.ravel(
            [
                [line[0][1], line[0][3]] for line in lines
            ]
        ) 
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
        # polyval()计算多项式的值
        point_min = (
            np.min(x_coords), 
            np.polyval(
                poly, 
                np.min(x_coords)
            )
        ) 
        point_max = (
            np.max(x_coords), 
            np.polyval(
                poly, 
                np.max(x_coords)
            )
        ) 
        return np.array([point_min, point_max], dtype=int) 

    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(
        edge_img, 
        1, 
        np.pi / 180, 
        15, 
        minLineLength=35, 
        maxLineGap=20
    )
    '''Hough, 参数可调'''
    if lines is None:
        # 如果没有检测到直线, 则使用上一帧的直线
        left_line = left_line_prev
        right_line = right_line_prev
    else:
        # 如果检测到直线, 则使用RANSAC拟合直线
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