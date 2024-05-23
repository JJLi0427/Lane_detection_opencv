import cv2
from lane_detection import get_edge_img, roi_mask, get_lines

def draw_lines(img, lines):
    left_line, right_line = lines
    # 用红色线表示检测到的车道线 
    cv2.line(
        img, 
        tuple(left_line[0]), 
        tuple(left_line[1]), 
        color=(0, 0, 255), 
        thickness=3
    )  
    cv2.line(
        img, 
        tuple(right_line[0]), 
        tuple(right_line[1]), 
        color=(0, 0, 255), 
        thickness=3
    ) 

    y_coords = [
        left_line[0][1], 
        left_line[1][1], 
        right_line[0][1], 
        right_line[1][1]
    ]
    y_min, y_max = min(y_coords), max(y_coords)
    slope_left = (left_line[1][1] - left_line[0][1]) / (left_line[1][0] - left_line[0][0])
    intercept_left = left_line[0][1] - slope_left * left_line[0][0]
    slope_right = (right_line[1][1] - right_line[0][1]) / (right_line[1][0] - right_line[0][0])
    # 根据最大和最小纵坐标计算新的端点横坐标
    intercept_right = right_line[0][1] - slope_right * right_line[0][0]

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

    cv2.line(
        img, 
        (int(x_min_left), y_min), 
        (int(x_max_left), y_max), 
        color=(0, 255, 0), 
        thickness=1
    )
    # 重新计算绘制延长的且高度平行的绿色线
    cv2.line(
        img, 
        (int(x_min_right), y_min), 
        (int(x_max_right), y_max), 
        color=(0, 255, 0), 
        thickness=1
    )

def show_lane(
    left_line_prev, 
    right_line_prev, 
    color_img, 
    selected_pts
):
    # 显示车道线
    edge_img = get_edge_img(color_img)
    mask_gray_img = roi_mask(edge_img, selected_pts)
    # 获取左右车道线
    lines = get_lines(mask_gray_img, left_line_prev, right_line_prev)
    left_line_prev = lines[0]
    right_line_prev = lines[1]

    draw_lines(color_img, lines)
    return color_img