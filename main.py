import cv2
import numpy as np
import sys
import math

def get_roi_params(timestamp):
    if timestamp < 28:
        return (300, 0, 440, 970, 1100, 200)
    elif timestamp < 58:
        return (400, 50,250, 900, 1050, 200)
    else:
        return (400, 70, 360, 920, 980, 320)

def reverse(img, tl, tr, bl, br):
    y, x = img.shape[:2]
    src = np.float32([tl, tr, bl, br])
    dst = np.float32([[0, 0], [x, 0], [0, y], [x, y]])
    trans = cv2.getPerspectiveTransform(dst, src)
    transformed = cv2.warpPerspective(img, trans, (x, y))
    return transformed

def w2g(img, timestamp, white_thresh=240, new_val=200):
    output = img.copy()
    xo, xa, x1, x2, y1, y2 = get_roi_params(timestamp)
    src = np.float32([[xo, y1], [xo + x1, y1], [xa + x2, y1 + y2], [xa, y1 + y2]])
    dst = np.float32([[0, 0], [x2, 0], [x2, y2], [0, y2]])
    trans = cv2.getPerspectiveTransform(src, dst)
    output = cv2.warpPerspective(img, trans, (x2, y2))
    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    lower_yellow = np.array([195, 195, 195])
    upper_yellow = np.array([255, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    lower_white = np.array([170, 170, 0])
    upper_white = np.array([255, 255, 165])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    kernel = np.ones((5, 5), np.uint8)
    binary_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("mask", mask)
    return binary_closed

def thr_black_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary_closed

def sw_polyfit(binary_img, nwindows=50, margin=40, minpix=20):
    histogram = np.sum(binary_img[binary_img.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    window_height = int(binary_img.shape[0] // nwindows)
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        win_y_low = binary_img.shape[0] - (window + 1) * window_height
        win_y_high = binary_img.shape[0] - window * window_height
        cv2.rectangle(binary_img, (leftx_current, win_y_high), (rightx_current, win_y_low), (255, 0, 0), 5)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    cv2.imshow("life", binary_img)
    left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else []
    right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else []
    leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else []
    lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else []
    rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else []
    righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else []
    left_fit = None
    right_fit = None
    if len(leftx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

def draw_ln(original_img, binary_img, left_fit, right_fit, timestamp):
    rarrow = cv2.imread('arrow.png', cv2.IMREAD_UNCHANGED)
    rarrow = cv2.resize(rarrow, (250,250))
    result = original_img.copy()
    overlay = np.zeros_like(original_img)
    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    frame_w = original_img.shape[1]
    if left_fit is None and right_fit is not None:
        right_fit_mirror = right_fit
        left_fit = np.array([-right_fit_mirror[0], -right_fit_mirror[1], frame_w - right_fit_mirror[2]])
    elif right_fit is None and left_fit is not None:
        left_fit_mirror = left_fit
        right_fit = np.array([-left_fit_mirror[0], -left_fit_mirror[1], frame_w - left_fit_mirror[2]])
    if left_fit is not None and right_fit is not None:
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
        pts_center = np.array([np.transpose(np.vstack([(left_fitx + right_fitx) / 2.0, ploty]))])
        cv2.polylines(overlay, np.int32(pts_left), False, (0, 0, 255), 5)
        cv2.polylines(overlay, np.int32(pts_right), False, (255, 0, 0), 5)
        cv2.polylines(overlay, np.int32(pts_center), False, (0, 255, 0), 3)
        center_fit = (left_fit + right_fit) / 2.0
        y0 = binary_img.shape[0] - 1
        y1 = max(binary_img.shape[0] - 51, 0)
        x0 = center_fit[0]*y0**2 + center_fit[1]*y0 + center_fit[2]
        x1 = center_fit[0]*y1**2 + center_fit[1]*y1 + center_fit[2]
        dx = x0 - x1
        dy = y0 - y1
        angle_rad = math.atan2(dx, dy)
        angle_deg = angle_rad * 180.0 / math.pi
        threshold = 10.0
        if angle_deg > threshold:
            arrow_type = "right"
        elif angle_deg < -threshold:
            arrow_type = "left"
        else:
            arrow_type = "forward"
    xo, xa, x1, x2, y1, y2 = get_roi_params(timestamp)
    src = np.float32([[xo, y1], [xo + x1, y1], [xa + x2, y1 + y2], [xa, y1 + y2]])
    dst = np.float32([[0, 0], [x2, 0], [x2, y2], [0, y2]])
    trans = cv2.getPerspectiveTransform(dst, src)
    overlay = cv2.warpPerspective(overlay, trans, (result.shape[1], result.shape[0]))
    result = cv2.addWeighted(overlay, 2, original_img, 0.5, 0)
    if rarrow is not None:
        if left_fit is not None and right_fit is not None:
            if arrow_type == "left":
                rarrow = cv2.flip(rarrow, 1)
            elif arrow_type == "forward":
                (h, w) = rarrow.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, -90, 1.0)
                rarrow = cv2.warpAffine(rarrow, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        arrow_h, arrow_w = rarrow.shape[:2]
        x_offset = result.shape[1] - arrow_w
        y_offset = 0
        if rarrow.shape[2] == 4:
            arrow_rgb = rarrow[:, :, :3]
            arrow_alpha = rarrow[:, :, 3] / 255.0
            roi = result[y_offset:y_offset+arrow_h, x_offset:x_offset+arrow_w]
            for c in range(3):
                roi[:, :, c] = arrow_alpha * arrow_rgb[:, :, c] + (1 - arrow_alpha) * roi[:, :, c]
            result[y_offset:y_offset+arrow_h, x_offset:x_offset+arrow_w] = roi
        else:
            result[y_offset:y_offset+arrow_h, x_offset:x_offset+arrow_w] = rarrow
    return result

def proc_img(img, timestamp):
    img_modified = w2g(img, timestamp, white_thresh=240, new_val=200)
    left_fit, right_fit = sw_polyfit(img_modified)
    if left_fit is None and right_fit is None:
        print("Warning: No lane pixels detected.")
        return img_modified
    result = draw_ln(img, img_modified, left_fit, right_fit, timestamp)
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python lane_detection.py <image_path>")
        sys.exit(1)
    cap = cv2.VideoCapture("lane.MP4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Out of frames")
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        print(timestamp)
        result = proc_img(frame, timestamp)
        xo, xa, x1, x2, y1, y2 = get_roi_params(timestamp)
        points = np.array([[xo,y1],[xo + x1, y1], [xa + x2, y1 + y2], [xa, y1 + y2]])
        points = points.reshape((-1, 1, 2))
        cv2.polylines(result, [points], isClosed = True, color = (255, 255, 255), thickness = 5)
        cv2.imshow("Lane Detection", result)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('w'):
            input('')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
