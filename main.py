import cv2
import numpy as np
import sys

def reverse(img, tl,tr,bl,br):
    y, x = img.shape[:2]
    src = np.float32([tl, tr, bl, br])
    dst = np.float32([[0, 0], [x, 0], [0, y], [x, y]])
    trans = cv2.getPerspectiveTransform(dst, src)
    transformed = cv2.warpPerspective(img, trans, (x, y))
    return transformed

def w2g(img, white_thresh=240, new_val=200):
    output = img.copy()
    # straight roi
    # xo, xa, x1, x2, y1, y2 = (350, 50, 350, 950, 1000, 300)
    # right curve
    xo, xa, x1, x2, y1, y2 = (300, 0, 500, 1000, 900, 300)
    src = np.float32([[xo,y1],[xo+x1,y1], [xa+x2, y1+y2], [xa, y1+y2]])
    dst = np.float32([[0, 0], [x2,0], [x2,y2],[0,y2]])
    trans = cv2.getPerspectiveTransform(src, dst)
    output = cv2.warpPerspective(img, trans, (x2, y2))
    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    lower_yellow = np.array([200, 200, 200])
    upper_yellow = np.array([255, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white = np.array([180,180,0])
    upper_white = np.array([255, 255, 150])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    mask = cv2.bitwise_or(mask_white, mask_yellow)
    kernel = np.ones((5, 5), np.uint8)
    binary_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("mask", mask)
    return mask

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
        cv2.rectangle(binary_img, (leftx_current, win_y_high), (rightx_current, win_y_low), (255, 0, 0),5)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    cv2.imshow("life", binary_img)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    if len(leftx) == 0 or len(rightx) == 0:
        return None, None
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

def draw_ln(original_img, binary_img, left_fit, right_fit):
    # overlay = (original_img.copy())[900:1200, 50:1000]
    overlay = np.zeros_like(original_img)
    result = original_img.copy()
    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    if left_fit is not None and right_fit is not None:
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        center_fitx = (left_fitx + right_fitx) / 2
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
        pts_center = np.array([np.transpose(np.vstack([center_fitx, ploty]))])
        cv2.polylines(overlay, np.int32(pts_left), isClosed=False, color=(0, 0, 255), thickness=5)
        cv2.polylines(overlay, np.int32(pts_right), isClosed=False, color=(255, 0, 0), thickness=5)
        cv2.polylines(overlay, np.int32(pts_center), isClosed=False, color=(0, 255, 0), thickness=3)
    # result[900:1200, 50:1000] = overlay
    y, x = original_img.shape[:2]
    # straight roi
    # xo, xa, x1, x2, y1, y2 = (350, 50, 350, 950, 1000, 300)
    # right curve
    xo, xa, x1, x2, y1, y2 = (300, 0, 500, 1000, 900, 300)
    src = np.float32([[xo, y1], [xo + x1, y1], [xa + x2, y1 + y2], [xa, y1 + y2]])
    dst = np.float32([[0, 0], [x2, 0], [x2, y2], [0, y2]])
    trans = cv2.getPerspectiveTransform(dst, src)
    overlay = cv2.warpPerspective(overlay, trans, (x, y))
    result = cv2.addWeighted(overlay, 2, original_img, 0.5, 0)
    return result

def proc_img(img):
    img_modified = w2g(img, white_thresh=240, new_val=200)
    cv2.imshow("img", img_modified)
    # binary = thr_black_lines(img_modified)
    left_fit, right_fit = sw_polyfit(img_modified)
    if left_fit is None or right_fit is None:
        print("Warning: Not enough lane pixels detected.")
        return img_modified
    result = draw_ln(img, img_modified, left_fit, right_fit)
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python lane_detection.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    cap = cv2.VideoCapture("taehong.mp4")
    while cap.isOpened():
        print("heloo")
        ret, frame = cap.read()
        if ret is None:
            print("out of frame")
            break
        result = proc_img(frame)
        cv2.imshow("Lane Detection", result)
        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.waitKey(1) == ord('w'):
            input('')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
