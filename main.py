##############################################################
# skeletonize(img)
#    - Description: Skeletonize the current frame
#    - Parameters:
#          img: Input image in BGR format (numpy array).
#    - Returns: Skeletonized image (numpy array).
#
# houghline(img, rho=1, theta=np.pi/180, threshold=50, min_line_length=50, max_line_gap=10)
#    - Description: Apply Hough line detection to return an array of the detected line in the video stream
#    - Parameters:
#          img: Input image (numpy array).
#          rho: Distance resolution in pixels.
#          theta: Angle resolution in radians.
#          threshold: Accumulator threshold parameter.
#          min_line_length: Minimum length of line segments.
#          max_line_gap: Maximum allowed gap between line segments.
#    - Returns: Detected lines (list of arrays).
#
# gaussian_blur(img, kernel_size=(5,5), sigma=0)
#    - Description: Blur the image for easier processing 
#    - Parameters:
#          img: Input image (numpy array).
#          kernel_size: Size of the Gaussian kernel.
#          sigma: Standard deviation for Gaussian kernel.
#    - Returns: Blurred image (numpy array).
#
# canny_edge(img, threshold1=100, threshold2=200)
#    - Description: Make the line more apparent 
#    - Parameters:
#          img: Input image (numpy array).
#          threshold1: Lower threshold for hysteresis.
#          threshold2: Upper threshold for hysteresis.
#    - Returns: Edge-detected image (numpy array).
#
# matplotlib_display(img, title='Image')
#    - Description: Displays an image using matplotlib.
#    - Parameters:
#          img: Input image (numpy array).
#          title: Title for the display window.
#    - Returns: None.
#
# get_roi_params(timestamp)
#    - Description: Get the current parameter for region of interest
#    - Parameters:
#          timestamp: Time in seconds (float).
#    - Returns: Tuple of parameters (xo, xa, x1, x2, y1, y2).
#
# reverse(img, tl, tr, bl, br)
#    - Description: reverse the ROI to add it to the original video stream
#    - Parameters:
#          img: Input image (numpy array).
#          tl, tr, bl, br: Coordinates for top-left, top-right, bottom-left, bottom-right points.
#    - Returns: Transformed image (numpy array).
#
# w2g(img, timestamp, white_thresh=240, new_val=200)
#    - Description: convert the frame into binary for easier processing 
#    - Parameters:
#          img: Input image (numpy array).
#          timestamp: Time in seconds (float).
#          white_thresh: Threshold for white color.
#          new_val: New value for thresholded pixels.
#    - Returns: Binary image (numpy array).
#
# thr_black_lines(img)
#    - Description: Convert the current frame into a black and white bit image for easier processing 
#    - Parameters:
#          img: Input image (numpy array).
#    - Returns: Binary image with black lines (numpy array).
#
# sw_polyfit(binary_img, nwindows=50, margin=40, minpix=20)
#    - Description: Calculate the center line through Sliding Window Algorithm 
#    - Parameters:
#          binary_img: Binary image (numpy array).
#          nwindows: Number of sliding windows.
#          margin: Margin around each window.
#          minpix: Minimum number of pixels to recenter a window.
#    - Returns: left_fit, right_fit (polynomial coefficients for left and right lanes).
#
# draw_ln(original_img, binary_img, left_fit, right_fit, timestamp)
#    - Description: Draw the line, center line, and the compass arrow 
#    - Parameters:
#          original_img: Original input image (numpy array).
#          binary_img: Processed binary image (numpy array).
#          left_fit, right_fit: Polynomial coefficients for left and right lanes.
#          timestamp: Time in seconds (float).
#    - Returns: Image with lane lines and arrow overlay (numpy array).
#
# proc_img(img, timestamp)
#    - Description: Process the current frame and apply all above mentioned function 
#    - Parameters:
#          img: Input image (numpy array).
#          timestamp: Time in seconds (float).
#    - Returns: Processed image with lane lines overlay (numpy array).
#
# LaneDetectionGUI class
#    - Description: the gui that will display everything 
#    - Methods being used aaa:
#         __init__(self, root): Initializes the GUI.
#         create_placeholder(width, height, color): Creates a placeholder image.
#         add_log(message): Adds a log entry to the text log.
#         move_up/down/left/right(self): Logs movement commands.
#         toggle_stream(self): Starts or stops the video stream.
#         update_placeholder(self): Updates the GUI with placeholder images.
#         update_frame(self): Updates the video frames in the GUI.
#
#
#
#         pseudo code
#  -- -- --- -- -- -- -- -- -- -- --
#  Initialize GUI
#  store video stream
#  Wait for start command
#  start displaying video stream
#  process video stream: 
#    Blur, Canny, Skeletonize, color mask frame
#    apply sliding window 
#    polyfit a line given the point position of each window 
#    calculate center line by taking mean of each window and polyfitting it
#    display the lane line 
#  display processed video 
#  continue till video complete processing 
##############################################################

import cv2
import numpy as np
import sys
import math
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

def skeletonize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(binary, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary, temp)
        skel = cv2.bitwise_or(skel, temp)
        binary = eroded.copy()
        if cv2.countNonZero(binary) == 0:
            break
    return skel

def houghline(img, rho=1, theta=np.pi/180, threshold=50, min_line_length=50, max_line_gap=10):
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

def gaussian_blur(img, kernel_size=(5,5), sigma=0):
    return cv2.GaussianBlur(img, kernel_size, sigma)

def canny_edge(img, threshold1=100, threshold2=200):
    return cv2.Canny(img, threshold1, threshold2)

def matplotlib_display(img, title='Image'):
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def get_roi_params(timestamp):
    if timestamp < 28:
        return (300, 0, 440, 970, 1100, 200)
    elif timestamp < 58:
        return (400, 50, 440, 900, 1050, 150)
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
    if rarrow is not None:
        rarrow = cv2.resize(rarrow, (250,250))
    rarrow = cv2.resize(rarrow, (250,250))
    result = original_img.copy()
    overlay = np.zeros_like(original_img)
    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    if left_fit is None and right_fit is not None:
        right_fit_mirror = right_fit
        left_fit = np.array([-right_fit_mirror[0], -right_fit_mirror[1], original_img.shape[1] - right_fit_mirror[2]])
    elif right_fit is None and left_fit is not None:
        left_fit_mirror = left_fit
        right_fit = np.array([-left_fit_mirror[0], -left_fit_mirror[1], original_img.shape[1] - left_fit_mirror[2]])
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
        y_tip = 10
        delta = 10
        x_tip = center_fit[0]*(y_tip**2) + center_fit[1]*y_tip + center_fit[2]
        y_next = y_tip + delta
        x_next = center_fit[0]*(y_next**2) + center_fit[1]*y_next + center_fit[2]
        dx = x_next - x_tip
        dy = y_next - y_tip
        angle_rad = math.atan2(dx, dy)
        arrow_angle = math.degrees(angle_rad)
    xo, xa, x1, x2, y1, y2 = get_roi_params(timestamp)
    src = np.float32([[xo, y1], [xo + x1, y1], [xa + x2, y1 + y2], [xa, y1 + y2]])
    dst = np.float32([[0, 0], [x2, 0], [x2, y2], [0, y2]])
    trans = cv2.getPerspectiveTransform(dst, src)
    overlay = cv2.warpPerspective(overlay, trans, (result.shape[1], result.shape[0]))
    result = cv2.addWeighted(overlay, 2, original_img, 0.5, 0)
    if rarrow is not None and left_fit is not None and right_fit is not None:
        (h, w) = rarrow.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, arrow_angle, 1.0)
        rarrow_rotated = cv2.warpAffine(rarrow, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        arrow_h, arrow_w = rarrow_rotated.shape[:2]
        x_offset = result.shape[1] - arrow_w
        y_offset = 0
        if rarrow_rotated.shape[2] == 4:
            arrow_rgb = rarrow_rotated[:, :, :3]
            arrow_alpha = rarrow_rotated[:, :, 3] / 255.0
            roi = result[y_offset:y_offset+arrow_h, x_offset:x_offset+arrow_w]
            for c in range(3):
                roi[:, :, c] = arrow_alpha * arrow_rgb[:, :, c] + (1 - arrow_alpha) * roi[:, :, c]
            result[y_offset:y_offset+arrow_h, x_offset:x_offset+arrow_w] = roi
        else:
            result[y_offset:y_offset+arrow_h, x_offset:x_offset+arrow_w] = rarrow_rotated
    return result

def proc_img(img, timestamp):
    img_modified = w2g(img, timestamp, white_thresh=240, new_val=200)
    left_fit, right_fit = sw_polyfit(img_modified)
    if left_fit is None and right_fit is None:
        print("Warning: No lane pixels detected.")
        return img_modified
    result = draw_ln(img, img_modified, left_fit, right_fit, timestamp)
    return result

class LaneDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lane Detection GUI")
        self.video_width = 600
        self.video_height = 450
        self.streaming = False
        self.cap = None
        self.placeholder = self.create_placeholder(self.video_width, self.video_height)
        self.raw_label = tk.Label(root, image=self.placeholder, width=self.video_width, height=self.video_height)
        self.raw_label.grid(row=0, column=0)
        self.proc_label = tk.Label(root, image=self.placeholder, width=self.video_width, height=self.video_height)
        self.proc_label.grid(row=1, column=0)
        self.ctrl_frame = tk.Frame(root, width=self.video_width, height=self.video_height)
        self.ctrl_frame.grid(row=0, column=1, sticky="nsew")
        self.ctrl_frame.grid_columnconfigure(0, weight=1)
        self.ctrl_frame.grid_columnconfigure(1, weight=1)
        self.ctrl_frame.grid_columnconfigure(2, weight=1)
        self.ctrl_frame.grid_rowconfigure(0, weight=1)
        self.ctrl_frame.grid_rowconfigure(1, weight=1)
        self.ctrl_frame.grid_rowconfigure(2, weight=1)
        self.up_button = tk.Button(self.ctrl_frame, text="Up", width=10, command=self.move_up)
        self.left_button = tk.Button(self.ctrl_frame, text="Left", width=10, command=self.move_left)
        self.center_button = tk.Button(self.ctrl_frame, text="Start", width=10, command=self.toggle_stream)
        self.right_button = tk.Button(self.ctrl_frame, text="Right", width=10, command=self.move_right)
        self.down_button = tk.Button(self.ctrl_frame, text="Down", width=10, command=self.move_down)
        self.up_button.grid(row=0, column=1, padx=5, pady=5)
        self.left_button.grid(row=1, column=0, padx=5, pady=5)
        self.center_button.grid(row=1, column=1, padx=5, pady=5)
        self.right_button.grid(row=1, column=2, padx=5, pady=5)
        self.down_button.grid(row=2, column=1, padx=5, pady=5)
        self.up_button.bind("<ButtonRelease-1>", lambda e: self.add_log("stop moving"))
        self.left_button.bind("<ButtonRelease-1>", lambda e: self.add_log("stop moving"))
        self.right_button.bind("<ButtonRelease-1>", lambda e: self.add_log("stop moving"))
        self.down_button.bind("<ButtonRelease-1>", lambda e: self.add_log("stop moving"))
        self.log_text = tk.Text(root, height=20, width=50)
        self.log_text.grid(row=1, column=1, sticky="nsew")
    def create_placeholder(self, width, height, color=(0, 0, 0)):
        from PIL import Image
        image = Image.new('RGB', (width, height), color)
        return ImageTk.PhotoImage(image)
    def add_log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
    def move_up(self):
        self.add_log("moving up")
    def move_down(self):
        self.add_log("moving down")
    def move_left(self):
        self.add_log("moving left")
    def move_right(self):
        self.add_log("moving right")
    def toggle_stream(self):
        if self.streaming:
            self.streaming = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.center_button.config(text="Start")
            self.add_log("end video stream")
            self.update_placeholder()
        else:
            self.cap = cv2.VideoCapture("lane.MP4")
            self.streaming = True
            self.center_button.config(text="Stop")
            self.add_log("start video stream")
            self.update_frame()
    def update_placeholder(self):
        self.raw_label.config(image=self.placeholder)
        self.raw_label.image = self.placeholder
        self.proc_label.config(image=self.placeholder)
        self.proc_label.image = self.placeholder
    def update_frame(self):
        if self.streaming and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                raw_frame = cv2.resize(frame, (self.video_width, self.video_height))
                raw_image = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                raw_image = Image.fromarray(raw_image)
                raw_photo = ImageTk.PhotoImage(raw_image)
                self.raw_label.config(image=raw_photo)
                self.raw_label.image = raw_photo
                proc_frame = proc_img(frame, timestamp)
                proc_frame = cv2.resize(proc_frame, (self.video_width, self.video_height))
                proc_image = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                proc_image = Image.fromarray(proc_image)
                proc_photo = ImageTk.PhotoImage(proc_image)
                self.proc_label.config(image=proc_photo)
                self.proc_label.image = proc_photo
            else:
                self.add_log("Out of frames")
                self.streaming = False
                self.center_button.config(text="Start")
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                self.update_placeholder()
            self.root.after(30, self.update_frame)
        else:
            self.update_placeholder()

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
    root = tk.Tk()
    app = LaneDetectionGUI(root)
    root.mainloop()
