import cv2
import imutils
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
import math


def main(filename, start_time, end_time, playback_speed):
    canny_low_treshold = 20
    canny_high_treshold = 100
    width, height = 800, 500
    ffmpeg_extract_subclip(filename, start_time, end_time, targetname="sub_vid.mp4")

    cap = cv2.VideoCapture("MLtask.mp4")
    if (cap.isOpened() == False):
        print("Cannot show movie")

    i = 0
    while (cap.isOpened()):
        k = cv2.waitKey(int(1000/playback_speed)) & 0xFF
        if k == ord('q'):
            break
        i+=1
        ret, frame = cap.read()
        if ret == True:
            image = imutils.resize(frame, width=1000, height=900)
            gray = gray_the_image(False, image)

            gray_blur = gauss_smooth(False, gray)

            canny_full_size = canny(False, gray_blur, canny_low_treshold, canny_high_treshold)

            road_only = get_road_only(False, canny_full_size, height, 400, width, 0)

            highlighted = get_highlighted_lines(False, road_only, cv2.cvtColor(road_only, cv2.COLOR_GRAY2BGR))

            final = put_hough_over_color(False, image, highlighted)

            cv2.imshow('Vid', final)

    cap.release()
    cv2.destroyAllWindows()
    print(f"{i} frames shown")


def gray_the_image(save, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if save:
        cv2.imwrite("gray_image.jpg", gray)
    return gray


def gauss_smooth(save, image):
    gray_img = image
    gray_blur = cv2.GaussianBlur(gray_img, (15, 15), 0)
    if save:
        cv2.imwrite("gray_blur.jpg", gray_blur)
    return gray_blur


def canny(save, img, low_tresh, high_tresh):
    gray_blur = img
    canny_img = cv2.Canny(gray_blur, low_tresh, high_tresh)
    if save:
        cv2.imwrite("canny.jpg", canny_img)
    return canny_img


def get_road_only(save, img, max_height, min_height, max_width, min_width):
    road_only = img[min_height:max_height + 400, min_width:max_width]
    road_only_full = np.zeros_like(img)
    road_only_full[min_height:max_height + 400, min_width:max_width] = road_only
    if save:
        cv2.imwrite("road_only.jpg", road_only_full)
    return road_only_full


def get_highlighted_lines(save, dst, cdst):
    cdstP = np.copy(cdst)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 - 1000 * b), int(y0 + 1000 * a))
            pt2 = (int(x0 + 1000 * b), int(y0 - 1000 * a))
            cv2.line(cdst, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            line = linesP[i][0]
            cv2.line(cdstP, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3, cv2.LINE_AA)
    if save:
        cv2.imwrite("Hough_lines.jpg", cdstP)
    return cdstP


def put_hough_over_color(save, color, houghed):
    font1 = cv2.FONT_ITALIC
    result = cv2.addWeighted(color, 0.8, houghed, 1, 0)
    cv2.putText(result, '', (700, 500), font1, 1, (0, 255, 0), 2, cv2.LINE_4)
    if save:
        cv2.imwrite("color_with_lines", result)
    return  result


if __name__ == '__main__':
    main("MLtask.mp4", 400, 430, 40)
