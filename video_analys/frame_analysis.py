import datetime
import io
import os
import cv2
import numpy as np
from constants import constant


def __write_analysis(message):
    with io.BufferedWriter(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "frame_log.txt"), "ab")) as file:
        file.write(
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').encode() +
            b'\n' +
            message.encode() +
            b'\n' +
            b'-' * 20 +
            b'\n'
        )


def lucas_kanade(prev_frame, current_frame):
    if constant.SHOW_OPTIC_FLOW and (cv2.getWindowProperty("Optical Flow", cv2.WND_PROP_VISIBLE) >= 0):
        cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Optical Flow", 800, 600)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    feature_params = dict(maxCorners=constant.MAX_CORNERS, qualityLevel=constant.QUALITY_LEVEL,
                          minDistance=constant.MIN_DISTANCE, blockSize=constant.BLOCK_SIZE)

    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    current_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_points, None)
    prev_points = prev_points[status == 1]
    current_points = current_points[status == 1]

    percent_shift = []

    for i, (prev_point, next_point) in enumerate(zip(prev_points, current_points)):
        x1, y1 = prev_point.ravel().astype(int)
        x2, y2 = next_point.ravel().astype(int)

        # Расчет расстояния между точками
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Вычисление процента смещения
        max_distance = np.sqrt(prev_frame.shape[0] ** 2 + prev_frame.shape[1] ** 2)
        percent_shift.append((distance / max_distance) * 100)

        # Отрисовка линий
        if constant.SHOW_OPTIC_FLOW:
            cv2.line(prev_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(prev_frame, (x1, y1), 5, (0, 0, 255), -1)

        if constant.WRITE_LOG_OPTIC_FLOW:
            __write_analysis(f"Процент смещения точки {i}: {percent_shift[-1]}%")

    if constant.SHOW_OPTIC_FLOW:
        cv2.imshow("Optical Flow", prev_frame)
        cv2.waitKey(1)

    return percent_shift


def flow_processing(result_lucas_kanade, result=0):
    if result == 0:
        if np.mean(result_lucas_kanade) > constant.TRESHOLD_MEAN:
            return False
        else:
            return True
    elif result == 1:
        if max(result_lucas_kanade) > constant.TRESHOLD_MAX:
            return False
        else:
            return True
    elif result == 2:
        count = sum(1 for num in result_lucas_kanade if num > constant.VALUE)
        percentage = (count / len(result_lucas_kanade)) * 100
        if percentage > constant.TRESHOLD_PERCENTAGE:
            return False
        else:
            return True
