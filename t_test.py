import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2()
if not cap.isOpened():
    print("Error opening video file")
while cap.isOpened():
    # Захват кадр за кадром
    ret, frame = cap.read()
    if ret:
        # Вычитание фона
        fg_mask = backSub.apply(frame)
        # Поиск контура
        contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        # показать итоговый кадр
        cv2.imshow('Frame_final', frame_ct)
        # устанавливаем глобальный порог для удаления теней
        retval, mask_thresh = cv2.threshold( fg_mask, 180, 255, cv2.THRESH_BINARY)
        # вычисление ядра
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Apply erosion
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
        min_contour_area = 500  # Define your minimum area threshold
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        frame_out = frame.copy()
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
        # отображаем результат
        cv2.imshow('Frame_final', frame_out)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break