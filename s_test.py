import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    next_ = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    #flow = cv.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = cv.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    #hsv[..., 0] = ang*180/np.pi/2
    #hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    #bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    #cv.imshow('frame2', bgr)

    #mag = cv.normalize(mag, None, 0, 63, cv.NORM_MINMAX)
    ret, thresh = cv.threshold(mag, 3, 255, cv.THRESH_BINARY)
    thresh_sum = thresh.sum(axis=0)
    thresh_sum_nz = np.where(thresh_sum != 0)[0]
    if len(thresh_sum_nz) != 0:
        thresh_sum_nz_min = thresh_sum_nz.min()
        thresh_sum_nz_max = thresh_sum_nz.max()
        thresh_sum_nz_centr = int(thresh_sum_nz_min + (thresh_sum_nz_max - thresh_sum_nz_min) / 2)
        div = thresh_sum_nz_centr - 320
    else:
        div = 0
        thresh_sum_nz_min = 0
        thresh_sum_nz_max = 640
        thresh_sum_nz_centr = 320

    # left line
    frame2 = cv.line(frame2, (thresh_sum_nz_min,0), (thresh_sum_nz_min,480), (250,0,0), 2)

    # right line
    frame2 = cv.line(frame2, (thresh_sum_nz_max,0), (thresh_sum_nz_max,480), (250,0,0), 2)

    # centre line
    frame2 = cv.line(frame2, (320,0), (320,480), (0,250,0), 2)

    # target line
    frame2 = cv.line(frame2, (thresh_sum_nz_centr,0), (thresh_sum_nz_centr,480), (0,0,250), 2)

    print(f'pix: {div}, percent: {round((div/320)*100, 2)}%')
    
    
    cv.imshow('frame2', frame2)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next_

cv.destroyAllWindows()