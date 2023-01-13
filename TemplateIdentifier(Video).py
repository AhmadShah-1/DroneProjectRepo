import cv2
import numpy as np

# cap = cv2.VideoCapture("vtest2.mp4")
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Images/testvideo1.mp4')
cap = cv2.VideoCapture('Images/testvideo2.mov')

while True:
    ret, frame = cap.read()

    img_rgb = cv2.resize(frame, (500, 500))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    H, W = img_rgb.shape[:2]

    template_height = int(H/5)
    template_width = int(W/5)


    template = cv2.resize(cv2.imread('Images/template1.png', 0), (template_height, template_width))
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    location = max_loc

    bottom_right = (location[0] + w, location[1] + h)
    cv2.rectangle(img_rgb, location, bottom_right, (128,128,128), 5)

    cv2.imshow('Detected', img_rgb)
    cv2.imshow('template', template)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()