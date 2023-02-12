
import itertools
import cv2
import numpy as np
import torch
from tracker import *
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('background-video-people-walking.mp4')
cap = cv2.VideoCapture("usian-bolt-running.mp4")

FrameW = 500
FrameH = 500
first_tracking = True
update_minimum = True
proceed1 = False
count_updates_for_minimum = 0
target_person_id = None
currentdistance = []
coodinates1 = []
# Find the center of mass of a cluster of pixels of a certain color that must reach a defined size to end the first part of the program

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (FrameW, FrameH))

    # Break the program if there is no input (Prevents error at the end of the run)
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([160, 100, 100])
    upper_red = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Find the center with the greatest cluster of the defined color:

    # Find all the connected clusters of pixels that are within the red hue range
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area, so that the largest contour is at the first position
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Draw the largest contour on the original image
    cv2.drawContours(frame, contours, 0, (0, 255, 0), 2)

    # Calculate the center of mass of the largest contour
    if len(contours) > 0 and cv2.contourArea(contours[0]) > 0:
        M = cv2.moments(contours[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Find the area of the largest contour
        area = cv2.contourArea(contours[0])
        # Set the threshold value for the area
        threshold = 3000
        if area > threshold:
            # Draw a circle around the center of mass (Going to use this to pass to the program
            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)

            # If testing Circle remove the break func here
            # The break func is to stop the program to pass to the next program once the fulfillment for the program has been fulfilled
            # The circle drawing at the top is merely for testing and optimization for the detection features
            break
            print(area)

    cv2.imshow('frame', frame)
    cv2.imshow('result', result)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) == ord('q'):
        break

# For Testing:
cX = 219
cY = 313

# cX = 72
# cY = 239
# Using Yolo find and track the person closest to the pixel location previously provided

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR, "ColorsBGR")


cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

tracker = Tracker()

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (FrameW, FrameH))

    # Detection
    results = model(frame)
    # frame = np.squeeze(results.render())

    # Stores the xmin xmax and ymin and ymax to create a bounding box
    # Also returns the confidence level and class (meaning what it is)
    list = []
    for index, row in results.pandas().xyxy[0].iterrows():
        # Just print it to show all that data
        # print(row)

        # Find sides of Bounding boxes
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        # Get name of Classes to name the boxes
        b = str(row['name'])

        # Because we only want to detect people we exclude everything else
        if 'person' in b:
            list.append([x1, y1, x2, y2])
        # use this to create a basic bounding box with id name on top
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        # cv2.putText(frame, b, (x1,y1), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)

    # Getting the Cords of the boxes and running it through the tracker algorithm
    boxes_ids = tracker.update(list)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 2)
        cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        # Calculate the distance from the cX, cY provided and only allow it to identify the target once so it doesn't keep identifying
        centerx = int((x + w)/2)
        centery = int((y + h)/2)
        cv2.circle(frame, (centerx, centery), 7, (255, 255, 255), -1)


    if target_person_id is None:
        # Store the distance of each centerpoint to th previous target, to then find the minimum distance to the pixel location given
        distances = []
        for i in range(len(boxes_ids)):
            distances.append(np.sqrt((cX - boxes_ids[i][0]) ** 2 + (cY - boxes_ids[i][1]) ** 2))
            coodinates1.append([boxes_ids[i][0], boxes_ids[i][1]])
            # print(distances, "DISTANCES")
            # print(np.argmin(distances), "ARGMIN")
        target_person_id = np.argmin(distances)
        # Start the x and y cooridinates of the centerpoint for the else statement
        targetx = coodinates1[target_person_id][0]
        targety = coodinates1[target_person_id][1]
    else:
        # Store the distances of all the centerpoints to find the closest ID to the last known location
        currentdistance = []
        for i in range(len(boxes_ids)):
            # If we already have a person to track, find the person with the same ID in the current frame
            if boxes_ids[i][-1] == target_person_id:
                x1, y1, w1, h1, id1 = boxes_ids[i]
                cX, cY = (boxes_ids[i][0], boxes_ids[i][1])
                cv2.rectangle(frame, (x1, y1), (w1, h1), (255, 0, 0), 2)
                proceed1 = False
                break
            else:
                # If we aren't able to find a person with the same id, keep appending the distances
                # so that we may repeat the same process as the previous "if target_person_id == None"
                x1, y1, w1, h1, id1 = boxes_ids[i]
                cX, cY = (boxes_ids[i][0], boxes_ids[i][1])
                currentdistance.append(np.sqrt((targetx - boxes_ids[i][0]) ** 2 + (targety - boxes_ids[i][1]) ** 2))
                proceed1 = True

        # If the ID wasn't found use distance to find the closest target in the event of occlusion or obstruction
        if proceed1:
            target_person_id = np.argmin(currentdistance)
            x2, y2, w2, h2, id2 = boxes_ids[target_person_id]
            cv2.rectangle(frame, (x2, y2), (w2, h2), (255, 0, 0), 2)
            # print(x2, y2, w2, h2, "COORDINATES")
            # print(target_person_id, "TARGETID")
            proceed1 = False

    # first_tracking = False
    cv2.imshow('FRAME', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()