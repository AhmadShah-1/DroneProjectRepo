
import cv2

# cap = cv2.VideoCapture("vtest2.mp4")
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Images/testvideo1.mp4')
cap = cv2.VideoCapture('Images/testvideo2.mov')

FrameW = 500
FrameH = 500

while True:
    ret, frame = cap.read()

    img_rgb = cv2.resize(frame, (FrameW, FrameH))
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
    cv2.rectangle(img_rgb, location, bottom_right, (128, 128, 128), 5)

    initial_rectangle = []
    initial_rectangle.append(location[0])
    initial_rectangle.append(location[1])
    initial_rectangle.append(location[0] + w)
    initial_rectangle.append(location[1] + h)
    initial_rectangle = tuple(initial_rectangle)

    cv2.imshow('Detected', img_rgb)
    cv2.imshow('template', template)

    if cv2.waitKey(1) == ord('q'):
        break




# Cod
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIAN-FLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]

# So far i think CSRT is the one im going with

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create()
if tracker_type == 'MEDIAN-FLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()


# UPCOMING EDIT
ret, frame = cap.read()
frame = cv2.resize(frame, (FrameW, FrameH))

frame_height, frame_width = frame.shape[:2]
output = cv2.VideoWriter(f'{tracker_type}.avi',
                         cv2.VideoWriter_fourcc(*'XVID'), 60.0,
                         (frame_width // 2, frame_height // 2), True)


# Select the bounding box in the first frame
bbox = initial_rectangle
ret = tracker.init(frame, bbox)


# Start tracking
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (FrameW, FrameH))
    # frame = cv2.resize(frame, [frame_width // 2, frame_height // 2])

    if not ret:
        print('something went wrong')
        break
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
        # Original Code: p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(frame, tracker_type + " Tracker", (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv2.imshow("Tracking", frame)
    output.write(frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
output.release()
cv2.destroyAllWindows()
