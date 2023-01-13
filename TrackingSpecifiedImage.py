import cv2

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIAN-FLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[0]

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


# Get the video file and read it
# video = cv2.VideoCapture("vtest2.mp4")
# video = cv2.VideoCapture(0)
video = cv2.VideoCapture('Images/testvideo1.mp4')
# video = cv2.VideoCapture('Images/testvideo2.mov')

FrameW = 500
FrameH = 500

ret, frame = video.read()
frame = cv2.resize(frame, (FrameW, FrameH))
frame_height, frame_width = frame.shape[:2]

# Resize the video for a more convenient view
#frame = cv2.resize(frame, [frame_width // 2, frame_height // 2])

# Initialize video writer to save the results

output = cv2.VideoWriter(f'{tracker_type}.avi',
                         cv2.VideoWriter_fourcc(*'XVID'), 60.0,
                         (frame_width // 2, frame_height // 2), True)

if not ret:
    print('cannot read the video')



    


# Select the bounding box in the first frame
bbox = cv2.selectROI(frame, False)
print(bbox)
ret = tracker.init(frame, bbox)

# Start tracking
while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (FrameW, FrameH))
    #frame = cv2.resize(frame, [frame_width // 2, frame_height // 2])
    if not ret:
        print('something went wrong')
        break
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
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

video.release()
output.release()
cv2.destroyAllWindows()