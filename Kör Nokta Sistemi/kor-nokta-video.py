import torch
import cv2
import numpy as np
import winsound
import time


def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

def play_alert_sound():
    winsound.PlaySound("alert.wav", winsound.SND_ASYNC)


cv2.namedWindow('KOR NOKTA')
cv2.setMouseCallback('KOR NOKTA', POINTS)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
count = 0
cap = cv2.VideoCapture('0107.mp4')

area=[(310,720),(500,540),(810,540),(970,720),]
circle_inside = False
last_alert_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1280, 720))
    results = model(frame)

    circle_inside = False

    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d = (row['name'])
        #print(d)
        cx=int(x1+x2)//2
        cy=int(y2)
        results=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        #print(results)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (181, 181, 181), 2)
        cv2.putText(frame, str(d), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 118, 72), 2)
        if results >= 0:
            alpha = 0.4
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), -1)
            cv2.putText(overlay, str(d), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 118, 72), 2)
            cv2.circle(overlay, (cx, cy), 4, (255, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            circle_inside = True

    cv2.polylines(frame,[np.array(area,np.int32)],True,(0,0,255),2)
    cv2.imshow("KOR NOKTA", frame)

    current_time = time.time()

    if circle_inside and (current_time - last_alert_time) >= 2:
        play_alert_sound()
        last_alert_time = current_time

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
# stream.release()
cv2.destroyAllWindows()