from detection import MIN_DISTANCE, detect_people
from scipy.spatial import distance as dist
import os
import cv2
import numpy as np
import imutils

MODEL_PATH="yolo_coco"

# define the minimum distance (in pixels) between two people 
MIN_DISTANCE=50
weight_path="D:\course\Object detection\Social Distancing YOLOv3\yolo_coco\yolov3.weights"
config_path="D:\course\Object detection\Social Distancing YOLOv3\yolo_coco\yolov3_config.txt"
#load coco classes
with open('yolo_coco/coco_classes.txt') as f:
	class_names=f.read().strip().split('\n')

#load Yolo Object detector 
net = cv2.dnn.readNetFromDarknet(config_path, weight_path)

# determine only the output layer names 
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

cap=cv2.VideoCapture('People.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('video_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, 
					  (frame_width, frame_height))

while cap.isOpened():
	ret,frame= cap.read()
	if ret:
		frame = imutils.resize(frame, width=700)
		results=detect_people(frame,net,ln,personId=class_names.index("person"))

		#intilize the set to store index that violate social distancing
		violate=set()
		if len(results) >= 2:
			centroids=np.array([r[2] for r in results])

			#calculate Euclidean distance to detect social distancing
			D=dist.cdist(centroids,centroids,metric='euclidean')
			for i in range(0,D.shape[0]):
				for j in range(i+1 , D.shape[1]):
					if D[i][j] < MIN_DISTANCE:
						violate.add(i)
						violate.add(j)
		#loop over results and get bounding box data
		for (i, (prob, bbox, centroid)) in enumerate(results):
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (0, 255, 0)

			#update color for violation people's bounding box 
			if i in violate:
				color=(0,0,255)

			cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
			#draw circle on centroids
			cv2.circle(frame,(cX,cY),5,color,1)
			text = "Social Distancing Violations: {}".format(len(violate))
			cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
		cv2.imshow('Social Distancing Detector',frame)
		out.write(frame)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
	else:
		break