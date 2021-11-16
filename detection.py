import numpy as np
import cv2
import os 

MODEL_PATH = "yolo-coco"

# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3


# define the minimum distance (in pixels) between two people 
MIN_DISTANCE = 50


def detect_people(frame, net, ln, personId=0):
	#get frame width and height
	(frame_height, frame_width) = frame.shape[:2]

	#create blob from image
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(ln)

	results=[]
	boxes = []
	centroids = []
	confidences = []
	#loop over layer outputs
	for output in outputs:
		#loop over each of detections
		for detection in output:
			#get classID and confidence
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			#extract only person detection with certain confidence
			if classID == personId and confidence > MIN_CONF:
				box = detection[0:4] * np.array([frame_width, frame_height, frame_width,frame_height])
				(center_x, center_y, width, height) = box.astype("int")

				#derive top left coordinate
				x = int(center_x - (width / 2))
				y = int(center_y - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((center_x, center_y))
				confidences.append(float(confidence))

				#non maxima suppression
				bound_box= cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
	#ensure at least one detection exits
	if len(bound_box) > 0:
		for i in bound_box.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	# return the list of results
	return results



