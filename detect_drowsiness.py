
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
	
	p = dist.euclidean(eye[1], eye[5])
	q = dist.euclidean(eye[2], eye[4])

	r = dist.euclidean(eye[0], eye[3])
	EAR = (p + q) / (2.0 * r)
	return EAR
 
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
 

EAR_LIMIT = 0.3
FRAMES_LIMIT = 48


COUNTER = 0
isALARM = False


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 0)

	for rect in rects:

		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		

		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

		left_eye = shape[lStart:lEnd]
		right_eye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(left_eye)
		rightEAR = eye_aspect_ratio(right_eye)

		ear = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(left_eye)
		rightEyeHull = cv2.convexHull(right_eye)
		cv2.drawContours(frame, [leftEyeHull], -1, (225, 0, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (225, 0, 0), 1)

		if ear < EAR_LIMIT:
			COUNTER += 1


			if COUNTER >= FRAMES_LIMIT:
				if not isALARM:
					isALARM = True

					# have to implement

				cv2.putText(frame, "!!!DROWSINESS ALERT!!!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		else:
			COUNTER = 0
			isALARM = False

		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()
vs.stop()