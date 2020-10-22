from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np 
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']="hide"
import pygame
import argparse
import imutils
import time
import dlib
import cv2
import matplotlib.gridspec as gridspec
from collections import OrderedDict
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import euclidean_distances
import multiprocessing

def cal_EAR(eye):
	dist1=dist.euclidean(eye[1],eye[5])
	dist2=dist.euclidean(eye[2],eye[4])
	dist3=dist.euclidean(eye[0],eye[3])

	ear=(dist1+dist2)/(2*dist3)

	return ear

def cal_MAR(m):
	dist4=dist.euclidean(mouth[13],mouth[19])
	dist5=dist.euclidean(mouth[14],mouth[18])
	dist6=dist.euclidean(mouth[15],mouth[17])

	mar=(dist4+dist5+dist6)/3

	return mar

def count(thresholded, segmented):
    chull = cv2.convexHull(segmented)

    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

 
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    radius = int(0.8 * maximum_distance)

    circumference = (2 * np.pi * radius)

    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    (cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

       
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)

   
def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)

    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

pygame.mixer.init()

ap=argparse.ArgumentParser()
ap.add_argument("-l","--landmark_predictor",required=True,help="path of landmark predictor")
ap.add_argument("-a","--alarm",type=str,default="",help="path to alarm.wav file")
ap.add_argument("-w","--webcam",type=int,default=0,help="webcam number")

args=vars(ap.parse_args())
print()
print("######### INPUT ARGUMENTS:- #########")
print()
print("Landmark identifier: ",args["landmark_predictor"])
print("Alarm file: ",args["alarm"])
print("Webcam: ",args["webcam"])

top,right,bottom,left=10, 450, 225, 690
aWeight=0.5
num_frames=0
eye_aspect_ratio_threshold=0.3
mouth_aspect_ratio_threshold=20
cont_frames=48
cont_frames_blink=20
cont_frames_yawn=25
count1=0
count2=0
count3=0
bg = None
alarm_trig=False
og_ear=[]

print()
print("######### LOADING FACIAL LANDMARK PREDICTOR #########")
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(args["landmark_predictor"])

(leftstart,leftend)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightstart,rightend)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart,mend)=face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print()
print("######### STARTING VIDEOSTREAM #########")
vs=VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

print()
print("######### INITIALIZING EAR THRESHOLD #########")
print()
print("Please sit in a position you would normally sit while driving for a few seconds.")
while count2<=100:
	frame=vs.read()
	frame=imutils.resize(frame,width=700)
	frame=cv2.flip(frame,1)

	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	gray=cv2.equalizeHist(gray)

	rects=detector(gray,0)

	for rect in rects:
		shape=predictor(gray,rect)
		shape=face_utils.shape_to_np(shape)

		for (name,(i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			for (x,y) in shape[i:j]:
				cv2.circle(frame,(x,y),1,(0,0,255),-1)

			(x,y,w,h)=cv2.boundingRect(np.array(shape[i:j]))
			region_of_interest=frame[y:y+h,x:x+w]
			region_of_interest=imutils.resize(region_of_interest,width=600,inter=cv2.INTER_CUBIC)

		lefteye=shape[leftstart:leftend]
		righteye=shape[rightstart:rightend]
		mouth=shape[mstart:mend]

		left_ear=cal_EAR(lefteye)
		right_ear=cal_EAR(righteye)

		avg_ear=(left_ear+right_ear)/2

		left_eye_hull=cv2.convexHull(lefteye)
		right_eye_hull=cv2.convexHull(righteye)

		cv2.drawContours(frame,[left_eye_hull],-1,(0,255,0),1)
		cv2.drawContours(frame,[right_eye_hull],-1,(0,255,0),1)

		og_ear.append(avg_ear)

		count2+=1

sum=0.0
for i in range(len(og_ear)):
	sum=sum+og_ear[i]

temp=sum/len(og_ear)
eye_aspect_ratio_threshold=0.83*temp

print()
print("######### EAR THRESHOLD DETERMINED #########")
print()
print("Average EAR of driver: {:.2f}".format(temp))
print("EAR threshold of the driver: {:.2f}".format(eye_aspect_ratio_threshold))
print()
print("If for a continuous of 48 frames the EAR of driver remains below the threshold then alarm is triggered.")
print()
print("Raise your hand in the indicated box to turn the alarm off.")

while True:
	frame=vs.read()
	frame=imutils.resize(frame, width=700)
	frame=cv2.flip(frame,1)

	clone = frame.copy()

	(height, width) = frame.shape[:2]

	roi = frame[top:bottom, right:left]

	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	gray=cv2.equalizeHist(gray)

	gray=cv2.GaussianBlur(gray,(7,7),0)

	gray_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
	gray_roi=cv2.GaussianBlur(gray_roi,(7,7),0)

	rects=detector(gray,0)
	cv2.putText(frame, "Press 'q' to exit", (500, 500),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

	for rect in rects:
		shape=predictor(gray,rect)
		shape=face_utils.shape_to_np(shape)

		for (name,(i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			for (x,y) in shape[i:j]:
				cv2.circle(frame,(x,y),1,(0,0,255),-1)

			(x,y,w,h)=cv2.boundingRect(np.array(shape[i:j]))
			region_of_interest=frame[y:y+h,x:x+w]
			region_of_interest=imutils.resize(region_of_interest,width=600,inter=cv2.INTER_CUBIC)

		lefteye=shape[leftstart:leftend]
		righteye=shape[rightstart:rightend]
		mouth=shape[mstart:mend]

		left_ear=cal_EAR(lefteye)
		right_ear=cal_EAR(righteye)

		avg_ear=(left_ear+right_ear)/2

		mar=cal_MAR(mouth)

		left_eye_hull=cv2.convexHull(lefteye)
		right_eye_hull=cv2.convexHull(righteye)
		# mouth_hull=cv2.convexHull(mouth)

		cv2.drawContours(frame,[left_eye_hull],-1,(0,255,0),1)
		cv2.drawContours(frame,[right_eye_hull],-1,(0,255,0),1)
		cv2.drawContours(frame,[mouth],-1,(0, 255, 0), 1)

		if num_frames < 30:
			run_avg(gray_roi, aWeight)
		else:
			hand = segment(gray_roi)
			if hand is not None:
				(thresholded, segmented) = hand
				cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
				cv2.imshow("Thesholded", thresholded)
				x=count(thresholded, segmented)
				if x>3:
					count1=0
					alarm_trig=False
					pygame.mixer.music.stop()

		if avg_ear<eye_aspect_ratio_threshold:
			count1+=1
			if count1>=cont_frames:
				if not alarm_trig:
					alarm_trig=True
					if args["alarm"]!="":
						pygame.mixer.music.load(args["alarm"])
						pygame.mixer.music.play()
						print()

				cv2.putText(frame,"ALERT: DRIVER SLEEPY!!!",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

			elif count1>=cont_frames_blink:
				cv2.putText(frame,"YOU JUST BLINKED",(10,470),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

		else:
			count1=0
			cv2.putText(frame,"DRIVER IS AWAKE",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

		cv2.putText(frame,"EAR: {:.2f}".format(avg_ear),(10,60),cv2.FONT_HERSHEY_TRIPLEX,0.7,(0,255,0),2)

		if mar>mouth_aspect_ratio_threshold:
			count3+=1
			if count3>=cont_frames_yawn:
				cv2.putText(frame,"YOU JUST YAWNED",(10,500),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

		cv2.putText(frame,"MAR: {:.2f}".format(mar),(10,90),cv2.FONT_HERSHEY_TRIPLEX,0.7,(0,255,0),2)
		
		cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
		
		num_frames += 1

		cv2.imshow("Frame",frame)
		key=cv2.waitKey(1) & 0xFF

		if key==ord("q"):
			break
cv2.destroyAllWindows()
vs.stop()

