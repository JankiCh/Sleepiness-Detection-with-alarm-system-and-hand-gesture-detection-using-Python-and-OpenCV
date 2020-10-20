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
import matplotlib.gridspec as gridspec
from collections import OrderedDict
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import euclidean_distances
import multiprocessing

# def cal_og_ear(counter):
# 	if counter<=50:
# 		counter+=1
# 		cv2.putText(frame,"Initializing threshold........",(600,500),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

# 	return counter

def sound_alarm(path):
	playsound.playsound(path)

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
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * maximum_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # take out the circular region of interest which has 
    # the palm and the fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    (cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # initalize the finger count
    count = 0

    # loop through the contours found
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    print(count)
    return count

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

   #---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

ap=argparse.ArgumentParser()
ap.add_argument("-l","--landmark_predictor",required=True,help="path of landmark predictor")
ap.add_argument("-a","--alarm",type=str,default="",help="path to alarm.wav file")
ap.add_argument("-w","--webcam",type=int,default=0,help="webcam number")

args=vars(ap.parse_args())

print("Printing input arguments passed:- ")
print("Landmark identifier: ",args["landmark_predictor"])
print("Alarm: ",args["alarm"])
print("Webcam: ",args["webcam"])

top,right,bottom,left=10, 450, 225, 690
aWeight=0.5
num_frames=0
eye_aspect_ratio_threshold=0.3
mouth_aspect_ratio_threshold=20
cont_frames=48
cont_frames_blink=20
count1=0
count2=0
yawncount=0
yawnthresh=4
bg = None
alarm_trig=False

print("Loading facial landmark predictor.........")
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(args["landmark_predictor"])

(leftstart,leftend)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightstart,rightend)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart,mend)=face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("Starting video stream......")
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
#call og ear
while True:
	frame=vs.read()
	frame=imutils.resize(frame, width=700)
	frame=cv2.flip(frame,1)

	clone = frame.copy()

	(height, width) = frame.shape[:2]

	roi = frame[top:bottom, right:left]

	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	gray=cv2.equalizeHist(gray)

	gray_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
	gray_roi=cv2.GaussianBlur(gray_roi,(7,7),0)

	rects=detector(gray,0)
	cv2.putText(frame, "Press 'q' to exit", (500, 500),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
	

	# count2=cal_og_ear(count2)

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
					# t.terminate()
					print("off")

		if avg_ear<eye_aspect_ratio_threshold:
			count1+=1
			if count1>=cont_frames:
				if not alarm_trig:
					alarm_trig=True
					if args["alarm"]!="":
						t=Thread(target=sound_alarm,args=(args["alarm"],))
						t.deamon=True
						t.start()

				cv2.putText(frame,"ALERT: DRIVER SLEEPY!!!",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

			elif count1>=cont_frames_blink:
				cv2.putText(frame,"YOU JUST BLINKED",(10,470),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

		else:
			count1=0
			cv2.putText(frame,"DRIVER IS AWAKE",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

		cv2.putText(frame,"EAR: {:.2f}".format(avg_ear),(10,60),cv2.FONT_HERSHEY_TRIPLEX,0.7,(0,255,0),2)

		if mar>mouth_aspect_ratio_threshold:
			yawncount+=1
			cv2.putText(frame,"YOU JUST YAWNED!!",(10,500),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

			if yawncount>=yawnthresh:
				if not alarm_trig:
					alarm_trig=True
					if args["alarm"]!="":
						t=Thread(target=sound_alarm,args=(args["alarm"],))
						t.deamon=True
						t.start()

				cv2.putText(frame,"ALERT: DRIVER SLEEPY!!!",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

		else: 
			# yawncount=0
			cv2.putText(frame,"DRIVER IS AWAKE",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
		
		cv2.putText(frame,"MAR: {:.2f}".format(mar),(10,90),cv2.FONT_HERSHEY_TRIPLEX,0.7,(0,255,0),2)
		
		cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
		
		num_frames += 1

		cv2.imshow("Frame",frame)
		key=cv2.waitKey(1) & 0xFF

		if key==ord("q"):
			break
cv2.destroyAllWindows()
vs.stop()

