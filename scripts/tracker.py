#!/usr/bin/env python3
# USAGE
# python multi_object_tracking_fast.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video race.mp4

# import the necessary packages
from imutils.video import FPS
from imutils.video import VideoStream
import time
import multiprocessing
import numpy as np
import argparse
import imutils
import dlib
import cv2

#import Ros packages
import rospy
import sys
import numpy as np

from std_msgs.msg import String,UInt16MultiArray,MultiArrayLayout,MultiArrayDimension,Int16
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def image_publisher():
	label_pub = rospy.Publisher('label', String, queue_size = 100)
	position_pub = rospy.Publisher('image_position', Int16, queue_size=100)
	rospy.init_node('image_publisher', anonymous = True)
	rate = rospy.Rate(200)
	bridge = CvBridge()

	def start_tracker(box, label, rgb, inputQueue, outputQueue):
		# construct a dlib rectangle object from the bounding box
		# coordinates and then start the correlation tracker
		t = dlib.correlation_tracker()
		rect = dlib.rectangle(box[0], box[1], box[2], box[3])
		t.start_track(rgb, rect)

		# loop indefinitely -- this function will be called as a daemon
		# process so we don't need to worry about joining it
		while True:
			# attempt to grab the next frame from the input queue
			rgb = inputQueue.get()

			# if there was an entry in our queue, process it
			if rgb is not None:
				# update the tracker and grab the position of the tracked
				# object
				t.update(rgb)
				pos = t.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the label + bounding box coordinates to the output
				# queue
				outputQueue.put((label, (startX, startY, endX, endY)))

	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", default='/home/dingxuan/catkin_ws/src/my_package/scripts/MobileNetSSD_deploy.prototxt',
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", default='/home/dingxuan/catkin_ws/src/my_package/scripts/MobileNetSSD_deploy.caffemodel',
		help="path to Caffe pre-trained model")
	ap.add_argument("-v", "--video", default='/home/dingxuan/Desktop/Videos/mbs_expo.mp4',
		help="path to input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	ap.add_argument("-c", "--confidence", type=float, default=0.2,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# initialize our list of queues -- both input queue and output queue
	# for *every* object that we will be tracking
	inputQueues = []
	outputQueues = []

	# initialize the list of class labels MobileNet SSD was trained to
	# detect
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# if a video path was not supplied, grab the reference to the web cam
	if not args.get("video", False):
		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start()
		time.sleep(1.0)

	# otherwise, grab a reference to the video file
	else:
		vs = cv2.VideoCapture(args["video"])
	writer = None

	# start the frames per second throughput estimator
	fps = FPS().start()

	# loop over frames from the video file stream
	while True:
		# grab the current frame, then handle if we are using a
		# VideoStream or VideoCapture object
		frame = vs.read()
		frame = frame[1] if args.get("video", False) else frame

		# check to see if we have reached the end of the video file
		if frame is None:
			break

		# resize the frame for faster processing and then convert the
		# frame from BGR to RGB ordering (dlib needs RGB ordering)
		frame = imutils.resize(frame, width=1600)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if we are supposed to be writing a video to disk, initialize
		# the writer
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(frame.shape[1], frame.shape[0]), True)

		# if our list of queues is empty then we know we have yet to
		# create our first object tracker
		if len(inputQueues) == 0:
			# grab the frame dimensions and convert the frame to a blob
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

			# pass the blob through the network and obtain the detections
			# and predictions
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated
				# with the prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by requiring a minimum
				# confidence
				if confidence > args["confidence"]:
					# extract the index of the class label from the
					# detections list
					idx = int(detections[0, 0, i, 1])
					label = CLASSES[idx]
					print(idx,label)

					# if the class label is not a person, ignore it
					if CLASSES[idx] != "person":
						continue

					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")
					bb = (startX, startY, endX, endY)

					# create two brand new input and output queues,
					# respectively
					iq = multiprocessing.Queue()
					oq = multiprocessing.Queue()
					inputQueues.append(iq)
					outputQueues.append(oq)

					# spawn a daemon process for a new object tracker
					p = multiprocessing.Process(
						target=start_tracker,
						args=(bb, label, rgb, iq, oq))
					p.daemon = True
					p.start()

					# grab the corresponding class label for the detection
					# and draw the bounding box
					cv2.rectangle(frame, (startX, startY), (endX, endY),
						(0, 255, 0), 2)
					cv2.putText(frame, label, (startX, startY - 15),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

		# otherwise, we've already performed detection so let's track
		# multiple objects
		else:
			# loop over each of our input ques and add the input RGB
			# frame to it, enabling us to update each of the respective
			# object trackers running in separate processes
			for iq in inputQueues:
				iq.put(rgb)

			# loop over each of the output queues
			for oq in outputQueues:
				# grab the updated bounding box coordinates for the
				# object -- the .get method is a blocking operation so
				# this will pause our execution until the respective
				# process finishes the tracking update
				(label, (startX, startY, endX, endY)) = oq.get()

				#print(label,startX,startY,endX,endY)

				# draw the bounding box from the correlation object
				# tracker
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				cv2.putText(frame, label, (startX, startY - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
				#print("RECEIVED IT"+ label,startX,startY,endX,endY)
	
		#send the label, positions as output
		#return label,startX,startY,endX,endY
				label_pub.publish(label)
				rospy.loginfo(label)
				position_names = ['startX', 'startY', 'endX', 'endY']
				position_list = [startX, startY, endX, endY]
				for i in range(4):
					position_pub.publish(position_list[i])
					rospy.loginfo(position_names[i])
					rate.sleep()

		# check to see if we should write the frame to disk
		if writer is not None:
			writer.write(frame)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# update the FPS counter
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	
	# check to see if we need to release the video writer pointer
	if writer is not None:
		writer.release()
	
	# do a bit of cleanup
	vs.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	try:
		image_publisher()
	except rospy.ROSInterruptException:
		pass
	
