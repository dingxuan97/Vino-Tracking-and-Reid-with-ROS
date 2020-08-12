#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import String,UInt16MultiArray,MultiArrayLayout,MultiArrayDimension,Int16,UInt8MultiArray
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError


def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "Received %s", data.data)

#def listener():

#    rospy.init_node('listener', anonymous=True)

#    rospy.Subscriber("image_position", Int16, callback)
#    rospy.Subscriber("label", String, callback)
#    
#    rospy.Subscriber("label_idx", Int16, callback)
#    
#    rospy.Subscriber("image", Image, showimage)

#    rospy.spin()
    
def showimage(data):
    bridge = CvBridge()
    rospy.loginfo("received an image!")
    try:
        image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        cv.imwrite("/home/dingxuan/Desktop/test.jpg", image)
        cv.imshow("subscribed", image)
    except CvBridgeError as e:
        print(e)
    
        
def listener():

    node_name = 'cv_bridge_subscribe'
    rospy.init_node(node_name, anonymous=True)
    
    bridge = CvBridge()
    rospy.loginfo("Loading subscribed images")    
    rospy.Subscriber('/image_pub', Image, showimage)
    rospy.Subscriber('/face_pub', Image, showimage)
    rospy.spin()
#    rate = rospy.Rate(5)
#    rate.sleep()
    
if __name__ == "__main__":
    try:
        listener()
    except:
        pass
    
    
