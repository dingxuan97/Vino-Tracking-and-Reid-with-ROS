#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import String,UInt16MultiArray,MultiArrayLayout,MultiArrayDimension,Int8,Int32MultiArray
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
from message_filters import TimeSynchronizer, Subscriber
from vino_reid.msg import face_roi

# Global Variables
face_label = "Unknown"
roi2 = [0,0,0,0]
vis = np.zeros(128)
index_roi = [0,0,0,0,0]


class Identity:
    def __init__(self, reid_index):
        self.index = reid_index
        self.roi = [0,0,0,0]
        self.links = {}
        self.misses = 0

class Person:
    def __init__(self):
        self.reid = []
        self.faces = []
        self.index = []
    
def callback2(data):
    global face_label, roi2
    face_label = data.name
    roi2 = data.roi
    #rospy.loginfo(rospy.get_caller_id() + "Received %s", data.data)
    
def callback5(data):
    global vis
    vis = CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")
    
def callback1(data):
    global index_roi
    index_roi = data.data
    
def check_overlap(R1, R2):
    #print(reid_roi, face_roi)
    # check if one rectangle is left of the other
    if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
        return False
    
    return True

    
def listener():
    global face_label, roi2, index_roi, vis
    rospy.init_node('listener', anonymous=True)
    
    final_vis = rospy.Publisher("/final_vis", Image, queue_size=1)
    img = Image()
    linked_identities = ""
    
    rospy.Subscriber("/index_roi", Int32MultiArray, callback1)
    rospy.Subscriber("/face_roi", face_roi, callback2)
    rospy.Subscriber("/vis_pub", Image, callback5) 
    
    # copy_id prevents value from changing when new value is heard from subscriber

    for idx in range(len(index_roi)//4):
        copy_index = index_roi[idx*5]
        if copy_index >= 0:
            if len(person.reid) > 0:
                for ids in person.reid:
                    if ids.index == copy_index:
                        ids.roi = index_roi[idx*5 +1: idx*5 +5]
                        if check_overlap(ids.roi, roi2):
                            if face_label in ids.links:
                                ids.links[face_label] += 1
                                ids.misses = 0	
                            elif face_label != 'Unknown':
                                ids.links[face_label] = 1
                            break
                        else:
                            continue
                        break
                    else:
                        ids.misses += 1
                        print(ids.misses)
                        if ids.misses >= 10000:
                            person.reid.remove(ids)
                            print("[INFO] {} removed!".format(max(ids.links, key=ids.links.get)))
                else:
                    if (person.index.count(copy_index) == 0):
                        print("[INFO] Added Index {}".format(copy_index))
                        person.reid.append(Identity(copy_index))
                        person.index.append(copy_index)
            else:
                person.reid.append(Identity(copy_index))
                person.index.append(copy_index)
                    
    for ids in person.reid:
        print(ids.index, ids.links)
        maxKey = max(ids.links, key=ids.links.get)
        text = "ID {} is {}.".format(ids.index, maxKey)
        text += '\n'
        linked_identities += text
    print(linked_identities)
    cv.putText(vis, linked_identities, (100,100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 0), 2)
    img = CvBridge().cv2_to_imgmsg(vis, 'bgr8')
    final_vis.publish(img)

if __name__ == '__main__':
    person = Person()
    while not rospy.is_shutdown():
        try:
            listener()
        except:
            pass
