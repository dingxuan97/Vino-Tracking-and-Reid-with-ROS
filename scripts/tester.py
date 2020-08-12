#!/usr/bin/env python

import unittest
import rospy
from std_msgs.msg import String
from time import sleep

class Test(unittest.TestCase):
	talker = False

	def callback(self, data):
		self.talker = True

	def test_publish(self):
		rospy.init_node(test_talker)
		rospy.Subscriber('image_position', String, self.callback)
	
		counter = 0 

		while not rospy.is_shutdown() and counter < 5 and (not self.talker):
			sleep(1)
			counter += 1

		self.assertTrue(self.talker)

if __name__ == '__main__':
	rostest.rosrun('my_package', 'tester', Test)
