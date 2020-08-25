#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import os.path as osp
import os
import sys
import time
from argparse import ArgumentParser

import cv2
import numpy as np

from openvino.inference_engine import IENetwork
from ie_module import InferenceContext
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
from util.misc import COLOR_PALETTE

# My Class
# from identity import Person
# from identity import Identity

# import ros packages
from vino_reid.msg import face_roi
import rospy
from std_msgs.msg import String,Int8,Int32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL']
MATCH_ALGO = ['HUNGARIAN', 'MIN_DIST']


# Global Variables
FRAME = np.zeros(128)
face_label = "Unknown"
roi2 = [0,0,0,0]
vis = np.zeros(128)
index_roi = [0,0,0,0,0]
linked_identities = ""


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


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', metavar="PATH", required=True,
                         help="(optional) Path to the input video " \
                         "('0' for the camera, default)")
    general.add_argument('-o', '--output', metavar="PATH", default="",
                         help="(optional) Path to save the output video to")
    general.add_argument('--no_show', action='store_true',
                         help="(optional) Do not display output")
    general.add_argument('-tl', '--timelapse', action='store_true',
                         help="(optional) Auto-pause after each frame")
    general.add_argument('-cw', '--crop_width', default=0, type=int,
                         help="(optional) Crop the input stream to this width " \
                         "(default: no crop). Both -cw and -ch parameters " \
                         "should be specified to use crop.")
    general.add_argument('-ch', '--crop_height', default=0, type=int,
                         help="(optional) Crop the input stream to this height " \
                         "(default: no crop). Both -cw and -ch parameters " \
                         "should be specified to use crop.")
    general.add_argument('--match_algo', default='HUNGARIAN', choices=MATCH_ALGO,
                         help="(optional)algorithm for face matching(default: %(default)s)")

    gallery = parser.add_argument_group('Faces database')
    gallery.add_argument('-fg', metavar="PATH", required=False, default="/home/dingxuan/Desktop/face/",
                         help="Path to the face images directory")
    gallery.add_argument('--run_detector', action='store_true',
                         help="(optional) Use Face Detection model to find faces" \
                         " on the face images, otherwise use full images.")

    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd', metavar="PATH", default="/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml", required=False,
                        help="Path to the Face Detection model XML file")
    models.add_argument('-m_lm', metavar="PATH", default="/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml", required=False,
                        help="Path to the Facial Landmarks Regression model XML file")
    models.add_argument('-m_reid', metavar="PATH", default="/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml", required=False,
                        help="Path to the Face Reidentification model XML file")
    models.add_argument('-fd_iw', '--fd_input_width', default=0, type=int,
                         help="(optional) specify the input width of detection model " \
                         "(default: use default input width of model). Both -fd_iw and -fd_ih parameters " \
                         "should be specified for reshape.")
    models.add_argument('-fd_ih', '--fd_input_height', default=0, type=int,
                         help="(optional) specify the input height of detection model " \
                         "(default: use default input height of model). Both -fd_iw and -fd_ih parameters " \
                         "should be specified for reshape.")
    
    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Face Detection model (default: %(default)s)")
    infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Facial Landmarks Regression model (default: %(default)s)")
    infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Face Reidentification model (default: %(default)s)")
    infer.add_argument('-l', '--cpu_lib', metavar="PATH", default="",
                       help="(optional) For MKLDNN (CPU)-targeted custom layers, if any. " \
                       "Path to a shared library with custom layers implementations")
    infer.add_argument('-c', '--gpu_lib', metavar="PATH", default="",
                       help="(optional) For clDNN (GPU)-targeted custom layers, if any. " \
                       "Path to the XML file with descriptions of the kernels")
    infer.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")
    infer.add_argument('-pc', '--perf_stats', action='store_true',
                       help="(optional) Output detailed per-layer performance stats")
    infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
                       help="(optional) Probability threshold for face detections" \
                       "(default: %(default)s)")

    # Will only display names of faces with similarity greater than 70%
    infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.3,
                       help="(optional) Cosine distance threshold between two vectors " \
                       "for face identification (default: %(default)s)")
    infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
                       help="(optional) Scaling ratio for bboxes passed to face recognition " \
                       "(default: %(default)s)")
    infer.add_argument('--allow_grow', action='store_true',
                       help="(optional) Allow to grow faces gallery and to dump on disk. " \
                       "Available only if --no_show option is off.")

    return parser



class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        used_devices = set([args.d_fd, args.d_lm, args.d_reid])
        self.context = InferenceContext(used_devices, args.cpu_lib, args.gpu_lib, args.perf_stats)
        context = self.context

        log.info("Loading models")
        face_detector_net = self.load_model(args.m_fd)
        
        assert (args.fd_input_height and args.fd_input_width) or \
               (args.fd_input_height==0 and args.fd_input_width==0), \
            "Both -fd_iw and -fd_ih parameters should be specified for reshape"
        
        if args.fd_input_height and args.fd_input_width :
            face_detector_net.reshape({"data": [1, 3, args.fd_input_height,args.fd_input_width]})
        landmarks_net = self.load_model(args.m_lm)
        face_reid_net = self.load_model(args.m_reid)

        self.face_detector = FaceDetector(face_detector_net,
                                          confidence_threshold=args.t_fd,
                                          roi_scale_factor=args.exp_r_fd)

        self.landmarks_detector = LandmarksDetector(landmarks_net)
        self.face_identifier = FaceIdentifier(face_reid_net,
                                              match_threshold=args.t_id,
                                              match_algo = args.match_algo)

        self.face_detector.deploy(args.d_fd, context)
        self.landmarks_detector.deploy(args.d_lm, context,
                                       queue_size=self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, context,
                                    queue_size=self.QUEUE_SIZE)
        log.info("Models are loaded")

        log.info("Building faces database using images from '%s'" % (args.fg))
        self.faces_database = FacesDatabase(args.fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if args.run_detector else None, args.no_show)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info("Database is built, registered %s identities" % \
            (len(self.faces_database)))

        self.allow_grow = args.allow_grow and not args.no_show
        
    def refresh_database(self, args):
    	log.info("Re-building face database from {}".format(args.fg))
    	self.faces_database = FacesDatabase(args.fg, self.face_identifier, self.landmarks_detector, self.face_detector if args.run_detector else None, args.no_show)
    	self.face_identifier.set_faces_database(self.faces_database)
    	log.info("Database is built, registered {} identities".format(len(self.faces_database)))
    	

    def load_model(self, model_path):
        model_path = osp.abspath(model_path)
        model_description_path = model_path
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        log.info("Loading the model from '%s'" % (model_description_path))
        assert osp.isfile(model_description_path), \
            "Model description is not found at '%s'" % (model_description_path)
        assert osp.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
        model = IENetwork(model_description_path, model_weights_path)
        log.info("Model is loaded")
        return model

    def process(self, frame):
        #print(frame.shape)
        assert len(frame.shape) == 3, \
            "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], \
            "Expected BGR or BGRA input"

        orig_image = frame.copy()
        frame = frame.transpose((2, 0, 1)) # HWC to CHW
        frame = np.expand_dims(frame, axis=0)

        self.face_detector.clear()
        self.landmarks_detector.clear()
        self.face_identifier.clear()

        self.face_detector.start_async(frame)
        rois = self.face_detector.get_roi_proposals(frame)
        if self.QUEUE_SIZE < len(rois):
            log.warning("Too many faces for processing." \
                    " Will be processed only %s of %s." % \
                    (self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]
        self.landmarks_detector.start_async(frame, rois)
        landmarks = self.landmarks_detector.get_landmarks()

        self.face_identifier.start_async(frame, rois, landmarks)
        face_identities, unknowns = self.face_identifier.get_matches()
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                    (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                    (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop = orig_image[int(rois[i].position[1]):int(rois[i].position[1]+rois[i].size[1]), int(rois[i].position[0]):int(rois[i].position[0]+rois[i].size[0])]
                name = self.faces_database.ask_to_save(crop)
                if name:
                    id = self.faces_database.dump_faces(crop, face_identities[i].descriptor, name)
                    face_identities[i].id = id

        outputs = [rois, landmarks, face_identities]

        return outputs


    def get_performance_stats(self):
        stats = {
            'face_detector': self.face_detector.get_performance_stats(),
            'landmarks': self.landmarks_detector.get_performance_stats(),
            'face_identifier': self.face_identifier.get_performance_stats(),
        }
        return stats


class Visualizer:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q'), 27}
    IMAGE = None
    def callback(data):
        global IMAGE
        IMAGE = CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")


    def __init__(self, args):
        self.frame_processor = FrameProcessor(args)
        self.display = not args.no_show
        self.print_perf_stats = args.perf_stats

        '''
        #Added features
        '''
        self.frames_snapped = 10
        self.name = None
        '''
        frames_snapped: Counts the number of frames to automatically save the image into database
        '''

        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.frame_num = 0
        self.frame_count = -1

        self.input_crop = None
        if args.crop_width and args.crop_height:
            self.input_crop = np.array((args.crop_width, args.crop_height))

        self.frame_timeout = 0 if args.timelapse else 1

    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_text_with_background(self, frame, text, origin,
                                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0,
                                  color=(0, 0, 0), thickness=1, bgcolor=(255, 255, 255)):
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame,
                      tuple((origin + (0, baseline)).astype(int)),
                      tuple((origin + (text_size[0], -text_size[1])).astype(int)),
                      bgcolor, cv2.FILLED)
        cv2.putText(frame, text,
                    tuple(origin.astype(int)),
                    font, scale, color, thickness)
        return text_size, baseline

    def draw_detection_roi(self, frame, roi, identity, person=None):
        '''
        Listens for the id and bbox from visualizer in [id,x1,y1,x2,y2,....] order
        Compares roi with the obtained faces and respective ROIs
        '''
        global face_label, roi2
        def listener():
            global index_roi, face_label, roi2, linked_identities
            def callback4(data):
                global vis
                vis = CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")
                
            def callback1(data):
                global index_roi
                index_roi = data.data
                
            def check_overlap(R1, R2):
                # check if one rectangle is left of or above the other
                if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
                    return False
                
                return True

            
            final_vis = rospy.Publisher("/final_vis", Image, queue_size=1)
            img = Image()
            linked_identities = ""
            
            rospy.Subscriber("/index_roi", Int32MultiArray, callback1)
            # rospy.Subscriber("/vis_pub", Image, callback4)

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
                if len(ids.links) != 0:
                    maxKey = max(ids.links, key=ids.links.get)
                    text = "ID {} is {}.".format(ids.index, maxKey)
                    text += '\n'
                    linked_identities += text
            print(linked_identities)
            #### END OF LISTENER SUBSCRIBER

        label = self.frame_processor \
            .face_identifier.get_identity_label(identity.id)

        # Draw face ROI border
        cv2.rectangle(frame,
                      tuple(roi.position), tuple(roi.position + roi.size),
                      (0, 220, 0), 2)

        # Draw identity label
        text_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("H1", font, text_scale, 1)
        line_height = np.array([0, text_size[0][1]])
        text = label.split('.')[0]
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))
        self.draw_text_with_background(frame, text,
                                       roi.position - line_height * 0.5,
                                       font, scale=text_scale)
        '''
        face_roi msg
        name: face_label
        roi: Array of the roi of the face
        '''
        x,y,w,h = int(roi.position[0]), int(roi.position[1]), int(roi.size[0]), int(roi.size[1])
        roi2 = [x,y,x+w,y+h]
        face_label = label.split('.')[0]
        listener()
        # label_roi = face_roi()
        # face_pub = rospy.Publisher('/face_roi', face_roi, queue_size=10)
        # rate = rospy.Rate(30)
        # label_roi.name = label.split('.')[0]
        # label_roi.roi = [x,y,x+w,y+h]
        # face_pub.publish(label_roi)
        # rate.sleep()

                

    def draw_detection_keypoints(self, frame, roi, landmarks):
        keypoints = [landmarks.left_eye,
                     landmarks.right_eye,
                     landmarks.nose_tip,
                     landmarks.left_lip_corner,
                     landmarks.right_lip_corner]

        for point in keypoints:
            center = roi.position + roi.size * point
            cv2.circle(frame, tuple(center.astype(int)), 2, (0, 255, 255), 2)

    def draw_detections(self, frame, detections, person=None):
        for roi, landmarks, identity in zip(*detections):
            self.draw_detection_roi(frame, roi, identity, person)
            self.draw_detection_keypoints(frame, roi, landmarks)

    def draw_status(self, frame, detections):
        origin = np.array([10, 10])
        color = (10, 160, 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text_size, _ = self.draw_text_with_background(frame,
                                                      "Frame time: %.3fs" % (self.frame_time),
                                                      origin, font, text_scale, color)
        self.draw_text_with_background(frame,
                                       "FPS: %.1f" % (self.fps),
                                       (origin + (0, text_size[1] * 1.5)), font, text_scale, color)

        log.debug('Frame: %s/%s, detections: %s, ' \
                  'frame time: %.3fs, fps: %.1f' % \
                     (self.frame_num, self.frame_count, len(detections[-1]), self.frame_time, self.fps))

        if self.print_perf_stats:
            log.info('Performance stats:')
            log.info(self.frame_processor.get_performance_stats())

    def display_interactive_window(self, frame, person):
        global linked_identities
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text = "Press '%s' key to exit" % (self.BREAK_KEY_LABELS)
        thickness = 2
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(frame, text,
                    tuple(origin.astype(int)), font, text_scale, color, thickness)
        
        # Draw the detections
                # Draw ROI
        cv2.putText(frame, linked_identities, (50,200), font, 2, (0,0,200), 3)
        for ids in person.reid: 
            if ids.roi != [0, 0, 0, 0]:
                box_color = COLOR_PALETTE[ids.index % len(COLOR_PALETTE)] if ids.index >= 0 else (0, 0, 0)
                cv2.rectangle(frame, (ids.roi[0], ids.roi[1]), (ids.roi[2], ids.roi[3]), box_color, thickness=3)
                cv2.putText(frame, str(ids.index), (ids.roi[0], ids.roi[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)
        cv2.imshow('Face recognition demo', frame)


    def should_stop_display(self):
        key = cv2.waitKey(self.frame_timeout) & 0xFF
        return key in self.BREAK_KEYS

    def process(self, input_stream, output_stream, args):
        self.input_stream = input_stream
        self.output_stream = output_stream
        
        def callback3(data):
            global FRAME
            FRAME = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

        # Init ROS
        bridge = CvBridge()
        rospy.init_node('face_publisher', anonymous = True)
        def listener():
            # From Multi-Cam Visualizer
            rospy.Subscriber('/frames_pub', Image, callback3)
        rate = rospy.Rate(30)

        # Init tracker 
        person = Person() 

        while not rospy.is_shutdown():
        #while input_stream.isOpened():
            listener()
            try:
                if FRAME.any() != 0:
                    frame = FRAME
                    '''
                    @Original function to register input data
                    '''
#                    has_frame, frame = input_stream.read()
#                    frame = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
#                    
#                    if not has_frame:
#                        break
                    
                    if self.input_crop is not None:
                        frame = Visualizer.center_crop(frame, self.input_crop)
                    detections = self.frame_processor.process(frame)

                    '''
                    @Manual cropping of faces
                    '''
                    if cv2.waitKey(1) & 0xFF == ord("a"):
                        name = input("Please input name")
                        cropped = cv2.selectROI(frame)
                        current_db = os.listdir(args.fg)
                        name_list = [f.split('.')[0] for f in current_db]
                        counter = name_list.count(self.name)
                        x,y,w,h = cropped[0], cropped[1], cropped[2], cropped[3]
                        cv2.imwrite("{}{}.{}.png".format(args.fg,name,str(counter)), frame[y:y+h, x:x+w])
                        self.frame_processor.refresh_database(args)
                    '''
                    @Added feature: Automated recording of faces
                    '''
                    if cv2.waitKey(1) & 0xFF == ord("s"):
                        self.name = input("Please input name and we will save 10 best images")
                    for roi, landmarks, identity in zip(*detections):
                        label = self.frame_processor.face_identifier.get_identity_label(identity.id)

                        # Saves if the reidentified face is same as name you typed and higher than 60% similarity
                        if self.name == label.split('.')[0] and (1-identity.distance) > 0.6:
                            x,y,w,h = int(roi.position[0]), int(roi.position[1]), int(roi.size[0]), int(roi.size[1])
                            current_db = os.listdir(args.fg)
                            counter = 0
                            name_list = [f.split('.')[0] for f in current_db]
                            counter = name_list.count(self.name)
                            cv2.imwrite("{}{}.{}.png".format(args.fg,self.name,str(counter)), frame[y:y+h, x:x+w])
                            self.frames_snapped -= 1
                            self.frame_processor.refresh_database(args)
                    if self.frames_snapped == 0:
                        self.frames_snapped = 10
                        self.name = None
                    self.draw_detections(frame, detections, person)
                    self.draw_status(frame, detections)
                    
                    if output_stream:
                        output_stream.write(frame)
                    if self.display:
                        self.display_interactive_window(frame, person)
                        if self.should_stop_display():
                            break
                    
                    self.update_fps()
                    self.frame_num += 1
            except CvBridgeError as e:
                rospy.loginfo(e)

    @staticmethod
    def center_crop(frame, crop_size):
        fh, fw, fc = frame.shape
        crop_size[0] = min(fw, crop_size[0])
        crop_size[1] = min(fh, crop_size[1])
        return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                     (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                     :]

    def run(self, args):
        input_stream = Visualizer.open_input_stream(args.input)
        if input_stream is None or not input_stream.isOpened():
            log.error("Cannot open input stream: %s" % args.input)
        fps = input_stream.get(cv2.CAP_PROP_FPS)
        frame_size = (int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.crop_width and args.crop_height:
            crop_size = (args.crop_width, args.crop_height)
            frame_size = tuple(np.minimum(frame_size, crop_size))
        log.info("Input stream info: %d x %d @ %.2f FPS" % \
            (frame_size[0], frame_size[1], fps))
        output_stream = Visualizer.open_output_stream(args.output, fps, frame_size)

        self.process(input_stream, output_stream, args)

        # Release resources
        if output_stream:
            output_stream.release()
        if input_stream:
            input_stream.release()

        cv2.destroyAllWindows()

    @staticmethod
    def open_input_stream(path):
        log.info("Reading input data from '%s'" % (path))
        stream = path
        try:
            stream = int(path)
        except ValueError:
            pass
        frame = cv2.VideoCapture(stream)
        return frame

    @staticmethod
    def open_output_stream(path, fps, frame_size):
        output_stream = None
        if path != "":
            if not path.endswith('.avi'):
                log.warning("Output file extension is not 'avi'. " \
                        "Some issues with output can occur, check logs.")
            log.info("Writing output to '%s'" % (path))
            output_stream = cv2.VideoWriter(path,
                                            cv2.VideoWriter.fourcc(*'MJPG'), fps, frame_size)
        return output_stream


def main():
    args = build_argparser().parse_args()

    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    log.debug(str(args))

    visualizer = Visualizer(args)
    visualizer.run(args)


if __name__ == '__main__':
    main()
