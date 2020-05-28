# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
Created on Thu Mar 28 13:45:00 2019

@author: xiaj
"""
import os
import tensorflow as tf
import numpy as np
from scipy import misc
from libface import tools
from libface.face_detection.mtcnn import detect_face
from libface.face_detection.base_detector import BaseDetector


class Detector(BaseDetector):
    def __init__(self):
        '''
        config = tf.ConfigProto(device_count={"CPU": 4},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=1,
                                intra_op_parallelism_threads=4,
                                log_device_placement=False)
        '''
        super(Detector, self).__init__()
        # with tf.device("/CPU:0"):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)  # local 0.333
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        file_root_path = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(file_root_path, 'models')
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, model_path)
        print('[face_detection.FaceDetector.__init__]:: successed!!!')

    def __del__(self):
        if self.sess is not None:
            self.sess.close()
        print('[FaceDetector.__del__]')

    def imread(self, impath):
        image = tools.imread(image_path, flags=0)
        return image

    def imwrite(self, image, impath):
        misc.imsave(impath, image)

    def detecting(self, image):
        """
        :param image:
        :return: return ndarray with shape[?, 15], 其中[?, 0:4]为人脸bbox, 依次是坐标(x1,y1),(x2,y2);
                                                      [?, 4]是检测为人脸的概率
                                                      [?, 5:15]为关键点坐标，依次是坐标(x1,y1),(x2,y2),(x3,y4), ...
        """
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor

        try:
            # imgsize = rgb_image.shape  # (width, height, channel)
            bounding_boxes, points = detect_face.detect_face(image, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
            shape = points.shape
            points = points.reshape((shape[1], 10)).transpose().reshape((shape[1], 10))
            # 反操作： points.reshape((10, 2)).transpose().reshape((10, 2))

            dets = np.hstack((bounding_boxes, points))

            return dets
        except Exception as err:
            print('[detect]: Exception.ERROR: %s' % err)
            return None

# --------------------------------------------------

import cv2


def show(image, dets):
    if type(image) == str:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    elif type(image) == np.ndarray:
        pass
    else:
        raise Exception('aa')

    for b in dets:
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(image, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)
        # save image

    name = "test.jpg"
    cv2.imwrite(name, image)
    cv2.imshow('PyTorch-Retinaface', image)
    cv2.waitKey()


if __name__ == '__main__':
    detector = Detector()

    image_path = '/disk1/home/xiaj/res/face/maskface/MAFA/tmp/test-images/test_00000394.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    dets = detector.detect(image)
    show(image, dets)
