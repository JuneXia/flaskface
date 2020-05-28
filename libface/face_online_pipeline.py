import os
import numpy as np
from libface.face_detection.mtcnn.detector import Detector
from libface.faceid.pipeline.identifier import FaceID
from libface.faceid.pipeline.verification import DistVerfication
from libface import tools
from libface.config import DeployConfig
from libface.config import SysConfig
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class FaceOnlinePipeline(object):
    def __init__(self):
        self.tmpdir = os.path.join(SysConfig['home_path'], DeployConfig['tmpdir'])
        if not os.path.exists(self.tmpdir):
            os.makedirs(self.tmpdir)

        self.detector = Detector()

        self.feature_extractor = []
        for faceid_model in DeployConfig['faceid_model_list']:
            model_path, model_ckpt = faceid_model
            model_path = os.path.join(DeployConfig['faceid_model_path'], model_path)
            self.feature_extractor.append(FaceID(model_path, specify_ckpt=model_ckpt))

        self.verifier = DistVerfication(distance_metric=1, is_dist_metric=False)

        image_path = './libface/lena.jpg'
        if os.path.exists(image_path):
            self.feature(image_path)

    def feature(self, image):
        """
        :param image: bgr image or impath
        :return:
        """
        if isinstance(image, str):
            image = self.detector.imread(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # TODO: just for mtcnn

        align_size = (160, 160)

        bounding_boxes = self.detector.detect(image, min_face_area=0, remove_inner_face=False)
        if bounding_boxes is None:
            return None
        aligned_bboxes, aligned_images = self.detector.align(image, bounding_boxes, align_size=align_size,
                                                             margin=0, center_det=False,
                                                             detect_multiple_faces=False,
                                                             detect_radius_factor=0.357)
        align_paths = list()
        for i in range(len(aligned_images)):
            impath = os.path.join(self.tmpdir, tools.get_strtime() + '_' + str(i) + '.png')
            self.detector.imwrite(aligned_images[i], impath)
            align_paths.append(impath)

        features = np.array([[]] * len(align_paths))
        for extractor in self.feature_extractor:
            feats = extractor.embedding(align_paths, random_rotate=False, random_crop=False, random_flip=False)
            features = np.concatenate((features, feats), axis=1)

        for impath in align_paths:
            if os.path.exists(impath):
                os.remove(image)

        return features

    def recognize(self, feature, base_features):
        """
        :param feature:
        :param base_features: data base features
        :return:
        """
        feature = feature.reshape([1, feature.shape[-1]])
        base_features = np.array(base_features)
        index, prob = self.verifier.identify(feature, base_features, use_max_simil=False)
        return index, prob


if __name__ == '__main__':
    # global g_pc
    print('[face_online_pipeline.__main__] start!!!!!!!!!!!!!!!!!!!!!')
    image_path = '/disk1/home/xiaj/res/lena.jpg'
    data_path = '/disk1/home/xiaj/res/face/gcface/Experiment/group3_crop_mtcnn_align160x160_margin32'
    images_path, images_label = tools.load_dataset(data_path, shuffle=False)

    pipeline = FaceOnlinePipeline()
    features = None
    for i, impath in enumerate(images_path):
        image = cv2.imread(impath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        feat = pipeline.feature(image)
        if features is None:
            features = feat
        else:
            features = np.concatenate((features, feat), axis=0)

    pipeline.recognize(features[0, :], features[1:, :])

    print('end')
