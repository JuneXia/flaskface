# -*- coding: utf-8 -*-
# !/usr/bin/env python2
"""
Created on Thu Mar 28 13:45:00 2019

@author: xiaj
"""
import random
from scipy import misc
import numpy as np


class BaseDetector:
    def __init__(self):
        pass

    def __del__(self):
        print('[BaseDetector.__del__]')

    def imread(self, impath):
        raise NotImplementedError

    def imwrite(self, image, impath):
        raise NotImplementedError

    def detect_image(self, image, min_face_area=0, remove_inner_face=True):
        raise Exception('废弃，被detect和detecting所替代')
        try:
            imgsize = image.shape  # (width, height, channel)
            dets = self.detector.detect(image)

            bounding_boxes, points = dets[:, 0:4], dets[:, 5:]
            shape = points.shape
            points = points.reshape((10, shape[0])).transpose().reshape((10, shape[0]))

            if len(bounding_boxes) > 0:
                bounding_boxes = bounding_boxes[:, 0:4].astype(int)
                bounding_boxes = bounding_boxes.tolist()
                points = points.transpose().tolist()
                bboxes = bounding_boxes.copy()
                points_meida = points.copy()
                for i, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox
                    if remove_inner_face and (x1 < 0 or y1 < 0 or x2 > imgsize[1] or y2 > imgsize[0]):
                        if bounding_boxes.count(bbox) > 0:
                            bounding_boxes.remove(bbox)
                            points.remove(points_meida[i])

                    face_area = (x2 - x1) * (y2 - y1)
                    if face_area < min_face_area:
                        if bounding_boxes.count(bbox) > 0:
                            bounding_boxes.remove(bbox)
                            points.remove(points_meida[i])

                points = np.array(points).transpose()

            if self.task == self.TASK_OUTPUT_KEY_POINT:
                return np.array(bounding_boxes), points

            return np.array(bounding_boxes)
        except Exception as err:
            print('[detect]: Exception.ERROR: %s' % err)
            return None

    def detecting(self, image):
        raise NotImplementedError

    def detect(self, image, min_face_area=None, remove_inner_face=True):
        """
        这里以1280x720尺寸的图像为基准，我们定义该尺寸图像下的最小人脸区域面积为500
        :param image:
        :param min_face_area: 用户也可以指定最小人脸区域的大小。
        :return:
        """
        try:
            factor = 5.425e-4  # factor = 500/(1280*720)

            if isinstance(image, str):
                image = self.imread(image)

            if min_face_area is None:
                min_face_area = image.shape[0] * image.shape[1] * factor

            # if self.task == self.TASK_OUTPUT_KEY_POINT:
            #     center_det = True
            #     bounding_boxes, points = self.detect_image(img, min_face_area, remove_inner_face)
            #     if len(bounding_boxes) > 0:
            #         det = bounding_boxes[:, 0:4]
            #         img_size = np.asarray(img.shape)[0:2]
            #
            #         bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            #         if center_det:
            #             img_center = img_size / 2
            #             offsets = np.vstack(
            #                 [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            #             offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            #             index = np.argmax(
            #                 bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
            #
            #             bounding_boxes = bounding_boxes[index, :]
            #             points = points[:, index]
            #         else:
            #             raise Exception('目前只支持中心检测！')
            #
            #     if len(bounding_boxes) > 0:
            #         bounding_boxes = bounding_boxes.reshape(1, -1)
            #         points = points.reshape(-1, 1)
            #     return bounding_boxes, points

            # bounding_boxes = self.detect_image(image, min_face_area, remove_inner_face)

            imgsize = image.shape  # (width, height, channel)
            dets = self.detecting(image)

            bounding_boxes, points = dets[:, 0:4], dets[:, 5:]
            shape = points.shape
            points = points.reshape((10, shape[0])).transpose().reshape((10, shape[0]))

            if len(bounding_boxes) > 0:
                bounding_boxes = bounding_boxes[:, 0:4].astype(int)
                bounding_boxes = bounding_boxes.tolist()
                points = points.transpose().tolist()
                bboxes = bounding_boxes.copy()
                points_meida = points.copy()
                for i, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox
                    if remove_inner_face and (x1 < 0 or y1 < 0 or x2 > imgsize[1] or y2 > imgsize[0]):
                        if bounding_boxes.count(bbox) > 0:
                            bounding_boxes.remove(bbox)
                            points.remove(points_meida[i])

                    face_area = (x2 - x1) * (y2 - y1)
                    if face_area < min_face_area:
                        if bounding_boxes.count(bbox) > 0:
                            bounding_boxes.remove(bbox)
                            points.remove(points_meida[i])

                points = np.array(points).transpose()

            # if self.task == self.TASK_OUTPUT_KEY_POINT:
            #     return np.array(bounding_boxes), points

            return np.array(bounding_boxes)
        except Exception as err:
            print('[detect]: Exception.ERROR: %s' % err)
            return None

    def margin_face(self, image, bboxes, align_size, margin):
        img_size = np.asarray(image.shape)[0:2]
        aligned_bboxes = []
        aligned_images = np.zeros((len(bboxes), align_size[0], align_size[1], 3))
        for i, bbox in enumerate(bboxes):
            bbox = np.squeeze(bbox)
            bb = np.zeros(4, dtype=np.int32)

            bb[0] = np.maximum(bbox[0] - margin / 2, 0)
            bb[1] = np.maximum(bbox[1] - margin / 2, 0)
            bb[2] = np.minimum(bbox[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(bbox[3] + margin / 2, img_size[0])

            aligned_bboxes.append(bb)
            cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = misc.imresize(cropped, (align_size[0], align_size[1]), interp='bilinear')
            aligned_images[i, :] = scaled
        aligned_bboxes = np.array(aligned_bboxes)

        return aligned_bboxes, aligned_images

    def align(self, img, bounding_boxes, align_size=(160, 160), margin=32,
              detect_multiple_faces=False, center_det=True,
              detect_radius_factor=None, random_margin=False):
        """
        Face align by image.
        :param img: image
        :param bounding_boxes: detected face bounding boxes
        :param align_size: destination align size, (align_height, align_width)
        :param margin:
        :param detect_multiple_faces:
        :param center_det:
        :param detect_radius_factor: It's usefully when there are detect multi-face.
        for compute detect radius is min(img_size)*detect_radius_factor, recommend 1/2.8
        :param random_margin:
        :return:
        """
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces < 1:
            print('[align_face]: not detected face!')
            return [], []

        if random_margin:
            margin_list = [0, 4, 9, 16, 19, 24, 27, 32]
            # margin_list = (np.array(margin_list).astype('float32') / 32.0) * margin
            # margin_list = margin_list.astype(int).tolist()
            margin_index = random.randint(0, len(margin_list) - 1)
            margin = margin_list[margin_index]

        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces > 1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                if detect_radius_factor is not None:
                    bboxes_center = np.array([det[:, 1] + (det[:, 3] - det[:, 1]) / 2,
                                              det[:, 0] + (det[:, 2] - det[:, 0]) / 2]).transpose()
                    center_point = img_size / 2
                    dist = (bboxes_center - center_point).__pow__(2)
                    proposal_index = np.where(np.sqrt(dist[:, 0] + dist[:, 1]) < min(img_size) * detect_radius_factor)[
                        0]
                    det = det[proposal_index]

                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                if center_det:
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    index = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                else:
                    index = np.argmax(bounding_box_size)
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))
        '''
        aligned_bboxes = []
        aligned_images = np.zeros((len(det_arr), align_size[0], align_size[1], 3))
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)

            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])

            aligned_bboxes.append(bb)
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = misc.imresize(cropped, (align_size[0], align_size[1]), interp='bilinear')
            aligned_images[i, :] = scaled

        aligned_bboxes = np.array(aligned_bboxes)
        '''
        det_arr = np.array(det_arr)
        aligned_bboxes, aligned_images = self.margin_face(img, det_arr, align_size, margin)
        return aligned_bboxes, aligned_images

    # def align_image(self, image, align_size=(160, 160), margin=32, min_face_area=0, remove_inner_face=True):
    #     '''
    #     :param image: image
    #     :param align_size:
    #     :param margin:
    #     :return:
    #     '''
    #     try:
    #         bboxes = self.detect_image(image, min_face_area=min_face_area, remove_inner_face=remove_inner_face)
    #         if self.task == self.TASK_MEGAFACE_DETECT:
    #             # 只要检测到人脸(不管检测到多少个人脸)，则将所有人脸都检测出并全部返回，交给ground-truth做overlap选取大于threshold的那个作为最终检测到的人脸；
    #             # 如果检测不到人脸，或者检测到的人脸和ground-truth的overlap小于threshold时，则使用ground-truth作为人脸框。
    #             aligned_bboxes, aligned_images = self.align_face(image, bboxes, align_size, margin, center_det=True,
    #                                                              detect_multiple_faces=True)
    #         elif self.task == self.TASK_FACESCRUB_DETECT:
    #             # FaceScrub没有ground-truth, 暂且每张图片只检测一个最大的人脸
    #             aligned_bboxes, aligned_images = self.align_face(image, bboxes, align_size, margin, center_det=False,
    #                                                              detect_multiple_faces=False)
    #         elif self.task == self.TASK_MASKFACE_DETECT:
    #             aligned_bboxes, aligned_images = self.align_face(image, bboxes, align_size, margin, center_det=False,
    #                                                              detect_multiple_faces=True,
    #                                                              detect_radius_factor=0.357)
    #         elif self.task == self.TASK_TRILLION_PAIRS:
    #             aligned_bboxes, aligned_images = self.align_face(image, bboxes, align_size, margin, center_det=True,
    #                                                              detect_multiple_faces=False,
    #                                                              detect_radius_factor=None)
    #         else:
    #             aligned_bboxes, aligned_images = self.align_face(image, bboxes, align_size, margin, center_det=False,
    #                                                              detect_multiple_faces=False,
    #                                                              detect_radius_factor=0.357)
    #
    #         return aligned_bboxes, aligned_images
    #     except Exception as err:
    #         print('[align_image]: Exception.ERROR: %s' % err)
    #         return None, None
    #
    # def align(self, image_path, align_size=(160, 160), margin=32, min_face_area=0, remove_inner_face=True):
    #     '''
    #     Align face by image_path.
    #     :param image_path:
    #     :param align_size: same as align_face
    #     :param margin: same as align_face
    #     :return:
    #     '''
    #     try:
    #         img = self.imread(image_path)
    #         aligned_bboxes, aligned_images = self.align_image(img, align_size=align_size, margin=margin,
    #                                                           min_face_area=min_face_area,
    #                                                           remove_inner_face=remove_inner_face)
    #
    #         return aligned_bboxes, aligned_images
    #     except Exception as err:
    #         print('[align]: Exception.ERROR: %s' % err)
    #         return None, None


"""
def face_crop(image, bbox):
    tools.imcrop(image, bbox, scale_ratio=2.0)


def batch_crop(face_detector, data_path, images_path, save_path):
    '''
    :param face_detector:
    :param data_path:
    :param images_path:
    :param save_path:
    :return:
    '''
    for i, impath in enumerate(images_path):
        iminfo = impath.replace(data_path, '')
        iminfo = iminfo[1:] if iminfo[0] == '/' else iminfo
        img_savepath = os.path.join(save_path, iminfo)
        img_savedir, img_name = img_savepath.rsplit('/', 1)
        if not os.path.exists(img_savedir):
            os.makedirs(img_savedir)

        bboxes = face_detector.detect(impath, remove_inner_face=False)
        if bboxes is not None:
            img_name = img_name.rsplit('.', 1)
            image = cv2.imread(impath)
            for j, bbox in enumerate(bboxes):
                face = tools.imcrop(image, bbox)
                imname = img_name[0] + '_' + str(j) + '.' + img_name[-1]
                img_savepath = os.path.join(img_savedir, imname)
                cv2.imwrite(img_savepath, face)
        tools.view_bar('batch croping: ', i+1, len(images_path))
    print('')


def _generate_aligned_path(images_info, data_path, save_path, subdir='', suffix=''):
    if len(suffix) > 0:
        assert '.' == suffix[0]

    images_path = []
    aligned_images_path = []
    for i, info in enumerate(images_info):
        person_name, imname = info
        person_path = os.path.join(save_path, person_name)
        if not os.path.exists(person_path):
            os.makedirs(person_path)
        if len(subdir) > 0:
            person_path = os.path.join(person_path, subdir)
            if not os.path.exists(person_path):
                os.makedirs(person_path)

        image_path = os.path.join(data_path, person_name, imname)
        aligned_image_path = os.path.join(save_path, person_name, imname)
        if len(suffix) > 0:
            aligned_image_path = tools.sufwith(aligned_image_path, suffix)

        images_path.append(image_path)
        aligned_images_path.append(aligned_image_path)

        tools.view_bar('make aligned dir: ', i + 1, len(images_info))
    print('')

    return images_path, aligned_images_path


def _del_perperson_min_nimage(images_path, subdir='', min_nimg_pclass4del=0):
    aligned_persons = os.listdir(images_path)
    for person in aligned_persons:
        num_images_per_person = 0
        if len(subdir) > 0:
            person_path = os.path.join(images_path, person, subdir)
            if os.path.exists(person_path):
                num_images_per_person += len(os.listdir(person_path))
        person_path = os.path.join(images_path, person)
        num_images_per_person += len(os.listdir(person_path))

        if num_images_per_person < min_nimg_pclass4del:
            print('[_del_perperson_min_nimage]:: num_images_per_person:{} < min_nimg_pclass4del:{}!'.format(
                num_images_per_person, min_nimg_pclass4del))
            shutil.rmtree(person_path)


def batch_align(face_detector, data_path, images_info, save_path, align_size=(160, 160, 3), margin=32, subdir='',
                aligned_override=False, aligned_min_nimg_pclass4del=0, min_face_area=2500, detect_multiface=True):
    '''
    :param face_detector:
    :param images_info: 源数据信息
    :param save_path: 对齐图片保存路径
    :param subdir: 子目录名
    :param aligned_override: 为 False 时:如果对齐的图片已经存在，则直接跳过而不再进行对齐操作。
    :param aligned_min_nimg_pclass4del: 小于等于0时该值无效；大于0时，则判断对齐后的每个人图片总量是否小鱼该值，如果是则删除该人的对齐目录。
    :return:
    '''
    print('[batch_align]:: generate aligned path!')
    images_path, aligned_images_path = _generate_aligned_path(images_info, data_path, save_path, subdir=subdir,
                                                              suffix='.png')

    aligned_none_count = 0
    aligned_tosamll_count = 0
    for i, (impath, aligned_impath) in enumerate(zip(images_path, aligned_images_path)):
        if (not aligned_override) and os.path.exists(aligned_impath):
            print('{} exists!'.format(aligned_impath))
            continue

        aligned_bboxes, aligned_images = face_detector.align(impath, align_size, margin=margin,
                                                             min_face_area=min_face_area, remove_inner_face=False)
        if aligned_bboxes is None or aligned_images is None:
            aligned_none_count += 1
            if False:
                print('[batch_align]:: align error, delete!')
                os.remove(impath)
            continue
        if len(aligned_images) == 0:
            aligned_tosamll_count += 1
            if False:
                print('[batch_align]:: face too small, delete!')
                os.remove(impath)
            continue

        if detect_multiface:
            aligned_impath = aligned_impath.rsplit('.', 1)
            for j, img in enumerate(aligned_images):
                aligned_impath = aligned_impath[0] + '_' + str(j) + '.' + aligned_impath[-1]
        else:
            misc.imsave(aligned_impath, aligned_images[0])

        if i % 100 == 0:
            print('[batch_align]:: aligned_none_count={}, aligned_tosamll_count={}'.format(aligned_none_count,
                                                                                           aligned_tosamll_count))
            tools.view_bar('aligning: ', i + 1, len(images_path))
    print('')

    if aligned_min_nimg_pclass4del > 0:
        print('[batch_align]:: min_nimg_pclass4del!')
        _del_perperson_min_nimage(save_path, subdir=subdir, min_nimg_pclass4del=aligned_min_nimg_pclass4del)


def batch_output_keypoint(face_detector, data_path, images_info, save_path, center_det=True, fixed_imsize=True):
    if fixed_imsize:
        person_name, imname = images_info[0]
        impath = os.path.join(data_path, person_name, imname)
        image = cv2.imread(impath)
        imshape = image.shape
    else:
        raise Exception('不是固定的图片尺寸那您应该读取每张图片，然后取出其image.shape')

    processed_handle = open(save_path, 'a')

    for i, info in enumerate(images_info):
        try:
            person_name, imname = info
            impath = os.path.join(data_path, person_name, imname)

            bboxes, points = face_detector.detect(impath)

            if len(bboxes) > 0:
                # points = points.reshape((2, 5)).transpose().astype(np.int)
                info.extend(points.flatten().tolist())
                points = tools.strcat(info)
                processed_handle.write(points)
                processed_handle.write('\n')
                processed_handle.flush()
            else:
                print('\n[face_detection.batch_output_keypoint]:: {} no detected face!'.format(impath))
        except Exception as e:
            print(e)

        tools.view_bar('[face_detection.batch_output_keypoint]:: loading: ', i + 1, len(images_info))
    print('')
    processed_handle.close()


def draw_image(image_path, bboxes):
    import cv2
    image = cv2.imread(image_path)
    img_size = np.array(image.shape[0:2])
    y1, x1 = img_size / 4
    y2, x2 = 3 * img_size / 4
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 5))
    cy, cx = img_size / 2
    rh, rw = img_size / 2.8
    cv2.circle(image, (int(cx), int(cy)), int(rw), (255, 0, 0))
    cv2.circle(image, (int(cx), int(cy)), int(rh), (0, 255, 0))

    cv2.imshow('show', image)
    cv2.waitKey(0)

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 5))
        cv2.imshow('show', image)
        cv2.waitKey(0)

    cv2.imwrite('debug_example.jpg', image)



if __name__ == '__main__tttt':  # 一般数据集的批量对齐
    # data_path = "/home/xiajun/res/face/MegaFace/FaceScrub/FaceScrub"
    # aligned_save_path = "/home/xiajun/res/face/MegaFace/FaceScrub/FaceScrub-mtcnn_align160x160_margin32"
    # data_path = "/disk1/home/xiaj/res/face/MegaFace/FaceScrub/FaceScrub"
    # aligned_save_path = "/disk1/home/xiaj/res/face/MegaFace/FaceScrub/FaceScrub-mtcnn_align160x160_margin32"

    # data_path = '/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned'
    # aligned_save_path = "/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32-tmptest"

    # data_path = '/disk2/res/CASIA-FaceV5/CASIA-FaceV5-000-499'
    # aligned_save_path = "/disk2/res/CASIA-FaceV5/CASIA-FaceV5-000-499-mtcnn_align160x160_margin32"

    # data_path = '/disk1/home/xiaj/res/face/maskface/MAFA/tmp'
    # aligned_save_path = "/disk1/home/xiaj/res/face/maskface/Experiment/MAFA-train-images-mtcnn_align182x182_margin44_tmp"

    data_path = '/disk2/res/face/Trillion Pairs/train_msra/msra'
    aligned_save_path = '/disk2/res/face/Trillion Pairs/Experiment/msra_retinaface_align182x182_margin44'

    align_size = (182, 182)
    margin = 44
    fdetector = FaceDetector(detector=FaceDetector.DETECTOR_RETINAFACE, task=FaceDetector.TASK_TRILLION_PAIRS)

    # subdir = 'clean'
    subdir = ''
    images_info = tools.load_image(data_path, subdir=subdir, min_num_image_per_class=1, del_under_min_num_class=False)
    print('len(images_info)={}'.format(len(images_info)))
    batch_align(fdetector, data_path, images_info, aligned_save_path, align_size=align_size, margin=margin,
                subdir=subdir, min_face_area=0, detect_multiface=False)


def load_megaface(data_path):
    label_count = 0
    images_path = []
    images_label = []
    images = os.listdir(data_path)
    images.sort()
    if DEBUG:
        images = images[0:500]
    for i, image in enumerate(images):
        cls_path = os.path.join(data_path, image)
        if os.path.isfile(cls_path):
            print('[load_dataset]:: {} is not dir!'.format(cls_path))
            continue

        imgs = os.listdir(cls_path)

        imgs_info = []
        for img in imgs:
            if img.endswith('.json'):
                continue
            imgs_info.append((image, img))

        images_path.extend(imgs)
        images_label.extend([label_count] * len(imgs))
        label_count += 1
        tools.view_bar('loading: ', i + 1, len(images))
    print('')


if __name__ == '__main__MegaFace数据集的对齐':  # MegaFace数据集的对齐
    def megaface_filter1(imgs):
        images = []
        for img in imgs:
            if img.endswith('.json'):
                continue
            images.append(img)

        return images


    def megaface_filter2(image_path):
        if image_path.endswith('.json'):
            return False
        else:
            return True


    data_path = "/disk1/home/xiaj/res/face/MegaFace/megafacedata/FlickrFinal2"
    aligned_save_path = "/disk1/home/xiaj/res/face/MegaFace/megafacedata/Experiment/mtcnn_align160x160_margin32"
    # fdetector = FaceDetector()
    images_info = tools.load_image(data_path, filter_cb=megaface_filter2)
    images_info, images_label = load_megaface(data_path)

    print('len(images_info)={}'.format(len(images_info)))
    batch_align(fdetector, data_path, images_info, aligned_save_path)

if __name__ == '__main__5':  # batch_output_keypoint
    # data_path = "/home/xiajun/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44"
    # save_path = "/home/xiajun/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44_keypoints.csv"

    # data_path = "/disk1/home/xiaj/res/face/GC-WebFace/Experiment/mtcnn_align182x182_margin44_happyjuzi_mainland_cleaning"
    # save_path = "/disk1/home/xiaj/res/face/GC-WebFace/Experiment/happyjuzi_mainland_cleaning_mtcnn_align182x182_margin44_keypoints.csv"

    # data_path = "/disk1/home/xiaj/res/face/GC-WebFace/Experiment/mtcnn_align182x182_margin44_happyjuzi_HongkongTaiwan_cleaning"
    # save_path = "/disk1/home/xiaj/res/face/GC-WebFace/Experiment/happyjuzi_HongkongTaiwan_cleaning_mtcnn_align182x182_margin44_keypoints.csv"

    # data_path = "/disk1/home/xiaj/res/face/GC-WebFace/Experiment/mtcnn_align182x182_margin44_happyjuzi_JapanKorea_cleaning"
    # save_path = "/disk1/home/xiaj/res/face/GC-WebFace/Experiment/happyjuzi_JapanKorea_cleaning_mtcnn_align182x182_margin44_keypoints.csv"

    data_path = '/disk2/res/CASIA-FaceV5/CASIA-FaceV5-000-499-mtcnn_align182x182_margin44'
    save_path = "/disk2/res/CASIA-FaceV5/CASIA-FaceV5-000-499-mtcnn_align182x182_margin44_mtcnn_align182x182_margin44_keypoints.csv"

    fdetector = FaceDetector()

    subdir = 'clean'
    images_info = tools.load_image(data_path, subdir=subdir, min_num_image_per_class=1, del_under_min_num_class=False,
                                   # min_area4del=2500
                                   )

    batch_output_keypoint(fdetector, data_path, images_info, save_path)


def glass_pad(image, points, prob=0.5):
    index = np.random.randint(0, 100)
    if index < prob * 100:
        offset = np.random.randint(6, 12)
        # offset = 15
        x1, y1 = points[0][0], points[0][1]
        x2, y2 = points[1][0], points[1][1]

        dist = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2)) / (offset + 10)
        _y1 = y1 + (y1 - y2) / dist
        _x1 = x1 - (x2 - x1) / dist
        x2 = x2 + x1 - _x1
        y2 = y2 - (_y1 - y1)
        x1, y1 = _x1, _y1

        fraction = (x2 - x1)
        if fraction == 0:
            x10 = x20 = x1 - offset
            y10, y20 = y1, y2
            x11 = x21 = x1 + offset
            y11, y21 = y1, y2
        else:
            x10 = x1 - (y1 - y2) * offset / fraction
            y10 = y1 - offset
            x11 = x10 + 2 * (x1 - x10)
            y11 = y1 + offset

            x20 = x2 - (y1 - y2) * offset / fraction
            y20 = y2 - offset
            x21 = x20 + 2 * (x2 - x20)
            y21 = y2 + offset

        pt1 = int(round(x10)), int(round(y10))
        pt2 = int(round(x11)), int(round(y11))
        pt3 = int(round(x20)), int(round(y20))
        pt4 = int(round(x21)), int(round(y21))

        pts = np.array([[pt1, pt2, pt4, pt3]], dtype=np.int32)
        # cv2.polylines(image, pts, isClosed=False, color=(128, 128, 128))
        cv2.fillPoly(image, pts, (127.5, 127.5, 127.5))

        if DEBUG:
            image = image.astype(np.uint8)
            centerpt1 = points[0].astype(np.int)
            centerpt2 = points[1].astype(np.int)

            cv2.circle(image, tuple(centerpt1), 5, (255, 0, 0), thickness=-1)
            cv2.circle(image, tuple(centerpt2), 5, (0, 255, 0), thickness=-1)
            # cv2.ellipse(image, tuple(centerpt2), (20, 10), 0, 0, 360, (255, 255, 255), 3)

            cv2.circle(image, tuple(pt1), 3, (255, 0, 0), thickness=-1)
            cv2.circle(image, tuple(pt2), 3, (0, 255, 0), thickness=-1)
            cv2.circle(image, tuple(pt3), 3, (0, 0, 255), thickness=-1)
            cv2.circle(image, tuple(pt4), 3, (255, 255, 0), thickness=-1)

    # return image


if __name__ == '__main__pad_glass':
    data_path = "/home/xiajun/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44"
    # image_path = '/home/xiajun/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44/n000020/0160_01.png'  # [[44, 79], [92, 83], [65, 105], [41, 116], [87, 121]]
    image_path = '/home/xiajun/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44/n000020/0016_01.png'
    fdetector = FaceDetector()

    persons = os.listdir(data_path)

    for person in persons:
        person_path = os.path.join(data_path, person)
        images = os.listdir(person_path)
        images = images[0::5]
        for imname in images:
            try:
                impath = os.path.join(person_path, imname)
                # impath = '/home/xiajun/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44/n000097/0154_01.png'

                bboxes, points = fdetector.detect(impath)
                if len(bboxes) == 1:
                    points = points.reshape((2, 5)).transpose()
                    points = points[:2, :]
                    image = cv2.imread(impath).astype(np.float32)

                    glass_pad(image, points)

                    if DEBUG:
                        cv2.imshow('show', image.astype(np.uint8))
                        cv2.waitKey(0)
                    print('debug')
            except Exception as e:
                print(e)

"""
