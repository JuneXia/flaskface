"""Functions for building the face recognition network.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# pylint: disable=missing-docstring
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile
import math
from six import iteritems
import cv2


RELEASE = True

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss
  
def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def shuffle_examples(image_paths, labels):
    shuffle_list = list(zip(image_paths, labels))
    random.shuffle(shuffle_list)
    image_paths_shuff, labels_shuff = zip(*shuffle_list)
    return image_paths_shuff, labels_shuff

def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')



def glass_pad(image, points, prob=0.5):
    index = np.random.randint(0, 100)
    if index < prob*100:
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

        if False:
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


g_FaceEyePoints = {}
def loading_face_keypoint():
    global g_FaceEyePoints

    try:
        if RELEASE:
            keypoints_file = '/home/yp/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44_keypoints.csv'
        else:
            keypoints_file = '/home/xiajun/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44_keypoints.csv'
        with open(keypoints_file, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip().split(',')
            key = os.path.join(line[0], line[1])
            points = np.array(line[2:12]).astype(np.float)
            points = points.reshape((2, 5)).transpose()
            points = points[:2, :]
            g_FaceEyePoints[key] = points

            print('[loading_face_keypoint]:: {}/{}'.format(i, len(lines)))
    except Exception as e:
        print(e)

def random_glass_padding(image, filename):
    global g_FaceEyePoints
    global g_count

    info = filename.decode('utf-8').split('/')[-2:]
    key = os.path.join(info[0], info[1])
    # print('filename: {}, info: {}, key: {}'.format(filename, info, key))
    if key in g_FaceEyePoints.keys():
        points = g_FaceEyePoints[key]
        glass_pad(image, points, 0.12)
    else:
        # points = np.array([[50, 70], [120, 80]])
        # glass_pad(image, points)
        pass

    return image


def random_color_failed(image):
    prob = 0.0
    random_index = np.random.randint(0, 101)  # TODO：100 or 101
    print('[random_color]:: random_index={}, type(random_index)={}, image.shape={}'.format(random_index, type(random_index), image.shape))
    if random_index < prob * 100:
        index = np.random.randint(0, 4)
        print('[random_color]:: index={}, type(index)={}'.format(index, type(index)))
        if index == 0:
            image = tf.image.random_brightness(image, 0.4)
        elif index == 1:
            image = tf.image.random_contrast(image, 0.8, 2)
        elif index == 2:
            image = tf.image.random_hue(image, 0.08)
        elif index == 3:
            image = tf.image.random_saturation(image, 0, 1)

    print('[random_color]:: image.shape={}'.format(image.shape))
    return image


def random_contract_failed(image):
    raw_image_size = 182
    contract_size = 132

    central_crop = tf.image.central_crop(image, contract_size / raw_image_size)

    # 随机的稍微放大一点
    h = w = tf.random_uniform([], contract_size, raw_image_size+1, dtype=tf.int32)
    resize_image = tf.image.resize_image_with_pad(central_crop, h, w)
    # resize_image = tf.cast(resize_image, dtype=tf.uint8)

    # 填充至目标大小
    pad_image = tf.image.resize_image_with_crop_or_pad(resize_image, raw_image_size, raw_image_size)

    # random_crop = tf.random_crop(pad_image, (160, 160) + (3,))
    return pad_image


def random_color(image, control):
    randcolor = tf.random_uniform([], 0, 7, dtype=tf.int32)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ctrl_bright = tf.cond(get_control_flag(control, RANDOM_COLOR),
                          lambda: tf.cond(tf.equal(randcolor, 0), lambda: True, lambda: False),
                          lambda: False)
    image = tf.cond(ctrl_bright,
                    lambda: tf.image.random_brightness(image, 0.4),
                    lambda: tf.identity(image))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ctrl_contrast = tf.cond(get_control_flag(control, RANDOM_COLOR),
                            lambda: tf.cond(tf.equal(randcolor, 1), lambda: True, lambda: False),
                            lambda: False)
    image = tf.cond(ctrl_contrast,
                    lambda: tf.image.random_contrast(image, 0.8, 2),
                    lambda: tf.identity(image))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ctrl_hue = tf.cond(get_control_flag(control, RANDOM_COLOR),
                       lambda: tf.cond(tf.equal(randcolor, 2), lambda: True, lambda: False),
                       lambda: False)
    image = tf.cond(ctrl_hue,
                    lambda: tf.image.random_hue(image, 0.08),
                    lambda: tf.identity(image))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ctrl_saturation = tf.cond(get_control_flag(control, RANDOM_COLOR),
                              lambda: tf.cond(tf.equal(randcolor, 3), lambda: True, lambda: False),
                              lambda: False)
    image = tf.cond(ctrl_saturation,
                    lambda: tf.image.random_saturation(image, 0, 1),
                    lambda: tf.identity(image))

    return image


def fixed_contract(image, control, raw_size=(182, 182)):
    if raw_size[0] != raw_size[1]:
        raise Exception('At present, the image width and height are equal. \n\
        If the width and height are not equal, part of the code of this function needs to be modified. \n\
        If you have known this idea, you can screen this exception.')

    contract_size = 132

    '''
    central_crop = tf.image.central_crop(image, contract_size / raw_image_size)
    print(
        '6 type(central_crop)={}, central_crop.shape={}, central_crop={}'.format(type(central_crop), central_crop.shape,
                                                                                 central_crop))

    # 随机的稍微放大一点
    h = w = tf.random_uniform([], contract_size, raw_image_size + 1, dtype=tf.int32)
    resize_image = tf.image.resize_image_with_pad(central_crop, h, w)
    # resize_image = tf.cast(resize_image, dtype=tf.uint8)

    # 填充至目标大小
    image = tf.image.resize_image_with_crop_or_pad(resize_image, raw_image_size, raw_image_size)
    '''

    # contract_size = tf.random_uniform([], 124, 144, dtype=tf.float64)
    image = tf.cond(get_control_flag(control, FIXED_CONTRACT),
                    # lambda: tf.image.central_crop(image, tf.divide(contract_size, raw_image_size)),  # 随机中心裁剪s
                    lambda: tf.image.central_crop(tf.ensure_shape(image, (None, None, 3)), contract_size / raw_size[0]),  # 固定中心裁剪
                    lambda: tf.identity(image))

    image = tf.cond(get_control_flag(control, FIXED_CONTRACT),
                    lambda: tf.cast(tf.image.resize_image_with_pad(image, raw_size[0], raw_size[1]), tf.uint8),
                    lambda: tf.identity(image))

    return image
from tensorflow.python.ops import array_ops


# 1: Random rotate 2: Random crop  4: Random flip  8:  Fixed image standardization  16: Flip
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16
RANDOM_GLASS = 32
RANDOM_COLOR = 64
FIXED_CONTRACT = 128
def create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder):
    # raise Exception('Your train image size must be 182x182, If you have known this idea, you can screen this exception.')

    raw_size = (182, 182)

    images_and_labels_list = []
    for _ in range(nrof_preprocess_threads):
        filenames, label, control = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, 3)
            # image = tf.image.decode_png(file_contents, 3)

            image = tf.cond(get_control_flag(control[0], RANDOM_GLASS),
                            lambda: tf.ensure_shape(tf.py_func(random_glass_padding, [image, filename], tf.uint8), (None, None, 3)),
                            lambda: tf.identity(image))

            image = random_color(image, control[0])

            image = fixed_contract(image, control[0], raw_size)

            image = tf.cond(get_control_flag(control[0], RANDOM_ROTATE),
                            lambda: tf.py_func(random_rotate_image, [image], tf.uint8),
                            lambda: tf.identity(image))

            image = tf.cond(get_control_flag(control[0], RANDOM_CROP),
                            lambda: tf.random_crop(image, image_size + (3,)),
                            lambda: tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))

            image = tf.cond(get_control_flag(control[0], RANDOM_FLIP),
                            lambda: tf.image.random_flip_left_right(image),
                            lambda: tf.identity(image))

            image = tf.cond(get_control_flag(control[0], FIXED_STANDARDIZATION),
                            lambda: (tf.cast(image, tf.float32) - 127.5)/128.0,
                            lambda: tf.image.per_image_standardization(image))

            if False:
                image = tf.cast(image, tf.float32)

            image = tf.cond(get_control_flag(control[0], FLIP),
                            lambda: tf.image.flip_left_right(image),
                            lambda: tf.identity(image))

            #pylint: disable=no-member
            image.set_shape(image_size + (3,))
            images.append(image)
        images_and_labels_list.append([images, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels_list, batch_size=batch_size_placeholder, 
        shapes=[image_size + (3,), ()], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * 100,
        allow_smaller_final_batch=True)
    
    return image_batch, label_batch

def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)
  
def _add_loss_summaries(total_loss):
    """Add summaries for losses.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op

def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
    
        grads = opt.compute_gradients(total_loss, update_gradient_vars)
        
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
  
    return train_op

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        cv2.imwrite('srcimg.jpg', img[:, :, 0:3])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
            cv2.imwrite('prewhtiten_img.jpg', img[:, :, 0:3])
        img = crop(img, do_random_crop, image_size)
        cv2.imwrite('crop_img.jpg', img[:, :, 0:3])
        img = flip(img, do_random_flip)
        cv2.imwrite('flip_img.jpg', img[:, :, 0:3])
        images[i,:,:,:] = img[:,:,0:3]  # BY xj
    return images

def get_label_batch(label_data, batch_size, batch_index):
    nrof_examples = np.size(label_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = label_data[j:j+batch_size]
    else:
        x1 = label_data[j:nrof_examples]
        x2 = label_data[0:nrof_examples-j]
        batch = np.vstack([x1,x2])
    batch_int = batch.astype(np.int64)
    return batch_int

def get_batch(image_data, batch_size, batch_index):
    nrof_examples = np.size(image_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = image_data[j:j+batch_size,:,:,:]
    else:
        x1 = image_data[j:nrof_examples,:,:,:]
        x2 = image_data[0:nrof_examples-j,:,:,:]
        batch = np.vstack([x1,x2])
    batch_float = batch.astype(np.float32)
    return batch_float

def get_triplet_batch(triplets, batch_index, batch_size):
    ax, px, nx = triplets
    a = get_batch(ax, int(batch_size/3), batch_index)
    p = get_batch(px, int(batch_size/3), batch_index)
    n = get_batch(nx, int(batch_size/3), batch_index)
    batch = np.vstack([a, p, n])
    return batch

def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                if par[1]=='-':
                    lr = -1
                else:
                    lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    print(path_exp, path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths
  
def split_dataset(dataset, split_ratio, min_nrof_images_per_class, mode):
    if mode=='SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*(1-split_ratio)))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode=='SPLIT_IMAGES':
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            nrof_images_in_class = len(paths)
            split = int(math.floor(nrof_images_in_class*(1-split_ratio)))
            if split==nrof_images_in_class:
                split = nrof_images_in_class-1
            if split>=min_nrof_images_per_class and nrof_images_in_class-split >= 1:
                train_set.append(ImageClass(cls.name, paths[:split]))
                test_set.append(ImageClass(cls.name, paths[split:]))
            else:
                raise ValueError('TODO: what happened!' % mode)
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set

def load_model(model, input_map=None, sess=None, specify_ckpt=None):  # TODO: add sess for debug
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp, specify_ckpt)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        if sess is not None:
            saver.restore(sess, os.path.join(model_exp, ckpt_file))
        else:
            saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
    
def get_model_filenames(model_dir, specify_ckpt=None):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        if specify_ckpt is not None:
            for ckpt_file in ckpt.all_model_checkpoint_paths:
                if specify_ckpt in ckpt_file:
                    return meta_file, ckpt_file
            raise ValueError('model dir {} have no one specified ckpt file {}'.format(model_dir, specify_ckpt))
        elif ckpt.model_checkpoint_path:
            ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
            return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file
  
def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        if len(embeddings1.shape) == 1:
            embeddings1 = np.reshape(embeddings1, (1, -1))
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        # similarity = np.clip(similarity, -0.999999, 0.999999)
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
        print('best_threshold: ', thresholds[best_threshold_index])
          
        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


  
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
    
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
  
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def store_revision_info(src_path, output_dir, arg_string):
    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' +  e.strerror
  
    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' +  e.strerror
    
    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('tensorflow version: %s\n--------------------\n' % tf.__version__)  # @UndefinedVariable
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)

def list_variables(filename):
    reader = training.NewCheckpointReader(filename)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    return names

def put_images_on_grid(images, shape=(16,8)):
    nrof_images = images.shape[0]
    img_size = images.shape[1]
    bw = 3
    img = np.zeros((shape[1]*(img_size+bw)+bw, shape[0]*(img_size+bw)+bw, 3), np.float32)
    for i in range(shape[1]):
        x_start = i*(img_size+bw)+bw
        for j in range(shape[0]):
            img_index = i*shape[0]+j
            if img_index>=nrof_images:
                break
            y_start = j*(img_size+bw)+bw
            img[x_start:x_start+img_size, y_start:y_start+img_size, :] = images[img_index, :, :, :]
        if img_index>=nrof_images:
            break
    return img

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))
