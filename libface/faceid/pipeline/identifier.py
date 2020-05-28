# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
# import argparse
from libface.faceid.pipeline import facenet
# import sys
from tensorflow.python.ops import data_flow_ops
# import pickle
from libml.utils import tools


class FaceID:
    def __init__(self, model_path=None, specify_ckpt=None):
        '''
        :param model_path:
        '''
        if model_path is None:
            file_root_path = os.path.abspath(os.path.dirname(__file__))
            model_path = os.path.join(file_root_path, 'save_model', '20180402-114759')

            # 本机
            model_path = '/disk1/home/xiaj/dev/alg_verify/face/facenet/pretrained_model/20180402-114759/20180402-114759.pb'

        # with tf.Graph().as_default():  # 这一行可以去掉
        # with tf.Session() as sess:
        self.graph = tf.Graph()
        with self.graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
            config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=self.graph)
            np.random.seed(seed=666)

            self.image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
            self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
            self.batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            self.control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')
            self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            nrof_preprocess_threads = 4
            image_nrom_size = 160
            image_size = (image_nrom_size, image_nrom_size)
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                                       dtypes=[tf.string, tf.int32, tf.int32],
                                                       shapes=[(1,), (1,), (1,)],
                                                       shared_name=None, name=None)
            self.eval_enqueue_op = eval_input_queue.enqueue_many(
                [self.image_paths_placeholder, self.labels_placeholder, self.control_placeholder],
                name='eval_enqueue_op')
            image_batch, self.label_batch = facenet.create_input_pipeline(eval_input_queue, image_size,
                                                                          nrof_preprocess_threads,
                                                                          self.batch_size_placeholder)

            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': self.label_batch,
                         'phase_train': self.phase_train_placeholder}
            facenet.load_model(model_path, input_map=input_map, sess=self.sess, specify_ckpt=specify_ckpt)

            # Get output tensor
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=self.sess)

            self.use_flipped_images = True
            self.lfw_batch_size = 10
            self.lfw_nrof_folds = 10
            self.distance_metric = 1
            self.subtract_mean = True

        print('[faceid_pipeline.FaceID.__init__]:: successed!!!')

    def __del__(self):
        if self.sess is not None:
            self.sess.close()
        print('[faceid_pipeline.FaceID.__del__]')

    def embedding(self, image_paths, batch_size=100, use_fixed_image_standardization=True, use_flipped_images=True,
                  random_rotate=True, random_crop=True, random_flip=True, fixed_contract=False):
        '''
        :param image_paths:
        :param use_flipped_images: use flipped image for extend embeddings.
        :param random_rotate:
        :param random_crop:
        :param random_flip: different form use_flipped_iamges
        :return:
        '''
        nrof_embeddings = len(image_paths)
        nrof_flips = 2 if use_flipped_images else 1
        nrof_images = nrof_embeddings * nrof_flips
        labels_array = np.expand_dims(np.arange(0, nrof_images), 1)
        image_paths_array = np.expand_dims(np.repeat(np.array(image_paths), nrof_flips), 1)

        control_value = facenet.RANDOM_ROTATE * random_rotate + \
                        facenet.RANDOM_CROP * random_crop + \
                        facenet.RANDOM_FLIP * random_flip + \
                        facenet.FIXED_CONTRACT * fixed_contract + \
                        facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
        control_array = np.ones_like(labels_array) * control_value

        if use_flipped_images:
            # Flip every second image
            control_array += (labels_array % 2) * facenet.FLIP

        self.sess.run(self.eval_enqueue_op, {self.image_paths_placeholder: image_paths_array, self.labels_placeholder: labels_array,
                                        self.control_placeholder: control_array})

        embedding_size = int(self.embeddings.get_shape()[1])
        nrof_batches = nrof_images // batch_size + (0 if nrof_images % batch_size == 0 else 1)
        if nrof_images < batch_size:
            batch_size = nrof_images

        emb_array = np.zeros((nrof_images, embedding_size))
        lab_array = np.zeros((nrof_images,))
        #print('[extract_embeddings]: nrof_embeddings={}, nrof_images={}, batch_size={}, nrof_batches={}'.format(nrof_embeddings, nrof_images, batch_size, nrof_batches))
        for i in range(nrof_batches):
            if i+1 == nrof_batches:
                batch_size = nrof_images - i*batch_size
            feed_dict = {self.phase_train_placeholder: False, self.batch_size_placeholder: batch_size}
            emb, lab = self.sess.run([self.embeddings, self.label_batch], feed_dict=feed_dict)
            lab_array[lab] = lab
            emb_array[lab, :] = emb
            tools.view_bar('Computing embedding: ', i + 1, nrof_batches)
        print('')
        embeddings = np.zeros((nrof_embeddings, embedding_size * nrof_flips))
        if use_flipped_images:
            # Concatenate embeddings for flipped and non flipped version of the images
            embeddings[:, :embedding_size] = emb_array[0::2, :]  # 取出所有未flip的
            embeddings[:, embedding_size:] = emb_array[1::2, :]  # 取出所有flip的
        else:
            embeddings = emb_array

        return embeddings

#
# def extract_embeddings_from_csv(netmodel, data_file, pkl_save_file, build_mode='pairs', shuffle=True):
#     '''
#     :param netmodel:
#     :param data_file:
#     :param pkl_save_file:
#     :param build_mode: reference data_split.load_image_label function
#     :return:
#     '''
#     from utils import data_split
#
#     image_paths, labels, names = data_split.load_image_label(data_file, build_mode=build_mode, shuffle=shuffle)
#     embeddings = netmodel.embedding(images_path, use_fixed_image_standardization=True, use_flipped_images=True,
#                                     random_rotate=False, random_crop=False, random_flip=False, fixed_contract=False)
#
#     print('save pkl path: ', pkl_save_file)
#     with open(pkl_save_file, 'wb') as f:
#         pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)
#         pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)
#         pickle.dump(names, f, pickle.HIGHEST_PROTOCOL)
#
#
# def visualize_embeddings(netmodel, data_file, pkl_save_file, build_mode='pairs', shuffle=True):
#     '''
#     :param netmodel:
#     :param data_file:
#     :param pkl_save_file:
#     :param build_mode: reference data_split.load_image_label function
#     :return:
#     '''
#     from utils import data_split
#
#     image_paths, labels, names = data_split.load_image_label(data_file, build_mode=build_mode, shuffle=shuffle)
#     embeddings = netmodel.embedding(images_path, use_fixed_image_standardization=True, use_flipped_images=True,
#                                     random_rotate=False, random_crop=False, random_flip=False, fixed_contract=False)
#
#     print('save pkl path: ', pkl_save_file)
#     with open(pkl_save_file, 'wb') as f:
#         pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)
#         pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)
#         pickle.dump(names, f, pickle.HIGHEST_PROTOCOL)
#
#
# def load_data(path_pkl):
#     with open(path_pkl, 'rb') as f:
#         embeddings = pickle.load(f)
#         labels = pickle.load(f)
#         names = pickle.load(f)
#         return embeddings, labels, names
#
#
# def parse_arguments(argv):
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('model', type=str,
#                         help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
#
#     parser.add_argument('--specify_ckpt', type=str,
#                         help='images path and label for validation.')
#
#     # 指定一个csv文件，该文件内容是图片路径和对应的label.
#     parser.add_argument('--image_csvfile', type=str,
#                         help='images path and label for validation.')
#     parser.add_argument('--build_mode', type=str,
#                         help='dataset build mode, example: pairs, path_label. reference data_split.load_image_label.')
#     # 将提取的特征存如该文件
#     parser.add_argument('--save_file', type=str,
#                         help='Path to the pkl data save directory.')
#     parser.add_argument('--shuffle',
#                         help='shuffle.', action='store_true')
#     return parser.parse_args(argv)
#
#
# if __name__ == '__main__1':  # 从csv文件提取特征，并保存到pkl文件
#     sys.argv = ['faceid_pipeline.py',
#                 '/home/xiajun/dev/facerec/facenet/mydataset/models/20180402-114759',
#                 '--image_csvfile', '/home/xiajun/dev/FlaskFace/face_verification/data/gcface_mtcnn_align160x160_margin32/test_database/database_set.csv',
#                 '--build_mode', 'path_label',
#                 '--save_file', '/home/xiajun/dev/FlaskFace/face_verification/data/gcface_mtcnn_align160x160_margin32/test_database/database_set.pkl',
#                 '--shuffle',
#                 ]
#
#     print(sys.argv)
#
#     """
#     save_pkl = ''
#     embeddings, actual_issame = load_data(save_pkl)
#     print(embeddings.shape, len(actual_issame))
#     print(embeddings, actual_issame)
#     """
#
#     args = parse_arguments(sys.argv[1:])
#     print(args)
#     model_path = args.model
#     image_csvfile = args.image_csvfile
#     build_mode = args.build_mode
#     pkl_save_file = args.save_file
#
#     netmodel = FaceID(model_path=model_path)
#
#     extract_embeddings_from_csv(netmodel, image_csvfile, pkl_save_file, build_mode=build_mode)
#
#
# from utils import dataset as Datset
#
# if __name__ == '__main__':
#     sys.argv = ['faceid_pipeline.py',
#                 '/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace/20191127-005344',
#                 '--specify_ckpt', 'model-20191127-005344-acc0.996167-val0.993333-prec0.998671.ckpt-341',
#                 ]
#
#     print(sys.argv)
#
#     """
#     save_pkl = ''
#     embeddings, actual_issame = load_data(save_pkl)
#     print(embeddings.shape, len(actual_issame))
#     print(embeddings, actual_issame)
#     """
#
#     data_path = '/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align160x160_margin32'
#
#     args = parse_arguments(sys.argv[1:])
#     print(args)
#     model_path = args.model
#     specify_ckpt = args.specify_ckpt
#     pkl_save_file = data_path + '.pkl'
#
#     netmodel = FaceID(model_path=model_path, specify_ckpt=specify_ckpt)
#
#     images_path, images_label = Datset.load_dataset(data_path, shuffle=True, min_nrof_cls=1)
#     embeddings = netmodel.embedding(images_path, use_fixed_image_standardization=True, use_flipped_images=True,
#                                     random_rotate=False, random_crop=False, random_flip=False, fixed_contract=False)
#
#     print('save pkl path: ', pkl_save_file)
#     with open(pkl_save_file, 'wb') as f:
#         pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)
#         pickle.dump(images_path, f, pickle.HIGHEST_PROTOCOL)
#         pickle.dump(images_label, f, pickle.HIGHEST_PROTOCOL)
#
#     print('end!')

