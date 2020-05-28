#! /usr/bin/python
import math
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate


def softmax(x):
    shift_x = x # - np.max(x)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)


def sigmoid1(x, delta=0.0):
    alpha = 1
    if delta != 0:
        alpha = math.log((2-1.999)/1.999)/delta
    val = 2*(1/(1+np.exp(-alpha*(x-delta)))-0.5)
    return val


def sigmoid(x, delta=0.0):
    alpha = 1
    if delta != 0:
        alpha = math.log((1-0.999)/0.999)/delta
    val = 1/(1+np.exp(-alpha*(x-delta)))
    return val


def distance(embeddings1, embeddings2, distance_metric=0, is_dist_metric=True):
    """
    :param embeddings1:
    :param embeddings2:
    :param distance_metric:
    :param is_dist_metric: It's usefull for distance_metric=1.
    :return:
    """
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        if is_dist_metric:
            dist = np.arccos(similarity) / math.pi
        else:
            dist = similarity
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds = thresholds[np.where(acc_train == acc_train[best_threshold_index])]
        best_threshold = best_thresholds[len(best_thresholds)//2]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(best_threshold, dist[test_set],
                                                      actual_issame[test_set])
        print('[calculate_roc]: best_threshold: ', best_threshold)

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
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
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
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


def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far

def metric_calc(tp, fn, fp, tn):
    tar = len(tp) / (len(tp) + len(fn))
    far = len(fp) / (len(fp) + len(tn))
    precision = len(tp) / (len(tp) + len(fp))
    acc = (len(tp) + len(tn)) / (len(tp) + len(fn) + len(fp) + len(tn))

    return acc, tar, precision, far


class DistVerfication(object):
    def __init__(self, distance_metric=0, dist_threshold=1.25, is_dist_metric=True):
        self.distance_metric = distance_metric
        self.dist_threshold = dist_threshold
        self.is_dist_metric = is_dist_metric

    def __del__(self):
        print('[DistVerfication.__del__]')

    def verify(self, verify_embs, emb_array):
        subtract_mean = True
        if subtract_mean:
            mean = np.mean(np.concatenate([verify_embs, emb_array]), axis=0)
        else:
            mean = 0.0
        dist = distance(verify_embs-mean, emb_array-mean, self.distance_metric, is_dist_metric=self.is_dist_metric)

        #print('type(dist)=', type(dist))
        #print('dist.shape=', dist.shape)
        #print('dist', dist)

        return dist

    def identify(self, verify_embs, emb_array, use_max_simil=True):
        dist = self.verify(verify_embs, emb_array)
        if self.is_dist_metric:
            if use_max_simil:
                index = dist.argmin()
                prob = sigmoid(-dist[index], delta=-self.dist_threshold)
            else:
                raise Exception('暂时没有在欧式距离度量准则下使用过二次校验策略的。不过以后欧氏距离应该也不会再用到了！')
        else:
            if use_max_simil:
                index = dist.argmax()
                prob = dist[index]
            else:
                index = np.argsort(-dist)
                prob = dist[index]

        return index, prob

    def prob_metric(self, prob, labels, prob_threshold):
        positive_sample_prob = prob[np.where((prob * labels) != 0)]
        negative_sample_prob = prob[np.where((prob * np.logical_not(labels)) != 0)]

        tp = positive_sample_prob[np.where(positive_sample_prob >= prob_threshold)]
        fn = positive_sample_prob[np.where(positive_sample_prob < prob_threshold)]
        fp = negative_sample_prob[np.where(negative_sample_prob >= prob_threshold)]
        tn = negative_sample_prob[np.where(negative_sample_prob < prob_threshold)]

        acc, tar, precision, far = metric_calc(tp, fn, fp, tn)
        print('prob_metric: acc={}, tar={}, far={}, precision={}'.format(acc, tar, far, precision))

    def dist_metric(self, dist, labels, dist_threshold):
        positive_sample_dist = dist[np.where((dist * labels) != 0)]
        negative_sample_dist = dist[np.where((dist * np.logical_not(labels)) != 0)]

        tp = positive_sample_dist[np.where(positive_sample_dist <= dist_threshold)]
        fn = positive_sample_dist[np.where(positive_sample_dist > dist_threshold)]
        fp = negative_sample_dist[np.where(negative_sample_dist <= dist_threshold)]
        tn = negative_sample_dist[np.where(negative_sample_dist > dist_threshold)]

        acc, tar, precision, far = metric_calc(tp, fn, fp, tn)
        print('dist_metirc: acc={}, tar={}, far={}, precision={}'.format(acc, tar, far, precision))

    def metric(self, embs1, embs2, labels, prob_threshold=0.8):
        dist = self.verify(embs1, embs2)

        prob = sigmoid(-dist, delta=-self.dist_threshold)
        print('prob: ', prob)

        self.prob_metric(prob, labels, prob_threshold)
        self.dist_metric(dist, labels, self.dist_threshold)

    def prob_verify(self, embs1, embs2):
        dist = self.verify(embs1, embs2)

        prob = sigmoid(-dist, delta=-self.dist_threshold)

        return prob


# def pairs_onehot(names, base_labels, base_names):
#     issame_onehot_list = []
#     base_labels = np.array(base_labels).reshape(len(base_labels))
#     for name in names:
#         if name in base_names:
#             label = base_labels[base_names.index(name)]
#         else:
#             label = -1
#         labels = np.ones([len(base_labels)]) * label
#         issame_array = labels == base_labels
#         issame_onehot_list.extend((np.arange(2) == issame_array[:, None]).astype(np.float32))
#
#     return np.array(issame_onehot_list)
#
#
# def parse_arguments(argv):
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--mode', type=str, help='verify mode: database_verify, lfw_verify, ppl_verify(path_path_label_verify).')
#     parser.add_argument('--faceid_model', type=str, help='faceid model path.')
#     parser.add_argument('--verify_model', type=str, help='verify model path.')
#     parser.add_argument('--base_embeddings_file', type=str, help='Path to the database embeddings file.')
#     parser.add_argument('--model_save_path', type=str, help='model save path.')
#     parser.add_argument('--images_path', type=str, help='aligned images path.')
#     parser.add_argument('--pairs_file', type=str, help='pairs file path.')
#     parser.add_argument('--dist_threshold', default=1.0, type=float, help='ver dist threshold')
#     parser.add_argument('--prob_threshold', default=0.80, type=float, help='ver prob threshold')
#     parser.add_argument('--distance_metric', default=0, type=float, help='0: Euclidian distance, 1: Cosine')
#     return parser.parse_args(argv)
#
#
# def database1vsN(verify_names, verify_impath, base_embeddings_file, faceid_model_path, distance_metric=0, dist_threshold=1.0):
#     base_embeddings, base_labels, base_names = faceid_pipeline.load_data(base_embeddings_file)
#     print('base_labels: ', base_labels)
#     print('base_names: ', base_names)
#
#     ""
#     labels = pairs_onehot(verify_names, base_labels, base_names)
#     faceid_model = faceid_pipeline.FaceID(faceid_model_path)
#     embeddings = faceid_model.embedding(verify_impath)
#     faceid_model.__del__()
#     ""
#
#     verify_model = DistVerfication(distance_metric=distance_metric, dist_threshold=dist_threshold)
#     index, prob = verify_model.identify(embeddings, base_embeddings)
#
#     print('database1vsN: base_names[{}]={}, prob={}'.format(index, base_names[index], prob))
#
#
# def verfiy1vs1(image_paths, actual_issame, faceid_model_path, distance_metric=0, dist_threshold=1.0, prob_threshold=0.8):
#     print('[verfiy1vs1]: test pairs num={}'.format(len(actual_issame)))
#
#     faceid_model = faceid_pipeline.FaceID(faceid_model_path)
#     embeddings = faceid_model.embedding(image_paths, random_rotate=False, random_crop=False, random_flip=False)
#     faceid_model.__del__()
#
#     embs1 = embeddings[0::2, :]
#     embs2 = embeddings[1::2, :]
#     labels = np.array(actual_issame) + 0
#
#     tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=distance_metric, subtract_mean=True)
#     print('accuracy=', accuracy)
#     print('val={}, val_std={}, far={}'.format(val, val_std, far))
#
#     verify_model = DistVerfication(distance_metric=distance_metric, dist_threshold=dist_threshold)
#     dist = verify_model.metric(embs1, embs2, labels, prob_threshold=prob_threshold)
#
#     if False:
#         with open('CosdistMetric_LFW.csv', "w") as f:
#             for actual, dst in zip(actual_issame, dist):
#                 f.write(str(actual) + ',' + str(dst) + '\n')
#
#     # acc = verify_model.metric(outputs, labels=labels, verify_threshold=verify_threshold)
#
#     # print('verfiy1vs1 acc: ', acc)
#
#
# if __name__ == '__main__':
#     sys.argv = ['nn_verification.py',
#                 '--mode', 'ppl_verify',
#                 '--faceid_model', '/home/xiajun/dev/facerec/facenet/mydataset/models/20180402-114759',
#                 '--verify_model', '/home/xiajun/dev/FlaskFace/face_verification/data/vggface2_train_mtcnn_align160x160_margin32_TrainPair_PerPairSize20/NNVerif_FixedStand_SubMean_Flip1024_RandomRotateCropFlip_epoch10/final_2019-05-06 21:05:34.202854.pb',
#                 '--base_embeddings_file', '/home/xiajun/dev/FlaskFace/face_verification/data/gcface_mtcnn_align160x160_margin32/test_database/database_set.pkl',
#                 '--images_path', '/home/xiajun/res/face/lfw/Experiment/facenet_mtcnn_align160x160_margin32',
#                 '--pairs_file', '/home/xiajun/dev/FlaskFace/face_identification/data/pairs.txt',
#                 ]
#
#     args = parse_arguments(sys.argv[1:])
#     print(args)
#     verify_mode = args.mode
#     faceid_model_path = args.faceid_model
#     verify_model_path = args.verify_model
#     base_embeddings_file = args.base_embeddings_file
#     images_path = args.images_path
#     pairs_file = args.pairs_file
#     distance_metric = args.distance_metric
#     dist_threshold = args.dist_threshold
#     prob_threshold = args.prob_threshold
#
#     if verify_mode == 'database_verify':
#         names = ['h4_id_119']
#         image_paths = ['/home/xiajun/dev/FlaskFace/data/jxl_align160_margin32.png']
#         database1vsN(names, image_paths, base_embeddings_file, faceid_model_path, distance_metric=distance_metric, dist_threshold=dist_threshold)
#     elif verify_mode == 'lfw_verify':
#         DEBUG = False
#         from face_identification import lfw
#         import random
#
#         pairs = lfw.read_pairs(pairs_file)
#         if DEBUG:
#             random.shuffle(pairs)
#             pairs = pairs[:100]
#         image_paths, actual_issame = lfw.get_paths(os.path.expanduser(images_path), pairs)
#
#         verfiy1vs1(image_paths, actual_issame, faceid_model_path, distance_metric=distance_metric, dist_threshold=dist_threshold, prob_threshold=prob_threshold)
#     elif verify_mode == 'ppl_verify':
#         from utils import face_dataset
#
#         # image_paths, actual_issame = face_dataset.read_csv_pair_file(pairs_file)
#         image_paths = ['/home/xiajun/res/face/GC-WebFace/tmp_label/张又天/clean/20161208102142zq6rxgwg2ftlrr6w2bse_3.jpg',
#                        '/home/xiajun/res/face/GC-WebFace/tmp_label/祝钒刚/clean/20161208102106xv23sglkfoomhh8s2kch_4.jpg']
#         actual_issame = [1]
#
#         verfiy1vs1(image_paths, actual_issame, faceid_model_path, distance_metric=distance_metric, dist_threshold=dist_threshold, prob_threshold=prob_threshold)
#
#     print('debug')
#
#
# def metric_tar(metric_file, threshold=0):
#     with open(metric_file, 'r') as f:
#         lines = f.readlines()
#         lines = [line.strip().split(',') for line in lines]
#         lines = np.array(lines).astype(np.float)
#
#     label = lines[:, 0]
#     dist = lines[:, 1]
#     # val = softmax(dist)
#     # val = sigmoid(dist)
#     prob = sigmoid(-dist, delta=-threshold)
#
#     positive_sample_prob = prob[np.where((prob * label) != 0)]
#     negative_sample_prob = prob[np.where((prob * np.logical_not(label)) != 0)]
#
#     prob_thresholds = np.linspace(0, 1, 100)
#     positive_tar = [len(positive_sample_prob[np.where(positive_sample_prob > prob)])/len(positive_sample_prob) for prob in prob_thresholds]
#     negative_tar = [len(negative_sample_prob[np.where(negative_sample_prob > prob)])/len(negative_sample_prob) for prob in prob_thresholds]
#
#     return prob_thresholds, positive_tar, negative_tar
#
#
# if __name__ == '__main__plot_metric':
#     prob_thresholds, positive_tar, negative_tar = metric_tar('data/EculidandistMetric_LFW.csv', threshold=2.28)
#     plt.plot(prob_thresholds, positive_tar, 'r', linewidth=2, marker='|', label='LFW-Euclidian-positive')
#     plt.plot(prob_thresholds, negative_tar, 'r', linewidth=2, marker='.', label='LFW-Euclidian-negative')
#     prob_thresholds, positive_tar, negative_tar = metric_tar('data/CosdistMetric_LFW.csv', threshold=0.37)
#     plt.plot(prob_thresholds, positive_tar, 'g', linewidth=2, marker='|', label='LFW-Cosine-positive')
#     plt.plot(prob_thresholds, negative_tar, 'g', linewidth=2, marker='.', label='LFW-Cosine-negative')
#     prob_thresholds, positive_tar, negative_tar = metric_tar('data/EculidandistMetric_GcTogether.csv', threshold=1.25)
#     plt.plot(prob_thresholds, positive_tar, 'b', linewidth=2, marker='|', label='GC-Together-Euclidian-positive')
#     plt.plot(prob_thresholds, negative_tar, 'b', linewidth=2, marker='.', label='GC-Together-Euclidian-negative')
#     prob_thresholds, positive_tar, negative_tar = metric_tar('data/CosdistMetric_GcTogether.csv', threshold=0.38)
#     plt.plot(prob_thresholds, positive_tar, 'y', linewidth=2, marker='|', label='GC-Together-Cosine-positive')
#     plt.plot(prob_thresholds, negative_tar, 'y', linewidth=2, marker='.', label='GC-Together-Cosine-negative')
#
#     plt.xlabel(r'$\rm{prob\_threshold}$', fontsize=16)
#     plt.ylabel(r'$\rm{tar}$', fontsize=16)
#     #plt.title(r'$f(x) \ \rm{is \ damping  \ with} \ x$', fontsize=16)
#     #plt.text(2.0, 0.5, r'$f(x) = \rm{sin}(2 \pi  x^2) e^{\sigma x}$', fontsize=20)
#     plt.legend()  # 显示图例
#     plt.savefig('latex.png', dpi=75)
#     plt.show()
#
#     print('debug')


# facenet 的验证结果
"""
# lfw verify
欧式距离
[calculate_roc]: best_threshold:  2.2800000000000002
[calculate_roc]: best_threshold:  2.2800000000000002
[calculate_roc]: best_threshold:  2.2800000000000002
[calculate_roc]: best_threshold:  2.2800000000000002
[calculate_roc]: best_threshold:  2.2800000000000002
[calculate_roc]: best_threshold:  2.2800000000000002
[calculate_roc]: best_threshold:  2.2800000000000002
[calculate_roc]: best_threshold:  2.2800000000000002
[calculate_roc]: best_threshold:  2.2800000000000002
[calculate_roc]: best_threshold:  2.2800000000000002
accuracy= [1.    0.99666667 0.99   0.99166667 0.99666667 1.0  0.99166667 0.99666667 0.99833333 0.99833333]
val=0.9826666666666666, val_std=0.010934146311237803, far=0.0006666666666666668


cos距离
[calculate_roc]: best_threshold:  0.37
[calculate_roc]: best_threshold:  0.37
[calculate_roc]: best_threshold:  0.37
[calculate_roc]: best_threshold:  0.37
[calculate_roc]: best_threshold:  0.37
[calculate_roc]: best_threshold:  0.37
[calculate_roc]: best_threshold:  0.36
[calculate_roc]: best_threshold:  0.37
[calculate_roc]: best_threshold:  0.37
[calculate_roc]: best_threshold:  0.37
accuracy= [1.     0.99666667 0.99333333 0.995  0.99166667 0.99833333  0.99166667 0.99666667 0.99666667 1. ]
val=0.99, val_std=0.007888106377466158, far=0.001



# gc_together ppl verify
欧式距离
[calculate_roc]: best_threshold:  1.3
[calculate_roc]: best_threshold:  1.31
[calculate_roc]: best_threshold:  1.24
[calculate_roc]: best_threshold:  1.24
[calculate_roc]: best_threshold:  1.3
[calculate_roc]: best_threshold:  1.3
[calculate_roc]: best_threshold:  1.24
[calculate_roc]: best_threshold:  1.24
[calculate_roc]: best_threshold:  1.3
[calculate_roc]: best_threshold:  1.3
accuracy= [0.95652174 0.9673913  0.97826087 0.97826087 0.9673913  0.98913043  0.95652174 0.97826087 0.9673913  0.9673913 ]
val=0.7151252268121369, val_std=0.06798304367714122, far=0.0
 
cos距离 
[calculate_roc]: best_threshold:  0.38
[calculate_roc]: best_threshold:  0.38
[calculate_roc]: best_threshold:  0.38
[calculate_roc]: best_threshold:  0.38
[calculate_roc]: best_threshold:  0.38
[calculate_roc]: best_threshold:  0.38
[calculate_roc]: best_threshold:  0.38
[calculate_roc]: best_threshold:  0.38
[calculate_roc]: best_threshold:  0.39
[calculate_roc]: best_threshold:  0.38
accuracy= [1.0   0.98913043  0.97826087  1.0   0.98913043  1.0  0.9673913  1.0   0.95652174   0.98913043]
val=0.5540525267722307, val_std=0.05423642304297514, far=0.0
"""


# 我的验证结果
"""
LFW
                sigmoid有负值
                '--distance_metric', '0',
                '--dist_threshold', '2.28',
                '--prob_threshold', '0.95',
prob_metric: acc=0.8915, tar=0.783, far=0.0, precision=1.0
dist_metric: acc=0.9953333333333333, tar=0.994, far=0.0033333333333333335, precision=0.9966577540106952

                sigmoid有负值
                '--distance_metric', '0',
                '--dist_threshold', '2.28',
                '--prob_threshold', '0.90',
prob_metric: acc=0.9395, tar=0.879, far=0.0, precision=1.0
dist_metric: acc=0.9956666666666667, tar=0.9936666666666667, far=0.0023333333333333335, precision=0.9976572958500669

                sigmoid有负值
                '--distance_metric', '0',
                '--dist_threshold', '2.28',
                '--prob_threshold', '0.65',
prob_metric: acc=0.9826666666666667, tar=0.966, far=0.0006666666666666666, precision=0.9993103448275862
dist_metirc: acc=0.9951666666666666, tar=0.9936666666666667, far=0.0033333333333333335, precision=0.9966566365763958



GC-Together
                sigmoid有负值
                '--distance_metric', '0',
                '--dist_threshold', '1.25',
                '--prob_threshold', '0.65',
prob_metric: acc=0.9532608695652174, tar=0.908695652173913, far=0.002173913043478261, precision=0.9976133651551312
dist_metirc: acc=0.975, tar=0.9652173913043478, far=0.015217391304347827, precision=0.9844789356984479

                sigmoid有负值
                '--distance_metric', '0',
                '--dist_threshold', '1.25',
                '--prob_threshold', '0.60',
prob_metric: acc=0.9586956521739131, tar=0.9195652173913044, far=0.002173913043478261, precision=0.9976415094339622
dist_metirc: acc=0.9717391304347827, tar=0.9586956521739131, far=0.015217391304347827, precision=0.984375

                sigmoid有负值
                '--distance_metric', '0',
                '--dist_threshold', '1.25',
                '--prob_threshold', '0.45',
prob_metric: acc=0.967391304347826, tar=0.941304347826087, far=0.006521739130434782, precision=0.9931192660550459
dist_metirc: acc=0.9684782608695652, tar=0.9586956521739131, far=0.021739130434782608, precision=0.9778270509977827

                sigmoid有负值
                '--distance_metric', '0',
                '--dist_threshold', '1.25',
                '--prob_threshold', '0.2',
prob_metric: acc=0.9706521739130435, tar=0.9521739130434783, far=0.010869565217391304, precision=0.9887133182844243
dist_metirc: acc=0.9717391304347827, tar=0.9630434782608696, far=0.01956521739130435, precision=0.9800884955752213


                sigmoid没有负值
                '--distance_metric', '0',
                '--dist_threshold', '1.25',
                '--prob_threshold', '0.6',
prob_metric: acc=0.9717391304347827, tar=0.9521739130434783, far=0.008695652173913044, precision=0.9909502262443439
dist_metirc: acc=0.975, tar=0.9630434782608696, far=0.013043478260869565, precision=0.9866369710467706
同一组参数，每次执行结果不一样：
prob_metric: acc=0.9717391304347827, tar=0.9565217391304348, far=0.013043478260869565, precision=0.9865470852017937
dist_metirc: acc=0.9717391304347827, tar=0.9608695652173913, far=0.017391304347826087, precision=0.9822222222222222


                sigmoid没有负值
                '--distance_metric', '0',
                '--dist_threshold', '1.25',
                '--prob_threshold', '0.75',
prob_metric: acc=0.9608695652173913, tar=0.9282608695652174, far=0.006521739130434782, precision=0.9930232558139535
dist_metirc: acc=0.9706521739130435, tar=0.9565217391304348, far=0.015217391304347827, precision=0.9843400447427293
# 这组数据是去除随机裁剪、旋转、翻转后重新做的。
prob_metric: acc=0.9554347826086956, tar=0.9130434782608695, far=0.002173913043478261, precision=0.997624703087886
dist_metirc: acc=0.9815217391304348, tar=0.9739130434782609, far=0.010869565217391304, precision=0.9889624724061811




                sigmoid没有负值
                '--distance_metric', '0',
                '--dist_threshold', '1.73',  # 对应20190719-050512
                '--prob_threshold', '0.75',
prob_metric: acc=0.9510869565217391, tar=0.9130434782608695, far=0.010869565217391304, precision=0.9882352941176471
dist_metirc: acc=0.9652173913043478, tar=0.9630434782608696, far=0.03260869565217391, precision=0.9672489082969432
这组数据是去除随机裁剪、旋转、翻转后重新做的:
                '--dist_threshold', '1.73',  # 对应20190719-050512
prob_metric: acc=0.9402173913043478, tar=0.8804347826086957, far=0.0, precision=1.0
dist_metirc: acc=0.967391304347826, tar=0.9521739130434783, far=0.017391304347826087, precision=0.9820627802690582

                sigmoid没有负值
                '--distance_metric', '0',
                '--dist_threshold', '1.73',  # 对应20190719-061604
                '--prob_threshold', '0.75',
prob_metric: acc=0.9630434782608696, tar=0.9282608695652174, far=0.002173913043478261, precision=0.9976635514018691
dist_metirc: acc=0.9706521739130435, tar=0.9630434782608696, far=0.021739130434782608, precision=0.977924944812362
这组数据是去除随机裁剪、旋转、翻转后重新做的:
prob_metric: acc=0.9597826086956521, tar=0.9195652173913044, far=0.0, precision=1.0
dist_metirc: acc=0.975, tar=0.9565217391304348, far=0.006521739130434782, precision=0.9932279909706546

                sigmoid没有负值
                '--distance_metric', '0',
                '--dist_threshold', '1.80',  # 对应20190719-062455
                '--prob_threshold', '0.75',
prob_metric: acc=0.966304347826087, tar=0.9369565217391305, far=0.004347826086956522, precision=0.9953810623556582
dist_metirc: acc=0.9771739130434782, tar=0.967391304347826, far=0.013043478260869565, precision=0.9866962305986696
这组数据是去除随机裁剪、旋转、翻转后重新做的: 目前这组效果是最好的，但far=0，评估标准得换，不能只看far数值。
prob_metric: acc=0.9608695652173913, tar=0.9217391304347826, far=0.0, precision=1.0
dist_metirc: acc=0.9760869565217392, tar=0.9652173913043478, far=0.013043478260869565, precision=0.9866666666666667

                sigmoid没有负值
                '--distance_metric', '0',
                '--dist_threshold', '1.73',  # 对应20190726-013308
                '--prob_threshold', '0.75',
prob_metric: acc=0.9532608695652174, tar=0.9108695652173913, far=0.004347826086956522, precision=0.995249406175772
dist_metirc: acc=0.9641304347826087, tar=0.9521739130434783, far=0.02391304347826087, precision=0.9755011135857461
这组数据是去除随机裁剪、旋转、翻转后重新做的:
                '--dist_threshold', '1.54',  # 对应20190726-013308
prob_metric: acc=0.9358695652173913, tar=0.8717391304347826, far=0.0, precision=1.0
dist_metirc: acc=0.9695652173913043, tar=0.9434782608695652, far=0.004347826086956522, precision=0.9954128440366973

                sigmoid没有负值
                '--distance_metric', '0',
                '--dist_threshold', '1.63',  # 对应20190726-013618
                '--prob_threshold', '0.75',
prob_metric: acc=0.9510869565217391, tar=0.9152173913043479, far=0.013043478260869565, precision=0.9859484777517564
dist_metirc: acc=0.9630434782608696, tar=0.95, far=0.02391304347826087, precision=0.9754464285714286
这组数据是去除随机裁剪、旋转、翻转后重新做的:
                '--dist_threshold', '1.71',  # 对应20190726-013618
prob_metric: acc=0.95, tar=0.9108695652173913, far=0.010869565217391304, precision=0.9882075471698113
dist_metirc: acc=0.966304347826087, tar=0.9586956521739131, far=0.02608695652173913, precision=0.9735099337748344


# 如果追求准确，上述评估结果最好重新测，因为上述评估可能使用了随机裁剪/旋转等操作

                '--distance_metric', '0',
                '--dist_threshold', '1.63',  # 对应20190730-020939
                '--prob_threshold', '0.75',
prob_metric: acc=0.9510869565217391, tar=0.9021739130434783, far=0.0, precision=1.0
dist_metirc: acc=0.9728260869565217, tar=0.95, far=0.004347826086956522, precision=0.9954441913439636


                '--distance_metric', '0',
                '--dist_threshold', '1.63',  # 对应20190730-021241
                '--prob_threshold', '0.75',
prob_metric: acc=0.9543478260869566, tar=0.9130434782608695, far=0.004347826086956522, precision=0.995260663507109
dist_metirc: acc=0.9641304347826087, tar=0.941304347826087, far=0.013043478260869565, precision=0.9863325740318907



                '--distance_metric', '0',
                '--dist_threshold', '1.23',  # 对应20190905-081601
                '--prob_threshold', '0.75',
prob_metric: acc=0.9630434782608696, tar=0.9260869565217391, far=0.0, precision=1.0
dist_metirc: acc=0.9782608695652174, tar=0.9608695652173913, far=0.004347826086956522, precision=0.9954954954954955



                '--distance_metric', '0',
                '--dist_threshold', '1.23',  # 对应20190909-050722
                '--prob_threshold', '0.75',
prob_metric: acc=0.9695652173913043, tar=0.9434782608695652, far=0.004347826086956522, precision=0.9954128440366973
dist_metirc: acc=0.9793478260869565, tar=0.967391304347826, far=0.008695652173913044, precision=0.9910913140311804



                '--distance_metric', '0',
                '--dist_threshold', '1.34',  # 对应20190911-061413
                '--prob_threshold', '0.75',
prob_metric: acc=0.9554347826086956, tar=0.9130434782608695, far=0.002173913043478261, precision=0.997624703087886
dist_metirc: acc=0.9771739130434782, tar=0.9630434782608696, far=0.008695652173913044, precision=0.9910514541387024

"""



