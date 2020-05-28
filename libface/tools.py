import os
import math
import sys
import skimage
import cv2
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# from sklearn.metrics import roc_curve, auc
import shutil
import datetime

DEBUG = False


def view_bar(message, num, total):
    """
    :param message:
    :param num:
    :param total:
    :return:

    Example:
    view_bar('loading: ', i + 1, len(list))
    """
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def show_rect(img_path, regions):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = skimage.io.imread(img_path)
    ax.imshow(img)
    for x, y, w, h in regions:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()


def strcat(strlist, cat_mark=','):
    '''
    :param strlist: a list for store string.
    :param cat_mark:
    :return:
    '''
    line = ''
    for ln in strlist:
        line += str(ln) + cat_mark
    line = line[0:line.rfind(cat_mark)]
    return line


def sufwith(s, suf):
    '''
    将字符串s的后缀改为以suf结尾
    :param s:
    :param suf:
    :return:
    '''
    return os.path.splitext(s)[0] + suf


def steps_per_epoch(num_samples, batch_size, allow_less_batsize=True):
    """
    :param num_samples: 样本总量
    :param batch_size:
    :param allow_less_batsize: True: 允许最后一个step的样本数量小于batch_size;
                               False: 如果最后一个step的样本数量小于batch_size，则丢弃这一step的样本。
    :return:
    """
    steps = 0
    steps += num_samples // batch_size
    if allow_less_batsize:
        offset = 0 if num_samples % batch_size == 0 else 1
        steps += offset

    return steps


def get_strtime():
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    return time_str


def imcrop(img, bbox, scale_ratio=2.0):
    '''
    :param img: ndarray or image_path
    :param bbox: bounding box
    :param scale_ratio: the scale_ratio used to control margin size around bbox.
    :return:
    '''
    if type(img) == str:
        img = cv2.imread(img)

    xmin, ymin, xmax, ymax = bbox
    hmax, wmax, _ = img.shape
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = (xmax - xmin) * scale_ratio
    h = (ymax - ymin) * scale_ratio

    xmin = x - w / 2
    xmax = x + w / 2
    ymin = y - h / 2
    ymax = y + h / 2

    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(wmax, int(xmax))
    ymax = min(hmax, int(ymax))

    face = img[ymin:ymax, xmin:xmax, :]
    return face


def named_standard(data_path, mark='', replace=''):
    class_list = os.listdir(data_path)
    for i, cls in enumerate(class_list):
        if cls.count(mark):
            new_cls = cls.replace(mark, replace)
            class_path = os.path.join(data_path, cls)
            new_class_path = os.path.join(data_path, new_cls)
            os.rename(class_path, new_class_path)
            print('[tools.named_standard]:: change {} to {}'.format(class_path, new_class_path))

        view_bar('[tools.load_image]:: loading: ', i + 1, len(class_list))
    print('')


def imread(image_path, flags=0):
    """
    :param image_path:
    :param flags: 0: 返回rgb图；1:返回bgr图
    :return:

    :下面是废弃的参考代码：
    image = misc.imread(image_path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        channel = image.shape[-1]
        if channel == 3:
            pass
        elif channel == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            raise Exception('没遇到过这种情况，排查下怎么回事！')

    return image
    """
    image = misc.imread(image_path)  # 有空可以试着将 misc 读图换成 cv2 读图
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        channel = image.shape[-1]
        if channel == 3:
            pass
        elif channel == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            raise Exception('没遇到过这种情况，排查下怎么回事！')

    if flags == 0:
        pass
    elif flags == 1:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        raise Exception('暂时没有其他颜色通道需求，若有需求可添加！')

    return image


def load_dataset(data_path, shuffle=True, validation_ratio=0.0, min_nrof_cls=1, max_nrof_cls=999999999, filter_cb=None):
    '''
data_path dir style:
data
├── folder1
│   ├── 00063.jpg
│   ├── 00068.jpg
└── folder2
    ├── 00070.jpg
    ├── 00072.jpg

    :param data_path:
    :param shuffle:
    :param validation_ratio: if > 0: will split train-set and validation-set,
           其中， validation-set accounts for is validation_ratio.
    :param min_nrof_cls: min number samples of each class
    :param max_nrof_cls: max number samples of each class
    :return:
    '''
    label_count = 0
    images_path = []
    images_label = []
    images = os.listdir(data_path)
    images = images[0:1000]
    images.sort()
    for i, image in enumerate(images):
        cls_path = os.path.join(data_path, image)
        if os.path.isfile(cls_path):
            print('[load_dataset]:: {} is not dir!'.format(cls_path))
            continue

        imgs = os.listdir(cls_path)
        if len(imgs) < min_nrof_cls:
            continue
        if len(imgs) > max_nrof_cls:
            np.random.shuffle(imgs)
            imgs = imgs[0:max_nrof_cls]

        imgs = [os.path.join(data_path, image, img) for img in imgs]
        if filter_cb is not None:
            imgs = filter_cb(imgs)
        images_path.extend(imgs)
        images_label.extend([label_count] * len(imgs))
        label_count += 1
        view_bar('loading: ', i + 1, len(images))
    print('')

    images = np.array([images_path, images_label]).transpose()

    if shuffle:
        np.random.shuffle(images)

    images_path = images[:, 0]
    images_label = images[:, 1].astype(np.int32)

    # if DEBUG and False:
    #     images_path = images_path[0:500]
    #     images_label = images_label[0:500]

    if validation_ratio > 0.0:
        if not shuffle:
            raise Exception('When there is a validation set split requirement, shuffle must be True.')
        validation_size = int(len(images_path) * validation_ratio)
        validation_images_path = images_path[0:validation_size]
        validation_images_label = images_label[0:validation_size]

        train_images_path = images_path[validation_size:]
        train_images_label = images_label[validation_size:]

        print('\n********************************样 本 总 量*************************************')
        print('len(train_images_path)={}, len(train_images_label)={}'.format(len(train_images_path), len(train_images_label)))
        print('len(validation_images_path)={}, len(validation_images_label)={}'.format(len(validation_images_path), len(validation_images_label)))
        print('num_train_class={}, num_validation_class={}'.format(len(set(train_images_label)), len(set(validation_images_label))))
        print('*******************************************************************************\n')

        return train_images_path, train_images_label, validation_images_path, validation_images_label
    else:
        print('\n********************************样 本 总 量*************************************')
        print('len(images_path)={}, len(images_label)={}'.format(len(images_path), len(images_label)))
        print('num_class={}'.format(len(set(images_label))))
        print('*******************************************************************************\n')

        return images_path, images_label


def load_image(data_path, subdir='', min_num_image_per_class=1, del_under_min_num_class=False, min_area4del=0, filter_cb=None):
    '''
    TODO: 本函数包含了很多判断删除操作，这一点很冗余。后续应当将该函数拆分成两个函数：1. 删除、判断处理数据；2. 递归加载数据
    :param data_path:
    :param subdir: 如果每个类别又子目录，则应该指定。
    :param min_num_image_per_class: 每个类至少min_num_image_per_class张图片，少于这个数量的类不会被加载。
    :param del_under_min_num_class: if True, 如果每个类别少于min_num_image_per_class张图片，则永久删除这个类。
    :param min_area4del: 如果图片尺寸小于该值则会被删除。设置为0即不适用该操作。
    :return:
    '''
    del_count = 0

    images_info = []
    class_list = os.listdir(data_path)
    if DEBUG:
        class_list = class_list[0:100]
    class_list.sort()
    for i, cls in enumerate(class_list):
        class_images_info = []
        class_path = os.path.join(data_path, cls)
        if os.path.isfile(class_path):
            print('[load_image]:: {} is not dir!'.format(class_path))
            continue

        image_list = os.listdir(class_path)
        image_list.sort()
        for image in image_list:
            image_path = os.path.join(class_path, image)
            if filter_cb is not None:
                if not filter_cb(image_path):
                    print('{} is filtered!'.format(image_path))
                    continue

            if os.path.isdir(image_path):
                if image == subdir:
                    sub_image_list = os.listdir(image_path)
                    sub_image_list.sort()
                    for sub_image in sub_image_list:
                        sub_image_path = os.path.join(image_path, sub_image)
                        if os.path.isfile(sub_image_path):
                            if min_area4del > 0:
                                img = cv2.imread(sub_image_path)
                                img_area = img.shape[0] * img.shape[1]
                                if img_area < min_area4del:
                                    os.remove(sub_image_path)
                                    del_count += 1

                            sub_image = os.path.join(image, sub_image)
                            class_images_info.append([cls, sub_image])
                        else:
                            # raise Exception("Error, Shouldn't exist! Please check your param or trying to reason for {}!".format(sub_image_path))
                            print(image_path)
                            if os.path.exists(class_path):
                                shutil.rmtree(class_path)
                else:
                    # raise Exception("Error, Shouldn't exist! Please check your param or trying to reason for {}!".format(image_path))
                    print(image_path)
                    if os.path.exists(class_path):
                        shutil.rmtree(class_path)
            else:
                if min_area4del > 0:
                    img = cv2.imread(image_path)
                    img_area = img.shape[0] * img.shape[1]
                    if img_area < min_area4del:
                        os.remove(image_path)
                        del_count += 1

                class_images_info.append([cls, image])

        if len(class_images_info) >= min_num_image_per_class:
            images_info.extend(class_images_info)
        elif del_under_min_num_class:
            if os.path.exists(class_path):
                del_count += 1
                shutil.rmtree(class_path)
            print('[tools.load_image]:: delete: {} image, {}'.format(len(class_images_info), class_path))
            continue
        elif del_under_min_num_class and len(cls) > 30:
            # del_count += 1
            # shutil.rmtree(class_path)
            print('[tools.load_image]:: delete: {} image, {}'.format(len(class_images_info), class_path))
        else:
            # raise Exception('为什么会出现这种情况，排查一下！！！！')
            pass

        view_bar('[tools.load_image]:: loading: ', i + 1, len(class_list))
    print('')
    print('[tools.load_image]:: delete {} images!'.format(del_count))
    return images_info


def load_csv(csv_file, start_idx=None, end_idx=None):
    """
    读取CSV文件
    :param csv_file:
    :param start_idx: 从start_idx这一行开始取数据
    :param end_idx: 取到end_idx这一行为止
    :return:
    """
    print('\n[load_csv]:: load {} '.format(csv_file))
    with open(csv_file) as fid:
        lines = fid.readlines()
        lines = lines[start_idx:end_idx]
        csv_info = []
        for i, line in enumerate(lines):
            csv_info.append(line.strip().split(","))
            view_bar('loading:', i + 1, len(lines))
        print('')

        csv_info = np.array(csv_info)
    return csv_info


def save_csv(templates, csv_file, title_line='\n'):
    assert len(templates.shape) >= 2
    assert len(title_line) >= 1
    assert title_line[-1] == '\n'

    print('\n[save_csv]:: save {} '.format(csv_file))
    with open(csv_file, "w") as fid:
        fid.write(title_line)
        for i in range(len(templates)):
            text = strcat(templates[i]) + "\n"
            fid.write(text)
            view_bar('saving:', i + 1, len(templates))
        print('')
    fid.close()


def concat_dataset(root_path, images, label_names):
    """
    数据路径拼接思想：root_path/images[i]/label_names[i]
    :param root_path: 数据根路径
    :param images: 特定类别的图片子目录，如：201912341.jpg、 subdir/201912341.jpg
    :param label_names: 在路径中的类别名
    :return:
    """
    print('\n[concat_dataset]:: load {} '.format(root_path))
    images_info = []
    imexist_count = 0
    imexts = ['jpg', 'png']
    for i, (image, label_name) in enumerate(zip(images, label_names)):
        imname, imext = image.rsplit('.', maxsplit=1)
        if imext not in imexts:
            raise Exception('Only support [jpg, png] currently!')

        imexist = False
        for ext in imexts:
            image_path = os.path.join(root_path, imname) + '.' + ext
            if os.path.exists(image_path):
                image_info = [label_name, image_path]
                images_info.append(image_info)
                imexist = True
                break
        if not imexist:
            imexist_count -= 1
            print('\t{}/{}.{} is not exist!'.format(root_path, imname, imexts))

        view_bar('concat dataset:', i + 1, len(images))
    print('')

    images_info = np.array(images_info)

    print('\n***********************************************')
    print('From {}, not existed images count: {}'.format(root_path, abs(imexist_count)))
    print('***********************************************\n')

    return images_info


def compute_auc(fpr, tpr):
    raise Exception('该函数废弃，转移至 libml.metrics.evaluate.py')
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_iou(bbox1, bbox2):
    """
    computing IoU
    :param bbox1: (x0, y0, x1, y1), which reflects (left, top, right, bottom)
    :param bbox2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (bbox1[3] - bbox1[1]) * (bbox1[2] - bbox1[0])
    S_rec2 = (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(bbox1[0], bbox2[0])
    right_line = min(bbox1[2], bbox2[2])
    top_line = max(bbox1[1], bbox2[1])
    bottom_line = min(bbox1[3], bbox2[3])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def get_learning_rate_from_file(filename, epoch):
    """
    :reference:
    :param filename:
    :param epoch:
    :return:
    """
    learning_rate = -1
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
    return learning_rate


def load_data_from_csv(feed_file, shuffle=True, start_idx=1, end_idx=None, all4train=False):
    ''' reference:  FlaskFace/utils/dataset.py
    加載csv文件內容，要求csv數據順序參考 make_feedata，即 train_val,label,person_name,image_path
    :param feed_file:
    :param shuffle:
    :param start_idx:
    :param end_idx:
    :param all4train:
    :return:
    '''
    imgs_info = load_csv(feed_file, start_idx=start_idx, end_idx=end_idx)  # , end_idx=10000
    if shuffle:
        np.random.shuffle(imgs_info)

    if False:
        # 如果feed_file中的数据是按照 'cls_name,image_path'这样的格式排列的，则需要通过下面的方法赋予 label.
        # ************************************************************
        images_path = imgs_info[:, 1]
        labels_name = imgs_info[:, 0]
        cls_name = set(labels_name)
        images_label = np.zeros_like(labels_name, dtype=np.int32)
        for i, cls in enumerate(cls_name):
            images_label[np.where(labels_name == cls)] = i

            view_bar('load feed data:', i + 1, len(cls_name))
        print('')
        # ************************************************************
    else:
        train_info = imgs_info[np.where(imgs_info[:, 0] == '0')]
        validation_info = imgs_info[np.where(imgs_info[:, 0] == '1')]

        train_images_label = train_info[:, 1].astype(np.int32).tolist()
        train_images_path = train_info[:, 3].tolist()

        validation_images_label = validation_info[:, 1].astype(np.int32).tolist()
        validation_images_path = validation_info[:, 3].tolist()

        if all4train:
            train_images_path = train_images_path + validation_images_path
            train_images_label = train_images_label + validation_images_label
            validation_images_path = []
            validation_images_label = []

            if shuffle:
                image_label = np.array([train_images_label, train_images_path]).transpose()
                np.random.shuffle(image_label)
                train_images_label = image_label[:, 0].astype(np.int32).tolist()
                train_images_path = image_label[:, 1].tolist()

    return train_images_path, train_images_label, validation_images_path, validation_images_label


def make_data_to_csv(datasets, save_file, title_line='\n', filter_cb=None):
    """reference: from FlaskFace/utils/dataset.py
    制作用于feed到网络的数据, 並按 train_val,label,person_name,image_path順序保存到csv文件。
    :param datasets: datasets must be a list, and every element in the list must be dict.
    :param save_file: save the resulting data to save_file.
    :param title_line: first line write to save_file, just remark.
    :return:
    """
    dataset_keys = ['root_path', 'csv_file']
    for dataset in datasets:
        if type(dataset) != dict:
            raise Exception('datasets must be a list, and every element in the list must be dict.')
        for key in dataset.keys():
            if key not in dataset_keys:
                raise Exception('dataset dict expect {}, but received {}'.format(dataset_keys, key))

    images_info = []
    for dataset in datasets:
        imgs_info = tools.load_csv(dataset['csv_file'], start_idx=1)
        imgs_path = [os.path.join(info[0], info[2]) for info in imgs_info]
        imgs_info = tools.concat_dataset(dataset['root_path'], imgs_path, imgs_info[:, 1])
        images_info.extend(imgs_info)
    images_info = np.array(images_info)

    # images_info = config_feedata(images_info, validation_ratio=-1, dup_base=(20, 80), filter_cb=filter_cb)
    images_info = config_feedata(images_info, validation_ratio=-1, dup_base=(80, 180), filter_cb=filter_cb)

    tools.save_csv(images_info, save_file, title_line=title_line)
