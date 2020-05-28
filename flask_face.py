# -*- coding=utf-8 -*-
import base64
import json
import cv2
import time
import logging
import traceback
import hmac
from scipy import misc
import numpy as np
from flask import Flask, jsonify, request, render_template, render_template_string, redirect, url_for, make_response, Response, g, session
import os
import random
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from libface.face_online_pipeline import FaceOnlinePipeline
import pymysql
# import pymysql.cursors
from mysql_connection import mysql_connection
from libface import tools
# from utils import dataset as Datset
import shutil

RELEASE = True

logging.getLogger().setLevel(logging.INFO)

app = Flask(__name__)
if RELEASE:
    app.config["SECRET_KEY"] = "ningbo_doppler_template"
else:
    app.config["SECRET_KEY"] = "nj_template"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=1)  # 配置session有效期是1天


config = {
    "mysql_host": "127.0.0.1",
    "mysql_user": "root",
    "mysql_pass": "xiajun",
    "mysql_db": "db_face_recognization",
    "log_root_dir": "./data/log",
    "template_root_dir": "./data/template",
    "verify_root_dir": "./data/verify",

    "multiface_concat_mark": ",",
    "feature_size": 2048,
    "faceid_best_threshold": 0.45,
    "featureinfo_ndim": 3,
    "use_second_verify": True
}
if RELEASE:
    config["mysql_pass"] = "GCDEV123"  # 云服务器
    config["mysql_pass"] = "ailab"  # ailab-server

redis_agency = None
face_online_pipline = None
mysql_db = None
mysql_conn = None

error_code = {
    'error_ok': 0,
    'error_add_template_error': 1001,
    'error_remove_template_error': 1002,
    'error_recognize_file_unmatched': 1003,
    'error_recognize_no_result': 1004,
    'error_database_index_error': 1005,
    'error_not_detect_face': 1006,
    'error_at_least_one_image_perperson': 1007,
    'error_login_error': 2001,
    'error_modify_user_error': 2002,
}

error_code_key_str = 'err_code'
error_desc_key_str = 'err_desc'
data_key_str = 'data'
token_key_str = 'token'

ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'JPG'}
ALLOWED_ANNOTATION_EXTENSIONS = {'xml'}

g_Features = []
g_FeatureInfo = []
g_PersonInfo = {}


class MyException(Exception):
    def __init__(self, msg):
        self.msg = msg
        # print(self.msg)

    #def __str__(self):
    #    return self.args


def allowed_file(filename, allowed_extensions):
    """
    用于判断文件后缀
    :param filename:
    :param allowed_extensions:
    :return:
    """
    return os.path.splitext(filename)[-1][1:] in allowed_extensions
    # return '.' in filename and os.path.splitext(filename)[-1][1:] in allowed_extensions


def generate_token(key, expire=3600):
    '''
        @Args:
            key: str (用户给定的key，需要用户保存以便之后验证token,每次产生token时的key 都可以是同一个key)
            expire: int(最大有效时间，单位为s)
        @Return:
            state: str
    '''
    # print('generate token: {} + {} = {}'.format(time.time(), expire, time.time() + expire))
    ts_str = str(time.time() + expire)
    # print('generated token: {}'.format(ts_str))
    ts_byte = ts_str.encode("utf-8")
    sha1_tshexstr = hmac.new(key.encode("utf-8"), ts_byte, 'sha1').hexdigest()
    token = ts_str+':'+sha1_tshexstr
    b64_token = base64.urlsafe_b64encode(token.encode("utf-8"))
    return b64_token.decode("utf-8")


def certify_token(key, token):
    '''
        @Args:
            key: str
            token: str
        @Returns:
            boolean
    '''
    token_str = base64.urlsafe_b64decode(token).decode('utf-8')
    token_list = token_str.split(':')
    if len(token_list) != 2:
        return False
    ts_str = token_list[0]
    # print('token >>> >>> >>> {} - {} = {}'.format(float(ts_str), time.time(), float(ts_str) - time.time()))
    if float(ts_str) < time.time():
        return False
    known_sha1_tsstr = token_list[1]
    sha1 = hmac.new(key.encode("utf-8"), ts_str.encode('utf-8'), 'sha1')
    calc_sha1_tsstr = sha1.hexdigest()
    if calc_sha1_tsstr != known_sha1_tsstr:
        # token certification failed
        return False
    # token certification success
    return True


def log_init():
    print('Init the log')
    # 创建存放日志的目录
    if not os.path.exists(config["log_root_dir"]):
        os.makedirs(config["log_root_dir"])

    log_path = os.path.join(config["log_root_dir"], 'flask.log')
    # 大小为10M的日志文件，2个备份文件
    handler = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=2, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    logging_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
    handler.setFormatter(logging_format)
    app.logger.addHandler(handler)


def print_cache():
    global g_Features
    global g_FeatureInfo
    global g_PersonInfo

    print('*********************************************************************')
    print('g_FeatureInfo.shape={}, g_Features.shape={}, len(g_PersonInfo)={}'.format(g_FeatureInfo.shape, g_Features.shape, len(g_PersonInfo)))
    print('g_PersonInfo: ', g_PersonInfo.values())
    # print('g_FeatureInfo: ', g_FeatureInfo)
    print('*********************************************************************')


def load_database():
    global g_Features
    global g_PersonInfo
    global g_FeatureInfo

    try:
        sql = "select * from tb_person"
        cursor = mysql_conn.query(sql)
        if cursor is not None:
            while True:
                query_ret = cursor.fetchone()
                if query_ret is None:
                    break
                g_PersonInfo[query_ret["person_id"]] = query_ret["person_name"]
            cursor.close()

        sql = "select * from tb_multiface_feature"
        cursor = mysql_conn.query(sql)
        if cursor is not None:
            while True:
                query_ret = cursor.fetchone()
                if query_ret is None:
                    break
                info = [query_ret["person_id"], query_ret["feature_mark"], query_ret["img"]]
                g_FeatureInfo.append(info)
                g_Features.append(np.fromstring(query_ret["feature"]))
            cursor.close()
        g_FeatureInfo = np.array(g_FeatureInfo)
        g_Features = np.array(g_Features)
        if len(g_Features.shape) != 2:
            print('g_Features have different dimension vector!')
            app.logger.error(traceback.format_exc())
            assert len(g_Features.shape) == 2
    except Exception as e:
        print('[flask_face.load_database]: error: ', e)

    if False:  # 将当前数据库人脸template拷贝到指定的目录
        tmp_save = '/user_data/home/xiaj/res/tmpface'
        for feat_info in g_FeatureInfo:
            person_id, mark, img = feat_info
            person_name = g_PersonInfo[int(person_id)]
            image_path = os.path.join(config['template_root_dir'], img)
            if os.path.exists(image_path):
                tmp_save_path = os.path.join(tmp_save, person_name)
                if not os.path.exists(tmp_save_path):
                    os.makedirs(tmp_save_path)
                tmp_save_file = os.path.join(tmp_save_path, img)
                shutil.copy(image_path, tmp_save_file)
            else:
                print('{} not exist!'.format(image_path))

    print_cache()


def _add_feature_cache(person_id, mark, impath, feature):
    global g_Features
    global g_FeatureInfo

    g_FeatureInfo = np.vstack((g_FeatureInfo, [person_id, mark, impath]))
    g_Features = np.vstack((g_Features, feature))


def _del_feature_cache(index):
    global g_Features
    global g_FeatureInfo

    g_FeatureInfo = np.delete(g_FeatureInfo, [index], axis=0)
    g_Features = np.delete(g_Features, [index], axis=0)


def _joint_find_index(array, person_id, mark=None):
    global g_FeatureInfo

    index = np.array([], dtype=np.int)
    index_person_id = np.where(array[:, 0] == str(person_id))
    if len(index_person_id[0]) > 0:
        if mark is None:
            index = index_person_id[0]
        else:
            index_mark = np.where(array[:, 1] == mark)
            index = np.intersect1d(index_person_id, index_mark)

    return index


def _get_CacheFeatureInfoIndex_byPersonidMarks(person_id, marks=None):
    global g_FeatureInfo

    indexs = []
    if len(g_FeatureInfo) > 0:
        if marks is None:
            idx = _joint_find_index(g_FeatureInfo, person_id)
            indexs.extend(idx)
        else:
            for mark in marks:
                idx = _joint_find_index(g_FeatureInfo, person_id, marks)
                if len(idx) > 0:
                    indexs.extend(idx)

    return indexs


def update_cache(person_id, person_name=None, features_mark=None, images_path=None, features=None, mode='insert'):
    '''
    :param person_id:
    :param images_path:
    :param features:
    :param mode: insert or delete from cache
    :return:
    '''
    global g_Features
    global g_FeatureInfo
    global g_PersonInfo

    person_id = int(person_id)
    if len(g_FeatureInfo) == 0:
        g_FeatureInfo = g_FeatureInfo.reshape((0, config["featureinfo_ndim"]))
    if len(g_Features) == 0:
        g_Features = g_Features.reshape((0, config["feature_size"]))

    if mode == 'insert':
        indexs = _get_CacheFeatureInfoIndex_byPersonidMarks(person_id, features_mark)
        g_FeatureInfo = np.delete(g_FeatureInfo, [indexs], axis=0)
        g_Features = np.delete(g_Features, [indexs], axis=0)

        for mark, impath, feature in zip(features_mark, images_path, features):
            g_FeatureInfo = np.vstack((g_FeatureInfo, [person_id, mark, impath]))
            g_Features = np.vstack((g_Features, feature))

        g_PersonInfo[person_id] = person_name
    elif mode == 'delete':
        try:
            if features_mark is None:
                if person_id in g_PersonInfo.keys():
                    g_PersonInfo.pop(person_id)

            indexs = _get_CacheFeatureInfoIndex_byPersonidMarks(person_id, features_mark)
            g_FeatureInfo = np.delete(g_FeatureInfo, [indexs], axis=0)
            g_Features = np.delete(g_Features, [indexs], axis=0)

        except Exception as e:
            print("[flask_face.update_cache]:exception::{}".format(e))
    elif mode == 'modify':
        for mark, impath, feature in zip(features_mark, images_path, features):
            index = _joint_find_index(g_FeatureInfo, person_id, mark)
            if len(index) > 0:
                # 如果已经存在于缓存，则直接覆盖
                g_FeatureInfo[index] = [person_id, mark, impath]
                g_Features[index, :] = feature.reshape((1, -1))
            else:
                # 如果缓存中不存在，则追加
                g_FeatureInfo = np.vstack((g_FeatureInfo, [person_id, mark, impath]))
                g_Features = np.vstack((g_Features, feature))

        if person_name is not None:
            g_PersonInfo[person_id] = person_name
    else:
        pass

    print_cache()


def mysql_init():
    print('Init the mysql database')
    global mysql_conn
    global mysql_db

    mysql_db = pymysql.connect(host=config["mysql_host"], user=config["mysql_user"], passwd=config["mysql_pass"], db=config["mysql_db"])
    mysql_conn = mysql_connection()
    mysql_conn.connect(host=config["mysql_host"], user=config["mysql_user"], password=config["mysql_pass"], db=config["mysql_db"])

    if not os.path.exists(config["template_root_dir"]):
        os.makedirs(config["template_root_dir"])
    if not os.path.exists(config["verify_root_dir"]):
        os.makedirs(config["verify_root_dir"])

    load_database()


def model_init():
    print('Init the face recognition deep learning model')
    global face_online_pipline
    face_online_pipline = FaceOnlinePipeline()


def _get_images_mark(imdict, front_end=False):
    t0 = time.time()
    keys = imdict.keys()
    t1 = time.time()
    print('[_get_images_mark]:: imdict.keys time={}'.format(t1 - t0))
    t0 = t1

    marks = list(keys)
    t1 = time.time()
    print('[_get_images_mark]:: list(keys) time={}'.format(t1 - t0))
    t0 = t1

    marks.sort()
    t1 = time.time()
    print('[_get_images_mark]:: marks.sort time={}'.format(t1 - t0))
    t0 = t1

    effect_images = []
    effect_marks = []
    for mark in marks:
        if front_end:
            if len(imdict[mark].filename) > 0:
                effect_images.append(imdict[mark])
                effect_marks.append(mark)
        else:
            effect_images = [imdict[mark] for mark in marks]
            effect_marks = marks

    t1 = time.time()
    print('[_get_images_mark]:: for mark in marks time={}'.format(t1 - t0))
    t0 = t1

    return effect_images, effect_marks


def _generate_image_path(person_name=None, person_num=1, img_ext='.jpg'):
    if person_name is not None:
        # 添加人脸：包括从form表单添加和base64加密图片添加。
        img_dir = config["template_root_dir"]
    else:
        # 识别阶段的测试图片
        img_dir = config["verify_root_dir"]

    images_path = []
    aligned_images_path = []
    for i in range(person_num):
        random_name = str(random.randint(1000, 9999))
        img_name = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') + random_name
        images_path.append(img_dir + '/' + img_name + img_ext)
        aligned_images_path.append(img_dir + '/' + img_name + '_aligned' + img_ext)

    return images_path, aligned_images_path


def _decode_image(raw_images, form_images_save_path=None):
    '''
    :param raw_images:
    :param form_images_save_path: 如果form_images_save_path为空，则表示raw_images数据来自base64编码，否则表示raw_images来自form表单图片文件。
    :return:
    '''
    bgrimages = []
    for i, image in enumerate(raw_images):
        if form_images_save_path is None:
            decoded_img = base64.b64decode(image)
            bgrimage = cv2.imdecode(np.fromstring(decoded_img, np.uint8), cv2.IMREAD_COLOR)
        else:
            image.save(form_images_save_path[i])
            bgrimage = cv2.imread(form_images_save_path[i])
        bgrimages.append(bgrimage)

    return bgrimages


def _save_raw_image(images_path, bgr_images):
    for impath, image in zip(images_path, bgr_images):
        cv2.imwrite(impath, image)


def _del_images(images, root_path=None):
    for image in images:
        if root_path is not None:
            image = os.path.join(root_path, image)
        if os.path.exists(image):
            os.remove(image)


def _get_identity(index, prob, use_second_verify=True):
    top_k = 1
    if use_second_verify:
        top_k = 2

    persons_id = []
    persons_name = []
    persons_prob = []
    for idx, prb in zip(index, prob):
        person_id = int(g_FeatureInfo[idx][0])
        person_name = g_PersonInfo[person_id]
        if person_id not in persons_id:
            persons_id.append(person_id)
            persons_name.append(person_name)
            persons_prob.append(prb)
        if len(persons_id) == top_k:
            break
    return persons_id, persons_name, persons_prob

    #     persons_id = []
    #     persons_name = []
    #     persons_prob = []
    #     for idx, prb in zip(index, prob):
    #         person_id = int(g_FeatureInfo[idx][0])
    #         person_name = g_PersonInfo[person_id]
    #         if person_id not in persons_id:
    #             persons_id.append(person_id)
    #             persons_name.append(person_name)
    #             persons_prob.append(prb)
    #         if len(persons_id) == 2:
    #             break
    #     return persons_id, persons_name, persons_prob
    # else:
    #     person_id = int(g_FeatureInfo[index][0])
    #     person_name = g_PersonInfo[person_id]
    #
    # return person_id, person_name


def _extract_feature(images, aligned_images_path):
    """
    :param image: bgr image or image path
    :param aligned_image_path: just temp file
    :return:
    """
    try:
        t0 = time.time()
        for i, (impath, image) in enumerate(zip(aligned_images_path, images)):
            if type(image) == np.ndarray:
                pass
            elif type(image) == np.str_:
                image = cv2.imread(image)
            else:
                raise Exception('unknow image type, please check your code!')
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            feature = face_online_pipline.feature(image)
            if feature is None:
                app.logger.info('TODO:Unknown Error while detect face!')
                raise Exception('detect face error!')

            if len(feature) == 0:
                raise MyException("not detect face!")

            # misc.imsave(impath, aligned_images[0])
            tools.view_bar('flask_face->detecting face:', i, len(aligned_images_path))
        print('')
        # t1 = time.time()
        # if RELEASE:
        #     feature = face_online_pipline.feature(aligned_images_path)
        # else:
        #     feature = np.random.random((len(aligned_images_path), config['feature_size']))
        #     if len(aligned_images_path) == 2:
        #         feature = np.array([np.zeros((1, config['feature_size'])), np.ones((1, config['feature_size']))])
        #         feature = feature.reshape(len(feature), -1)
        # t2 = time.time()
        _del_images(aligned_images_path)

        # print('{} face detect time: {}, face embedding time: {}'.format(len(images), t1 - t0, t2 - t1))
        return feature

    except Exception as e:
        print("[flask_face._extract_feature]:exception::{}".format(e))
        _del_images(aligned_images_path)

        raise e


# TODO: 查询语句基本都可以用封装好的_sql_query函数执行
def _sql_query(sql):
    global mysql_conn
    tps = []
    try:
        cursor = mysql_conn.query(sql)
        if cursor != None:
            tps = cursor.fetchall()
            cursor.close()
    except Exception as e:
        print("[flask_face.query]:exception::{}".format(e))

    return tps


def _get_feature_insert_sql(person_id, marks, images_path):
    values = []
    for key, name in zip(marks, images_path):
        value = "({}, '{}', '{}', %s)".format(person_id, key, name)
        values.append(value)
    sql = tools.strcat(values)

    sql = "insert into tb_multiface_feature(person_id, feature_mark, img, feature) values" + sql + ";"

    return sql


def _get_record(person_id):
    """
    根据person_id获取记录
    :param person_id:
    :return:
    """
    global mysql_conn
    sql = "select img from tb_multiface_feature where person_id={}".format(person_id)
    try:
        cursor = mysql_conn.query(sql)
        if cursor != None:
            query_ret = cursor.fetchall()
            cursor.close()
            if query_ret is not None:
                return query_ret
        return None
    except Exception as e:
        print("[flask_face._get_record]:execute sql fail:{}, exception:{}".format(sql, e))
        return None


def _del_record(person_id, feature_mark=None):
    """
    根据person_id和feature_mark删除记录,如果feature_mark不为空，则表示删除tb_face_featrue表中关于该人的feature_mark对应的记录；
    如果feature_mark为空，则表示删除该人的所有记录，
    :param person_id:
    :param feature_mark:
    :return:
    """
    global mysql_conn
    if feature_mark is not None:
        sql = "delete from tb_multiface_feature where person_id={} and feature_mark='{}'".format(person_id, feature_mark)
    else:
        sql = "delete tbp.*, tbf.* from tb_person tbp, tb_multiface_feature tbf where tbp.person_id=tbf.person_id and tbp.person_id={}".format(person_id)
    try:
        ret = mysql_conn.execute(sql)
        if ret >= 0:
            return True
        return False
    except Exception as e:
        mysql_db.rollback()
        print("[flask_face._del_record]:execute sql fail:{}, exception:{}".format(sql, e))
        return False


def _get_template_list(records, front_end=False):
    '''
    TODO: 下一版本中，本函数都统一改成前端的格式。
    :param records:
    :param front_end:
    :return:
    '''
    templates = []
    for record in records:
        info = {'person_id': record['person_id'], 'person_name': record['person_name'], 'image': {}}
        marks = record['feature_mark'].split(',')
        images = record['img'].split(',')

        if front_end:
            # 因为前端是从‘img’字段提取图片的，为了兼容flask_face.py也要使用temp_list.html前端页面，
            # 所以这里将‘image’字段替换为‘img’字段。
            info.pop('image')
            imgs = []
            for i, mark in enumerate(marks):
                imgs.append({'mark': mark, 'img_name': images[i]})
            info['img'] = imgs # tools.strcat(marks)
        else:
            for i, mark in enumerate(marks):
                info['image'][mark] = images[i]

        templates.append(info)

    return templates


def _move_verify_file2specify_path(persons_id, persons_name, persons_prob, image_path):
    # 只是用于保存测试样本到对应目录下，此处暂时可不使用二次校验机制。
    person_id, person_name, person_prob = _strategy_verify(persons_id, persons_name, persons_prob, use_second_verify=False)

    if person_prob < config['faceid_best_threshold']:
        return

    ver_save_path = os.path.join(config["verify_root_dir"], tools.strcat([person_id, person_name], cat_mark='_'))
    imname_filed = image_path.split('/')[-1].split('.')
    imname_filed[0] = tools.strcat([imname_filed[0], 'p' + str(person_prob)], cat_mark='_')
    imname = tools.strcat(imname_filed, cat_mark='.')
    if not os.path.exists(ver_save_path):
        os.makedirs(ver_save_path)

    ver_save_path = os.path.join(ver_save_path, imname)
    shutil.move(image_path, ver_save_path)


def _strategy_verify(persons_id, persons_name, persons_prob, use_second_verify=False, second_verify_threshold=0.9):
    if use_second_verify:
        offset = persons_prob[1] / persons_prob[0]
        if offset >= second_verify_threshold:
            return None, None, None
        else:
            return persons_id[0], persons_name[0], persons_prob[0]
    else:
        return persons_id[0], persons_name[0], persons_prob[0]


def update_feature():
    sql = "select tbp.person_name, tbf.* from tb_person tbp inner join \
                    (select feature_id, person_id, feature_mark, img \
                    from tb_multiface_feature) tbf \
                where tbp.person_id=tbf.person_id"
    tbs = _sql_query(sql)

    img_dir = config["template_root_dir"]
    images_info = []
    for record in tbs:
        feature_id = record['feature_id']
        img_name = record['img']
        impath = os.path.join(img_dir, img_name)
        if not os.path.exists(impath):
            print(record)
            # raise Exception('竟然没有和数据对应的原始图片:{}。你需要编写处理这种情况的异常，可考虑直接将这条特征删除！'.format(impath))
            print('竟然没有和数据对应的原始图片。你需要编写处理这种情况的异常，可考虑直接将这条特征删除！')

        img_name, img_ext = img_name.split('.')
        aligned_impath = os.path.join(img_dir, img_name + '_aligned.' + img_ext)

        images_info.append([feature_id, impath, aligned_impath])

    images_info = np.array(images_info)

    images_path = images_info[:, 1]
    aligned_images_path = images_info[:, 2]
    features = _extract_feature(images_path, aligned_images_path)

    raw_conn = mysql_conn.get_raw_conn()
    try:
        sql_list = []
        features_id = images_info[:, 0]
        for feature_id in features_id:
            sql = "update tb_multiface_feature set feature=%s where feature_id={}".format(feature_id)
            sql_list.append(sql)

        cursor = raw_conn.cursor()
        for i, sql in enumerate(sql_list):
            bin_feature = mysql_connection.Binary(features[i])
            ret = cursor.execute(sql, (bin_feature,))
            if ret < 0:
                raise Exception("execute {} error!".format(sql))
        raw_conn.commit()
        cursor.close()
    except Exception as e:
        raw_conn.rollback()
        print('update error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    print('update end!')


def evaluate1():
    data_path = '/user_data/home/xiaj/res/20190912-cleaned'
    align_path = '/user_data/home/xiaj/res/20190912-cleaned-aligned'
    images_path, images_label = Datset.load_dataset(data_path, shuffle=False)

    img_dir = config["template_root_dir"]
    images_info = []
    for image_path in images_path:
        image_align = image_path.replace(data_path, align_path)
        image_align_path = image_align[:image_align.rfind('/')]
        if not os.path.exists(image_align_path):
            os.makedirs(image_align_path)
        images_info.append([image_path, image_align])
    images_info = np.array(images_info)

    images_info = images_info[0:1000, :]

    images_path = images_info[:, 0]
    aligned_images_path = images_info[:, 1]
    features = _extract_feature(images_path, aligned_images_path)

    print('debug')


def evaluate():
    # from PIL import Image
    # import matplotlib.pyplot as plt
    # import pickle
    # import pylab

    data_path = '/user_data/home/xiaj/res/20190912-cleaned-mtcnn_align160x160_margin32'
    # save_pkl_file = '/user_data/home/xiaj/res/20190912-cleaned-mtcnn_align160x160_margin32.pkl'
    images_path, images_label = Datset.load_dataset(data_path, shuffle=False)

    # images_path = images_path[0:100]
    features = face_online_pipline.feature(images_path)

    rslt = []
    for i, (feature, image_path) in enumerate(zip(features, images_path)):
        index, prob = face_online_pipline.recognize(feature, g_Features)
        # prob = round(prob, 4)
        persons_id, persons_name, persons_prob = _get_identity(index, prob, use_second_verify=False)
        # print('[flask_face.api_recognize]: index={}, person_id={}, person_name={}, confidence={}'.format(index, person_id, person_name, prob))

        rslt.append((image_path, persons_id, persons_name, persons_prob))

        # img = Image.open(image_path)
        # plt.figure("Image")  # 图像窗口名称
        # plt.imshow(img)
        # plt.axis('on')  # 关掉坐标轴为 off
        # plt.title('image')  # 图像题目
        # # plt.show()
        # pylab.show()
        # print('debug')

        # if prob > 0.5:
        #     image = cv2.imread(image_path)
        #     cv2.imshow('show', image)
        #     cv2.waitKey(0)
        tools.view_bar('rec identity: ', i, len(images_path))
    print('')

    rslt = np.array(rslt)


    # with open(save_pkl_file, 'wb') as f:
    #     pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(images_path, f, pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(images_label, f, pickle.HIGHEST_PROTOCOL)
    #
    #     pickle.dump(g_Features, f, pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(g_FeatureInfo, f, pickle.HIGHEST_PROTOCOL)

    print('debug')


@app.route("/")
def index():
    return render_template('index.html', admin_user=session.get("user_id"))


@app.route("/login", methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        user_name = request.form["user_name"]
        pass_wd = request.form["pass_wd"]
        if user_name is None or pass_wd is None:
            error = "用户名或者密码不能为空"
        else:
            try:
                sql = "select * from tb_admin_user where user_name='{}' and pass_wd=md5('{}') limit 1".format(user_name,
                                                                                                              pass_wd)
                cursor = mysql_conn.query(sql)
                if cursor is not None:
                    result = cursor.fetchone()
                    cursor.close()
                    if result is not None:
                        session['user_id'] = user_name
                        session.permanent = True
                        g.user = user_name
                        return redirect(url_for("temp_list"))
                    else:
                        error = "验证失败"
                else:
                    error = "验证失败"
                pass
            except Exception as e:
                print("[flask_face.login]:exception::{}".format(e))
                #return render_template_string("login.html")
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    if session.get("user_id") is not None:
        session.pop("user_id", None)
        g.user = None
    return redirect(url_for("index"))


@app.route("/temp_list", methods=['POST', 'GET'])
def temp_list():
    if not session.get("user_id"):
        error = "请先登录"
        return render_template("login.html")

    global mysql_conn
    try:
        sql = "select tbp.person_name, tbf.* from tb_person tbp inner join \
                (select person_id, group_concat(feature_mark) feature_mark, group_concat(img) img from tb_multiface_feature group by person_id) tbf \
                where tbp.person_id=tbf.person_id"
        tps = _sql_query(sql)
        templates = _get_template_list(tps, front_end=True)

        return render_template("temp_list.html", tps=templates, admin_user=session.get("user_id"))
        # return render_template("temp_list.html", tps=tps)
    except Exception as e:
        print("[flask_face.temp_list]:exception::{}".format(e))

        json_result = {error_code_key_str: error_code['error_database_index_error'],
                       error_desc_key_str: 'Params error'}
        return jsonify(json_result)


@app.route("/modify_pass", methods=['POST', 'GET'])
def modify_pass():
    if not session.get("user_id"):
        error = "请先登录"
        return redirect(url_for("index"))
    # if session.get("user_id") is not "admin":
    #     error = "无权操作"
    #     return redirect(url_for("index"))
    global mysql_conn
    error = None
    if request.method == 'POST':
        user_id = session.get("user_id")
        old_pass_wd = request.form["old_pass_wd"]
        pass_wd1 = request.form["pass_wd1"]
        pass_wd2 = request.form["pass_wd2"]
        if old_pass_wd is None or pass_wd1 is None or pass_wd2 is None:
            error = "密码不能为空"
        elif pass_wd1 != pass_wd2:
            error = "两次输入的密码不一样"
        elif old_pass_wd == pass_wd2:
            error = "新密码和老密码相同"
        else:
            sql = "update tb_admin_user set pass_wd=md5('{}') where user_name='{}' and pass_wd=md5('{}') ".format(
                pass_wd2, user_id, old_pass_wd)
            try:
                sql_ret = mysql_conn.execute(sql)
                if sql_ret <= 0:
                    error = "修改用户密码失败"
                else:
                    error = "修改用户密码成功"
            except Exception as e:
                error = "修改用户密码失败"
    return render_template("modify_pass.html", error=error, admin_user=session.get("user_id"))


@app.route("/add_user", methods=['POST', 'GET'])
def add_user():
    if not session.get("user_id"):
        error = "请先登录"
        return redirect(url_for("index"))
    if session.get("user_id") != "admin":
        error = "无权操作"
        return redirect(url_for("index"))
    error = None
    if request.method == 'POST':
        user_id = request.form["user_name"]
        pass_wd = request.form["pass_wd"]
        if user_id is None or pass_wd is None:
            error = "用户名或者密码不能为空"
        else:
            sql = "insert into tb_admin_user(user_name, pass_wd, role) values('{}', md5('{}'), 2)".format(user_id,
                                                                                                          pass_wd)
            try:
                sql_ret = mysql_conn.execute(sql)
                if sql_ret <= 0:
                    error = "添加管理员用户失败"
                else:
                    error = "添加管理员用户成功, 用户名:" + user_id
            except Exception as e:
                error = "添加管理员用户失败, 用户名：" + user_id
                # return render_template("add_user.html", error=error, admin_user=session.get("user_id"))
    return render_template("add_user.html", error=error, admin_user=session.get("user_id"))


@app.route("/show_app_log")
def show_app_log():
    global config
    log_content=None
    if not session.get("user_id"):
        error = "请先登录"
        return render_template("login.html", error=error)
    try:
        file_path = os.path.join(config["log_root_dir"], "flask.log")
        error = None
        if os.path.exists(file_path):
            file = open(file_path, "r")
            log_content = file.read()
            file.close()
        else:
            error = "no log file exist"
    except Exception as e:
        error = e
        return render_template("show_app_log.html", error=error, log_content=log_content)
    # return render_template("show_app_log.html", error=error, log_content=log_content)
    resp = make_response(log_content)
    resp.headers["Content-type"]="application/json;charset=UTF-8"
    return resp


@app.route("/add_template", methods=['POST', 'GET'])
def add_template():
    if not session.get("user_id"):
        error = "请先登录"
        return render_template("login.html")

    global mysql_conn
    error = None
    if request.method == 'POST':
        images_path = []
        raw_conn = mysql_conn.get_raw_conn()
        try:
            images, marks = _get_images_mark(request.files, front_end=True)
            if not request.form["person_name"]:
                error = "person_name empty"
            elif len(images) == 0:
                error = "template image empty"
            else:
                person_name = request.form["person_name"]
                images_path, aligned_images_path = _generate_image_path(person_name=person_name,
                                                                        person_num=len(marks))
                images = _decode_image(images, images_path)

                features = _extract_feature(images, aligned_images_path)
                images_name = [impath.split('/')[-1] for impath in images_path]

                sql = "insert into tb_person(person_name) values('{}') on duplicate key update person_name='{}'".format(
                    person_name, person_name)
                cursor = raw_conn.cursor()
                ret = cursor.execute(sql)

                person_id = cursor.lastrowid

                sql = _get_feature_insert_sql(person_id, marks, images_name)

                bin_features = [mysql_connection.Binary(feature) for feature in features]
                ret = cursor.execute(sql, bin_features)

                raw_conn.commit()  # TODO: 等两个表都插入完成再提交。
                cursor.close()
                if ret > 0:
                    update_cache(person_id, person_name, marks, images_name, features)

                app.logger.info('success add template, person_id={}, person_name={}'.format(person_id, person_name))
                return redirect(url_for("add_template"))
        except Exception as e:
            print("[flask_face.add_template]:exception::{}".format(e))
            if type(e) == MyException:
                app.logger.info('not detect face!')
                error = '没有检测到人脸'
            else:
                app.logger.error(traceback.format_exc())
                error = '参数错误'

            raw_conn.rollback()
            _del_images(images_path)

            # return render_template("add_multitemplate.html", error=error)
    return render_template("add_multitemplate.html", error=error)


@app.route("/query", methods=['POST', 'GET'])
def query():
    if not session.get("user_id"):
        error = "请先登录"
        return render_template("login.html")

    global mysql_conn
    error = None
    if request.method == 'POST':
        if not request.form["keyword"]:
            error = "keyword empty"
        else:
            keyword = request.form["keyword"]

            try:
                # sql = "select person_id, person_name, img from tb_multiface_feature where person_id='{}' or person_name like '%{}%'".format(keyword, keyword)
                sql = "select tbp.person_name, tbf.* from tb_person tbp inner join \
                                    (select person_id, group_concat(feature_mark) feature_mark, group_concat(img) img \
                                    from tb_multiface_feature group by person_id) tbf \
                                where tbp.person_id=tbf.person_id and (tbp.person_id='{}' or tbp.person_name like '%{}%')"
                sql = sql.format(keyword, keyword)

                cursor = mysql_conn.query(sql)
                tps = {}
                if cursor != None:
                    tps = cursor.fetchall()
                    cursor.close()

                templates = _get_template_list(tps, front_end=True)
                return render_template("temp_list.html", tps=templates)
            except Exception as e:
                print("[flask_face.query]:exception::{}".format(e))

                return render_template_string("query.html")
    return render_template("query.html")


@app.route("/modify_template", methods=['POST', 'GET'])
def modify_template():
    try:
        person_id = request.args.get("person_id")
        if not person_id:
            return redirect("/")
        ret = _get_record(person_id)
        if ret is not None:
            img = ret["img"]
            if os.path.exists(img):
                os.remove(img)
        ret = _del_record(person_id)
        if ret:
            update_cache(int(person_id), mode='delete')
    except Exception as e:
        print("[flask_face.remove_template]:exception::{}".format(e))
        return redirect("/")
    return redirect("/")


@app.route("/recognize", methods=['POST', 'GET'])
def recognize():
    print('[recognize]:: start !!!!!!!!!!!!!!!!')
    t0 = time.time()
    if not session.get("user_id"):
        error = "请先登录"
        return render_template("login.html")
    t1 = time.time()
    print('[recognize]:: session time={}'.format(t1 - t0))
    t0 = t1

    global mysql_conn
    error = None
    if request.method == 'POST':
        t1 = time.time()
        print('[recognize]:: request.method time={}'.format(t1 - t0))
        t0 = t1

        if not request.files["img"]:
            error = "image empty"
        else:
            t1 = time.time()
            print('[recognize]:: request.files[img] time={}'.format(t1 - t0))
            t0 = t1

            images_path = []
            try:
                images, marks = _get_images_mark(request.files, front_end=True)
                t1 = time.time()
                print('[recognize]:: _get_images_mark time={}'.format(t1 - t0))
                t0 = t1

                images_path, aligned_images_path = _generate_image_path()
                t1 = time.time()
                print('[recognize]:: _generate_image_path time={}'.format(t1 - t0))
                t0 = t1

                images = _decode_image(images, images_path)
                t1 = time.time()
                print('[recognize]:: _decode_image time={}'.format(t1 - t0))
                t0 = t1

                feature = _extract_feature(images, aligned_images_path)
                t1 = time.time()
                print('[recognize]:: _extract_feature time={}'.format(t1 - t0))
                t0 = t1

                index, prob = face_online_pipline.recognize(feature, g_Features)
                t1 = time.time()
                print('[recognize]:: recognize time={}'.format(t1 - t0))
                t0 = t1

                # prob = round(prob, 4)
                persons_id, persons_name, persons_prob = _get_identity(index, prob, use_second_verify=config['use_second_verify'])
                print(
                    '[flask_face.api_recognize]: index={}, persons_id={}, persons_name={}, confidence={}'.format(index,
                                                                                                               persons_id,
                                                                                                               persons_name,
                                                                                                               persons_prob))

                data = []
                for person_id, person_name, person_prob in zip(persons_id, persons_name, persons_prob):
                    data.append({'person_id': person_id, 'person_name': person_name, 'confidence': person_prob})

                json_result = {error_code_key_str: error_code['error_ok'], 'data': data}
                t1 = time.time()
                print('[recognize]:: _get_identity time={}'.format(t1 - t0))

                # ver_save_name = tools.strcat([person_id, person_name, prob], cat_mark='_') + '.' + images_path[0].split('.')[-1]
                # ver_save_path = os.path.join(config["verify_root_dir"], ver_save_name)
                # shutil.move(images_path[0], ver_save_path)

                # 前端网页添加的基本都是历史照片，所以不必重复存储了。
                # _move_verify_file2specify_path(persons_id, persons_name, persons_prob, images_path[0])
                _del_images(images_path)

                return json.dumps(json_result, ensure_ascii=False)
            except Exception as e:
                print("[flask_face.recognize]:exception::{}".format(e))
                if type(e) == MyException:
                    app.logger.info('not detect face!')
                    error = '没有检测到人脸'
                else:
                    app.logger.error(traceback.format_exc())
                    error = '参数错误'
                _del_images(images_path)

                # return render_template("recognize.html", error=error)
    t1 = time.time()
    print('[recognize]:: total time={}'.format(t1 - t0))
    return render_template("recognize.html", error=error)


@app.route("/remove_template")
def remove_template():
    if not session.get("user_id"):
        error = "请先登录"
        return render_template("login.html")

    try:
        person_id = request.args.get("person_id")
        if not person_id:
            return redirect(url_for("temp_list"))

        ret = _get_record(person_id)
        if ret is not None:
            images = [record['img'] for record in ret]
            _del_images(images, root_path=config["template_root_dir"])
        ret = _del_record(person_id)
        if ret:
            update_cache(person_id, mode='delete')

        app.logger.info('success remove template, person_id={}!'.format(person_id))
    except Exception as e:
        app.logger.error(traceback.format_exc())
        print("[flask_face.remove_template]:exception::{}".format(e))
        return redirect(url_for("temp_list"))
    return redirect(url_for("temp_list"))


@app.route("/show_img")
def show_img():
    #if not session.get("user_id"):
    #    error = "请先登录"
    #    return render_template("login.html")

    try:
        img_name = request.args.get("img_name")
        img_path = config['template_root_dir'] + '/' + img_name

        if not img_path:
            return redirect("/")
        img_file = open(img_path, 'rb')
        img_data = img_file.read()
        response = make_response(img_data)
        response.headers["Content-Type"] = "image/jpeg"
        return response
        # return Response()

        # return render_template("show_img.html", img=img_data)
    except Exception as e:
        print("[flask_face.show_img]:exception::{}".format(e))
        return redirect("/")


@app.route("/api/face/login", methods=['POST', 'GET'])
def api_login():
    try:
        recv_data = json.loads(request.get_data().decode('utf-8'))

        user_name = recv_data['user_name']
        pass_wd = recv_data['pass_wd']

        # token = generate_token(user_name + ':' + pass_wd)
        # ret = certify_token(user_name + ':' + pass_wd, token)

        sql = "select * from tb_admin_user where user_name='{}' and pass_wd=md5('{}') limit 1".format(user_name,
                                                                                                      pass_wd)
        cursor = mysql_conn.query(sql)
        if cursor is not None:
            result = cursor.fetchone()
            cursor.close()
            if result is not None:
                user_id = result['user_id']
                token = generate_token(str(user_id), 3600*24)
                session['user_id'] = user_name
                session.permanent = True
                g.user = user_name

                data = {"user_id": user_id, token_key_str: token}
                json_result = {error_code_key_str: error_code['error_ok'], data_key_str: data}
                app.logger.info('success add template, user_name={}!'.format(user_name))
                return jsonify(json_result)
            else:
                error = "验证失败"
        else:
            error = "验证失败"
        pass
    except Exception as e:
        print("[flask_face.login]:exception::{}".format(e))

    app.logger.error("Login error!")
    json_result = {error_code_key_str: error_code['error_login_error'],
                   error_desc_key_str: 'Login error'}
    return jsonify(json_result)


@app.route("/api/face/logout", methods=['POST', 'GET'])
def api_logout():
    try:
        recv_data = json.loads(request.get_data().decode('utf-8'))
        user_id = recv_data['user_id']
        token = recv_data['token']
        if not certify_token(str(user_id), token):
            raise Exception('token certify error!')
    except Exception as e:
        json_result = {error_code_key_str: error_code['error_login_error']}
        app.logger.info(e)
        return jsonify(json_result)

    # TODO: logout接口校验成功后暂时先直接返回ok, 这样客户端的程序逻辑不用变，后期待服务器用redis机制完善后再来处理logout.
    json_result = {error_code_key_str: error_code['error_ok']}
    app.logger.info('logout!')
    return jsonify(json_result)


@app.route("/api/face/modify_pass", methods=['POST', 'GET'])
def api_modify_pass():
    try:
        recv_data = json.loads(request.get_data().decode('utf-8'))
        user_id = recv_data['user_id']
        token = recv_data['token']
        if not certify_token(str(user_id), token):
            raise Exception('token certify error!')
    except Exception as e:
        json_result = {error_code_key_str: error_code['error_login_error']}
        app.logger.info(e)
        return jsonify(json_result)

    # if not session.get("user_id"):
    #     error = "请先登录"
    #     return redirect(url_for("index"))
    # if session.get("user_id") is not "admin":
    #     error = "无权操作"
    #     return redirect(url_for("index"))
    global mysql_conn
    error = None
    # if request.method == 'POST':
    try:
        # user_id = session.get("user_id")
        # old_pass_wd = request.form["old_pass_wd"]
        # pass_wd1 = request.form["pass_wd1"]
        # pass_wd2 = request.form["pass_wd2"]

        old_pass_wd = recv_data["old_pass_wd"]
        pass_wd1 = recv_data["pass_wd1"]
        pass_wd2 = recv_data["pass_wd2"]
        if old_pass_wd is None or pass_wd1 is None or pass_wd2 is None:
            error = "密码不能为空"
        elif pass_wd1 != pass_wd2:
            error = "两次输入的密码不一样"
        elif old_pass_wd == pass_wd2:
            error = "新密码和老密码相同"
        else:
            sql = "update tb_admin_user set pass_wd=md5('{}') where user_id='{}' and pass_wd=md5('{}') ".format(
                pass_wd2, user_id, old_pass_wd)
            sql_ret = mysql_conn.execute(sql)
            if sql_ret <= 0:
                error = "修改用户密码失败"
                json_result = {error_code_key_str: error_code['error_modify_user_error']}
            else:
                error = "修改用户密码成功"
                json_result = {error_code_key_str: error_code['error_ok']}

            return jsonify(json_result)
    except Exception as e:
        error = "修改用户密码失败"

        json_result = {error_code_key_str: error_code['error_modify_user_error']}
        return jsonify(json_result)


@app.route('/api/face/add_template', methods=['POST', "GET"])
def api_add_template():
    """
    :return:
    """
    try:
        recv_data = json.loads(request.get_data().decode('utf-8'))
        user_id = recv_data['user_id']
        token = recv_data['token']
        if not certify_token(str(user_id), token):
            raise Exception('token certify error!')
    except Exception as e:
        json_result = {error_code_key_str: error_code['error_login_error']}
        app.logger.info(e)
        return jsonify(json_result)

    global mysql_conn

    images_path = []
    raw_conn = mysql_conn.get_raw_conn()
    try:
        person_name = recv_data['person_name']

        images, marks = _get_images_mark(recv_data["image"])
        images_path, aligned_images_path = _generate_image_path(person_name=person_name, person_num=len(marks))
        images = _decode_image(images)
        _save_raw_image(images_path, images)

        features = _extract_feature(images, aligned_images_path)
        images_name = [impath.split('/')[-1] for impath in images_path]

        sql = "insert into tb_person(person_name) values('{}') on duplicate key update person_name='{}'".format(person_name, person_name)
        cursor = raw_conn.cursor()
        ret = cursor.execute(sql)

        person_id = cursor.lastrowid

        sql = _get_feature_insert_sql(person_id, marks, images_name)

        bin_features = [mysql_connection.Binary(feature) for feature in features]
        ret = cursor.execute(sql, bin_features)

        raw_conn.commit()  # TODO: 等两个表都插入完成再提交。
        cursor.close()
        if ret > 0:
            update_cache(person_id, person_name, marks, images_name, features)

        json_result = {error_code_key_str: error_code['error_ok']}
        app.logger.info('success add template, person_id={}, person_name={}!'.format(person_id, person_name))
        return jsonify(json_result)
    except Exception as e:
        print("[flask_face.api_add_template]:exception::{}".format(e))
        if type(e) == MyException:
            app.logger.info('not detect face!')
            json_result = {error_code_key_str: error_code['error_not_detect_face']}
        else:
            app.logger.error(traceback.format_exc())
            json_result = {error_code_key_str: error_code['error_add_template_error'],
                           error_desc_key_str: 'Insert error'}

        raw_conn.rollback()
        _del_images(images_path)

        return jsonify(json_result)


@app.route('/api/face/modify_template', methods=['POST'])
def api_modify_template():
    """
    :return:
    """
    try:
        recv_data = json.loads(request.get_data().decode('utf-8'))
        user_id = recv_data['user_id']
        token = recv_data['token']
        if not certify_token(str(user_id), token):
            raise Exception('token certify error!')
    except Exception as e:
        json_result = {error_code_key_str: error_code['error_login_error']}
        app.logger.info(e)
        return jsonify(json_result)

    features = person_name = None
    images_path = []
    raw_conn = mysql_conn.get_raw_conn()
    try:
        images_name = marks = old_images_del = []

        person_id = int(recv_data["person_id"])
        sql_dict = {"tb_person": [], "tb_multiface_feature": []}
        recv_keys = list(recv_data.keys())
        if 'person_name' in recv_keys:
            person_name = recv_data['person_name']
            sql_dict["tb_person"].append("update tb_person set person_name='{}' where person_id={}".format(person_name, person_id))

        if 'image' in recv_keys:
            sql_query_person = "select feature_mark, img from tb_multiface_feature where person_id={}".format(person_id)
            tps = _sql_query(sql_query_person)
            old_images = [record['img'] for record in tps]
            old_marks = [record['feature_mark'] for record in tps]
            if person_name is None:
                person_name = g_PersonInfo[person_id]

            images, marks = _get_images_mark(recv_data["image"])
            images_path, aligned_images_path = _generate_image_path(person_name=person_name, person_num=len(marks))
            images = _decode_image(images)
            _save_raw_image(images_path, images)
            features = _extract_feature(images, aligned_images_path)

            images_name = [impath.split('/')[-1] for impath in images_path]
            for mark, imname in zip(marks, images_name):
                if mark in old_marks:
                    # TODO: 按照客户端现在的逻辑，这个条件下的语句是不会被执行到的，因为对于一个mark已经存在的图片，如果用户想要更新这张图片，则客户端会先删除这个mark的图片，然后再添加新的图片。
                    # TODO: 但是postman在调试的时候，可以直接覆盖原有mark对应的图片，故这里暂且还是维持这种写法。
                    index = old_marks.index(mark)
                    old_images_del.append(old_images[index])
                    sql = "update tb_multiface_feature set img='{}', feature=%s where person_id={} and feature_mark='{}'".format(imname, person_id, mark)
                else:
                    sql = _get_feature_insert_sql(person_id, [mark], [imname])
                sql_dict["tb_multiface_feature"].append(sql)

        cursor = raw_conn.cursor()
        for sql in sql_dict["tb_person"]:
            ret = cursor.execute(sql)  # TODO: cursor.execute执行结果返回0也是ok的吗？
            if ret < 0:
                raise Exception("execute {} error!".format(sql))
        for i, sql in enumerate(sql_dict["tb_multiface_feature"]):
            bin_feature = mysql_connection.Binary(features[i])
            ret = cursor.execute(sql, (bin_feature,))
            if ret < 0:
                raise Exception("execute {} error!".format(sql))
        raw_conn.commit()
        cursor.close()
        update_cache(person_id, person_name=person_name, features_mark=marks, images_path=images_name, features=features, mode='modify')

        _del_images(old_images_del, config["template_root_dir"])

        app.logger.info('success modify template, person_id={}!'.format(person_id))
        json_result = {error_code_key_str: error_code['error_ok']}
        return jsonify(json_result)
    except Exception as e:
        print("[flask_face.api_modify_template]:exception::{}".format(e))
        if type(e) == MyException:
            app.logger.info('not detect face!')
            json_result = {error_code_key_str: error_code['error_not_detect_face']}
        else:
            app.logger.error(traceback.format_exc())
            json_result = {error_code_key_str: error_code['error_add_template_error'],
                           error_desc_key_str: 'Modify error'}

        raw_conn.rollback()

        _del_images(images_path)

        return jsonify(json_result)


@app.route('/api/face/modify_template/del', methods=['POST'])
def api_modify_template_del():
    """
    :return:
    """
    global g_FeatureInfo

    try:
        recv_data = json.loads(request.get_data().decode('utf-8'))
        user_id = recv_data['user_id']
        token = recv_data['token']
        if not certify_token(str(user_id), token):
            raise Exception('token certify error!')
    except Exception as e:
        json_result = {error_code_key_str: error_code['error_login_error']}
        app.logger.info(e)
        return jsonify(json_result)

    raw_conn = mysql_conn.get_raw_conn()
    try:
        person_id = int(recv_data["person_id"])
        sql = "select feature_mark from tb_multiface_feature where person_id={}".format(person_id)
        tps = _sql_query(sql)
        db_marks = set([t['feature_mark'] for t in tps])
        marks = recv_data['delete']

        if db_marks.issubset(marks):
            db_marks = list(db_marks)
            db_marks.sort()
            marks.remove(db_marks[0])

        if len(marks) > 0:
            for mark in marks:
                _del_record(person_id, mark)

            indexs = _get_CacheFeatureInfoIndex_byPersonidMarks(person_id, marks)
            del_feature_info = g_FeatureInfo[indexs]
            if len(del_feature_info):
                _del_images(list(del_feature_info[:, -1]), config["template_root_dir"])
            update_cache(person_id, features_mark=marks, mode='delete')

            app.logger.info('success modify template, person_id={}!'.format(person_id))
            json_result = {error_code_key_str: error_code['error_ok']}
        else:
            print('[api_modify_template_del]: can not delete the last one. At least one image per-person!')
            json_result = {error_code_key_str: error_code['error_at_least_one_image_perperson']}

        return jsonify(json_result)
    except Exception as e:
        print("[flask_face.api_remove_template_del]:exception::{}".format(e))
        json_result = {error_code_key_str: error_code['error_remove_template_error'],
                       error_desc_key_str: 'Params error'}

        raw_conn.rollback()

        return jsonify(json_result)


@app.route('/api/face/remove_template', methods=['POST'])
def api_remove_template():
    """
    :return:
    """
    try:
        recv_data = json.loads(request.get_data().decode('utf-8'))
        user_id = recv_data['user_id']
        token = recv_data['token']
        if not certify_token(str(user_id), token):
            raise Exception('token certify error!')
    except Exception as e:
        json_result = {error_code_key_str: error_code['error_login_error']}
        app.logger.info(e)
        return jsonify(json_result)

    try:
        all_person_id = recv_data["person_id"].split(',')
        for person_id in all_person_id:
            person_id = int(person_id)
            ret = _get_record(person_id)
            if ret is not None:
                images = [record['img'] for record in ret]
                _del_images(images, root_path=config["template_root_dir"])
            ret = _del_record(person_id)
            if ret:
                update_cache(person_id, mode='delete')

        app.logger.info('success remove template, person_id={}!'.format(all_person_id))
        json_result = {error_code_key_str: error_code['error_ok']}
        return jsonify(json_result)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        print("[flask_face.api_remove_template]:exception::{}".format(e))
        json_result = {error_code_key_str: error_code['error_remove_template_error'],
                       error_desc_key_str: 'Params error'}
        return jsonify(json_result)


@app.route('/api/face/recognize', methods=['POST'])
def api_recognize():
    print('[api_recognize]:: start !!!!!!!!!!!!!!!!')
    t0 = time.time()
    try:
        data = request.get_data()
        t1 = time.time()
        print('[api_recognize]:: request.get_data time={}'.format(t1 - t0))
        t0 = t1

        data = data.decode('utf-8')
        t1 = time.time()
        print('[api_recognize]:: data.decode(utf-8) time={}'.format(t1 - t0))
        t0 = t1

        recv_data = json.loads(data)
        # recv_data = json.loads(request.get_data().decode('utf-8'))
        t1 = time.time()
        print('[api_recognize]:: json.loads time={}'.format(t1 - t0))
        t0 = t1

        user_id = recv_data['user_id']
        token = recv_data['token']
        if not certify_token(str(user_id), token):
            raise Exception('token certify error!')
    except Exception as e:
        json_result = {error_code_key_str: error_code['error_login_error']}
        app.logger.info(e)
        return jsonify(json_result)
    t1 = time.time()
    print('[api_recognize]:: token certify time={}'.format(t1 - t0))
    t0 = t1

    images_path = []
    try:
        images_path, aligned_images_path = _generate_image_path()
        images = _decode_image(recv_data['image'])
        t1 = time.time()
        print('[api_recognize]:: decode image time={}'.format(t1 - t0))
        t0 = t1

        _save_raw_image(images_path, images)
        t1 = time.time()
        print('[api_recognize]:: save image time={}'.format(t1 - t0))
        t0 = t1

        feature = _extract_feature(images, aligned_images_path)
        t1 = time.time()
        print('[api_recognize]:: extract feature time={}'.format(t1 - t0))
        t0 = t1

        index, prob = face_online_pipline.recognize(feature, g_Features)
        t1 = time.time()
        print('[api_recognize]:: recognize time={}'.format(t1 - t0))
        t0 = t1

        # prob = round(prob, 4)
        persons_id, persons_name, persons_prob = _get_identity(index, prob, use_second_verify=config['use_second_verify'])
        print(
            '[flask_face.api_recognize]: index={}, person_id={}, person_name={}, confidence={}'.format(index, persons_id,
                                                                                                       persons_name,
                                                                                                       persons_prob))

        data = []
        for person_id, person_name, person_prob in zip(persons_id, persons_name, persons_prob):
            data.append({'person_id': person_id, 'person_name': person_name, 'confidence': person_prob})
        json_result = {error_code_key_str: error_code['error_ok'], 'data': data}

        t1 = time.time()
        print('[api_recognize]:: get identity time={}'.format(t1 - t0))

        _move_verify_file2specify_path(persons_id, persons_name, persons_prob, images_path[0])

        app.logger.info(json_result)

        return json.dumps(json_result, ensure_ascii=False)
    except Exception as e:
        print("[flask_face.api_recognize]:exception::{}".format(e))
        if type(e) == MyException:
            app.logger.info('not detect face!')
            json_result = {error_code_key_str: error_code['error_not_detect_face']}
        else:
            app.logger.error(traceback.format_exc())
            json_result = {error_code_key_str: error_code['error_recognize_no_result'],
                           error_desc_key_str: 'Params error'}

        _del_images(images_path)

        return jsonify(json_result)


@app.route('/api/face/temp_list', methods=['POST', 'GET'])
def api_temp_list():
    try:
        recv_data = json.loads(request.get_data().decode('utf-8'))
        user_id = recv_data['user_id']
        token = recv_data['token']
        if not certify_token(str(user_id), token):
            raise Exception('token certify error!')
    except Exception as e:
        json_result = {error_code_key_str: error_code['error_login_error']}
        app.logger.info(e)
        return jsonify(json_result)

    global mysql_conn
    try:
        sql = "select tbp.person_name, tbf.* from tb_person tbp inner join \
        (select person_id, group_concat(feature_mark) feature_mark, group_concat(img) img from tb_multiface_feature group by person_id) tbf \
        where tbp.person_id=tbf.person_id"

        tps = _sql_query(sql)
        templates = _get_template_list(tps)

        json_result = {error_code_key_str: error_code['error_ok'], data_key_str: templates}

        return json.dumps(json_result, ensure_ascii=False)
    except Exception as e:
        print("[flask_face.api_temp_list]:exception::{}".format(e))

        json_result = {error_code_key_str: error_code['error_database_index_error'],
                       error_desc_key_str: 'Params error'}
        return jsonify(json_result)


@app.route("/api/face/query", methods=['POST'])
def api_query():
    try:
        recv_data = json.loads(request.get_data().decode('utf-8'))
        user_id = recv_data['user_id']
        token = recv_data['token']
        if not certify_token(str(user_id), token):
            raise Exception('token certify error!')
    except Exception as e:
        json_result = {error_code_key_str: error_code['error_login_error']}
        app.logger.info(e)
        return jsonify(json_result)

    global mysql_conn
    try:
        recv_data = json.loads(request.get_data().decode('utf-8'))
        keyword = recv_data['keyword']
        sql = "select tbp.person_name, tbf.* from tb_person tbp inner join \
                    (select person_id, group_concat(feature_mark) feature_mark, group_concat(img) img \
                    from tb_multiface_feature group by person_id) tbf \
                where tbp.person_id=tbf.person_id and (tbp.person_id='{}' or tbp.person_name like '%{}%')"
        sql = sql.format(keyword, keyword)

        cursor = mysql_conn.query(sql)
        tps = []
        if cursor != None:
            tps = cursor.fetchall()
            cursor.close()
        templates = _get_template_list(tps)

        json_result = {error_code_key_str: error_code['error_ok'], data_key_str: templates}

        return json.dumps(json_result, ensure_ascii=False)
    except Exception as e:
        print("[flask_face.query]:exception::{}".format(e))

        json_result = {error_code_key_str: error_code['error_database_index_error'],
                       error_desc_key_str: 'Params error'}
        return jsonify(json_result)


if __name__ == '__main__':
    print('[flask_multiface.__main__] start!!!!!!!!!!!!!!!!!!!!!')
    log_init()
    model_init()
    mysql_init()
    # update_feature()
    # evaluate()

    app.run(debug=False, host="0.0.0.0", port=10013)
