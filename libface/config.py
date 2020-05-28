# config.py
import os
import socket
import getpass


DeployConfig = {
    'device': 'cuda:0',
    'faceid_model_path': '/disk1/home/xiaj/dev/alg_verify/face/facenet/save_model/VGGFace2_GCWebFace',
    'faceid_model_list': [
        #('20191122-044343', 'model-20191122-044343-acc0.996000-val0.991667-prec0.998999.ckpt-235'),
        #('20191016-203214', 'model-20191016-203214.ckpt-275'),
        #('20190719-062455', 'model-20190719-062455.ckpt-498'),
        #('20191127-005344', 'model-20191127-005344-acc0.996167-val0.993333-prec0.998671.ckpt-341'),  # 0.5151
        #('20191127-005344', 'model-20191127-005344-acc0.994333-val0.991667-prec0.999001.ckpt-391'),  # 0.5279
        #('20190517-231453', 'model-20190517-231453.ckpt-4'),
        #('20191203-083355', 'model-20191203-083355-acc0.996667-val0.992667-prec0.998668.ckpt-330'),  # 0.6003
        #('20191203-083355', 'model-20191203-083355-final.ckpt-400'),  # 0.5985
        #('20191204-001947', 'model-20191204-001947-acc0.996500-val0.994000-prec0.998668.ckpt-314'),  # 0.615
        #('20191204-001947', 'model-20191204-001947-acc0.996333-val0.993000-prec0.999336.ckpt-393'),  # 0.4861, but 通过率下降了点
        ('20191208-201809', 'model-20191208-201809-acc0.996833-val0.989000-prec0.999338.ckpt-352'),  # 测试良久
        #('20191212-213848', 'model-20191212-213848-acc0.996833-val0.991667-prec0.998670.ckpt-386'),  训练包含gcface
    ],
    'tmpdir': '.tmpface',
}

SysConfig = {
    "home_path": os.environ['HOME'],
    "user_name": getpass.getuser(),
    "host_name": socket.gethostname()
}