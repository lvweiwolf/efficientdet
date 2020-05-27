r"""Convert Custom(labelme dataset) dataset to TFRecord.

Example usage:
    python create_custom_tfrecord.py  --data_dir=/tmp/销钉裁剪_train  \
        --class_name_path=VOC2012  --output_path=/tmp/custom
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os

from absl import app
from absl import flags
from absl import logging
from glob import glob

from lxml import etree
import PIL.Image
import tensorflow as tf

import tfrecord_util


flags.DEFINE_string('data_dir', '', '包含jpg 和.json 标记文件的数据集目录.')
flags.DEFINE_string('output_path', '', '.tfrecord 文件输出路径.')
flags.DEFINE_string('class_name_path', None, '.names 类别和标签索引文件.')
flags.DEFINE_integer('num_shards', 32, '.tfrecord 分页文件数.')
flags.DEFINE_integer('num_images', None, '处理图片最大数量.')
FLAGS = flags.FLAGS

unknow_label_map_dict = {
    'Unknow-1': 0,
    'Unknow-2': 1,
    'Unknow-3': 2,
    'Unknow-4': 3,
    'Unknow-5': 4,
    'Unknow-6': 5,
    'Unknow-7': 6,
    'Unknow-8': 7,
    'Unknow-9': 8,
    'Unknow-10': 9
}

GLOBAL_IMG_ID = 0  # global image id.
GLOBAL_ANN_ID = 0  # global annotation id.

def get_image_id(filename):
    """Convert a string to a integer."""
    # Warning: this function is highly specific to pascal filename!!
    # Given filename like '2008_000002', we cannot use id 2008000002 because our
    # code internally will convert the int value to float32 and back to int, which
    # would cause value mismatch int(float32(2008000002)) != int(2008000002).
    # COCO needs int values, here we just use a incremental global_id, but
    # users should customize their own ways to generate filename.
    del filename
    global GLOBAL_IMG_ID
    GLOBAL_IMG_ID += 1
    return GLOBAL_IMG_ID

def get_ann_id():
    """Return unique annotation id across images."""
    global GLOBAL_ANN_ID
    GLOBAL_ANN_ID += 1
    return GLOBAL_ANN_ID

def _load_class_dict(data):
    '''
    {
        id : label_name
        ...
    }
    '''
    names = {}
    for ID, name in enumerate(data):
        names[ID] = name.strip('\n')
        
    return names

def _load_label_map_dict(data):
    '''
    {
        label_name : id
        ...
    }
    '''
    label_map_dict = {}
    for ID, name in enumerate(data):
        label_map_dict[name.strip('\n')] = ID
        
    return label_map_dict

def _get_boundingbox(list):
    import numpy as np
    bbox_coords = np.array(list, dtype=np.float)
    # Boundingbox
    x = bbox_coords[:, 0]
    y = bbox_coords[:, 1]
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    
    return xmin, ymin, xmax, ymax


def json_to_tf_example(example,
                       label_map_dict,
                       ann_json_dict=None):
   
    img_path = os.path.splitext(example)[0] + '.jpg'
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    # SHA校验码
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = image.width
    height = image.height
    
    with open(example, 'r') as f:
        data = json.load(f)
    
    image_id = get_image_id(data['imagePath'])
    if ann_json_dict:
        image = {
            'file_name': data['imagePath'],
            'height': height,
            'width': width,
            'id': image_id,
        }
        ann_json_dict['images'].append(image)
    
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    # truncated = []
    # poses = []
    # difficult_obj = []
    
    if 'shapes' in data:
        for obj in data['shapes']:
            # 忽略.names文件中不存在的类别 label
            if obj['label'] not in label_map_dict:
                continue
        
            # 转化bbox坐标
            x0, y0, x1, y1 = _get_boundingbox(obj['points'])
            xmin.append(x0 / width)
            ymin.append(y0 / height)
            xmax.append(x1 / width)
            ymax.append(y1 / height)
            # 类别标签设置utf8编码，以支持中文
            classes_text.append(obj['label'].encode('utf-8'))
            classes.append(label_map_dict[obj['label']])
            
            if ann_json_dict:
                abs_xmin = int(x0)
                abs_ymin = int(y0)
                abs_xmax = int(x1)
                abs_ymax = int(y1)
                abs_width = abs_xmax - abs_xmin
                abs_height = abs_ymax - abs_ymin
                ann = {
                    'area': abs_width * abs_height,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': [abs_xmin, abs_ymin, abs_width, abs_height],
                    'category_id': label_map_dict[obj['label']],
                    'id': get_ann_id(),
                    'ignore': 0,
                    'segmentation': [],
                }
                ann_json_dict['annotations'].append(ann)
    
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height':
                    tfrecord_util.int64_feature(height),
                'image/width':
                    tfrecord_util.int64_feature(width),
                'image/filename':
                    tfrecord_util.bytes_feature(
                        data['imagePath'].encode('utf8')),
                'image/source_id':
                    tfrecord_util.bytes_feature(str(image_id).encode('utf8')),
                'image/key/sha256':
                    tfrecord_util.bytes_feature(key.encode('utf8')),
                'image/encoded':
                    tfrecord_util.bytes_feature(encoded_jpg),
                'image/format':
                    tfrecord_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin':
                    tfrecord_util.float_list_feature(xmin),
                'image/object/bbox/xmax':
                    tfrecord_util.float_list_feature(xmax),
                'image/object/bbox/ymin':
                    tfrecord_util.float_list_feature(ymin),
                'image/object/bbox/ymax':
                    tfrecord_util.float_list_feature(ymax),
                'image/object/class/text':
                    tfrecord_util.bytes_list_feature(classes_text),
                'image/object/class/label':
                    tfrecord_util.int64_list_feature(classes),
            }))
    
    return tf_example
    
def main(_):
    # 确认必要参数
    if not FLAGS.output_path:
        raise ValueError('tfrecord 输出路径不允许为空.')
    
    logging.info('写入tfrecord 到路径: %s', FLAGS.output_path)
    
    # 创建tfrecord文件写入工具
    writers = [
        tf.python_io.TFRecordWriter(FLAGS.output_path + '-%05d-of-%05d.tfrecord' %
                                    (i, FLAGS.num_shards))
        for i in range(FLAGS.num_shards)
    ]
    
    if FLAGS.class_name_path:
        with tf.io.gfile.GFile(FLAGS.class_name_path, 'r') as f:
            label_map_dict = _load_label_map_dict(f)
    else:
        label_map_dict = unknow_label_map_dict
        
    ann_json_dict = {
        'images': [],
        'type': 'instances',
        'annotations': [],
        'categories': []
    }
    
    for class_name, class_id in label_map_dict.items():
        cls = {'supercategory': 'none', 'id': class_id, 'name': class_name}
        ann_json_dict['categories'].append(cls)
        
    logging.info('从路径 %s 读取数据集.', FLAGS.data_dir)
    example_list = glob(os.path.join(FLAGS.data_dir, '**/*.json'), recursive=True)
    for idx, example in enumerate(example_list):
        if FLAGS.num_images and idx >= FLAGS.num_images:
            break 
                   
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(example_list))

        # 转换数据到tfrecord
        tf_example = json_to_tf_example(example,
                                        label_map_dict,
                                        ann_json_dict=ann_json_dict)
        writers[idx % FLAGS.num_shards].write(tf_example.SerializeToString())
    
    # 关闭文件
    for writer in writers:
        writer.close()
        
    json_file_path = os.path.join(
        os.path.dirname(FLAGS.output_path),
        'json_' + os.path.basename(FLAGS.output_path) + '.json')
    with tf.io.gfile.GFile(json_file_path, 'w') as f:
        json.dump(ann_json_dict, f, indent=4, ensure_ascii=False)
        
    return 0

if __name__ == '__main__':
    app.run(main)   # 入口，需要返回码
