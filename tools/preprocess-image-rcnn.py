import argparse
import base64
import os, sys
import csv
import itertools

csv.field_size_limit(sys.maxsize)
sys.path.append(os.getcwd())

import h5py
import numpy as np
from tqdm import tqdm

from utilities import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

    features_shape = (
        82783 + 40504 if not args.test else 81434,  # number of images in trainval or in test
        config.rcnn_output_size,
        config.output_features,
    )
    boxes_shape = (
        features_shape[0],
        config.rcnn_output_size,
        4,
    )

    h5_path = config.rcnn_trainval_path if not args.test else config.rcnn_test_path
    tsv_path = config.bottom_up_trainval_path if not args.test else config.bottom_up_test_path

    with h5py.File(h5_path, libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float32')
        boxes = fd.create_dataset('boxes', shape=boxes_shape, dtype='float32')
        coco_ids = fd.create_dataset('ids', shape=(features_shape[0],), dtype='int32')
        widths = fd.create_dataset('widths', shape=(features_shape[0],), dtype='int32')
        heights = fd.create_dataset('heights', shape=(features_shape[0],), dtype='int32')

        readers = []
        for filename in os.listdir(tsv_path):
            if not '.tsv' in filename:
                continue
            full_filename = os.path.join(tsv_path, filename)
            fd = open(full_filename, 'r')
            reader = csv.DictReader(fd, delimiter='\t', fieldnames=FIELDNAMES)
            readers.append(reader)

        reader = itertools.chain.from_iterable(readers)
        for i, item in enumerate(tqdm(reader, total=features_shape[0])):
            img_id = item['image_id']
            coco_ids[i] = int(img_id)
            widths[i] = int(item['image_w'])
            heights[i] = int(item['image_h'])

            buf = base64.decodestring(item['features'].encode('utf8'))
            array = np.frombuffer(buf, dtype='float32')
            array = array.reshape((-1, config.output_features))
            objects = array.shape[0]
            features[i, :objects, :] = array

            buf = base64.decodestring(item['boxes'].encode('utf8'))
            array = np.frombuffer(buf, dtype='float32')
            array = array.reshape((-1, 4))
            boxes[i, :objects, :] = array


if __name__ == '__main__':
    main()
