import os
import copy
import pickle
import argparse
from utils import *
import numpy as np
import PIL.Image as Image
import tensorflow as tf
from file_handler import *

parser = argparse.ArgumentParser()
parser.add_argument('--test-dir', type=str, default='/home/ld/TSD_dataset/TSD-signal')
parser.add_argument('--test-label-dir', type=str, default='/home/ld/TSD_dataset/TSD-signal-GT')
parser.add_argument('--cat-path', type=str, default='/home/ld/TSD_dataset/cat_dict.pkl')
parser.add_argument('--ckpt-path', type=str, default='./models/posnet/ssd-mobilenet/frozen_inference_graph.pb')
args = parser.parse_args()

'''File reader'''
print('1. Reading and parsing testing directory...')
test_dir = args.test_dir
test_files_list = read_and_parse_dir(test_dir)
test_labels = get_annotation(test_files_list, args.test_label_dir, args.cat_path)


'''Position network'''
# load inference model
print('2.1 Loading inference model...')
ckpt_path = args.ckpt_path
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ckpt_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print('2.2 Evaluating on test files...')
raw_test_position_results = []
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for i, dir_files_list in enumerate(test_files_list):
        print('Processing directory %d/%d'%(i+1, len(test_files_list)))
        raw_dir_position_results = []
        dir_files_list = sorted(dir_files_list)
        for image_path in dir_files_list:
            image = Image.open(image_path)
            w, h = image.size
            h = int(h/2)
            image_np = pil_to_array(image)
            # NOTE: numpy array shape is (H,W,C)
            image_np = image_np[:h, :, :]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (bboxes, scores) = sess.run([detection_boxes, detection_scores],
                                       feed_dict={image_tensor: image_np_expanded})
            bboxes = bboxes.squeeze()
            scores = scores.squeeze()
            raw_dir_position_results.append({'bboxes': bboxes,
                                             'scores': scores,
                                             # we also store image w,h to
                                             # restore bbox
                                             'metadata': (w, h)})
        raw_test_position_results.append(raw_dir_position_results)
# with open('raw_test_position_results.pkl', 'wb') as f:
#     pickle.dump(raw_test_position_results, f)

# with open('raw_test_position_results.pkl', 'rb') as f:
#     raw_test_position_results = pickle.load(f)

print('Computing statistics...')
thresh = 0.5
test_position_results = wash_with_threshold(raw_test_position_results, thresh)
stats = []
for dir_pos, dir_gts in zip(test_position_results, test_labels):
    for frame_pos, frame_gt in zip(dir_pos, dir_gts):
        prec, rec, F = compute_frame_bbox_F(frame_pos, frame_gt)
        stats.append([prec, rec, F])
stats = np.asarray(stats)
stats = np.mean(stats, axis=0)
print('precision = %.3f, recall = %.3f, F = %.3f'%(stats[0], stats[1], stats[2]))