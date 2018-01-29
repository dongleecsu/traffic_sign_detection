import os
import copy
import pickle
import argparse
from utils import *
import numpy as np
from tqdm import tqdm
from patchnet import PatchNet
import PIL.Image as Image
import tensorflow as tf
from file_handler import *
from repairer import *

parser = argparse.ArgumentParser()
parser.add_argument('--test-dir', type=str, default='/home/ld/TSD_dataset/TSD-signal')
parser.add_argument('--test-label-dir', type=str, default='/home/ld/TSD_dataset/TSD-signal-GT')
parser.add_argument('--cat-path', type=str, default='/home/ld/TSD_dataset/cat_dict.pkl')
parser.add_argument('--ckpt-path', type=str, default='./models/posnet/ssd-mobilenet/frozen_inference_graph.pb')
parser.add_argument('--patchnet-model', type=str, default='./models/patchnet/try15_new.h5')
args = parser.parse_args()

'''File reader'''
print('1. Reading and parsing testing directory...')
test_dir = args.test_dir
test_files_list = read_and_parse_dir(test_dir)
test_labels = get_annotation(test_files_list, args.test_label_dir, args.cat_path)


'''Position network'''
# load inference model
# print('2.1 Loading inference model...')
# ckpt_path = args.ckpt_path
# detection_graph = tf.Graph()
# with detection_graph.as_default():
#     od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile(ckpt_path, 'rb') as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name='')

# print('2.2 Evaluating on test files...')
# raw_test_position_results = []
# with detection_graph.as_default():
#     with tf.Session(graph=detection_graph) as sess:
#         image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#         detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#         detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#         detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#         num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#     for i, dir_files_list in enumerate(test_files_list):
#         print('Processing directory %d/%d'%(i+1, len(test_files_list)))
#         raw_dir_position_results = []
#         dir_files_list = sorted(dir_files_list)
#         for image_path in dir_files_list:
#             image = Image.open(image_path)
#             w, h = image.size
#             h = int(h/2)
#             image_np = pil_to_array(image)
#             # NOTE: numpy array shape is (H,W,C)
#             image_np = image_np[:h, :, :]
#             image_np_expanded = np.expand_dims(image_np, axis=0)
#             (bboxes, scores) = sess.run([detection_boxes, detection_scores],
#                                        feed_dict={image_tensor: image_np_expanded})
#             bboxes = bboxes.squeeze()
#             scores = scores.squeeze()
#             raw_dir_position_results.append({'bboxes': bboxes,
#                                              'scores': scores,
#                                              # we also store image w,h to
#                                              # restore bbox
#                                              'metadata': (w, h)})
#         raw_test_position_results.append(raw_dir_position_results)
# with open('raw_test_position_results.pkl', 'wb') as f:
#     pickle.dump(raw_test_position_results, f)

with open('raw_test_position_results.pkl', 'rb') as f:
    raw_test_position_results = pickle.load(f)

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
print('***** PosNet Performance *****')
print('precision = %.3f, recall = %.3f, F = %.3f'%(stats[0], stats[1], stats[2]))

'''Crop patches'''
print('3. Cropping patches...')
test_patches = []
for i, dir_files_list in tqdm(enumerate(test_files_list), ncols=64):
    dir_position_results = test_position_results[i]
    dir_files_list = sorted(dir_files_list)
    dir_crops = []
    for j, image_path in enumerate(dir_files_list):
        image = Image.open(image_path)
        w, h = image.size
        h = int(h / 2)
        image_np = pil_to_array(image)
        arr = pil_to_array(image)
        image_np = image_np[:h, :, :]
        frame_position_results = dir_position_results[j]
        crops_per_image = []
        for bbox in frame_position_results:
            crop = image_np[bbox[1]:bbox[1] + bbox[3],
                            bbox[0]:bbox[0] + bbox[2], :]
            crop = array_to_pil(crop).resize((50, 50))
            crop = pil_to_array(crop)
            crops_per_image.append(np.asarray(crop))
        crops_per_image = np.array(crops_per_image)
        dir_crops.append(crops_per_image)
    test_patches.append(dir_crops)

'''Classification network'''
print('4.1 Building patchnet...')
# TODO: set patchnet model_path here
patchnet_weights = args.patchnet_model
patchnet = PatchNet(input_shape = (50,50,1),
                    num_filters=[32, 32, 64],
                    filter_size=(5, 5),
                    num_fc_units = [128],
                    drop_keep_prob = [0.5], # only for FC layers
                    num_classes=74)
patchnet.load_weights(patchnet_weights)

print('4.2 Evaluating on test patches...')
test_class_results = []
for dir_patches in tqdm(test_patches, ncols=64):
    dir_preds = []
    for patch in dir_patches:
        if len(patch) == 0:
            dir_preds.append(np.array([]))
            continue
        gray = rgb2gray(patch)
        clahe = to_clahe(gray).reshape(-1, 50, 50, 1)
        y_hat = patchnet.predict(clahe)
        preds = np.argmax(y_hat, axis=1) + 1
        dir_preds.append(preds)
    test_class_results.append(dir_preds)

test_class_results_ = copy.deepcopy(test_class_results)
try:
    print('Pass')
    test_class_results = wash_one_tar_detection(test_class_results)
except:
    print('WARNING: rep with exception')
    test_class_results = test_class_results_

test_dets = []
for i, dir_class_results in enumerate(test_class_results):
    dir_position_results = test_position_results[i]
    dir_dets = []
    for j, frame_class in enumerate(dir_class_results):
        frame_position = dir_position_results[j]
        frame_dets = []
        for k, cls in enumerate(frame_class):
            position = frame_position[k]
            frame_dets.append({'Position': position,
                               'Type': cls})
        dir_dets.append(frame_dets)
    test_dets.append(dir_dets)

# detections without temporal filter
test_dets_ = []
for i, dir_class_results in enumerate(test_class_results_):
    dir_position_results = test_position_results[i]
    dir_dets = []
    for j, frame_class in enumerate(dir_class_results):
        frame_position = dir_position_results[j]
        frame_dets = []
        for k, cls in enumerate(frame_class):
            position = frame_position[k]
            frame_dets.append({'Position': position,
                               'Type': cls})
        dir_dets.append(frame_dets)
    test_dets_.append(dir_dets)

stats = []
for dir_dets, dir_gts in zip(test_dets, test_labels):
    for frame_dets, frame_gts in zip(dir_dets, dir_gts):
        det_pos = []
        det_cls = []
        for det in frame_dets:
            det_pos.append(det['Position'])
            det_cls.append(det['Type'])
        gt_pos = []
        gt_cls = []
        for gt_ in frame_gts:
            gt_pos.append(gt_['Position'])
            gt_cls.append(gt_['Type'])
        prec, rec, F = compute_pre_rec_F(det_pos, det_cls, gt_pos, gt_cls)
        stats.append([prec, rec, F])
stats = np.asarray(stats)
stats = np.mean(stats, axis=0)
print('***** Total Performance with filter*****')
print('precision = %.3f, recall = %.3f, F = %.3f'%(stats[0], stats[1], stats[2]))

stats = []
for dir_dets, dir_gts in zip(test_dets_, test_labels):
    for frame_dets, frame_gts in zip(dir_dets, dir_gts):
        det_pos = []
        det_cls = []
        for det in frame_dets:
            det_pos.append(det['Position'])
            det_cls.append(det['Type'])
        gt_pos = []
        gt_cls = []
        for gt_ in frame_gts:
            gt_pos.append(gt_['Position'])
            gt_cls.append(gt_['Type'])
        prec, rec, F = compute_pre_rec_F(det_pos, det_cls, gt_pos, gt_cls)
        stats.append([prec, rec, F])
stats = np.asarray(stats)
stats = np.mean(stats, axis=0)
print('***** Total Performance without filter*****')
print('precision = %.3f, recall = %.3f, F = %.3f'%(stats[0], stats[1], stats[2]))