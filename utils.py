import os
# import cv2
import pickle as pkl
import matplotlib
from lxml import etree
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image as Image
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def to_clahe(arr):
    # Clahe (Contrast limit adaptive histogram equalizationg) for
    # gray images.
    arr_ = np.asarray(arr, dtype=np.uint8)
    clahe = cv2.createCLAHE(tileGridSize=(5, 5), clipLimit=10.0)
    clahe_arr = np.asarray(list(map(lambda img: clahe.apply(img), arr_)))
    return clahe_arr

def pil_to_array(image):
    return np.asarray(image)

def array_to_pil(arr):
    return Image.fromarray(arr)

def show_bbox_on_image(img, bbox, path):
    # print(score)
    name = path.split('/')[-1]
    save_base = 'test_dataset/posnet_show'
    save_path = os.path.join(save_base, 'det_'+name)
    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(img)
    for bb in bbox:
        rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.savefig(save_path)

def wash_with_threshold(raw, thresh):
    clean_results = []
    for dir_results in raw:
        # directory level
        clean_dir_results = []
        for frame_results_dict in dir_results:
            # frame level
            bboxes = frame_results_dict['bboxes']
            scores = frame_results_dict['scores']
            w, h = frame_results_dict['metadata']
            cutpoint = np.argmax(scores < thresh)
            th_bboxes = bboxes[:cutpoint]
            for i, bbox in enumerate(th_bboxes):
                xmin = bbox[1] * w
                ymin = bbox[0] * h
                width = (bbox[3] - bbox[1]) * w
                height = (bbox[2] - bbox[0]) * h
                th_bboxes[i] = [xmin, ymin, width, height]
            th_bboxes = np.asarray(th_bboxes, dtype=int)
            clean_dir_results.append(th_bboxes.tolist())
        clean_results.append(clean_dir_results)
    return clean_results

def recursive_parse_xml_to_dict(xml):
  """Recursively parses XML contents to python dict.
  Credit from tensorflow models object detection API codebase.
  """
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

def get_annotation_kv_clean(ann_list, cat):
    clean_ann_list = []
    for ann in ann_list:
        xs = ann['Position'].strip().split()
        xs = [int(x) for x in xs]
        idx = cat[ann['Type']]
        clean_ann_list.append({'Position': xs,
                               'Type': idx})
    return clean_ann_list

def get_annotation(test_files_list, test_label_dir, cat_path):
    with open(cat_path, 'rb') as f:
        cat = pkl.load(f)
    annotations = []
    for dir_names in test_files_list:
        dir_annotations = []
        for frame_name in dir_names:
            video_name = frame_name.split('/')[-2]
            xml_path = os.path.join(test_label_dir, video_name+'-GT.xml')
            with open(xml_path, 'rb') as f:
                context = f.read()
            xml = etree.fromstring(context)
            data = recursive_parse_xml_to_dict(xml)['opencv_storage']
            frame_idx = frame_name.split('/')[-1].split('.')[0].split('-')[-1]
            target_num = data['Frame'+frame_idx+'TargetNumber']
            annotation_list = []
            for i in range(int(target_num)):
                target_name = 'Frame'+frame_idx+'Target%0.5d'%(i)
                try:
                    annotation_list.append(data[target_name])
                except:
                    print('check file ', sample_path)
            clean_ann_list =  get_annotation_kv_clean(annotation_list, cat)
            dir_annotations.append(clean_ann_list)
        annotations.append(dir_annotations)
    return annotations

def compute_frame_bbox_F(frame_pos, frame_gt):
    tp = 0
    num_det = len(frame_pos)
    num_gt = len(frame_gt)
    for pos, gt in zip(frame_pos, frame_gt):
        x_center = pos[0] + pos[2] / 2.
        y_center = pos[1] + pos[3] / 2.
        gt_bbox = gt['Position']
        x_flag = x_center >= gt_bbox[0] and x_center <= gt_bbox[0] + gt_bbox[2]
        y_flag = y_center >= gt_bbox[1] and y_center <= gt_bbox[1] + gt_bbox[3]
        if x_flag and y_flag:
            tp += 1
    fp = num_det - tp
    fn = num_gt - tp
    assert(tp >= 0 and fp >=0 and fn >= 0, 'tp, fp, fn computing error')
    if tp == 0 and fp == 0 and fn == 0:
        # no target in the image
        return 1.0, 1.0, 1.0
    try:
        precision = float(tp) / (tp + fp)
    except ZeroDivisionError:
        precision = 1.0
    try:
        recall = float(tp) / (tp + fn)
    except ZeroDivisionError:
        recall = 1.0
    if tp == 0:
        # supress divide zero error
        F = 0.0
    else:
        F = 2*precision*recall / (precision + recall)
    return precision, recall, F

