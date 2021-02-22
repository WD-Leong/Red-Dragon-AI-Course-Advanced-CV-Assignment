import os
import numpy as np
import pickle as pkl
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
import tf_mobilenetv2_object_detection as tf_obj_detector

# Custom function to parse the data. #
def _parse_image(
    filename, img_rows, img_cols):
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = \
        tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize(
        image_decoded, [img_rows, img_cols])
    image_resized = tf.ensure_shape(
        image_resized, shape=(img_rows, img_cols, 3))
    return tf.expand_dims(image_resized, axis=0)

def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    
    boxes1_area = np.multiply(
        boxes1[..., 2] - boxes1[..., 0], 
        boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = np.multiply(
        boxes2[..., 2] - boxes2[..., 0], 
        boxes2[..., 3] - boxes2[..., 1])
    
    left_up   = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_dwn = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_dwn - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    
    ious = np.maximum(
        1.0 * inter_area / union_area, np.finfo(np.float32).eps)
    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, width, height, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    
    tmp_bboxes = bboxes
    tmp_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    tmp_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    
    best_bboxes = []
    for tmp_cls in classes_in_img:
        cls_mask = (tmp_bboxes[:, 5] == tmp_cls)
        cls_bboxes = tmp_bboxes[cls_mask]
        
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            
            cls_bboxes = np.concatenate(
                [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou_bboxes = bboxes_iou(
                best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            box_weight = np.ones((len(iou_bboxes),), dtype=np.float32)
            
            assert method in ['nms', 'soft-nms']
            
            if method == 'nms':
                iou_mask = iou_bboxes > iou_threshold
                box_weight[iou_mask] = 0.0
            
            if method == 'soft-nms':
                box_weight = np.exp(-(1.0 * iou_bboxes ** 2 / sigma))
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * box_weight
            
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]
    return best_bboxes

def obj_detect_results(
    img_in_file, voc_model, labels, 
    thresh=0.50, iou_thresh=0.15, downsample=8, 
    num_scale=4, img_rows=448, img_cols=448, img_scale=None, 
    img_title=None, save_img_file="object_detection_result.jpg"):
    if img_scale is None:
        min_scale = min(img_rows, img_cols)
        img_scale = [int(
            min_scale/(2**x)) for x in range(num_scale)]
    else:
        if len(img_scale) != num_scale:
            raise ValueError(
                "img_scale must be size " + str(num_scale) + ".")
    strides = [8, 16, 32, 64]
    
    # Read the image. #
    image_resized = _parse_image(
        img_in_file, img_rows, img_cols)
    
    tmp_output = \
        voc_model.predict(image_resized)
    n_classes  = tmp_output[0].shape[3]
    
    # Plot the bounding boxes on the image. #
    fig, ax = plt.subplots(1)
    tmp_img = np.array(
        Image.open(img_in_file), dtype=np.uint8)
    ax.imshow(tmp_img)
    
    img_width   = tmp_img.shape[0]
    img_height  = tmp_img.shape[1]
    tmp_w_ratio = img_width / img_rows
    tmp_h_ratio = img_height / img_cols
    
    tmp_obj_detect = []
    for n_sc in range(num_scale):
        stride = strides[n_sc]
        box_scale = img_scale[n_sc]
        
        reg_output = tmp_output[n_sc][0, :, :, :4]
        cls_output = tmp_output[n_sc][0, :, :, 5:]
        
        cls_probs = tf.nn.sigmoid(cls_output)
        cls_probs = cls_probs.numpy()
        box_scale = img_scale[n_sc]
        
        if n_classes > 1:
            prob_max = tf.reduce_max(cls_probs, axis=2)
            pred_label = 1 + tf.math.argmax(cls_probs, axis=2)
        else:
            prob_max = cls_probs[:, :, 0]
        tmp_thresh = \
            np.where(prob_max >= thresh, 1, 0)
        tmp_coords = np.nonzero(tmp_thresh)
        
        for n_box in range(len(tmp_coords[0])):
            x_coord = tmp_coords[0][n_box]
            y_coord = tmp_coords[1][n_box]
            
            tmp_boxes = reg_output[x_coord, y_coord, :]
            tmp_probs = int(
                prob_max[x_coord, y_coord].numpy()*100)
            if n_classes > 1:
                tmp_label = pred_label[x_coord, y_coord].numpy()
            else:
                tmp_label = str(labels[0])
            
            x_centroid = \
                tmp_w_ratio * (x_coord + tmp_boxes[0])*stride
            y_centroid = \
                tmp_h_ratio * (y_coord + tmp_boxes[1])*stride
            box_width  = \
                tmp_w_ratio * box_scale * tmp_boxes[2]
            box_height = \
                tmp_h_ratio * box_scale * tmp_boxes[3]
            
            if box_width > img_width:
                box_width = img_width
            if box_height > img_height:
                box_height = img_height
            
            # Output prediction is transposed. #
            x_lower = x_centroid - box_width/2
            y_lower = y_centroid - box_height/2
            if x_lower < 0:
                x_lower = 0
            if y_lower < 0:
                y_lower = 0
            
            tmp_bbox = np.array([
                y_lower, x_lower, 
                box_height, box_width, tmp_probs, tmp_label])
            tmp_obj_detect.append(np.expand_dims(tmp_bbox, axis=0))
    
    bboxes_raw = np.concatenate(
        tuple(tmp_obj_detect), axis=0)
    bboxes_nms = nms(bboxes_raw, iou_thresh, method='nms')
    for tmp_obj in bboxes_nms:
        box_width  = tmp_obj[2] - tmp_obj[0]
        box_height = tmp_obj[3] - tmp_obj[1]
        box_patch  = plt.Rectangle(
            (tmp_obj[0], tmp_obj[1]), box_width, box_height, 
            linewidth=1, edgecolor="red", fill=None)
        
        tmp_label = str(labels[int(tmp_obj[5])])
        tmp_text  = \
            tmp_label + ": " + str(tmp_obj[4]) + "%"
        ax.add_patch(box_patch)
        ax.text(tmp_obj[0], tmp_obj[1], 
                tmp_text, fontsize=5, color="red")
    print(str(len(bboxes_nms)), "objects detected.")
    
    if img_title is not None:
        fig.suptitle(img_title)
    fig.savefig(save_img_file, dpi=199)
    plt.close()
    del fig, ax
    return None

# Load the VOC 2012 dataset. #
tmp_path = "C:/Users/admin/Desktop/Data/VOCdevkit/VOC2012/"
load_pkl_file = tmp_path + "voc_annotations_320.pkl"
with open(load_pkl_file, "rb") as tmp_load:
    img_scale = pkl.load(tmp_load)
    label_dict = pkl.load(tmp_load)
    voc_object_list = pkl.load(tmp_load)

# Split into train and validation dataset. #
num_train = int(0.80 * len(voc_object_list))

np.random.seed(1234)
idx_perm   = np.random.permutation(len(voc_object_list))
train_data = [voc_object_list[x] for x in idx_perm[:num_train]]
test_data  = [voc_object_list[x] for x in idx_perm[num_train:]]
del train_data

# Define the Neural Network. #
compute_map = False
box_thresh  = 0.10

img_rows  = 320
img_cols  = 320
n_classes = len(label_dict)

# Define the checkpoint callback function. #
voc_path = "C:/Users/admin/Desktop/TF_Models/voc_2012_model/"
train_loss  = voc_path + "voc_losses_v5.csv"
ckpt_path   = voc_path + "voc_v5.ckpt"
ckpt_dir    = os.path.dirname(ckpt_path)
ckpt_model  = voc_path + "voc_keras_model_v5"

# Load the weights if continuing from a previous checkpoint. #
voc_model = tf_obj_detector.build_model(
    n_classes, img_rows=img_rows, img_cols=img_cols)
optimizer = tf.keras.optimizers.Adam()

checkpoint = tf.train.Checkpoint(
    step=tf.Variable(0), 
    voc_model=voc_model, 
    optimizer=optimizer)
ck_manager = tf.train.CheckpointManager(
    checkpoint, directory=ckpt_model, max_to_keep=1)

checkpoint.restore(ck_manager.latest_checkpoint)
if ck_manager.latest_checkpoint:
    print("Model restored from {}".format(
        ck_manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")
st_step = checkpoint.step.numpy().astype(np.int32)

print("Testing model (" +\
      str(st_step) + " iterations).")

save_img_file = "C:/Users/admin/Desktop/Red Dragon/"
save_img_file += "Advanced CV/assignment_2/"
save_img_file += "voc_lite_object_detection_1.jpg"

img_disp  = "C:/Users/admin/Desktop/Red Dragon/"
img_disp  += "Advanced CV/assignment_2/cat.jpg"
img_title = "Lite Object Detection Result"
obj_detect_results(
    img_disp, voc_model, label_dict, 
    thresh=0.50, img_scale=img_scale, 
    img_rows=img_rows, img_cols=img_cols, 
    img_title=img_title, save_img_file=save_img_file)

