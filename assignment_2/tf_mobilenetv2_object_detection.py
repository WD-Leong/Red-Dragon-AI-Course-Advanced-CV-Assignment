
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tf_bias_layer import BiasLayer

from PIL import Image
import matplotlib.pyplot as plt

def _parse_image(
    filename, img_rows=448, img_cols=448):
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = \
        tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize(
        image_decoded, [img_rows, img_cols])
    image_resized = tf.ensure_shape(
        image_resized ,shape=(img_rows, img_cols, 3))
    return image_resized

def build_model(
    n_classes, tmp_pi=0.99, img_rows=448, img_cols=448):
    b_init  = tf.math.log((1.0-tmp_pi) / tmp_pi)
    b_focal = BiasLayer(bias_init=b_init, name="b_focal")
    
    # Get the MobileNet weights. #
    mobilenet_v2 = tf.keras.applications.MobileNetV2(
        input_shape=(img_rows, img_cols, 3), 
        include_top=False, weights="imagenet", pooling=None)
    
    # Extract the feature maps. #
    x_blk3_out = \
        mobilenet_v2.get_layer("block_3_expand_relu").output
    x_blk4_out = \
        mobilenet_v2.get_layer("block_6_expand_relu").output
    x_blk5_out = \
        mobilenet_v2.get_layer("block_13_expand_relu").output
    x_blk6_out = \
        mobilenet_v2.get_layer("out_relu").output
    
    # Regression and Classification heads. #
    x_reg_small = tf.nn.sigmoid(layers.Conv2D(
        4, (3, 3), strides=(2, 2), padding="same", 
        activation="linear", name="reg_small")(x_blk3_out))
    x_cls_small = layers.Conv2D(
        n_classes, (3, 3), strides=(2, 2), padding="same", 
        activation="linear", name="cls_small")(x_blk3_out)
    x_out_small = tf.concat(
        [x_reg_small, b_focal(x_cls_small)], axis=3)
    
    x_reg_medium = tf.nn.sigmoid(layers.Conv2D(
        4, (3, 3), strides=(2, 2), padding="same", 
        activation="linear", name="reg_medium")(x_blk4_out))
    x_cls_medium = layers.Conv2D(
        n_classes, (3, 3), strides=(2, 2), padding="same", 
        activation="linear", name="cls_medium")(x_blk4_out)
    x_out_medium = tf.concat(
        [x_reg_medium, b_focal(x_cls_medium)], axis=3)
    
    x_reg_large = tf.nn.sigmoid(layers.Conv2D(
        4, (3, 3), strides=(2, 2), padding="same", 
        activation="linear", name="reg_large")(x_blk5_out))
    x_cls_large = layers.Conv2D(
        n_classes, (3, 3), strides=(2, 2), padding="same", 
        activation="linear", name="cls_large")(x_blk5_out)
    x_out_large = tf.concat(
        [x_reg_large, b_focal(x_cls_large)], axis=3)
    
    x_reg_vlarge = tf.nn.sigmoid(layers.Conv2D(
        4, (3, 3), strides=(2, 2), padding="same", 
        activation="linear", name="reg_vlarge")(x_blk6_out))
    x_cls_vlarge = layers.Conv2D(
        n_classes, (3, 3), strides=(2, 2), padding="same", 
        activation="linear", name="cls_vlarge")(x_blk6_out)
    x_out_vlarge = tf.concat(
        [x_reg_vlarge, b_focal(x_cls_vlarge)], axis=3)
    
    # Concatenate the outputs. #
    x_output = [
        x_out_small, x_out_medium, 
        x_out_large, x_out_vlarge]
    obj_model = tf.keras.Model(
        inputs=mobilenet_v2.input, outputs=x_output)
    return obj_model

def sigmoid_loss(labels, logits):
    return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(labels, tf.float32), logits=logits)

def focal_loss(
    labels, logits, alpha=0.25, gamma=2.0):
    labels = tf.cast(labels, tf.float32)
    tmp_log_logits  = tf.math.log(1.0 + tf.exp(-1.0 * tf.abs(logits)))
    
    tmp_abs_term = tf.math.add(
        tf.multiply(labels * alpha * tmp_log_logits, 
                    tf.pow(1.0 - tf.nn.sigmoid(logits), gamma)), 
        tf.multiply(tf.pow(tf.nn.sigmoid(logits), gamma), 
                    (1.0 - labels) * (1.0 - alpha) * tmp_log_logits))
    
    tmp_x_neg = tf.multiply(
        labels * alpha * tf.minimum(logits, 0), 
        tf.pow(1.0 - tf.nn.sigmoid(logits), gamma))
    tmp_x_pos = tf.multiply(
        (1.0 - labels) * (1.0 - alpha), 
        tf.maximum(logits, 0) * tf.pow(tf.nn.sigmoid(logits), gamma))
    
    foc_loss_stable = tmp_abs_term + tmp_x_pos - tmp_x_neg
    return tf.reduce_sum(foc_loss_stable, axis=[1, 2, 3])

def model_loss(
    bboxes, masks, outputs, loss_type="sigmoid"):
    total_reg_loss = 0.0
    total_cls_loss = 0.0
    for id_sc in range(len(outputs)):
        reg_weight = tf.expand_dims(masks[id_sc], axis=3)
        reg_output = outputs[id_sc][:, :, :, :4]
        cls_output = outputs[id_sc][:, :, :, 4:]
        cls_labels = tf.cast(
            bboxes[id_sc][:, :, :, 4:], tf.int32)
        
        if loss_type == "sigmoid":
            total_cls_loss += tf.reduce_sum(
                sigmoid_loss(cls_labels, cls_output))
        else:
            total_cls_loss += tf.reduce_sum(
                focal_loss(cls_labels, cls_output))
        total_reg_loss += tf.reduce_sum(tf.multiply(tf.abs(
            bboxes[id_sc][:, :, :, :4] - reg_output), reg_weight))
    return total_cls_loss, total_reg_loss

def train_step(
    voc_model, sub_batch_sz, 
    images, bboxes, masks, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0, 
    cls_lambda=5.0, loss_type="focal"):
    optimizer.lr.assign(learning_rate)
    
    batch_size = images.shape[0]
    if batch_size <= sub_batch_sz:
        n_sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        n_sub_batch = int(batch_size / sub_batch_sz)
    else:
        n_sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = voc_model.trainable_variables
    acc_gradients = [tf.zeros_like(var) for var in model_params]
    
    tmp_reg_loss = 0.0
    tmp_cls_loss = 0.0
    for n_sub in range(n_sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (n_sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        tmp_images = images[id_st:id_en, :, :, :]
        
        tmp_bboxes = []
        tmp_masks  = []
        for id_sc in range(len(bboxes)):
            tmp_masks.append(masks[id_sc][id_st:id_en, :, :])
            tmp_bboxes.append(bboxes[id_sc][id_st:id_en, :, :, :])
        
        with tf.GradientTape() as voc_tape:
            tmp_output = voc_model(tmp_images, training=True)
            tmp_losses = model_loss(
                tmp_bboxes, tmp_masks, tmp_output, loss_type=loss_type)
            
            tmp_cls_loss += tmp_losses[0]
            tmp_reg_loss += tmp_losses[1]
            total_losses = \
                cls_lambda*tmp_losses[0] + tmp_losses[1]
        
        # Accumulate the gradients. #
        tmp_gradients = \
            voc_tape.gradient(total_losses, model_params)
        acc_gradients = [
            (acc_grad+grad) for \
            acc_grad, grad in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_reg_loss  = tmp_reg_loss / batch_size
    avg_cls_loss  = tmp_cls_loss / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clipped_gradients, _ = \
        tf.clip_by_global_norm(acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model_params))
    return avg_cls_loss, avg_reg_loss

def obj_detect_results(
    img_in_file, voc_model, labels, 
    heatmap=True, img_box=None, thresh=0.50, 
    img_rows=448, img_cols=448, img_scale=None, 
    img_title=None, save_img_file="object_detection_result.jpg"):
    if img_scale is None:
        if max(img_rows, img_cols) >= 512:
            max_scale = max(img_rows, img_cols)
        else:
            max_scale = 512
        img_scale = [64, 128, 256, max_scale]
    else:
        if len(img_scale) != 4:
            raise ValueError("img_scale must be size 4.")
    dwn_scale = [8, 16, 32, 64]
    
    # Read the image. #
    image_resized = tf.expand_dims(_parse_image(
        img_in_file, img_rows=img_rows, img_cols=img_cols), axis=0)
    
    tmp_output = \
        voc_model.predict(image_resized)
    n_classes  = tmp_output[0][0, :, :, 4:].shape[2]
    
    # Plot the bounding boxes on the image. #
    fig, ax = plt.subplots(1)
    tmp_img = np.array(
        Image.open(img_in_file), dtype=np.uint8)
    ax.imshow(tmp_img)
    
    img_width   = tmp_img.shape[0]
    img_height  = tmp_img.shape[1]
    tmp_w_ratio = img_width / img_rows
    tmp_h_ratio = img_height / img_cols
    
    if heatmap:
        tmp_probs = []
        for n_sc in range(len(tmp_output)):
            tmp_array = np.zeros(
                [int(img_rows/8), int(img_cols/8)])
            down_scale = int(dwn_scale[n_sc] / 8)
            cls_output = tmp_output[n_sc][0, :, :, 4:]
            cls_probs  = tf.nn.sigmoid(cls_output)
            
            if n_classes > 1:
                obj_probs = tf.reduce_max(
                    cls_probs[:, :, 1:], axis=2)
            else:
                obj_probs = cls_probs[:, :, 0]
            tmp_array[int(down_scale/2)::down_scale, 
                      int(down_scale/2)::down_scale] = obj_probs
            
            tmp_array = tf.expand_dims(tmp_array, axis=2)
            obj_probs = tf.squeeze(tf.image.resize(tf.expand_dims(
                tmp_array, axis=0), [img_width, img_height]), axis=3)
            tmp_probs.append(obj_probs)
        
        tmp_probs = tf.concat(tmp_probs, axis=0)
        tmp_probs = tf.reduce_max(tmp_probs, axis=0)
        
        tmp = ax.imshow(tmp_probs, "jet", alpha=0.50)
        fig.colorbar(tmp, ax=ax)
    
    n_obj_detected = 0
    for n_sc in range(4):
        down_scale = dwn_scale[n_sc]
        
        reg_output = tmp_output[n_sc][0, :, :, :4]
        cls_output = tmp_output[n_sc][0, :, :, 4:]
        cls_probs  = tf.nn.sigmoid(cls_output)
        if n_sc == 3:
            if max(img_rows, img_cols) <= img_scale[n_sc]:
                box_scale = max(img_rows, img_cols)
            else:
                box_scale = img_scale[n_sc]
        else:
            box_scale = img_scale[n_sc]
        
        if n_classes > 1:
            prob_max = tf.reduce_max(
                cls_probs[:, :, 1:], axis=2)
            pred_label = 1 + tf.math.argmax(
                cls_probs[:, :, 1:], axis=2)
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
                tmp_label = str(labels[
                    pred_label[x_coord, y_coord].numpy()])
            else:
                tmp_label = str(labels[0])
            
            x_centroid = \
                tmp_w_ratio * (x_coord + tmp_boxes[0])*down_scale
            y_centroid = \
                tmp_h_ratio * (y_coord + tmp_boxes[1])*down_scale
            box_width  = tmp_w_ratio * box_scale * tmp_boxes[2]
            box_height = tmp_h_ratio * box_scale * tmp_boxes[3]
            
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
            
            box_patch = plt.Rectangle(
                (y_lower, x_lower), box_height, box_width, 
                linewidth=1, edgecolor="red", fill=None)
            
            n_obj_detected += 1
            tmp_text = \
                tmp_label + ": " + str(tmp_probs) + "%"
            ax.add_patch(box_patch)
            ax.text(y_lower, x_lower, tmp_text, 
                    fontsize=10, color="red")
    print(str(n_obj_detected), "objects detected.")
    
    # True image is not transposed. #
    if img_box is not None:
        for n_sc in range(4):
            down_scale = dwn_scale[n_sc]
            
            if n_sc == 3:
                if max(img_rows, img_cols) <= img_scale[n_sc]:
                    box_scale = max(img_rows, img_cols)
                else:
                    box_scale = img_scale[n_sc]
            else:
                box_scale = img_scale[n_sc]
            
            tmp_true_box = np.nonzero(img_box[n_sc][:, :, 4])
            for n_box in range(len(tmp_true_box[0])):
                x_coord = tmp_true_box[0][n_box]
                y_coord = tmp_true_box[1][n_box]
                tmp_boxes = img_box[n_sc][x_coord, y_coord, :4]
                
                x_centroid = \
                    tmp_w_ratio * (x_coord + tmp_boxes[0])*down_scale
                y_centroid = \
                    tmp_h_ratio * (y_coord + tmp_boxes[1])*down_scale
                box_width  = tmp_w_ratio * box_scale * tmp_boxes[2]
                box_height = tmp_h_ratio * box_scale * tmp_boxes[3]
                
                x_lower = x_centroid - box_width/2
                y_lower = y_centroid - box_height/2
                box_patch = plt.Rectangle(
                    (y_lower.numpy(), x_lower.numpy()), 
                    box_height.numpy(), box_width.numpy(), 
                    linewidth=1, edgecolor="black", fill=None)
                ax.add_patch(box_patch)
    
    if img_title is not None:
        fig.suptitle(img_title)
    fig.savefig(save_img_file, dpi=199)
    plt.close()
    del fig, ax
    return None

