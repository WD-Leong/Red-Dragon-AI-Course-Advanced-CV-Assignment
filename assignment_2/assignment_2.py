import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
import tf_mobilenetv2_object_detection as tf_obj_detector

from PIL import Image
import matplotlib.pyplot as plt

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

def train(
    voc_model, sub_batch_sz, batch_size, 
    train_dataset, training_loss, img_disp, img_box, 
    st_step, max_steps, optimizer, ckpt, ck_manager, 
    label_dict, decay=0.75, init_lr=1.0e-3, 
    img_rows=320, img_cols=320, img_scale=None, 
    display_step=100, step_cool=50, min_lr=1.0e-6, 
    save_flag=False, save_train_loss_file="train_losses.csv"):
    n_data  = len(train_dataset)
    n_batch = 0
    
    start_time = time.time()
    tot_reg_loss  = 0.0
    tot_cls_loss  = 0.0
    for step in range(st_step, max_steps):
        batch_sample = np.random.choice(
            n_data, size=batch_size, replace=False)
        
        img_files = [train_dataset[x][0] for x in batch_sample]
        img_batch = tf.concat([_parse_image(
            x, img_rows, img_cols) for x in img_files], axis=0)
        
        img_bbox = []
        img_mask = []
        for n_sc in range(4):
            tmp_bbox = tf.concat([tf.cast(tf.expand_dims(
                tf.sparse.to_dense(tf.sparse.reorder(
                    train_dataset[x][1][n_sc])), axis=0), 
                        tf.float32) for x in batch_sample], axis=0)
            
            img_bbox.append(tmp_bbox)
            img_mask.append(tmp_bbox[:, :, :, 4])
        
        epoch = int(step * batch_size / n_data)
        lrate = max(decay**epoch * init_lr, min_lr)
        
        tmp_losses = tf_obj_detector.train_step(
            voc_model, sub_batch_sz, img_batch, 
            img_bbox, img_mask, optimizer, learning_rate=lrate)
        
        ckpt.step.assign_add(1)
        n_batch += 1
        tot_cls_loss += tmp_losses[0]
        tot_reg_loss += tmp_losses[1]
        
        if (step+1) % display_step == 0:
            avg_reg_loss = tot_reg_loss.numpy() / display_step
            avg_cls_loss = tot_cls_loss.numpy() / display_step
            training_loss.append((step+1, avg_cls_loss, avg_reg_loss))
            
            tot_reg_loss = 0.0
            tot_cls_loss = 0.0
            
            print("Step", str(step+1), "Summary:")
            print("Learning Rate:", str(optimizer.lr.numpy()))
            print("Average Epoch Cls. Loss:", str(avg_cls_loss) + ".")
            print("Average Epoch Reg. Loss:", str(avg_reg_loss) + ".")
            
            elapsed_time = (time.time() - start_time) / 60.0
            print("Elapsed time:", str(elapsed_time), "mins.")
            
            start_time = time.time()
            if (step+1) % step_cool != 0:
                tf_obj_detector.obj_detect_results(
                    img_disp, voc_model, label_dict, 
                    heatmap=True, img_box=img_box, 
                    img_rows=img_rows, img_cols=img_cols, 
                    img_scale=img_scale, thresh=0.50)
                print("-" * 50)
        
        if (step+1) % step_cool == 0:
            if save_flag:
                # Save the training losses. #
                train_loss_df = pd.DataFrame(
                    training_loss, columns=["step", "cls_loss", "reg_loss"])
                train_loss_df.to_csv(save_train_loss_file, index=False)
                
                # Save the model. #
                save_path = ck_manager.save()
                print("Saved model to {}".format(save_path))
            print("-" * 50)
            
            save_img_file = "C:/Users/admin/Desktop/Data/"
            save_img_file += "Results/VOC_Lite_Object_Detection/"
            save_img_file += "voc_obj_detect_" + str(step+1) + ".jpg"
            
            img_title = "Lite YOLO Object Detection Result "
            img_title += "at Step " + str(step+1)
            
            tf_obj_detector.obj_detect_results(
                img_disp, voc_model, label_dict, heatmap=True, 
                img_scale=img_scale, img_box=img_box, 
                img_rows=img_rows, img_cols=img_cols, thresh=0.50, 
                img_title=img_title, save_img_file=save_img_file)
            
            print("Cooling GPU for 2 minutes.")
            time.sleep(120)

# Load the VOC 2012 dataset. #
tmp_path = "C:/Users/admin/Desktop/Data/VOCdevkit/VOC2012/"
load_pkl_file = tmp_path + "voc_annotations_320.pkl"
with open(load_pkl_file, "rb") as tmp_load:
    img_scale = pkl.load(tmp_load)
    label_dict = pkl.load(tmp_load)
    voc_object_list = pkl.load(tmp_load)

# Define the Neural Network. #
restore_flag = True
img_width  = 320
img_height = 320
n_filters  = 16
step_cool  = 100

init_lr    = 1.0e-3
decay_rate = 0.99
max_steps  = 7500
batch_size = 96
sub_batch  = 8
batch_test = 10
n_classes  = len(label_dict)
display_step = 50

# Define the checkpoint callback function. #
voc_path = "C:/Users/admin/Desktop/TF_Models/voc_2012_model/"
train_loss  = voc_path + "voc_losses_v5.csv"
ckpt_path   = voc_path + "voc_v5.ckpt"
ckpt_dir    = os.path.dirname(ckpt_path)
ckpt_model  = voc_path + "voc_keras_model_v5"

# Load the weights if continuing from a previous checkpoint. #
voc_model = tf_obj_detector.build_model(
    n_classes, img_rows=img_width, img_cols=img_height)
optimizer = tf.keras.optimizers.Adam()

checkpoint = tf.train.Checkpoint(
    step=tf.Variable(0), 
    voc_model=voc_model, 
    optimizer=optimizer)
ck_manager = tf.train.CheckpointManager(
    checkpoint, directory=ckpt_model, max_to_keep=1)

if restore_flag:
    train_loss_df = pd.read_csv(train_loss)
    training_loss = [tuple(
        train_loss_df.iloc[x].values) \
        for x in range(len(train_loss_df))]
    checkpoint.restore(ck_manager.latest_checkpoint)
    if ck_manager.latest_checkpoint:
        print("Model restored from {}".format(
            ck_manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
else:
    training_loss = []
st_step = checkpoint.step.numpy().astype(np.int32)

# Split into train and validation dataset. #
num_train = int(0.80 * len(voc_object_list))

np.random.seed(1234)
idx_perm  = np.random.permutation(len(voc_object_list))
train_data = [voc_object_list[x] for x in idx_perm[:num_train]]
test_data  = [voc_object_list[x] for x in idx_perm[num_train:]]

# Print out the model summary. #
model_summary = voc_path + "voc_model_v5_summary.txt"
with open(model_summary, "w") as fh:
    voc_model.summary(print_fn=lambda x: fh.write(x + '\n'))

print("Fit model on training data (" +\
      str(len(train_data)) + " training samples).")
img_disp = voc_object_list[25][0]
img_box  = []
for n_sc in range(4):
    tmp_bbox = tf.cast(
        tf.sparse.to_dense(tf.sparse.reorder(
            voc_object_list[25][1][n_sc])), tf.float32)
    img_box.append(tmp_bbox)

fig, ax = plt.subplots(1)
tmp_img = np.array(
    Image.open(img_disp), dtype=np.uint8)
ax.imshow(tmp_img)
fig.savefig("test_image.jpg", dpi=199)

plt.close()
del fig, ax

train(voc_model, sub_batch, batch_size, 
      train_data, training_loss, img_disp, img_box, 
      st_step, max_steps, optimizer, checkpoint, ck_manager, 
      label_dict, init_lr=init_lr, decay=decay_rate, 
      step_cool=step_cool, display_step=display_step, 
      img_rows=img_width, img_cols=img_height, img_scale=img_scale, 
      min_lr=1.0e-5, save_flag=True, save_train_loss_file=train_loss)
print("Model fitted.")


