
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Load the data. #
tmp_path = \
    "C:/Users/admin/Desktop/Red Dragon/Advanced CV/quickdraw/quickdraw/"
file_names = list(sorted(glob(tmp_path + "*.npy")))
label_dict = []

x_data  = np.array([])
img_max = 12500
for i, filename in enumerate(file_names):
    arr = np.load(filename)
    arr = arr[:img_max]
    arr = arr.reshape((-1, 28, 28, 1))
    
    labels = [i] * img_max
    label_dict.append((i, filename.split("\\")[1].split(".")[0]))
    if len(x_data) == 0:
        x_data  = arr
        y_label = np.asarray(labels)
    else:
        x_data = np.concatenate((x_data, arr))
        y_label = np.concatenate((y_label, labels))

# Form the mapping from integer label to actual class. #
label_dict = dict(label_dict)
n_classes  = len(label_dict)

# Split the data into a training and test set. #
x_data, y_label = shuffle(x_data, y_label, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_label, test_size=0.2, random_state=42)

# Build the model. #
num_epochs = 50
batch_size = 256
learning_rate = 0.01

x_inputs  = tf.keras.Input(shape=(28,28,1,), name='img_input')
x_layer_1 = layers.Conv2D(
    32, (3, 3), activation='relu', name='cnn_1')(x_inputs)
x_pool_1  = layers.MaxPooling2D((2, 2))(x_layer_1)
x_layer_2 = layers.Conv2D(
    64, (3, 3), activation='relu', name='cnn_2')(x_pool_1)
x_pool_2  = layers.MaxPooling2D((2, 2))(x_layer_2)
x_flatten = layers.Flatten()(x_pool_2)
x_linear  = layers.Dense(
    64, activation="relu", name="linear")(x_flatten)
x_outputs = \
    layers.Dense(n_classes, name='logits')(x_linear)

# Compile the Keras model. #
quickdraw_model = tf.keras.Model(inputs=x_inputs, outputs=x_outputs)
quickdraw_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["sparse_categorical_accuracy"])

# Train the model. #
print('Fit model on training data.')
history = quickdraw_model.fit(
    X_train, y_train, batch_size=batch_size, 
    epochs=num_epochs, validation_data=(X_test, y_test))
history_df = pd.DataFrame(history.history)
history_df["epoch"] = [(x+1) for x in range(num_epochs)]

print("Model trained.")
tmp_loss_file = \
    "C:/Users/admin/Desktop/Red Dragon/Advanced CV/quickdraw_losses.csv"
history_df.to_csv(tmp_loss_file, index=False)

# Plot the losses. #
plot_img_file = "C:/Users/admin/Desktop/" +\
    "Red Dragon/Advanced CV/quickdraw_losses.jpg"

fig, ax = plt.subplots()
ax.plot(history_df["epoch"], history_df["loss"])
ax.set(xlabel="Number of Epochs", 
       ylabel="Cross-Entropy Training Loss")

ax_twin = ax.twinx()
ax_twin.plot(
    history_df["epoch"], 
    history_df["val_sparse_categorical_accuracy"], color="green")
ax_twin.set(xlabel="Number of Epochs", 
            ylabel="Validation Accuracy")
fig.suptitle("Training Progress of Quickdraw Classifier")
fig.savefig(plot_img_file, dpi=199)
del fig, ax

# Show an example of predicted label. #
test_matrix = np.expand_dims(X_test[15], axis=0)
pred_logits = quickdraw_model.predict(test_matrix)
pred_label  = tf.argmax(pred_logits, axis=1)
pred_label  = label_dict[pred_label.numpy()[0]]

tmp_img_file = \
    "C:/Users/admin/Desktop/Red Dragon/Advanced CV/quickdraw_img.jpg"

tmp_title = "Predicted Label:" + pred_label + "\n"
tmp_title += "Actual Label:" + label_dict[y_test[15]]

plt.close("all")
plt.imshow(test_matrix[0, :, :, 0])
plt.title(tmp_title)
plt.savefig(tmp_img_file, dpi=199)
