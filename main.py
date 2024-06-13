import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# DATASET PREPARATION
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
zip_file = tf.keras.utils.get_file(origin=_URL, fname="flower_photos.tgz", extract=True)
base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

# PARAMETERS
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
dataset_split_percentage = 0.8
epochs = 5
batch_size = 128
IMG_SHAPE = 224
learning_rate = 0.01
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
modelName = "ResNetfinetune_" + "augmentation_" + "E" + str(epochs) + "_LR" + str(learning_rate) + "_" + timestamp
log_dir = os.path.join("logs", "fit", modelName)
checkpoint_path = os.path.join("checkpoints", modelName, "model.keras")

# STRATIFIED SPLIT OF DATASET AND DIRECTORY PREPARATION
for cl in classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + '/*.jpg')
    print("{}: {} Images".format(cl, len(images)))
    num_train = int(round(len(images) * dataset_split_percentage))
    train, val = images[:num_train], images[num_train:]

    for t in train:
        if not os.path.exists(os.path.join(base_dir, 'train', cl)):
            os.makedirs(os.path.join(base_dir, 'train', cl))
        bn=os.path.basename(t)
        if not os.path.exists(os.path.join(base_dir, 'train', cl, bn)):
            shutil.move(t, os.path.join(base_dir, 'train', cl))

    for v in val:
        if not os.path.exists(os.path.join(base_dir, 'val', cl)):
            os.makedirs(os.path.join(base_dir, 'val', cl))
        bn = os.path.basename(v)
        if not os.path.exists(os.path.join(base_dir, 'val', cl, bn)):
            shutil.move(v, os.path.join(base_dir, 'val', cl))
    print(" training images:", num_train, "; validation images:", len(images)-num_train)

# DATA GENERATORS AND AUGMENTATION
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

image_gen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5)
train_data_gen = image_gen_train.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='sparse')

image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(
    batch_size=batch_size,
    directory=val_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='sparse')

# LOAD RESNET MODEL
modelResNet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SHAPE, IMG_SHAPE, 3))
modelResNet.trainable = False

# NEW LAEYRS
model = tf.keras.Sequential([
    modelResNet,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation='softmax')
])

# COMPILE MODEL
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# CALLBACKS
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min')

# TRAINING
history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(batch_size))),
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback]
)

# EVALUATION
model.load_weights(checkpoint_path)
val_data_gen.reset()
test_loss, test_acc = model.evaluate(val_data_gen)
print(f"Validation Accuracy: {test_acc:.4f}")

# PREDICTIONS
test_pred_raw = model.predict(val_data_gen, verbose=1)
test_pred = np.argmax(test_pred_raw, axis=-1)
test_labels = val_data_gen.classes

# METRICS
conf_matrix = confusion_matrix(test_labels, test_pred)
precision = precision_score(test_labels, test_pred, average='weighted')
recall = recall_score(test_labels, test_pred, average='weighted')
f1 = f1_score(test_labels, test_pred, average='weighted')
accuracy = accuracy_score(test_labels, test_pred)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# RESULTS
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
