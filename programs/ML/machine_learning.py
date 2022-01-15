
import matplotlib.pyplot as plt
import datetime
# from keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os

from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# info
IMG_SIZE = (80, 80)
MODEL_SAVE_DIR = "./model"
TRAINING_DIR = "./train-dataset"
VALIDATION_DIR = "./valid-dataset"
NUM_CLASSES = 75

# params
data_generator_param = {
    "rescale": 1 / 256,  # 各ピクセルの値を0以上1以下に正規化(一般的な画像データは256階調で画素の明るさを表現しているから)
    "rotation_range": 10,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "shear_range": 0.1,
    "zoom_range": 0.1,
    "fill_mode": "nearest",
}
TRAIN_BATCHSIZE = 1
VAL_BATCHSIZE = 1

LOSS = 'categorical_crossentropy'
OPTIMIZER = 'rmsprop'
METRICS = 'accuracy'

EPOCHS = 60
STEPS_PER_EPOCH = 1260
VERBOSE = 1
VALIDATION_STEPS = 225


training_datagenerator = ImageDataGenerator(**data_generator_param)
validation_datagenerator = ImageDataGenerator(**data_generator_param)
generated_train_data = training_datagenerator.flow_from_directory(  # training_datagenの定義を用いて写真を変更し保存
    TRAINING_DIR,  # 学習する画像のフォルダを指定
    target_size=IMG_SIZE,  # 画像データのサイズ
    # "categorical"か"binary"か"sparse"か"input"か"other"か"None"のいずれか1つ．"categorical"は2次元のone-hotにエンコード化されたラベル
    class_mode='categorical',
    batch_size=TRAIN_BATCHSIZE  # batch_sizeはミニバッチ学習に使われるバッチのサイズ。
)
generated_vali_data = validation_datagenerator.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    class_mode='categorical',
    batch_size=VAL_BATCHSIZE
)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        64, (3, 3), activation='relu', input_shape=(
            80, 80, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
model.summary()
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=[METRICS])
history = model.fit(generated_train_data,
                    epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=generated_vali_data,
                    verbose=VERBOSE,
                    validation_steps=VALIDATION_STEPS
                    )


date_now = str(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
model.save(
    os.path.join(
        MODEL_SAVE_DIR,
        date_now +
        "_model.h5"))  # モデルの保存(h5形式)

# モデルの保存(png形式)  expand_nestedはデフォルトではFalse
# plot_model(model, to_file=date_now + "_model.png", show_shapes=True)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.plot(epochs, loss, 'g', label="Training loss")
plt.plot(epochs, val_loss, 'm', label="Validation loss")
plt.title('Training and validation accuracy and loss')
plt.legend(loc=0)
plt.savefig(f'{MODEL_SAVE_DIR}/' + date_now + '_acc.png')
plt.show()
