from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import matplotlib.pyplot as plt
import datetime
# from keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
import keras_preprocessing
import tensorflow as tf
import os
import time
start = time.time()


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()

IMG_SIZE = (80, 80)
MODEL_SAVE_DIR = "./model"

data_generator_param = {
    "rescale": 1 / 256,  # 各ピクセルの値を0以上1以下に正規化(一般的な画像データは256階調で画素の明るさを表現しているから)
    "rotation_range": 10,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "shear_range": 0.1,
    "zoom_range": 0.1,
    "fill_mode": "nearest",
}
TRAINING_DIR = "./train-dataset"
training_datagen = ImageDataGenerator(**data_generator_param)
VALIDATION_DIR = "./valid-dataset"
validation_datagen = ImageDataGenerator(**data_generator_param)


"""
「バッチサイズについて」
  学習データから設定したサイズごとにデータを取り出し、計算を行う
  値は任意に調整すれば良いが、テスト用データの枚数を割り切れる数にしないとおかしなことになる可能性があるので注意する。
  参考：https://qiita.com/kenta1984/items/bad75a37d552510e4682
"""
TRAIN_BATCHSIZE = 100
train_generator = training_datagen.flow_from_directory(  # training_datagenの定義を用いて写真を変更し保存
    TRAINING_DIR,  # 学習する画像のフォルダを指定
    target_size=IMG_SIZE,  # 画像データのサイズ
    # "categorical"か"binary"か"sparse"か"input"か"other"か"None"のいずれか1つ．"categorical"は2次元のone-hotにエンコード化されたラベル
    class_mode='categorical',
    batch_size=TRAIN_BATCHSIZE  # batch_sizeはミニバッチ学習に使われるバッチのサイズ。
)

VAL_BATCHSIZE = 100
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    class_mode='categorical',
    batch_size=VAL_BATCHSIZE
)


# モデルの定義(構築)
# 以下に関連するドキュメント
"""
Conv2Dについて（https://keras.io/ja/layers/convolutional/）
  kernel_sizeで指定した範囲の画像を見て、畳み込みを行っています。画像における畳み込み計算は、簡単にいうとフィルター処理です（このフィルターは、カーネルとも呼ばれます）。
  https://github.com/vdumoulin/conv_arithmeticのREADME.mdにあるgifを見るとわかりやすい

  Conv2D(filters, kernel_size, strides=(1, 1),...)
    filters: 整数で，出力空間の次元（つまり畳み込みにおける出力フィルタの数）
    kernel_size: 2次元の畳み込みウィンドウの幅と高さを指定します. 単一の整数の場合は正方形のカーネル.=> カーネル(フィルター)のサイズを指定
    activation: 使用する活性化関数の名前(活性化関数一覧：https://keras.io/ja/activations/)
    input_shape: 128*128 RGB画像の時は(128, 128, 3)   3=チャンネル数を意味、3ch=RGBを意味
  =>特徴量をうまく抽出できるようなfilterをつくる

活性化関数:
  reru=負の数を0に変換する
  softmax=最大値を返す
"""

"""
MaxPooling2Dについて(https://keras.io/ja/layers/pooling/)
  pool_sizeの範囲を見て、その中で最も大きな値を次の層に渡します

  MaxPooling2D(pool_size=(2, 2), strides=None,...)
    pool_size: ダウンスケールする係数を決める 2つの整数のタプル（垂直，水平）． (2, 2) は画像をそれぞれの次元で半分にします． 1つの整数しか指定ないと，それぞれの次元に対して同じ値が用いられます．
    =>pllo_sizeごとにマスを見て、その中の最大値を返す=(2,2)で実行すると画像サイズは半分になる
"""
# 上の2つはCNN（Convolutional Neural Network、畳み込みニューラルネットワーク）と呼ばれる操作

"""
Flattenについて（https://keras.io/ja/layers/core/）
  入力を平坦化する．バッチサイズに影響を与えません．
  ＝>2次元（2D）の特徴を1次元に引き延ばす操作
"""

"""
Dropoutについて（https://keras.io/ja/layers/core/）
  入力にドロップアウトを適用する．過学習の防止に役立つ
  全結合の層とのつながりをランダムに切断してあげることで、過学習を防ぐ。rateはその切断する割合を示している。

  Dropout(rate, noise_shape=None, seed=None)
    rate： 0と1の間の浮動小数点数．入力ユニットをドロップする割合
"""

"""
Denseについて(https://keras.io/ja/layers/core/)
  通常の全結合ニューラルネットワークレイヤー．

  Dense(units, activation=None, use_bias=True,...)
      units：正の整数，出力空間の次元数＝ニューロンの数
      activation： 使用する活性化関数名 （活性化関数一覧：https://keras.io/ja/activations/） 何も指定しなければ，活性化は適用されない（すなわち，"線形"活性化： a(x) = x）
"""
# SequentialAPI(シンプルな一直線のモデル)によって構築
# Functional APIやSubclassing APIなども存在
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # ↑ おそらくinput_shape=(150, 150, 3)を適宜変更してという意味
    # This is the first convolution
    tf.keras.layers.Conv2D(
        64, (3, 3), activation='relu', input_shape=(
            128, 128, 3)),  # 3*3のフィルター(カーネル)を64枚
    tf.keras.layers.MaxPooling2D(2, 2),  # 2*2マスごとに見て最大値を返す => 画像サイズは半分になる
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# summary()=作成したニューラルネットワークのモデル形状の概要を表示(上で定義したモデルの概要を表示(可視化))
model.summary()


"""
compileについて（https://keras.io/ja/models/model/）
  学習のためのモデルを設定=生成したモデルに訓練（学習）プロセスを設定する

  compile(optimizer, loss=None, metrics=None,...)
    optimizer: 最適化アルゴリズム(詳細：https://keras.io/api/optimizers/)
    loss: 損失関数（詳細：https://keras.io/api/losses/）
    metrics: 評価関数．一般的にはmetrics=['accuracy']を使う．(詳細：https://keras.io/ja/metrics/)

"""
# パラメータ
LOSS = 'categorical_crossentropy'
OPTIMIZER = 'rmsprop'
METRICS = 'accuracy'

model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=[METRICS])


"""
fitについて（https://keras.io/ja/models/model/）
  固定回数（データセットの反復）の試行でモデルを学習＝訓練の実行

  fit(epochs=1, verbose=1,  validation_data=None, validation_steps=None,steps_per_epoch=None,...)
    epochs: 訓練データ配列の反復回数を示す整数．<=>モデルを学習するエポック数（学習データ全体を何回繰り返し学習させるか）
    verbose: 進行状況の表示モード．0 = 表示なし，1 = プログレスバー，2 = 各試行毎に一行の出力．
    validation_data: 各試行の最後に損失とモデル評価関数を評価するために用いられる(x_val, y_val)のタプル
    steps_per_epoch: 終了した1エポックを宣言して次のエポックを始めるまでのステップ数の合計（サンプルのバッチ）．
    validation_steps: steps_per_epochを指定している場合のみ関係します．停止する前にバリデーションするステップの総数（サンプルのバッチ）．

お役立ち情報
  ModelCheckpoint:	各種タイミング（毎エポック終了時や、今まででlossが一番小さかったエポック終了時など）に、学習したモデルを保存できる。保存する名前を、そのエポック数や、lossの値にすることもできる
  EarlyStopping:	lossや正答率が変化しなくなったところで、学習を打ち切ることができる
  LearningRateScheduler:	学習率のスケジューリングができる
  CSVLogger:	各エポックの結果をCSVファイルに保存する
"""
# パラメータ
EPOCHS = 20
STEPS_PER_EPOCH = 15
VALIDATION_DATA = validation_generator
VERBOSE = 1
VALIDATION_STEPS = 3

history = model.fit(train_generator,
                    epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=VALIDATION_DATA,
                    verbose=VERBOSE,
                    validation_steps=VALIDATION_STEPS
                    )


date_now = str(datetime.datetime.now().strftime(
    '%Y%m%d-%H%M%S'))  # 現在時刻 (UTC or GMTのどちらか忘れた)
elapsed_time = time.time() - start  # 経過時間の算出
elapsed_time = datetime.timedelta(seconds=elapsed_time)  # 経過時間の換算 秒->時間or分


"""
.h5形式のファイルについて(一応)
  Hierarchical Data Formatの略.階層化された形でデータを保存することができるファイル形式

  メリット
    csvよりも読み書きが早い(pickleよりは遅い)
    他の言語でも使える(pickleとは違う点)

"""
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
# plt.figure()


plt.savefig(f'{MODEL_SAVE_DIR}/' + date_now + '_acc.png')

plt.show()
