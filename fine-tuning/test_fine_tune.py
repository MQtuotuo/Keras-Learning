WEIGHTS_PATH = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

from keras.applications.vgg16 import VGG16
model = VGG16(include_top=False, weights='imagenet')

#如何提取bottleneck feature
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# （1）载入图片
# 图像生成器初始化
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

nb_train_samples = 500
nb_validation_samples = 100
epochs = 50
batch_size = 4


datagen = ImageDataGenerator(rescale=1./255)

# 训练集图像生成器
generator = datagen.flow_from_directory(
        'data/animal5/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode=None,
        shuffle=False)



#（2）灌入pre-model的权重
model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

#（3）得到bottleneck feature
bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)
# 核心，steps是生成器要返回数据的轮数，每个epoch含有500张图片，与model.fit(samples_per_epoch)相对
np.save(open('5_bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


#　验证集图像生成器
generator = datagen.flow_from_directory(
        'data/animal5/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode=None,
        shuffle=False)

bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples // batch_size)
# 与model.fit(nb_val_samples)相对，一个epoch有100张图片，验证集
np.save(open('5_bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

# （1）导入bottleneck_features数据
train_data = np.load(open('5_bottleneck_features_train.npy', 'rb'))
# the features were saved in order, so recreating the labels is easy
train_labels = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 96)  # matt,打标签

validation_data = np.load(open('5_bottleneck_features_validation.npy', 'rb'))
validation_labels = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 16)  # matt,打标签

# （2）设置标签，并规范成Keras默认格式
import keras.utils as utils
train_labels = utils.to_categorical(train_labels, 5)
validation_labels = utils.to_categorical(validation_labels, 5)

# （3）写“小网络”的网络结构
model = Sequential()
#train_data.shape[1:]
model.add(Flatten(input_shape=train_data.shape[1:]))
#model.add(Flatten(input_shape=(4,4,512)))# 4*4*512
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(1, activation='sigmoid'))  # 二分类
model.add(Dense(5, activation='softmax'))  # matt,多分类
#model.add(Dense(1))
#model.add(Dense(5))
#model.add(Activation('softmax'))

# （4）设置参数并训练
model.compile(loss='categorical_crossentropy',
# matt，多分类，不是binary_crossentropy
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=epochs, batch_size=batch_size,
          validation_data=(validation_data, validation_labels))
model.save_weights('5_bottleneck_fc_model.h5')

