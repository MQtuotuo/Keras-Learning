from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.models import Model

'''
model= Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
# Dense(32) is a fully-connected layer with 32 hidden units.
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))

labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)
'''
import numpy as np
data = np.random.random((1000, 784))

labels = np.random.randint(2, size=(1000, 10))


# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
# 输入inputs，输出x
# (inputs)代表输入
x = Dense(64, activation='relu')(x)
# 输入x，输出x
predictions = Dense(10, activation='softmax')(x)
# 输入x，输出分类

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, epochs=10)  # starts training
