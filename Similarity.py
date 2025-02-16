import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 加载并预处理MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_samples = 10000
random_indices = np.random.choice(x_train.shape[0], num_samples, replace=False)
x_train, y_train = x_train[random_indices], y_train[random_indices]
# 归一化数据并调整形状
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 创建卷积神经网络模型

input_layer = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
feature_output = Dense(128, activation='relu')(x)

model = Model(inputs=input_layer, outputs=feature_output)

#
model.compile(optimizer=Adam(), loss='mean_squared_error')


features_train = model.predict(x_train)

model.fit(x_train, features_train, epochs=5, batch_size=64, validation_data=(x_test,features_train))

# 保存训练后的模型
model.save('mnist_similarity_model.keras')

# 加载模型并定义相似度计算函数
def get_features(image, model):
    image = np.expand_dims(image, axis=-1)  # 增加通道维度
    image = np.expand_dims(image, axis=0)  # 增加批量维度
    return model.predict(image)

# 加载模型
model = tf.keras.models.load_model('mnist_similarity_model.keras')

# 选择两张图片来计算相似度
indices = np.random.choice(len(x_test), 2, replace=False)
image1 = x_test[indices[0]]  # 选择一张测试集图片
image2 = x_test[indices[1]]  # 选择另一张测试集图片

# 获取两张图片的特征
features1 = get_features(image1, model)
features2 = get_features(image2, model)

# 计算余弦相似度
similarity = cosine_similarity(features1, features2)
print(f"Similarity between image1 and image2: {similarity[0][0]}")


def plot_images(image1, image2, similarity):
    # 设置画布
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 绘制第一张图片
    axes[0].imshow(image1.squeeze(), cmap='gray')
    axes[0].set_title("Image 1")
    axes[0].axis('off')

    # 绘制第二张图片
    axes[1].imshow(image2.squeeze(), cmap='gray')
    axes[1].set_title("Image 2")
    axes[1].axis('off')

    # 显示相似度
    plt.suptitle(f"Cosine Similarity: {similarity[0][0]:.4f}", fontsize=16)
    plt.show()

plot_images(image1, image2, similarity)