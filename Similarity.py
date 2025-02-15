import numpy as np
import pickle
import sklearn as skl
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# import mnist datasets
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int)
X = X / 255.0
X = X.values.reshape(-1, 28, 28, 1)
y = np.eye(10)[y]  # one-hot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

subset_size = 10000
random_indices = np.random.choice(len(X_train), subset_size, replace=False)

X_train = X_train[random_indices]
y_train = y_train[random_indices]

print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

input_depth = 1
input_size = 28
num_kernels = 3
kernel_size = 3
stride = 1
padding = 1  # (3x3) therefore padding = 1

weights = np.random.randn(num_kernels, input_depth, kernel_size, kernel_size) * 0.1
biases = np.zeros(num_kernels)



# fully connected layer initialization

out_height = (input_size + 2 * padding - kernel_size) // stride + 1
out_width = (input_size + 2 * padding - kernel_size) // stride + 1

hidden_size = num_kernels * (out_height * out_width)


fc_output_size = 128

W = np.random.randn(hidden_size, fc_output_size) * np.sqrt(2.0 / hidden_size)
b = np.zeros(fc_output_size)


def conv_forward(X):
    pad_X = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    batch_size, height, width, _ = pad_X.shape
    out_height = (height - kernel_size) // stride + 1
    out_width = (width - kernel_size) // stride + 1
    output = np.zeros((batch_size, out_height, out_width, num_kernels))

    for k in range(num_kernels):
        for h in range(out_height):
            for w in range(out_width):
                region = pad_X[:, h * stride: h * stride + kernel_size, w * stride: w * stride + kernel_size, :]
                output[:, h, w, k] = np.sum(region * weights[k], axis=(1, 2, 3)) + biases[k]

    return np.maximum(0, output)


def extract_features(X, batch_size=32):
    num_samples = X.shape[0]
    all_features = []

    for i in range(0, num_samples, batch_size):
        batch_X = X[i:i + batch_size]
        conv_output = conv_forward(batch_X)
        flatten_output = conv_output.reshape(batch_X.shape[0], -1)
        batch_features = np.dot(flatten_output, W) + b
        all_features.append(batch_features)

    return np.vstack(all_features)

train_features = extract_features(X_train,batch_size=32)
test_features = extract_features(X_test, batch_size=32)



similarity_method = "cosine"


if similarity_method == "cosine":
    similarities = cosine_similarity(test_features, train_features)
    predictions = y_train[np.argmax(similarities, axis=1)]
    accuracy = np.mean(predictions == y_test)
    print(f"cosine相似度模型分类准确率: {accuracy:.4f} ")

    plt.figure(figsize=(10, 10))
    num_images = 16  # Show 16 images
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {np.argmax(predictions[i])} / True: {np.argmax(y_test[i])}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

model_params = {
    "weights": weights,
    "biases": biases,
    "W": W,
    "b": b
}
with open("cnn_model.pkl", "wb") as f:
    pickle.dump(model_params, f)

print("模型已保存到 cnn_model.pkl")

# with open("cnn_model.pkl", "rb") as f:
#     loaded_model = pickle.load(f)
#
# weights = loaded_model["weights"]
# biases = loaded_model["biases"]
# W = loaded_model["W"]
# b = loaded_model["b"]
#
# print("模型已加载成功")
#
# def predict(X):
#     test_features = extract_features(X, batch_size=32)
#     similarities = cosine_similarity(test_features, train_features)
#     predictions = y_train[np.argmax(similarities, axis=1)]
#     return predictions
#
# predictions = predict(X_test)
# accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
# print(f"测试集分类准确率: {accuracy:.4f}")
#
# plt.figure(figsize=(10, 10))
# num_images = 16
# for i in range(num_images):
#     plt.subplot(4, 4, i + 1)
#     plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
#     plt.title(f"Pred: {np.argmax(predictions[i])} / True: {np.argmax(y_test[i])}")
#     plt.axis('off')
#
# plt.tight_layout()
# plt.show()


# similarity_method = "KMeans"
# if similarity_method == "KMeans":
#     kmeans = KMeans(n_clusters=10, random_state=0)
#     kmeans.fit(train_features)
#
#     kmeans_predictions = kmeans.predict(test_features)
#     kmeans_predictions = np.eye(10)[kmeans_predictions]
#
#     kmeans_accuracy = np.mean(np.argmax(kmeans_predictions, axis=1) == np.argmax(y_test, axis=1))
#     print(f"KMeans模型分类准确率: {kmeans_accuracy:.4f}")
#
# similarity_method = "EM"
# if similarity_method == "EM":
#     gmm = GaussianMixture(n_components=10, random_state=0)
#     gmm.fit(train_features)
#     gmm_predictions = gmm.predict(test_features)
#
#     gmm_predictions = np.eye(10)[gmm_predictions]
#     gmm_accuracy = np.mean(np.argmax(gmm_predictions, axis=1) == np.argmax(y_test, axis=1))
#     print(f"EM模型（高斯混合）分类准确率: {gmm_accuracy:.4f}")










