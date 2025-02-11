import numpy as np
import sklearn as skl
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split



if __name__ == "__main__":

    #import mnist datasets
    mnist = fetch_openml('mnist_784',version=1)
    X,y = mnist.data, mnist.target.astype(int)
    X = X /255.0
    X = X.values.reshape(-1, 28, 28, 1)
    y = X.copy()



    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=0)

    subset_size = 10000
    random_indices = np.random.choice(len(X_train), subset_size, replace=False)

    X_train = X_train[random_indices]
    y_train = y_train[random_indices]

    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

    noise_factor = 0.3
    X_noisy_train = np.clip(X_train + np.random.normal(0,noise_factor,X_train.shape),0,1)
    X_noisy_test = np.clip(X_test + np.random.normal(0,noise_factor,X_test.shape),0,1)

    num_epochs = 5
    batch_size = 32
    input_depth = 1
    num_kernels = 3
    kernel_size = 3
    stride = 1
    padding = 1   #(3x3) therefore padding = 1


    weights = np.random.randn(num_kernels, input_depth, kernel_size, kernel_size) * 0.1
    biases = np.zeros(num_kernels)

    hidden_size = 588
    input_size = 28 * 28
    output_size = 28 * 28

    #fully connected layer initialization
    W1 = np.random.randn(input_size * num_kernels, hidden_size) * np.sqrt(2.0 / input_size * num_kernels)
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size , output_size) * np.sqrt(2.0 / input_size )
    b2 = np.zeros(output_size)


    for epoch in range(num_epochs):
        for i in range(0,len(X_train),batch_size):
            batch_X = X_noisy_train[i:i + batch_size]
            batch_y = X_train[i:i + batch_size]   #images that without noise

            #padding
            pad_X = np.pad(batch_X, ((0,0),(padding, padding), (padding, padding), (0, 0)), mode='constant')
            batch_size, height,width,_ = pad_X.shape

            out_height = (height  - kernel_size) // stride + 1   #(origin_height + 2 * padding - ker_size) // stride +1
            out_width = (width  - kernel_size) // stride + 1
            output = np.zeros((batch_size,out_height, out_width, num_kernels))

            for k in range(num_kernels):
                for h in range(0, out_height):
                    for w in range(0, out_width):
                        region = pad_X[:,h * stride:h * stride + kernel_size,
                                 w * stride:w * stride + kernel_size, :]
                        output[:,h, w, k] = np.sum(region * weights[k], axis=(1,2,3)) + biases[k]


            relu_output = np.where( output > 0, output, 0.01*output)

            flatten_output = relu_output.reshape(batch_size, -1)


            input_size = flatten_output.shape[1]
            hidden_size = 588
            output_size =  28 * 28  # reconstruct images

            # fully connection layer


            hidden_layer = np.dot(flatten_output, W1) + b1
            hidden_layer = np.maximum(0, hidden_layer)

            output_layer = np.dot(hidden_layer, W2) + b2
            output_layer = 1 / (1 + np.exp(-output_layer))
            output_layer = output_layer.reshape(batch_size, 28, 28, 1)

            # loss calculation
            loss = np.mean((output_layer - batch_y) **2)
            print(f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Loss: {loss:.4f}")

            # backpropagation
            d_output = 2 * (output_layer - batch_y) / batch_size

            dW2 = np.dot(hidden_layer.T , d_output.reshape(batch_size, -1))
            db2 = np.sum(d_output, axis=(0,1,2))

            d_hidden = np.dot(d_output.reshape(batch_size, -1), W2.T)
            d_hidden[hidden_layer <= 0] = 0

            dW1 = np.dot(flatten_output.T, d_hidden)
            db1 = np.sum(d_hidden, axis=0)

            #Update Fully Connected Layer Parameters
            learning_rate = 0.01

            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1

            if np.isnan(dW2).any() or np.isnan(db2).any():
                print("NaN detected in dW2 or db2!")
                break


    print("Trainning Completed.")
    test_sample = X_noisy_test[:5].reshape(5,28,28,1)
    pad_X = np.pad(test_sample, ((0,0),(padding, padding), (padding, padding), (0, 0)), mode='constant')

    batch_size, height, width, _ = pad_X.shape

    out_height = (height - kernel_size) // stride + 1
    out_width = (width - kernel_size) // stride + 1
    output = np.zeros((batch_size, out_height, out_width, num_kernels))
    for k in range(num_kernels):
        for h in range(0, out_height):
            for w in range(0, out_width):
                region = pad_X[:, h * stride:h * stride + kernel_size,
                         w * stride:w * stride + kernel_size, :]
                output[:, h, w, k] = np.sum(region * weights[k], axis=(1, 2, 3)) + biases[k]

    relu_output = np.where(output > 0, output, 0.01 * output)
    flatten_output = relu_output.reshape(batch_size, -1)
    hidden_layer = np.dot(flatten_output, W1) + b1
    hidden_layer = np.maximum(0, hidden_layer)
    output_layer = np.dot(hidden_layer, W2) + b2
    output_layer = 1 / (1 + np.exp(-output_layer))
    reconstructed = output_layer.reshape(5, 28, 28)


    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for j in range(5):
        axes[0, j].imshow(test_sample[j].reshape(28, 28), cmap='gray')
        axes[0, j].set_title("Noisy Input")
        axes[0, j].axis("off")

        axes[1, j].imshow(reconstructed[j], cmap='gray')
        axes[1, j].set_title("Reconstructed")
        axes[1, j].axis("off")

    plt.show()









