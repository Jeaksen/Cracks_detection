import os
import random
import time
import math
import mahotas
import threading
import mahotas.features
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi


try:
    from cv2 import cv2

except ImportError:
    pass


def run_tests():
    prepare_gabor_kernels()
    file = open("results/output.csv", "a+")
    # file.write("Method;count;accuracy;time\n")
    functions_list = [first_order_features, hog_features, haralick, lbp_hist, gabor_features, first_order_haralick, first_order_hog]

    for count in range(1000, 21000, 1000):
        for method in functions_list:
            classify(method, count, file)
    file.close()


def run_tests_thread(method):
    prepare_gabor_kernels()
    file = open(f"results/output_{method.__name__}.csv", "a+")
    file.write("Method;count;accuracy;time\n")

    for count in range(1000, 21000, 1000):
        classify(method, count, file)
    file.close()


# This methods steers the process of testing a feature accuracy with a given size of input files
def classify(method, count, file):
    positive_images, negative_images = get_pictures(count)
    start = time.time_ns()
    p_features = extract_features(positive_images, method)
    n_features = extract_features(negative_images, method)

    features_df, results = create_dataset(p_features, n_features)
    x_train, x_test, y_train, y_test = train_test_split(features_df, results, test_size=0.2)

    classifier = RandomForestClassifier(n_estimators=10)
    classifier = classifier.fit(x_train, y_train)

    accuracy = round(classifier.score(x_test, y_test), 5)
    classification_time = round((time.time_ns() - start) / 1e9, 3)
    log_results(file, method, count, accuracy, classification_time)
    file.flush()


def log_results(file, method, count, accuracy, classification_time):
    file.write(f'{method.__name__};{count};{accuracy * 100};{classification_time}\n')
    print(f"Method: {method.__name__}")
    print(f"Pictures count: {count}")
    print(f"Accuracy: {accuracy * 100}%")
    print(f"time:{classification_time}s\n")


# Returns $count random pictures with cracks and count without cracks
def get_pictures(count):
    negative_dir = "./archive/Negative/"
    positive_dir = "./archive/Positive/"

    positive = random.sample(os.listdir(positive_dir), count)
    negative = random.sample(os.listdir(negative_dir), count)
    positive = list(map(lambda file_name: positive_dir + file_name, positive))
    negative = list(map(lambda file_name: negative_dir + file_name, negative))
    return positive, negative


# Return a combined DataFrame with the features and a result vector with assignment to classes
def create_dataset(pos_features, neg_features):
    df = pd.DataFrame(pos_features + neg_features)
    res = [1] * len(pos_features) + [0] * len(neg_features)
    return df, res


# Return a list of features computed for each image from the list using the given method
def extract_features(image_names, method):
    return list(map(lambda name: features_for(name, method), image_names))


# Return features computed for given image using the given method
def features_for(img_name, method):
    gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    return method(gray)


# Return a histogram of local binary pattern image from the given image
def lbp_hist(img):
    lbp = local_binary_pattern(img, 8, 1, 'ror')
    lbp = np.ndarray.astype(lbp, np.uint8)
    return cv2.calcHist([lbp], [0], None, [256], [0, 256]).reshape([256])


# Returns a vector of haralick features for the image
def haralick(img):
    return mahotas.features.haralick(img).mean(0)


# Returns a vector of HOG features for the image
def hog_features(img):
    resized_img = cv2.resize(img, (64, 64))
    return hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))


# Returns a vector of first order histogram based features for the image
def first_order_features(img):
    intensity = cv2.calcHist([img], [0], None, [256], [0, 256]).reshape([256]) / np.size(img)
    ft_vec = []
    mean = np.sum([i * intensity[i] for i in range(256)])
    ft_vec.append(mean)
    # variance
    var = np.sum([pow((i - mean), 2) * intensity[i] for i in range(256)])
    ft_vec.append(var)
    # skewness
    ft_vec.append(np.sum([pow((i - mean), 3) * intensity[i] for i in range(256)]) / (var * math.sqrt(var)))
    # kurtosis
    ft_vec.append(np.sum([pow((i - mean), 4) * intensity[i] for i in range(256)]) / (var * var) - 3)
    # energy
    ft_vec.append(np.sum(pow(intensity, 2)))
    return ft_vec


# Initializes gabor kernels which are used in the Gabor filter features extraction
def prepare_gabor_kernels():
    for theta in range(3):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)


# Returns a features vector based on convolution of the image with defined Gabor filters
def gabor_features(img):
    feats = np.zeros(2 * (len(kernels)), dtype=np.double)
    img = cv2.resize(img, (64, 64))
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(img, kernel, mode='wrap')
        feats[2*k] = filtered.mean()
        feats[2*k + 1] = filtered.var()
    return feats


# Returns a vector of features which is a combination of First order and Haralick methods
def first_order_haralick(img):
    return first_order_features(img) + list(haralick(img))


# Returns a vector of features which is a combination of First order and HOG methods
def first_order_hog(img):
    return first_order_features(img) + list(hog_features(img))


#Global variables
kernels = []
# Program start
run_tests()
#     functions_list = [first_order_features, hog_features, haralick, lbp_hist, gabor_features, first_order_haralick]
#     threads = []
#     s = time.time()
#     for fun in functions_list:
#         x = threading.Thread(target=run_tests_thread, args=(fun, ))
#         x.start()
#         threads.append(x)
#
#     for index, thread in enumerate(threads):
#         thread.join()
#     print(f"time: {time.time() - s}s")
