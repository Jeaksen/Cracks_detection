import os
import random
import time
import math
import mahotas
import mahotas.features
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from mahotas.features import surf

try:
    from cv2 import cv2

except ImportError:
    pass

#Global variables
kernels = []


# Returns count random pictures with cracks and count without cracks
def get_pictures(count):
    negative_dir = "./archive/Negative/"
    positive_dir = "./archive/Positive/"

    positive = random.sample(os.listdir(positive_dir), count)
    negative = random.sample(os.listdir(negative_dir), count)
    positive = list(map(lambda file_name: positive_dir + file_name, positive))
    negative = list(map(lambda file_name: negative_dir + file_name, negative))
    return positive, negative


def create_dataset(pos_features, neg_features):
    df = pd.DataFrame(pos_features + neg_features)
    res = [1] * len(pos_features) + [0] * len(neg_features)
    return df, res


def extract_features(image_names, method):
    return list(map(lambda name: features_for(name, method), image_names))


def features_for(img_name, method):
    gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    return method(gray)


def lbp_hist(img):
    lbp = local_binary_pattern(img, 8, 1, 'ror')
    lbp = np.ndarray.astype(lbp, np.uint8)
    return cv2.calcHist([lbp], [0], None, [256], [0, 256]).reshape([256])


def haralick(img):
    return mahotas.features.haralick(img).mean(0)


def hog_features(img):
    resized_img = cv2.resize(img, (32, 32))
    return hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))


def first_order_features(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).reshape([256]).reshape([256])
    intensity = hist / np.size(img)
    ft_vec = []
    # mean
    h_mean = np.sum([i * intensity[i] for i in range(256)])
    ft_vec.append(h_mean)
    # variance
    h_var = np.sum([pow((i - h_mean), 2) * intensity[i] for i in range(256)])
    ft_vec.append(h_var)
    # # skewness
    h_skew = np.sum([pow((i - h_mean), 3) * intensity[i] for i in range(256)]) / (h_var * math.sqrt(h_var))
    ft_vec.append(h_skew)
    # kurtosis
    h_kurt = np.sum([pow((i - h_mean), 4) * intensity[i] for i in range(256)]) / (h_var * h_var) - 3
    ft_vec.append(h_kurt)
    h_energy = np.sum(pow(intensity, 2))
    ft_vec.append(h_energy)
    return ft_vec


def prepare_gabor_kernels():
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)


def gabor_features(img):
    feats = np.zeros(2 * (len(kernels)), dtype=np.double)
    resized_img = cv2.resize(img, (64, 64))
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(resized_img, kernel, mode='wrap')
        feats[2*k] = filtered.mean()
        feats[2*k + 1] = filtered.var()
    return feats


def run_tests(debug):
    start = 0
    prepare_gabor_kernels()
    file = open("output.txt", "w+")
    functions_list = [first_order_features, hog_features, haralick, lbp_hist, gabor_features]
    for method in functions_list:
        for count in range(100, 500, 100):
            if debug:
                print("---------------------------START-------------------------------")
                print(f'Getting random {count} positive and {count} negative pictures')
                start = time.time_ns()
            positive_images, negative_images = get_pictures(count)
            if debug:
                print(f"time:{(time.time_ns() - start) / 1e9}s")
                print("\n----------Extracting features----------\n")

            start = time.time_ns()
            p_features = extract_features(positive_images, method)
            if debug:
                print(f"positive time:{(time.time_ns() - start) / 1e9}s")
                start = time.time_ns()
            n_features = extract_features(negative_images, method)
            if debug:
                print(f"negative time:{(time.time_ns() - start) / 1e9}s")
                print("\n----------Creating data frame----------\n")
                start = time.time_ns()

            features_df, results = create_dataset(p_features, n_features)
            x_train, x_test, y_train, y_test = train_test_split(features_df, results, test_size=0.2)
            if debug:
                print(f"time:{(time.time_ns() - start) / 1e9}s")
                print("\n----------Testing----------\n")
                start = time.time_ns()

            classifier = RandomForestClassifier(n_estimators=10)
            classifier = classifier.fit(x_train, y_train)
            # predictions = classifier.predict(x_test)
            # indexes = list(X_test.index)
            # names_list = positive_images + negative_images
            # print([names_list[i] for i in indexes])
            # print(y_test)
            # print("Predictions:", predictions)
            accuracy = classifier.score(x_test, y_test)
            classification_time = (time.time_ns() - start) / 1e9
            print(f"Method: {method.__name__}")
            print(f"Pictures count: {count}")
            print(f"Accuracy: {accuracy * 100}%")
            print(f"time:{classification_time}s")
            print("")



if __name__ == '__main__':
    run_tests(False)
