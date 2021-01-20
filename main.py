import os
import random
import time
import math
import mahotas
import mahotas.features
import numpy as np
import pandas as pd
import scipy.stats as stat
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import local_binary_pattern

try:
    from cv2 import cv2

except ImportError:
    pass


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
    vector_size = len(pos_features[0])
    df = pd.DataFrame(columns=list(range(vector_size)))

    res = [1] * len(pos_features) + [0] * len(neg_features)
    for i in range(len(pos_features) + len(neg_features)):
        if i < len(pos_features):
            df.loc[i] = pos_features[i]
        else:
            df.loc[i] = neg_features[i - len(pos_features)]

    return df, res


def extract_features(image_names):
    return list(map(features_for, image_names))


def lbp_hist(img):
    lbp = local_binary_pattern(img, 8, 1, 'ror')
    lbp = np.ndarray.astype(lbp, np.uint8)
    return cv2.calcHist([lbp], [0], None, [256], [0, 256]).reshape([256])


def haralick(img):
    return mahotas.features.haralick(img).mean(0)


def first_order_features(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).reshape([256]).reshape([256])
    intensity = hist / np.size(img)
    # mean
    h_mean = np.sum([i * intensity[i] for i in range(256)])
    ft_vec = [h_mean]
    # variance
    h_var = np.sum([pow((i - h_mean), 2) * intensity[i] for i in range(256)])
    ft_vec += h_var
    # skewness
    h_skew = np.sum([pow((i - h_mean), 3) * intensity[i] for i in range(256)]) / (h_var * math.sqrt(h_var))
    ft_vec += h_skew
    # kurtosis
    h_kurt = np.sum([pow((i - h_mean), 4) * intensity[i] for i in range(256)]) / (h_var * h_var) - 3
    ft_vec += h_kurt
    h_energy = np.sum(pow(intensity, 2))
    ft_vec += h_energy
    return ft_vec


def features_for(img_name):
    # rgb = cv2.imread(img_name)
    gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    # return haralick(rgb)
    # return lbp_hist(gray)
    return first_order_features(gray)


if __name__ == '__main__':
    files_count = 1000
    testing = False
    print(f'Getting random {files_count} positive and {files_count} negative pictures')

    start = time.time_ns()
    positive_images, negative_images = get_pictures(files_count)
    print("time:", (time.time_ns() - start) / 1e9, "s")
    # print(positive_images)
    # print(negative_images)

    print("\n----------Extracting features----------\n")

    start = time.time_ns()
    p_features = extract_features(positive_images)
    print("positive time:", (time.time_ns() - start) / 1e9, "s")
    start = time.time_ns()
    n_features = extract_features(negative_images)
    print("negative time:", (time.time_ns() - start) / 1e9, "s")

    if not testing:
        print("\n----------Creating data frame----------\n")

        start = time.time_ns()
        features_df, results = create_dataset(p_features, n_features)
        X_train, X_test, y_train, y_test = train_test_split(features_df, results, test_size=0.2)
        print("time:", (time.time_ns() - start) / 1e9, "s")

        # print("Training:", X_train, "Test:", X_test, sep='\n')
        # print("Training labels:", y_train, "Test labels", y_test, sep='\n')
        print("Test labels", y_test, sep='\n')

        print("\n----------Testing----------\n")

        start = time.time_ns()
        classifier = RandomForestClassifier(n_estimators=10)
        predictions = classifier.fit(X_train, y_train).predict(X_test)
        print("Predictions:", predictions)
        print("Accuracy:", classifier.score(X_test, y_test) * 100, "%")
        print("time:", (time.time_ns() - start) / 1e9, "s")
