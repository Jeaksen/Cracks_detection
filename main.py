import os
import random

import mahotas
import mahotas.features
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import local_binary_pattern

try:
    from cv2 import cv2

except ImportError:
    pass




def features_for(img_name):
    # img = cv2.imread(img_name)
    # return mahotas.features.haralick(img).mean(0)
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    # return mahotas.features.lbp(img, 1, 8)
    lbp = local_binary_pattern(img, 8, 1, 'ror')
    lbp = np.ndarray.astype(lbp, np.uint8)
    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    return hist


# Returns count random pictures with cracks and count without cracks
def get_pictures(count):
    negative_dir = "archive/Negative/"
    positive_dir = "archive/Positive/"

    positive = random.sample(os.listdir(positive_dir), count)
    negative = random.sample(os.listdir(negative_dir), count)
    positive = list(map(lambda file_name: positive_dir + file_name, positive))
    negative = list(map(lambda file_name: negative_dir + file_name, negative))
    return positive, negative


def extract_features(image_names):
    return list(map(features_for, image_names))


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


if __name__ == '__main__':
    files_count = 1

    print(f'Getting random {files_count} positive and {files_count} negative pictures\n')
    positive_images, negative_images = get_pictures(files_count)
    # print(positive_images)
    # print(negative_images)
    print("Extracting features\n")
    p_features = extract_features(positive_images)
    n_features = extract_features(negative_images)

    print("Creating data frame\n")

    features_df, results = create_dataset(p_features, n_features)

    # print(features_df)

    X_train, X_test, y_train, y_test = train_test_split(features_df, results, test_size=0.2)

    # print("Training:", X_train, "Test:", X_test, sep='\n')
    # print("Training labels:", y_train, "Test labels", y_test, sep='\n')
    print("Test labels", y_test, sep='\n')

    print("Testing\n")
    classifier = RandomForestClassifier(n_estimators=10)
    predictions = classifier.fit(X_train, y_train).predict(X_test)
    print(predictions)
    print(classifier.score(X_test, y_test))
