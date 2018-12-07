import os
import cv2 as cv 
import numpy as np 
import autokeras
from autokeras.image_supervised import ImageClassifier
import csv
import time

def classify():
    #preprocess_data
    x_train_data = []
    y_train_data = []
    x_test_data = []
    name_file_list =[]
    for i in os.listdir(DATA_PATH_TRAIN):
        if 'cat' in i:
            image = cv.imread(os.path.join(DATA_PATH_TRAIN, i))
            resized_img = cv.resize(image,(64, 64))
            x_train_data.append(resized_img)
            y_train_data.append([0])
        elif 'dog' in i:
            image = cv.imread(os.path.join(DATA_PATH_TRAIN, i))
            resized_img = cv.resize(image,(64, 64))
            x_train_data.append(resized_img)
            y_train_data.append([1])

    for i in os.listdir(DATA_PATH_TEST):
        image = cv.imread(os.path.join(DATA_PATH_TEST, i))
        resized_img = cv.resize(image, (64, 64))
        name_file = os.path.splitext(os.path.basename(i)[0])
        x_test_data.append(resized_img)
        name_file_list.append(name_file)
    
    x_train_data = np.array(x_train_data)
    y_train_data = np.array(y_train_data)
    x_test_data = np.array(x_test_data)
    # build_model
    clf = ImageClassifier(verbose=True)
    clf.fit(x_train_data, y_train_data, time_limit=12 * 60 * 60)
    clf.load_searcher().load_best_model().produce_keras_model().save(os.path.join(DIR_NAME, 'util/tmp/model.h5'))
    training_time = (time.time() - start_time) / 60
    # classify_test_data
    start_predict_time = time.time()
    with open(os.path.join(DIR_NAME, 'util/tmp/submission.csv'), 'a') as f:
        field_names = ['id', 'label']
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for num_x, x in x_test_data:
            label = clf.predict(x)
            writer.writerow({'id' : name_file_list[num_x] , 'label': label})
    predict_time = (time.time() - start_predict_time) / 60
    with open(os.path.join(DIR_NAME, 'util/report.txt'), 'a') as txt_file:
        txt_file.writelines(f'Training time: {training_time} min')
        txt_file.writelines(f'Predict time: {predict_time} min')


if __name__ == "__main__":
    start_time = time.time()
    DIR_NAME = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))
    DATA_PATH_TRAIN = os.path.join(DIR_NAME, 'data/train')
    DATA_PATH_TEST = os.path.join(DIR_NAME, 'data/test')
    classify()
    
