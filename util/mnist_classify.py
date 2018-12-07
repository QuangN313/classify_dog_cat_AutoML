from keras.datasets import mnist
import autokeras
import os

if __name__ == '__main__':
    DIR_NAME = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    clf = autokeras.ImageClassifier(verbose=True)
    clf.fit(x_train, y_train, time_limit=30 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    clf.load_searcher().load_best_model().produce_keras_model().save(os.join.path(DIR_NAME, 'util/tmp/model.h5'))
    y = clf.evaluate(x_test, y_test)
    with open(os.path.join(DIR_NAME, 'util/report.txt'), 'a') as f:
        f.write(y)