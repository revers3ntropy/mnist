from keras.datasets import mnist
import json
import os


def install_depends():
    print('> Installing Python dependencies')
    with open('requirements.txt') as req:
        for r in req:
            print(f'> Installing {r}')
            os.system("pip3 install " + str(r))

    print('> Installing Node dependencies')
    os.system("cd tensorflow.js && npm i")


def load_mnist():
    print('> Loading NMIST data')
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train = [[], list(map(int, train_y))]
    for i in range(train_X.shape[0]):
        train[0].append([list(map(int, row)) for row in train_X[i]])

    print('> Writing training data to train.json')
    with open('train.json', 'w') as f:
        f.write(json.dumps(train))

    test = [[], list(map(int, test_y))]
    for i in range(test_X.shape[0]):
        test[0].append([list(map(int, row)) for row in test_X[i]])

    print('> Writing test data to test.json')
    with open('test.json', 'w') as f:
        f.write(json.dumps(test))


if __name__ == '__main__':
    install_depends()
    load_mnist()
    print('> Finished')
