from keras.datasets import mnist
import json


def main():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train = [[], list(map(int, train_y))]
    for i in range(train_X.shape[0]):
        train[0].append([list(map(int, row)) for row in train_X[i]])

    with open('train.json', 'w') as f:
        f.write(json.dumps(train))

    test = [[], list(map(int, test_y))]
    for i in range(test_X.shape[0]):
        test[0].append([list(map(int, row)) for row in test_X[i]])

    with open('test.json', 'w') as f:
        f.write(json.dumps(test))


if __name__ == '__main__':
    main()
