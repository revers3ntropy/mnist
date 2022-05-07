import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflowjs as tfjs

ALPHA = 0.001
EPOCHS = 2


def test(model, X_test, y_test):
    n = 100

    totalError = 0
    incorrects = 0

    for i in range(n):
        X = tf.constant(X_test[i]).set_shape([1, 28, 28])
        y = list(model.predict(X))[0]
        for j in range(10):
            totalError += (y[j] - y_test[i][j]) ** 2

        if max(y)[0] != max(y_test[i])[0]:
            incorrects += 1

    print(f'{incorrects}/{n} incorrect')
    print(f'{totalError / n} avg error')


def main():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(ALPHA),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_test,
        verbose=False
    )

    model.save('file:///home/joseph/dev/mnist/tensorflow/py-model')
    tfjs.converters.save_keras_model(model, 'model')


if __name__ == '__main__':
    main()
