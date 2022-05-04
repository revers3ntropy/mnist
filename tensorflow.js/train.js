import * as tf from '@tensorflow/tfjs-node';

const model = tf.sequential();

model.add(
	tf.layers.dense({
		inputShape: [50, 50],
		units: 2,
		activation: "tanh",
	})
);
model.add(
	tf.layers.dense({
		units: 2,
		activation: "tanh",
	})
);
model.add(
	tf.layers.dense({
		units: 10,
	})
);

