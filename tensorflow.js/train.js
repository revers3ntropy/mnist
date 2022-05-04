import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';

const ALPHA = 0.001;

async function main () {
	let [X, y] = JSON.parse(String(fs.readFileSync('../train.json')));

	y = y.map(v => {
		let arr = Array(10).fill(0);
		arr[v] = 1;
		return tf.tensor(arr, [10]);
	});

	X = X.map(v => {
		return tf.tensor(v, [28, 28]);
	});

	const HIDDEN_SIZE = 4
	const model = tf.sequential();

	model.add(
		tf.layers.dense({
			inputShape: [28, 28],
			units: HIDDEN_SIZE,
			activation: "tanh",
		})
	);
	model.add(
		tf.layers.dense({
			units: HIDDEN_SIZE,
			activation: "tanh",
		})
	);
	model.add(tf.layers.flatten());
	model.add(
		tf.layers.dense({
			units: 10,
		})
	);

	model.compile({
		optimizer: tf.train.sgd(ALPHA),
		loss: "meanSquaredError",
	});

	await model.fit(X, y, {
		epochs: 50,
		callbacks: {
			onEpochEnd: async (epoch, logs) => {
				if (epoch % 10 === 0) {
					console.log(`Epoch ${epoch}: error: ${logs?.loss}`)
				}
			},
		}
	});
}

main();
