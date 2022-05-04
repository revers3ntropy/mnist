import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import now from 'performance-now';

const ALPHA = 0.001,
	EPOCHS = 100,
	BATCH_SIZE = 350,
	IMG_X = 28,
	IMG_Y = 28;

function getData () {
	let [X, y] = JSON.parse(String(fs.readFileSync('../train.json')));
	let [X_test, y_test] = JSON.parse(String(fs.readFileSync('../test.json')));

	y = y.map(v => {
		let arr = Array(10).fill(0);
		arr[v] = 1;
		return arr;
	});

	y_test = y_test.map(v => {
		let arr = Array(10).fill(0);
		arr[v] = 1;
		return arr;
	});

	y = tf.tensor(y).reshape([X.length, 10]);
	X = tf.tensor(X, [X.length, IMG_X, IMG_Y]);

	y_test = tf.tensor(y_test).reshape([X_test.length, 10]);
	X_test = tf.tensor(X_test, [X_test.length, IMG_X, IMG_Y]);

	return [X, y, X_test, y_test];
}

async function main () {
	const [X, y, X_test, y_test] = getData();

	const HIDDEN_SIZE = 4
	const model = tf.sequential();

	model.add(
		tf.layers.dense({
			inputShape: [IMG_X, IMG_Y],
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

	let losses = [];
	let times = [];

	let start = now();

	await model.fit(X, y, {
		epochs: EPOCHS,
		callbacks: {
			onEpochEnd: async (epoch, logs) => {
				losses.push(logs.loss);
				times.push(now() - start);
				start = now();
				console.log(`Finished epoch ${epoch+1} / ${EPOCHS}: ${logs.loss}`);
			},
		},
		batchSize: BATCH_SIZE,
		verbose: false,
		shuffle: true,
		validationData: [X_test, y_test]
	});

	console.log(times.reduce((a, b) => a + b, 0) / times.length, 'ms on average');
}

main();
