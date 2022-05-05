import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import now from 'performance-now';

const ALPHA = 0.005,
	EPOCHS = 100,
	BATCH_SIZE = 300,
	IMG_X = 28,
	IMG_Y = 28,
	HIDDEN_SIZE = 16,
	NUM_DIGITS = 10;

function max (arr) {
	let maxIdx = -1;
	let maxV = -Infinity;

	for (let i = 0; i < arr.length; i++) {
		if (arr[i] > maxV) {
			maxIdx = i;
			maxV = arr[i];
		}
	}

	return [ maxIdx, maxV ];
}


function getData () {
	let [ X, y ] = JSON.parse(String(fs.readFileSync('../train.json')));
	let [ X_test, y_test ] = JSON.parse(String(fs.readFileSync('../test.json')));

	y = y.map(v => {
		let arr = Array(NUM_DIGITS).fill(-.1);
		arr[v] = 1;
		return arr;
	});

	y_test = y_test.map(v => {
		let arr = Array(NUM_DIGITS).fill(-.1);
		arr[v] = 1;
		return arr;
	});

	y = tf.tensor(y).reshape([X.length, NUM_DIGITS]);
	X = tf.tensor(X, [X.length, IMG_X, IMG_Y]);

	y_test = tf.tensor(y_test).reshape([X_test.length, NUM_DIGITS]);
	X_test = tf.tensor(X_test, [X_test.length, IMG_X, IMG_Y]);

	return [ X, y, X_test, y_test ];
}

async function test (model, X_test, y_test) {
	const n = Math.min(1000, X_test.shape[0]);

	y_test = await y_test.array();
	X_test = await X_test.array();

	let totalError = 0;
	let incorrects = 0;

	for (let i = 0; i < n; i++) {
		const X = tf.tensor(X_test[i]).reshape([1, 28, 28]);
		let y = (await model.predict(X).array())[0];
		for (let j = 0; j < NUM_DIGITS; j++) {
			totalError += (y[j] - y_test[i][j])**2;
		}

		if (max(y)[0] !== max(y_test[i])[0]) {
			incorrects++;
		}
	}

	console.log(`${incorrects}/${n} incorrect`);
	console.log(`${totalError/n} avg error`);
}

async function main () {
	const [X, y, X_test, y_test] = getData();
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
			units: NUM_DIGITS,
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
				if (epoch % 100 === 0 || epoch === EPOCHS) {
					console.log(`Finished epoch ${epoch+1} / ${EPOCHS}: ${logs.loss}`);
				}
			},
		},
		batchSize: BATCH_SIZE,
		verbose: false,
		shuffle: true,
		validationData: [X_test, y_test]
	});

	console.log((times.reduce((a, b) => a + b, 0) / times.length).toFixed(5), 'ms on average');

	await test(model, X_test, y_test);

	await model.save('file:///home/joseph/dev/mnist/tensorflow.js/model');

	console.log('Saved model');
}

main();
