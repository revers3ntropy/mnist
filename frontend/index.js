const canvas = document.getElementById("canvas");
const ctx = canvas.getContext('2d');
const result = document.getElementById('result');
const prediction = document.getElementById('prediction');
const imgDiv = document.getElementById('image');

let dragging = false;
const pos = [0, 0];

tf ??= tensorflow;

let model;

canvas.addEventListener('mousedown', () => {
    dragging = true;
});
canvas.addEventListener('mouseup', () => {
    dragging = false;
});

canvas.addEventListener('mousedown', setPosition);
canvas.addEventListener('mousemove', async (e) => {
    e.preventDefault();
    e.stopPropagation();

    if (!dragging) return;

    ctx.beginPath();

    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'rgb(16,190,190)';

    ctx.moveTo(pos.x, pos.y);
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);

    ctx.stroke();
    predictModel();
});

function setPosition (e) {
    pos.x = e.clientX - ctx.canvas.offsetLeft;
    pos.y = e.clientY - ctx.canvas.offsetTop;
}

// clear canvas
function erase () {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictModel();
}

async function loadModel () {
    model = await tf.loadLayersModel('../tensorflow.js/model/model.json');
    // warm up
    model.predict(tf.zeros([1, 28, 28]))
}

function getData () {
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

async function predictModel () {
    const imageData = getData();

    // converts from a canvas data object to a tensor
    let image = tf.browser.fromPixels(imageData)

    // pre-process image
    image = tf.image.resizeBilinear(image, [28,28]).sum(2).expandDims(0);

    let imgDivHTML = '';
    for (let row of (await image.array())[0]) {
        for (let n of row) {
            imgDivHTML += `
                <div style="background: rgb(${n}, ${n}, ${n})"></div>
            `;
        }
    }
    imgDiv.innerHTML = imgDivHTML;

    // gets model prediction
    const y = (await model.predict(image).array())[0];

    prediction.innerHTML = '';

    let max = 0;
    let maxV = -Infinity;

    for (let i = 0; i < 10; i++) {
        if (y[i] > maxV) {
            max = i;
            maxV = y[i];
        }

        if (y[i] < 0) y[i] = 0;

        prediction.innerHTML += `
            <div style="width: 100%; margin: 10px;">
                <div class=bar style="width: ${y[i] * 100}%"> </div>
                <span style="float: left; transform: translate(20px, -20px)">
                    ${i}: ${(y[i] * 100).toFixed(2)}%
                </span>
            </div>
        `;
    }

    result.innerText = max.toString();
}

loadModel();