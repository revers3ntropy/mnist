const canvas = document.getElementById("canvas");
const ctx = canvas.getContext('2d');
const result = document.getElementById('result');

let dragging = false;
const pos = [0, 0];

let model;

canvas.addEventListener('mousedown',  engage);
canvas.addEventListener('mousedown',  setPosition);
canvas.addEventListener('mousemove',  draw);
canvas.addEventListener('mouseup', disengage);

// touch
canvas.addEventListener('touchstart', engage);
canvas.addEventListener('touchmove', setPosition);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', disengage);

function isTouchDevice() {
  return (
        ('ontouchstart' in window) ||
        (navigator.maxTouchPoints > 0) ||
        (navigator['msMaxTouchPoints'] ?? 0 > 0)
    );
}

function engage () {
    dragging = true;
}

function disengage () {
    dragging = false;
}

function setPosition (e) {
    if (isTouchDevice()) {
        const touch = e.touches[0];
        pos.x = touch.clientX - ctx.canvas.offsetLeft;
        pos.y = touch.clientY - ctx.canvas.offsetTop;
    } else {
        pos.x = e.clientX - ctx.canvas.offsetLeft;
        pos.y = e.clientY - ctx.canvas.offsetTop;
    }
}

function draw (e) {
    e.preventDefault();
    e.stopPropagation();

    if (!dragging) return;

    ctx.beginPath();

    ctx.lineWidth = 40;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'red';

    ctx.moveTo(pos.x, pos.y);
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);

    ctx.stroke();
}

// clear canvas
function erase () {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

async function loadModel () {
    model = await tf.loadLayersModel('../tensorflow.js/model');
    // warm up
    model.predict(tf.zeros([1, 28, 28, 1]))
}

function getData(){
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

async function predictModel() {
    const imageData = getData();

    // converts from a canvas data object to a tensor
    let image = tf.browser.fromPixels(imageData)

    // pre-process image
    image = tf.image.resizeBilinear(image, [28,28]).sum(2).expandDims(0).expandDims(-1)

    // gets model prediction
    const y = model.predict(image);

    // replaces the text in the result tag by the model prediction
    result.innerHTML = y.argMax(1).dataSync();
}

loadModel();