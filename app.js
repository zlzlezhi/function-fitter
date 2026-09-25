// ---------- Parametric function library ----------
const FUNCTION_SPECS = {
    polynomial: {
        name: '多项式',
        formula: (p) => `y = ${fmt(p.a)}x³ + ${fmt(p.b)}x² + ${fmt(p.c)}x + ${fmt(p.d)}`,
        params: {
            a: { label: 'x³ 系数', min: -2, max: 2, step: 0.1, value: 0 },
            b: { label: 'x² 系数', min: -2, max: 2, step: 0.1, value: 1 },
            c: { label: 'x 系数',  min: -2, max: 2, step: 0.1, value: 0 },
            d: { label: '常数项',  min: -2, max: 2, step: 0.1, value: 0 },
        },
        fn: (x, p) => p.a * x * x * x + p.b * x * x + p.c * x + p.d,
    },
    sine: {
        name: '正弦波',
        formula: (p) => `y = ${fmt(p.amp)}·sin(${fmt(p.freq)}x + ${fmt(p.phase)}) + ${fmt(p.offset)}`,
        params: {
            amp: { label: '振幅 A', min: 0.1, max: 3, step: 0.1, value: 2 },
            freq: { label: '角频率 ω', min: 0.2, max: 4, step: 0.1, value: 1.5 },
            phase: { label: '相位 φ', min: -3.14, max: 3.14, step: 0.1, value: 0 },
            offset: { label: '偏移 b', min: -2, max: 2, step: 0.1, value: 0 },
        },
        fn: (x, p) => p.amp * Math.sin(p.freq * x + p.phase) + p.offset,
    },
    tanh: {
        name: '双曲正切',
        formula: (p) => `y = ${fmt(p.A)}·tanh(${fmt(p.ω)}x) + ${fmt(p.b)}`,
        params: {
            A: { label: '振幅 A', min: 0.1, max: 3, step: 0.1, value: 1.5 },
            ω: { label: '宽度 ω', min: 0.2, max: 4, step: 0.1, value: 1 },
            b: { label: '偏移 b', min: -2, max: 2, step: 0.1, value: 0 },
        },
        fn: (x, p) => p.A * Math.tanh(p.ω * x) + p.b,
    },
    abs: {
        name: '绝对值',
        formula: (p) => `y = ${fmt(p.A)}·|x - ${fmt(p.c)}| + ${fmt(p.b)}`,
        params: {
            A: { label: '斜率 A', min: 0.1, max: 3, step: 0.1, value: 1 },
            c: { label: '中心 c', min: -2, max: 2, step: 0.1, value: 0 },
            b: { label: '偏移 b', min: -2, max: 2, step: 0.1, value: 0 },
        },
        fn: (x, p) => p.A * Math.abs(x - p.c) + p.b,
    },
    cube: {
        name: '三次函数',
        formula: (p) => `y = ${fmt(p.a)}x³`,
        params: {
            a: { label: '系数 a', min: 0.1, max: 2, step: 0.1, value: 1 },
        },
        fn: (x, p) => p.a * x * x * x,
    },
    gaussian: {
        name: '高斯',
        formula: (p) => `y = ${fmt(p.A)}·exp(-(x-${fmt(p.μ)})²/(2·${fmt(p.σ)}²))`,
        params: {
            A: { label: '幅度 A', min: 0.1, max: 3, step: 0.1, value: 2 },
            μ: { label: '中心 μ', min: -2, max: 2, step: 0.1, value: 0 },
            σ: { label: '宽度 σ', min: 0.2, max: 2, step: 0.05, value: 0.8 },
        },
        fn: (x, p) => p.A * Math.exp(-((x - p.μ) ** 2) / (2 * p.σ ** 2)),
    },
    combo: {
        name: '混合波',
        formula: (p) => `y = ${fmt(p.A)}·sin(${fmt(p.ω1)}x) + ${fmt(p.B)}·cos(${fmt(p.ω2)}x)`,
        params: {
            A:  { label: 'sin 振幅 A', min: -3, max: 3, step: 0.1, value: 1.2 },
            ω1: { label: 'sin 频率 ω₁', min: 0.2, max: 4, step: 0.1, value: 1.5 },
            B:  { label: 'cos 振幅 B', min: -3, max: 3, step: 0.1, value: 1 },
            ω2: { label: 'cos 频率 ω₂', min: 0.2, max: 4, step: 0.1, value: 1 },
        },
        fn: (x, p) => p.A * Math.sin(p.ω1 * x) + p.B * Math.cos(p.ω2 * x),
    },
};

function fmt(v) {
    const s = v.toFixed(2);
    return s.replace('-', '−');
}

const DOMAIN = { min: -3, max: 3 };
const N_SAMPLES = 150;

const ACTIVATIONS = {
    tanh: {
        name: 'Tanh',
        fn: z => Math.tanh(z),
        deriv: a => 1 - a * a,
    },
    relu: {
        name: 'ReLU',
        fn: z => Math.max(0, z),
        deriv: a => a > 0 ? 1 : 0,
    },
    leaky_relu: {
        name: 'Leaky ReLU',
        fn: z => z > 0 ? z : 0.1 * z,
        deriv: a => a > 0 ? 1 : 0.1,
    },
    sigmoid: {
        name: 'Sigmoid',
        fn: z => 1 / (1 + Math.exp(-z)),
        deriv: a => a * (1 - a),
    },
};

// ---------- DOM elements ----------
const fnTypeSelect   = document.getElementById('function-type');
const paramsContainer = document.getElementById('params-container');
const formulaEl      = document.getElementById('formula');

const hiddenInput    = document.getElementById('hidden-size');
const layersInput    = document.getElementById('hidden-layers');
const activationInput = document.getElementById('activation');
const lrInput        = document.getElementById('learning-rate');
const epochsInput    = document.getElementById('epochs');

const hiddenVal      = document.getElementById('hidden-val');
const layersVal      = document.getElementById('layers-val');
const lrVal          = document.getElementById('lr-val');
const epochsVal      = document.getElementById('epochs-val');
const diagH1         = document.getElementById('diag-h1');
const diagH2         = document.getElementById('diag-h2');
const diagH3         = document.getElementById('diag-h3');
const diagArrow2     = document.getElementById('diag-arrow2');
const diagArrow3     = document.getElementById('diag-arrow3');
const diagArrow4     = document.getElementById('diag-arrow4');

const trainBtn       = document.getElementById('train-btn');
const resetBtn       = document.getElementById('reset-btn');
const themeToggle    = document.getElementById('theme-toggle');
const epochDisplay   = document.getElementById('epoch-display');
const epochTotal     = document.getElementById('epoch-total');
const lossDisplay    = document.getElementById('loss-display');

const fitCanvas      = document.getElementById('fit-canvas');
const fitCtx         = fitCanvas.getContext('2d');
const lossCanvas     = document.getElementById('loss-canvas');
const lossCtx        = lossCanvas.getContext('2d');

// ---------- Hi-DPI canvas setup ----------
const LOGICAL_FIT = { width: 900, height: 380 };
const LOGICAL_LOSS = { width: 900, height: 150 };

function resizeCanvas(canvas, ctx, logical) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.floor(logical.width * dpr);
    const height = Math.floor(logical.height * dpr);
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
    }
    return { width: logical.width, height: logical.height };
}

function resizeAllCanvases() {
    fitSize = resizeCanvas(fitCanvas, fitCtx, LOGICAL_FIT);
    lossSize = resizeCanvas(lossCanvas, lossCtx, LOGICAL_LOSS);
}

let fitSize = { ...LOGICAL_FIT };
let lossSize = { ...LOGICAL_LOSS };

function getCssVar(name, fallback) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

function clearCanvasPhysically(canvas, ctx) {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = getCssVar('--panel-2', '#23232d');
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.restore();
}

// ---------- State ----------
let currentParams = {};
let net = null;
let trainingData = [];
let yMean = 0;
let yStd = 1;
let lossHistory = [];
let isTraining = false;
let stopTraining = false;
let animationId = null;

// Cached visuals for the current target function.
let cachedTargetCurve = null;
let cachedTargetRange = null;
let referenceYRange = null;

// ---------- Utilities ----------
function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function xavierInit(inSize, outSize) {
    const scale = Math.sqrt(2 / (inSize + outSize));
    return () => randn() * scale;
}

function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

// ---------- MLP ----------
class MLP {
    constructor(layerSizes, activation = 'tanh') {
        this.activation = ACTIVATIONS[activation];
        this.layers = [];
        for (let i = 0; i < layerSizes.length - 1; i++) {
            const inSize = layerSizes[i];
            const outSize = layerSizes[i + 1];
            const init = xavierInit(inSize, outSize);
            this.layers.push({
                W: Array.from({ length: outSize }, () => Array.from({ length: inSize }, init)),
                b: new Float64Array(outSize).fill(0),
            });
        }
    }

    activate(z) {
        const out = new Float64Array(z.length);
        for (let i = 0; i < z.length; i++) out[i] = this.activation.fn(z[i]);
        return out;
    }

    forward(x) {
        let a = x;
        const cache = [{ a }];
        for (let l = 0; l < this.layers.length; l++) {
            const layer = this.layers[l];
            const outSize = layer.b.length;
            const inSize = a.length;
            const z = new Float64Array(outSize);
            for (let i = 0; i < outSize; i++) {
                let sum = layer.b[i];
                const row = layer.W[i];
                for (let j = 0; j < inSize; j++) sum += row[j] * a[j];
                z[i] = sum;
            }
            const isLast = l === this.layers.length - 1;
            const nextA = isLast ? z : this.activate(z);
            a = nextA;
            cache.push({ z, a });
        }
        return { output: a, cache };
    }

    backward(target, cache) {
        const L = this.layers.length;
        const grads = this.layers.map(layer => ({
            W: layer.W.map(row => new Float64Array(row.length).fill(0)),
            b: new Float64Array(layer.b.length).fill(0),
        }));

        let delta = new Float64Array(target.length);
        const pred = cache[L].a;
        for (let i = 0; i < target.length; i++) delta[i] = pred[i] - target[i];

        for (let l = L - 1; l >= 0; l--) {
            const layer = this.layers[l];
            const prevA = cache[l].a;

            for (let i = 0; i < layer.b.length; i++) {
                grads[l].b[i] = delta[i];
                for (let j = 0; j < prevA.length; j++) {
                    grads[l].W[i][j] = delta[i] * prevA[j];
                }
            }

            if (l > 0) {
                const prevDelta = new Float64Array(prevA.length);
                for (let j = 0; j < prevA.length; j++) {
                    let sum = 0;
                    for (let i = 0; i < layer.b.length; i++) sum += layer.W[i][j] * delta[i];
                    prevDelta[j] = sum * this.activation.deriv(prevA[j]);
                }
                delta = prevDelta;
            }
        }
        return grads;
    }

    update(grads, lr, wd = 0) {
        for (let l = 0; l < this.layers.length; l++) {
            const layer = this.layers[l];
            const g = grads[l];
            for (let i = 0; i < layer.b.length; i++) {
                layer.b[i] -= lr * g.b[i];
                for (let j = 0; j < layer.W[i].length; j++) {
                    layer.W[i][j] -= lr * (g.W[i][j] + wd * layer.W[i][j]);
                }
            }
        }
    }
}

// ---------- Function & data ----------
function currentSpec() {
    return FUNCTION_SPECS[fnTypeSelect.value];
}

function generateData() {
    const spec = currentSpec();
    const raw = [];
    const step = (DOMAIN.max - DOMAIN.min) / (N_SAMPLES - 1);
    for (let i = 0; i < N_SAMPLES; i++) {
        const x = DOMAIN.min + i * step;
        raw.push({ x, y: spec.fn(x, currentParams) });
    }

    let sum = 0;
    for (const r of raw) sum += r.y;
    yMean = sum / raw.length;

    let sq = 0;
    for (const r of raw) sq += (r.y - yMean) ** 2;
    yStd = Math.sqrt(sq / raw.length) || 1;

    return raw.map(r => ({ x: r.x, y: r.y, yNorm: (r.y - yMean) / yStd }));
}

function buildParamsControls() {
    const spec = currentSpec();
    currentParams = {};
    paramsContainer.innerHTML = '';

    for (const [key, cfg] of Object.entries(spec.params)) {
        currentParams[key] = cfg.value;

        const row = document.createElement('div');
        row.className = 'control-row';

        const label = document.createElement('label');
        label.innerHTML = `${cfg.label} <span id="param-${key}-val">${cfg.value}</span>`;

        const input = document.createElement('input');
        input.type = 'range';
        input.min = cfg.min;
        input.max = cfg.max;
        input.step = cfg.step;
        input.value = cfg.value;
        input.dataset.key = key;

        input.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            currentParams[key] = val;
            document.getElementById(`param-${key}-val`).textContent = val.toFixed(2);
            formulaEl.textContent = spec.formula(currentParams);
            if (!isTraining) {
                trainingData = generateData();
                invalidateTargetCache();
                drawAll();
            }
        });

        row.appendChild(label);
        row.appendChild(input);
        paramsContainer.appendChild(row);
    }

    formulaEl.textContent = spec.formula(currentParams);
}

function invalidateTargetCache() {
    cachedTargetCurve = null;
    // referenceYRange is intentionally kept so parameter changes remain visible.
}

function getTargetCurve(resolution = 150) {
    if (cachedTargetCurve && cachedTargetCurve.length === resolution) {
        return cachedTargetCurve;
    }
    const spec = currentSpec();
    const curve = [];
    const step = (DOMAIN.max - DOMAIN.min) / (resolution - 1);
    for (let i = 0; i < resolution; i++) {
        const x = DOMAIN.min + i * step;
        curve.push({ x, y: spec.fn(x, currentParams) });
    }
    cachedTargetCurve = curve;
    return curve;
}

function getTargetYRange() {
    if (!referenceYRange) {
        const curve = getTargetCurve(150);
        referenceYRange = computeRange(curve.map(p => p.y), 0.25);
    }
    // Ensure the current curve fits by expanding the envelope if needed.
    const currentCurve = getTargetCurve(150);
    const currentRange = computeRange(currentCurve.map(p => p.y), 0.12);
    let expanded = false;
    if (currentRange.min < referenceYRange.min) {
        referenceYRange = { ...referenceYRange, min: currentRange.min };
        expanded = true;
    }
    if (currentRange.max > referenceYRange.max) {
        referenceYRange = { ...referenceYRange, max: currentRange.max };
        expanded = true;
    }
    if (expanded) {
        // Add a little extra padding when expanding to reduce future expansions.
        const pad = (referenceYRange.max - referenceYRange.min) * 0.05;
        referenceYRange = {
            min: referenceYRange.min - pad,
            max: referenceYRange.max + pad,
        };
    }
    return referenceYRange;
}

function getPredictionCurve(resolution = 150) {
    if (!net) return [];
    const curve = [];
    const step = (DOMAIN.max - DOMAIN.min) / (resolution - 1);
    for (let i = 0; i < resolution; i++) {
        const x = DOMAIN.min + i * step;
        const { output } = net.forward(new Float64Array([x]));
        curve.push({ x, y: output[0] * yStd + yMean });
    }
    return curve;
}

// ---------- Training ----------
function getLayerSizes() {
    const hidden = parseInt(hiddenInput.value, 10);
    const nLayers = parseInt(layersInput.value, 10);
    const sizes = [1];
    for (let i = 0; i < nLayers; i++) sizes.push(hidden);
    sizes.push(1);
    return sizes;
}

function makeNetwork() {
    net = new MLP(getLayerSizes(), activationInput.value);
}

function reset() {
    stopTraining = true;
    if (animationId) cancelAnimationFrame(animationId);
    isTraining = false;
    lossHistory = [];
    epochDisplay.textContent = '0';
    lossDisplay.textContent = '—';
    cachedTargetCurve = null;
    referenceYRange = null;
    makeNetwork();
    updateUIState();
    drawAll();
}

async function train() {
    if (isTraining) return;
    isTraining = true;
    stopTraining = false;
    updateUIState();

    try {
        const lr = parseFloat(lrInput.value);
        const totalEpochs = parseInt(epochsInput.value, 10);
        const batchSize = 32;
        const weightDecay = 0.0005;

        lossHistory = [];

        for (let epoch = 1; epoch <= totalEpochs && !stopTraining; epoch++) {
            shuffle(trainingData);
            let epochLoss = 0;

            for (let i = 0; i < trainingData.length; i += batchSize) {
                const batch = trainingData.slice(i, i + batchSize);
                let batchLoss = 0;
                const acc = net.layers.map(layer => ({
                    W: layer.W.map(row => new Float64Array(row.length).fill(0)),
                    b: new Float64Array(layer.b.length).fill(0),
                }));

                for (const sample of batch) {
                    const x = new Float64Array([sample.x]);
                    const y = new Float64Array([sample.yNorm]);
                    const { output, cache } = net.forward(x);
                    const err = output[0] - y[0];
                    batchLoss += err * err;

                    const grads = net.backward(y, cache);
                    for (let l = 0; l < net.layers.length; l++) {
                        for (let j = 0; j < net.layers[l].b.length; j++) {
                            acc[l].b[j] += grads[l].b[j];
                            for (let k = 0; k < net.layers[l].W[j].length; k++) {
                                acc[l].W[j][k] += grads[l].W[j][k];
                            }
                        }
                    }
                }

                const n = batch.length;
                for (let l = 0; l < net.layers.length; l++) {
                    for (let j = 0; j < net.layers[l].b.length; j++) {
                        acc[l].b[j] /= n;
                        for (let k = 0; k < net.layers[l].W[j].length; k++) {
                            acc[l].W[j][k] /= n;
                        }
                    }
                }
                net.update(acc, lr, weightDecay);
                epochLoss += batchLoss / n;
            }

            const avgLoss = epochLoss / Math.ceil(trainingData.length / batchSize);
            lossHistory.push(avgLoss * yStd * yStd);

            epochDisplay.textContent = epoch;
            lossDisplay.textContent = (avgLoss * yStd * yStd).toExponential(4);

            // Lower resolution during training for speed; full resolution at the end.
            try { drawAll(epoch < totalEpochs ? 80 : 150); } catch (e) { console.error('drawAll error:', e); }
            await new Promise(r => { animationId = requestAnimationFrame(r); });
        }
    } catch (err) {
        console.error('Training error:', err);
        lossDisplay.textContent = '错误';
    } finally {
        isTraining = false;
        updateUIState();
        try { drawAll(150); } catch (e) { console.error('final draw error:', e); }
    }
}

// ---------- Plotting ----------
function computeRange(values, pad = 0.1) {
    let min = Infinity, max = -Infinity;
    for (const v of values) {
        if (v < min) min = v;
        if (v > max) max = v;
    }
    if (!isFinite(min) || !isFinite(max)) return { min: -1, max: 1 };
    const margin = (max - min) * pad || 0.5;
    return { min: min - margin, max: max + margin };
}

function mapX(x, width, padding = 50) {
    const p = padding;
    return p + (x - DOMAIN.min) / (DOMAIN.max - DOMAIN.min) * (width - 2 * p);
}

function mapY(y, height, yRange, padding = 40) {
    const p = padding;
    const span = yRange.max - yRange.min || 1;
    return height - p - (y - yRange.min) / span * (height - 2 * p);
}

function drawAxes(ctx, width, height, yRange) {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const axisColor = isLight ? 'rgba(0,0,0,0.15)' : 'rgba(255,255,255,0.12)';
    const zeroColor = isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.06)';
    const padX = 50;
    const padY = 40;
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1;

    // X axis
    ctx.beginPath();
    ctx.moveTo(padX, height - padY);
    ctx.lineTo(width - padX, height - padY);
    ctx.stroke();

    // Y axis
    ctx.beginPath();
    ctx.moveTo(padX, padY);
    ctx.lineTo(padX, height - padY);
    ctx.stroke();

    // Zero line and label
    if (yRange.min <= 0 && yRange.max >= 0) {
        const y0 = mapY(0, height, yRange);
        ctx.strokeStyle = zeroColor;
        ctx.beginPath();
        ctx.moveTo(padX, y0);
        ctx.lineTo(width - padX, y0);
        ctx.stroke();
    }

    ctx.fillStyle = getCssVar('--text-dim', '#9ca3af');
    ctx.font = '12px sans-serif';

    // X labels
    ctx.textAlign = 'center';
    ctx.fillText(DOMAIN.min.toFixed(1), padX, height - 12);
    ctx.fillText(DOMAIN.max.toFixed(1), width - padX, height - 12);

    // Y labels (top, middle, bottom)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yTicks = [yRange.max, 0, yRange.min].filter((v, i, a) => {
        // Keep 0 only if it is not too close to min or max
        if (v === 0) {
            const span = yRange.max - yRange.min;
            return Math.abs(v - yRange.min) > span * 0.08 && Math.abs(yRange.max - v) > span * 0.08;
        }
        return true;
    });
    for (const yVal of yTicks) {
        const y = mapY(yVal, height, yRange);
        // small tick mark
        ctx.strokeStyle = axisColor;
        ctx.beginPath();
        ctx.moveTo(padX - 4, y);
        ctx.lineTo(padX, y);
        ctx.stroke();
        // label
        ctx.fillText(yVal.toFixed(1), padX - 8, y);
    }
    ctx.textBaseline = 'alphabetic';
}

function drawFit(predResolution = 150) {
    const width = fitSize.width;
    const height = fitSize.height;
    clearCanvasPhysically(fitCanvas, fitCtx);

    const targetCurve = getTargetCurve(150);
    const predCurve = net ? getPredictionCurve(predResolution) : [];
    const yRange = getTargetYRange();

    drawAxes(fitCtx, width, height, yRange);

    fitCtx.strokeStyle = getCssVar('--accent', '#22d3ee');
    fitCtx.lineWidth = 3;
    fitCtx.lineJoin = 'round';
    fitCtx.lineCap = 'round';
    fitCtx.beginPath();
    for (let i = 0; i < targetCurve.length; i++) {
        const x = mapX(targetCurve[i].x, width);
        const y = mapY(targetCurve[i].y, height, yRange);
        if (i === 0) fitCtx.moveTo(x, y);
        else fitCtx.lineTo(x, y);
    }
    fitCtx.stroke();

    if (predCurve.length) {
        fitCtx.strokeStyle = getCssVar('--accent-warm', '#f59e0b');
        fitCtx.lineWidth = 2.5;
        fitCtx.beginPath();
        for (let i = 0; i < predCurve.length; i++) {
            const x = mapX(predCurve[i].x, width);
            const y = mapY(predCurve[i].y, height, yRange);
            if (i === 0) fitCtx.moveTo(x, y);
            else fitCtx.lineTo(x, y);
        }
        fitCtx.stroke();
    }

    fitCtx.fillStyle = document.documentElement.getAttribute('data-theme') === 'light'
        ? 'rgba(0,0,0,0.15)'
        : 'rgba(255,255,255,0.22)';
    for (let i = 0; i < trainingData.length; i += 8) {
        const x = mapX(trainingData[i].x, width);
        const y = mapY(trainingData[i].y, height, yRange);
        fitCtx.beginPath();
        fitCtx.arc(x, y, 3, 0, Math.PI * 2);
        fitCtx.fill();
    }
}

function drawLoss() {
    const width = lossSize.width;
    const height = lossSize.height;
    clearCanvasPhysically(lossCanvas, lossCtx);

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    lossCtx.strokeStyle = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.05)';
    lossCtx.lineWidth = 1;
    lossCtx.strokeRect(0, 0, width, height);

    if (lossHistory.length === 0) {
        lossCtx.fillStyle = getCssVar('--text-dim', '#71717a');
        lossCtx.font = '14px sans-serif';
        lossCtx.textAlign = 'center';
        lossCtx.fillText('训练损失将在此显示', width / 2, height / 2 + 5);
        return;
    }

    const padX = 50;
    const padY = 25;

    const logLoss = lossHistory.map(v => Math.log10(Math.max(v, 1e-10)));
    const yRange = computeRange(logLoss, 0.1);

    lossCtx.strokeStyle = isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)';
    lossCtx.lineWidth = 1;
    lossCtx.beginPath();
    lossCtx.moveTo(padX, height - padY);
    lossCtx.lineTo(width - padX, height - padY);
    lossCtx.stroke();

    lossCtx.beginPath();
    lossCtx.moveTo(padX, padY);
    lossCtx.lineTo(padX, height - padY);
    lossCtx.stroke();

    lossCtx.strokeStyle = getCssVar('--accent-hot', '#f472b6');
    lossCtx.lineWidth = 2;
    lossCtx.beginPath();
    const denom = Math.max(lossHistory.length - 1, 1);
    for (let i = 0; i < lossHistory.length; i++) {
        const x = padX + (i / denom) * (width - 2 * padX);
        const y = mapY(logLoss[i], height, yRange, padY);
        if (i === 0) lossCtx.moveTo(x, y);
        else lossCtx.lineTo(x, y);
    }
    lossCtx.stroke();

    lossCtx.fillStyle = getCssVar('--text-dim', '#9ca3af');
    lossCtx.font = '11px sans-serif';
    lossCtx.textAlign = 'right';
    lossCtx.fillText(Math.pow(10, yRange.max).toExponential(1), padX - 6, padY + 4);
    lossCtx.fillText(Math.pow(10, yRange.min).toExponential(1), padX - 6, height - padY - 2);
    lossCtx.textAlign = 'center';
    lossCtx.fillText('0', padX, height - 8);
    lossCtx.fillText(String(lossHistory.length), width - padX, height - 8);
}

function drawAll(predResolution = 150) {
    resizeAllCanvases();
    drawFit(predResolution);
    drawLoss();
}

// ---------- UI ----------
function updateNetworkDiagram() {
    const hidden = parseInt(hiddenInput.value, 10);
    const nLayers = parseInt(layersInput.value, 10);

    diagH1.textContent = hidden;
    diagH2.textContent = hidden;
    diagH3.textContent = hidden;

    if (nLayers === 1) {
        diagH2.style.visibility = 'hidden';
        diagH3.style.visibility = 'hidden';
        diagArrow2.style.visibility = 'hidden';
        diagArrow3.style.visibility = 'hidden';
        diagArrow4.style.visibility = 'hidden';
    } else if (nLayers === 2) {
        diagH2.style.visibility = 'visible';
        diagH3.style.visibility = 'hidden';
        diagArrow2.style.visibility = 'visible';
        diagArrow3.style.visibility = 'hidden';
        diagArrow4.style.visibility = 'hidden';
    } else {
        diagH2.style.visibility = 'visible';
        diagH3.style.visibility = 'visible';
        diagArrow2.style.visibility = 'visible';
        diagArrow3.style.visibility = 'visible';
        diagArrow4.style.visibility = 'visible';
    }
}

function updateUIState() {
    trainBtn.disabled = isTraining;
    fnTypeSelect.disabled = isTraining;
    hiddenInput.disabled = isTraining;
    layersInput.disabled = isTraining;
    activationInput.disabled = isTraining;
    lrInput.disabled = isTraining;
    epochsInput.disabled = isTraining;

    const inputs = paramsContainer.querySelectorAll('input');
    inputs.forEach(input => input.disabled = isTraining);

    trainBtn.textContent = isTraining ? '训练中…' : '开始训练';
    epochTotal.textContent = epochsInput.value;
}

// ---------- Theme ----------
function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    themeToggle.title = theme === 'light' ? '切换到夜间模式' : '切换到日间模式';
    try { localStorage.setItem('function-fitter-theme', theme); } catch (e) {}
    invalidateTargetCache();
    drawAll();
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    applyTheme(current === 'light' ? 'dark' : 'light');
}

function initTheme() {
    let theme = 'light';
    try {
        const saved = localStorage.getItem('function-fitter-theme');
        if (saved) theme = saved;
    } catch (e) {}
    applyTheme(theme);
}

function onNetParamsChange() {
    hiddenVal.textContent = hiddenInput.value;
    layersVal.textContent = layersInput.value;
    lrVal.textContent = lrInput.value;
    epochsVal.textContent = epochsInput.value;
    updateNetworkDiagram();
    if (!isTraining) {
        makeNetwork();
        drawAll();
    }
}

function onFunctionTypeChange() {
    buildParamsControls();
    trainingData = generateData();
    cachedTargetCurve = null;
    referenceYRange = null; // recompute reference range for new function
    lossHistory = [];
    epochDisplay.textContent = '0';
    lossDisplay.textContent = '—';
    if (!isTraining) {
        makeNetwork();
        drawAll();
    }
}

// ---------- Init ----------
for (const [key, spec] of Object.entries(FUNCTION_SPECS)) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = spec.name;
    fnTypeSelect.appendChild(opt);
}

fnTypeSelect.addEventListener('change', onFunctionTypeChange);
hiddenInput.addEventListener('input', onNetParamsChange);
layersInput.addEventListener('input', onNetParamsChange);
activationInput.addEventListener('change', onNetParamsChange);
lrInput.addEventListener('input', onNetParamsChange);
epochsInput.addEventListener('input', onNetParamsChange);
trainBtn.addEventListener('click', train);
resetBtn.addEventListener('click', reset);
themeToggle.addEventListener('click', toggleTheme);

let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(drawAll, 100);
});

initTheme();
onFunctionTypeChange();
