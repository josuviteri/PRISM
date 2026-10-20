const checkpoint = document.querySelector('#checkpoint');
const methods = [...document.querySelectorAll('input[name="method"]')];
const steps = document.querySelector('#steps');
const samples = document.querySelector('#samples');
const seed = document.querySelector('#seed');
const status = document.querySelector('#status');
const generate = document.querySelector('#generate');
let models = [];
let trajectory = [];
let playback;

const selectedModel = () => models.find(model => model.id === checkpoint.value);
const selectedMethod = () => methods.find(method => method.checked).value;

function setRangeFill(input) {
  const fill = ((input.value - input.min) / (input.max - input.min)) * 100;
  input.style.setProperty('--range', `${fill}%`);
}

function setStatus(message, state = '') {
  status.className = `status ${state}`;
  status.querySelector('p').textContent = message;
}

function updateControls() {
  const model = selectedModel();
  if (!model) return;
  const ddim = selectedMethod() === 'ddim';
  document.querySelector('#model-type').textContent = model.dataset.toUpperCase();
  document.querySelector('#model-details').textContent = `${model.dataset.toUpperCase()} · trained with ${model.timesteps.toLocaleString()} diffusion steps`;
  steps.max = model.timesteps;
  if (+steps.value > model.timesteps) steps.value = model.timesteps;
  steps.disabled = !ddim;
  document.querySelector('#steps-field').classList.toggle('is-disabled', !ddim);
  document.querySelector('#steps-description').textContent = ddim
    ? 'Fewer steps generate faster; more preserve the learned trajectory.'
    : `DDPM follows all ${model.timesteps.toLocaleString()} training steps.`;
  document.querySelector('#steps-output').textContent = ddim ? steps.value : model.timesteps;
  setRangeFill(steps);
}

function prepareDownload(anchor, source) {
  anchor.href = source;
}

function showFrame(index) {
  const frame = trajectory[index];
  if (!frame) return;
  const slider = document.querySelector('#trajectory-slider');
  slider.value = index;
  setRangeFill(slider);
  document.querySelector('#trajectory-image').src = frame.image;
  document.querySelector('#frame-label').textContent = `Reverse step ${frame.iteration} · t = ${frame.timestep}`;
}

function stopPlayback() {
  window.clearInterval(playback);
  playback = undefined;
  document.querySelector('#play-trajectory').textContent = 'Play trajectory';
}

async function initialise() {
  const response = await fetch('/api/models');
  if (!response.ok) throw new Error('Could not read checkpoint metadata.');
  models = await response.json();
  if (!models.length) throw new Error('No compatible checkpoints were found.');
  for (const model of models) checkpoint.add(new Option(`${model.id} · ${model.dataset.toUpperCase()}`, model.id));
  updateControls();
}

checkpoint.addEventListener('change', updateControls);
methods.forEach(method => method.addEventListener('change', updateControls));
steps.addEventListener('input', () => { document.querySelector('#steps-output').textContent = steps.value; setRangeFill(steps); });
samples.addEventListener('input', () => { document.querySelector('#samples-output').textContent = samples.value; setRangeFill(samples); });
document.querySelector('#randomise').addEventListener('click', () => { seed.value = Math.floor(Math.random() * 2 ** 31); seed.focus(); });

document.querySelector('#form').addEventListener('submit', async event => {
  event.preventDefault();
  const method = selectedMethod();
  generate.disabled = true;
  setStatus(method === 'ddpm' ? 'Running the complete DDPM schedule. This can take a while…' : 'Traversing the DDIM denoising trajectory…', 'loading');
  try {
    const response = await fetch('/api/generate', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({checkpoint: checkpoint.value, method, steps: +steps.value, samples: +samples.value, seed: +seed.value}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Sampling failed.');
    document.querySelector('#noise').src = data.noise;
    document.querySelector('#output').src = data.samples;
    prepareDownload(document.querySelector('#download-noise'), data.noise);
    prepareDownload(document.querySelector('#download-output'), data.samples);
    document.querySelector('#empty').hidden = true;
    document.querySelector('#results').hidden = false;
    document.querySelector('#run-summary').textContent = `${data.dataset.toUpperCase()} · ${data.method.toUpperCase()} · ${data.steps.toLocaleString()} steps · seed ${data.seed}`;
    trajectory = data.trajectory;
    const slider = document.querySelector('#trajectory-slider');
    slider.max = Math.max(trajectory.length - 1, 0);
    stopPlayback();
    showFrame(0);
    document.querySelector('#trajectory-view').hidden = false;
    setStatus('Generation complete. Change the seed or sampler to compare trajectories.', 'ready');
  } catch (error) {
    setStatus(error.message, 'error');
  } finally {
    generate.disabled = false;
  }
});

document.querySelector('#trajectory-slider').addEventListener('input', event => { stopPlayback(); showFrame(+event.target.value); });
document.querySelector('#previous-frame').addEventListener('click', () => { const slider = document.querySelector('#trajectory-slider'); showFrame(Math.max(0, +slider.value - 1)); });
document.querySelector('#next-frame').addEventListener('click', () => { const slider = document.querySelector('#trajectory-slider'); showFrame(Math.min(trajectory.length - 1, +slider.value + 1)); });
document.querySelector('#play-trajectory').addEventListener('click', event => {
  if (playback) { stopPlayback(); return; }
  const slider = document.querySelector('#trajectory-slider');
  if (+slider.value >= trajectory.length - 1) showFrame(0);
  event.target.textContent = 'Pause trajectory';
  playback = window.setInterval(() => {
    if (+slider.value >= trajectory.length - 1) { stopPlayback(); return; }
    showFrame(+slider.value + 1);
  }, 650);
});

setRangeFill(steps); setRangeFill(samples);
initialise().catch(error => setStatus(error.message, 'error'));
