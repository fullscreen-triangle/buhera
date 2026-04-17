import { computeSEntropy, storeEntry, findByDescription, getMemoryCount } from './engine.js';
import { renderChart, renderTable, renderSCoords } from './renderer.js';
import { responses } from './demos.js';

const history = document.getElementById('history');
const input = document.getElementById('input');
const entropyDisplay = document.getElementById('entropy-display');
const os = document.getElementById('os');

let welcomeEl = null;
let entryCount = 0;
let ready = true;

function init() {
    welcomeEl = document.createElement('div');
    welcomeEl.className = 'welcome';
    welcomeEl.innerHTML = '<h1>BUHERA</h1><p>a research operating system</p>';
    os.appendChild(welcomeEl);

    input.addEventListener('keydown', onKeyDown);
    input.addEventListener('input', autoResize);
    updateEntropy({ sk: 0, st: 0, se: 0, address: '000000000' });
    input.focus();

    document.addEventListener('click', () => input.focus());
}

function autoResize() {
    input.style.height = 'auto';
    input.style.height = input.scrollHeight + 'px';
}

function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const text = input.value.trim();
        if (text && ready) processInput(text);
    }
}

async function processInput(text) {
    ready = false;

    if (welcomeEl) {
        welcomeEl.classList.add('hidden');
        setTimeout(() => { if (welcomeEl) welcomeEl.remove(); welcomeEl = null; }, 500);
    }

    const coords = computeSEntropy(text);
    updateEntropy(coords);

    const entry = document.createElement('div');
    entry.className = 'entry';

    const inputEl = document.createElement('div');
    inputEl.className = 'entry-input';
    inputEl.textContent = text;
    entry.appendChild(inputEl);

    history.appendChild(entry);
    input.value = '';
    input.style.height = 'auto';
    scrollToBottom();

    const loadingEl = document.createElement('span');
    loadingEl.className = 'loading';
    loadingEl.textContent = 'navigating';
    entry.appendChild(loadingEl);
    scrollToBottom();

    await delay(400 + Math.random() * 600);

    loadingEl.remove();

    const response = matchResponse(text);
    const responseEl = document.createElement('div');
    responseEl.className = 'entry-response';

    if (response.text) {
        const p = document.createElement('p');
        p.textContent = response.text;
        responseEl.appendChild(p);
    }

    if (response.tag) {
        const tag = document.createElement('span');
        tag.className = 'synthesized-tag';
        tag.textContent = response.tag;
        responseEl.appendChild(tag);
    }

    entry.appendChild(responseEl);
    scrollToBottom();

    if (response.chart) {
        await delay(200);
        renderChart(responseEl, response.chart.type, response.chart.data);
        scrollToBottom();
    }

    if (response.table) {
        await delay(150);
        renderTable(responseEl, response.table);
        scrollToBottom();
    }

    await delay(100);
    renderSCoords(responseEl, coords);
    scrollToBottom();

    storeEntry(text, response.text || '', coords);

    entryCount++;
    updateEntropy(coords);

    ready = true;
    input.focus();
}

function matchResponse(text) {
    for (const pattern of responses.patterns) {
        const m = text.match(pattern.match);
        if (m) {
            return pattern.handler(m, findByDescription);
        }
    }
    return responses.fallback(text);
}

function updateEntropy(coords) {
    const mem = getMemoryCount();
    entropyDisplay.textContent =
        `Sk:${coords.sk.toFixed(3)} St:${coords.st.toFixed(3)} Se:${coords.se.toFixed(3)}` +
        ` | addr:${coords.address}` +
        ` | mem:${mem}`;
}

function scrollToBottom() {
    history.scrollTop = history.scrollHeight;
}

function delay(ms) {
    return new Promise(r => setTimeout(r, ms));
}

init();
