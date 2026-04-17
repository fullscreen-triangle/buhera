export function renderChart(container, type, data) {
    const wrapper = document.createElement('div');
    wrapper.className = 'inline-chart';
    const canvas = document.createElement('canvas');
    const w = Math.min(700, window.innerWidth - 160);
    const h = 220;
    canvas.width = w * 2;
    canvas.height = h * 2;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(2, 2);

    if (type === 'line') drawLineChart(ctx, w, h, data);
    else if (type === 'bar') drawBarChart(ctx, w, h, data);
    else if (type === 'scatter') drawScatterChart(ctx, w, h, data);
    else if (type === 'trajectory') drawTrajectory(ctx, w, h, data);

    wrapper.appendChild(canvas);
    container.appendChild(wrapper);
}

function drawLineChart(ctx, w, h, data) {
    const pad = { t: 20, r: 20, b: 35, l: 50 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, w, h);

    const yMin = Math.min(...data.values);
    const yMax = Math.max(...data.values);
    const yRange = yMax - yMin || 1;

    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + (ph * i / 4);
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(pad.l + pw, y);
        ctx.stroke();
        ctx.fillStyle = '#555';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        const val = yMax - (yRange * i / 4);
        ctx.fillText(val.toFixed(1), pad.l - 6, y + 3);
    }

    ctx.strokeStyle = '#2a9d8f';
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.values.forEach((v, i) => {
        const x = pad.l + (pw * i / (data.values.length - 1));
        const y = pad.t + ph - ((v - yMin) / yRange * ph);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    data.values.forEach((v, i) => {
        const x = pad.l + (pw * i / (data.values.length - 1));
        const y = pad.t + ph - ((v - yMin) / yRange * ph);
        ctx.fillStyle = '#2a9d8f';
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
    });

    if (data.labels) {
        ctx.fillStyle = '#555';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        data.labels.forEach((l, i) => {
            if (i % Math.ceil(data.labels.length / 8) === 0) {
                const x = pad.l + (pw * i / (data.labels.length - 1));
                ctx.fillText(l, x, h - 8);
            }
        });
    }

    if (data.title) {
        ctx.fillStyle = '#777';
        ctx.font = '11px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(data.title, pad.l, 14);
    }
}

function drawBarChart(ctx, w, h, data) {
    const pad = { t: 20, r: 20, b: 45, l: 50 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, w, h);

    const yMax = Math.max(...data.values) * 1.1;
    const barW = pw / data.values.length * 0.7;
    const gap = pw / data.values.length * 0.3;

    const colors = ['#2a9d8f', '#e76f51', '#457b9d', '#e9c46a', '#6a4c93', '#264653'];

    data.values.forEach((v, i) => {
        const x = pad.l + (pw * i / data.values.length) + gap / 2;
        const barH = (v / yMax) * ph;
        const y = pad.t + ph - barH;
        ctx.fillStyle = colors[i % colors.length];
        ctx.fillRect(x, y, barW, barH);

        ctx.fillStyle = '#aaa';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(v.toFixed(2), x + barW / 2, y - 5);
    });

    if (data.labels) {
        ctx.fillStyle = '#555';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        data.labels.forEach((l, i) => {
            const x = pad.l + (pw * i / data.values.length) + barW / 2 + gap / 2;
            ctx.save();
            ctx.translate(x, h - 5);
            ctx.rotate(-0.4);
            ctx.fillText(l, 0, 0);
            ctx.restore();
        });
    }

    if (data.title) {
        ctx.fillStyle = '#777';
        ctx.font = '11px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(data.title, pad.l, 14);
    }
}

function drawScatterChart(ctx, w, h, data) {
    const pad = { t: 20, r: 20, b: 35, l: 50 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, w, h);

    const xMin = Math.min(...data.x);
    const xMax = Math.max(...data.x);
    const yMin = Math.min(...data.y);
    const yMax = Math.max(...data.y);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

    data.x.forEach((xv, i) => {
        const px = pad.l + ((xv - xMin) / xRange) * pw;
        const py = pad.t + ph - ((data.y[i] - yMin) / yRange) * ph;
        const c = data.colors ? data.colors[i] : '#2a9d8f';
        ctx.fillStyle = c;
        ctx.globalAlpha = 0.8;
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1.0;
    });

    if (data.title) {
        ctx.fillStyle = '#777';
        ctx.font = '11px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(data.title, pad.l, 14);
    }
}

function drawTrajectory(ctx, w, h, data) {
    const pad = { t: 20, r: 20, b: 35, l: 50 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + (ph * i / 4);
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(pad.l + pw, y);
        ctx.stroke();
    }

    const points = data.trajectory;
    const n = points.length;

    ctx.lineWidth = 2;
    for (let i = 1; i < n; i++) {
        const t = i / n;
        const r = Math.round(42 + t * 189);
        const g = Math.round(157 - t * 50);
        const b = Math.round(143 - t * 62);
        ctx.strokeStyle = `rgb(${r},${g},${b})`;
        const x0 = pad.l + ((i - 1) / (n - 1)) * pw;
        const y0 = pad.t + ph - points[i - 1] * ph;
        const x1 = pad.l + (i / (n - 1)) * pw;
        const y1 = pad.t + ph - points[i] * ph;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
    }

    ctx.fillStyle = '#e76f51';
    ctx.beginPath();
    ctx.arc(pad.l, pad.t + ph - points[0] * ph, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#2a9d8f';
    ctx.beginPath();
    ctx.arc(pad.l + pw, pad.t + ph - points[n - 1] * ph, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#777';
    ctx.font = '11px monospace';
    ctx.fillText(data.title || 'backward trajectory', pad.l, 14);
    ctx.fillStyle = '#e76f51';
    ctx.font = '10px monospace';
    ctx.fillText('initial', pad.l, h - 8);
    ctx.fillStyle = '#2a9d8f';
    ctx.textAlign = 'right';
    ctx.fillText('penultimate', pad.l + pw, h - 8);
}

export function renderTable(container, data) {
    const wrapper = document.createElement('div');
    wrapper.className = 'inline-data';
    const table = document.createElement('table');

    if (data.headers) {
        const thead = document.createElement('thead');
        const tr = document.createElement('tr');
        data.headers.forEach(h => {
            const th = document.createElement('th');
            th.textContent = h;
            tr.appendChild(th);
        });
        thead.appendChild(tr);
        table.appendChild(thead);
    }

    const tbody = document.createElement('tbody');
    data.rows.forEach(row => {
        const tr = document.createElement('tr');
        row.forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrapper.appendChild(table);
    container.appendChild(wrapper);
}

export function renderSCoords(container, coords) {
    const span = document.createElement('span');
    span.className = 's-coords';
    span.innerHTML =
        `<span class="label">S</span> = (`
        + `<span class="label">k:</span>${coords.sk.toFixed(3)}, `
        + `<span class="label">t:</span>${coords.st.toFixed(3)}, `
        + `<span class="label">e:</span>${coords.se.toFixed(3)}`
        + `) addr: ${coords.address}`;
    container.appendChild(span);
}
