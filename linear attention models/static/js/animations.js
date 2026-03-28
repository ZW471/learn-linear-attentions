/**
 * Shared animation and visualization utilities for linear attention models.
 */

// ===== Color Scale =====
function valueToColor(val, minVal, maxVal) {
    if (maxVal === minVal) return 'rgba(96, 165, 250, 0.3)';
    const normalized = (val - minVal) / (maxVal - minVal);

    if (val >= 0) {
        const intensity = Math.min(normalized, 1);
        return `rgba(96, 165, 250, ${0.1 + intensity * 0.8})`;
    } else {
        const intensity = Math.min(Math.abs(normalized), 1);
        return `rgba(248, 113, 113, ${0.1 + intensity * 0.8})`;
    }
}

function divergingColor(val, absMax) {
    if (absMax === 0) return 'rgba(96, 165, 250, 0.1)';
    const norm = val / absMax;
    if (norm >= 0) {
        return `rgba(96, 165, 250, ${0.05 + norm * 0.85})`;
    } else {
        return `rgba(248, 113, 113, ${0.05 + Math.abs(norm) * 0.85})`;
    }
}

// ===== Matrix Rendering =====
function renderMatrix(containerId, matrix, label, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const rows = matrix.length;
    const cols = matrix[0] ? (Array.isArray(matrix[0]) ? matrix[0].length : 1) : 0;

    // Flatten for min/max
    let flat = [];
    for (let i = 0; i < rows; i++) {
        if (Array.isArray(matrix[i])) {
            flat = flat.concat(matrix[i]);
        } else {
            flat.push(matrix[i]);
        }
    }
    const absMax = Math.max(...flat.map(Math.abs), 0.001);

    let html = '<div class="matrix-vis">';
    if (label) {
        html += `<div class="matrix-label">${label}</div>`;
    }
    html += '<table>';

    for (let i = 0; i < rows; i++) {
        html += '<tr>';
        const row = Array.isArray(matrix[i]) ? matrix[i] : [matrix[i]];
        for (let j = 0; j < row.length; j++) {
            const val = row[j];
            const color = divergingColor(val, absMax);
            const textColor = Math.abs(val / absMax) > 0.5 ? '#fff' : 'var(--text-secondary)';
            const highlighted = options.highlightRow === i || options.highlightCol === j;
            const cls = highlighted ? ' class="highlight"' : '';
            html += `<td${cls} style="background:${color};color:${textColor}" title="[${i},${j}] = ${val.toFixed(4)}">${val.toFixed(2)}</td>`;
        }
        html += '</tr>';
    }
    html += '</table></div>';
    container.innerHTML = html;
}

function renderMatrixEquation(containerId, matrices) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.className = 'matrix-container';
    let html = '';

    matrices.forEach((item, idx) => {
        if (item.type === 'matrix') {
            const id = `matrix-eq-${containerId}-${idx}`;
            html += `<div id="${id}"></div>`;
            // Defer rendering
            setTimeout(() => renderMatrix(id, item.data, item.label, item.options || {}), 0);
        } else if (item.type === 'operator') {
            html += `<span class="matrix-operator">${item.symbol}</span>`;
        } else if (item.type === 'text') {
            html += `<span style="font-family:var(--font-mono);color:var(--text-secondary);font-size:0.9rem;padding:0 0.5rem">${item.text}</span>`;
        }
    });

    container.innerHTML = html;
    // Re-render matrices after HTML is set
    matrices.forEach((item, idx) => {
        if (item.type === 'matrix') {
            const id = `matrix-eq-${containerId}-${idx}`;
            renderMatrix(id, item.data, item.label, item.options || {});
        }
    });
}

// ===== Step-by-Step Controller =====
class StepAnimator {
    constructor(containerId, steps) {
        this.container = document.getElementById(containerId);
        this.steps = steps;
        this.currentStep = 0;
        this.render();
    }

    render() {
        const step = this.steps[this.currentStep];

        let html = `
            <div class="step-controls">
                <button class="step-btn" onclick="stepAnimators['${this.container.id}'].prev()" ${this.currentStep === 0 ? 'disabled' : ''}>&#9664; Prev</button>
                <button class="step-btn primary" onclick="stepAnimators['${this.container.id}'].next()" ${this.currentStep === this.steps.length - 1 ? 'disabled' : ''}>Next &#9654;</button>
                <button class="step-btn" onclick="stepAnimators['${this.container.id}'].reset()">Reset</button>
                <span class="step-indicator">Step ${this.currentStep + 1} / ${this.steps.length}</span>
            </div>
            <div class="step-description">
                <strong>${step.title}</strong><br>${step.description}
            </div>
        `;

        this.container.innerHTML = html;

        if (step.render) {
            const vizDiv = document.createElement('div');
            vizDiv.style.marginTop = '1rem';
            this.container.appendChild(vizDiv);
            step.render(vizDiv);
        }
    }

    next() {
        if (this.currentStep < this.steps.length - 1) {
            this.currentStep++;
            this.render();
        }
    }

    prev() {
        if (this.currentStep > 0) {
            this.currentStep--;
            this.render();
        }
    }

    reset() {
        this.currentStep = 0;
        this.render();
    }
}

// Global registry for step animators
const stepAnimators = {};

// ===== Plotly Helpers =====
const PLOTLY_DARK_LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(26,29,46,0.5)',
    font: { color: '#9ea4c1', family: 'Inter, sans-serif', size: 12 },
    xaxis: { gridcolor: 'rgba(45,50,90,0.5)', zerolinecolor: 'rgba(45,50,90,0.8)' },
    yaxis: { gridcolor: 'rgba(45,50,90,0.5)', zerolinecolor: 'rgba(45,50,90,0.8)' },
    margin: { l: 50, r: 20, t: 40, b: 40 },
    legend: { bgcolor: 'rgba(0,0,0,0)' },
};

function plotHeatmap(containerId, matrix, title, xLabels, yLabels) {
    const data = [{
        z: matrix,
        type: 'heatmap',
        colorscale: [
            [0, '#1a1d2e'],
            [0.25, 'rgba(96,165,250,0.3)'],
            [0.5, 'rgba(96,165,250,0.5)'],
            [0.75, 'rgba(96,165,250,0.75)'],
            [1, '#60a5fa']
        ],
        x: xLabels,
        y: yLabels,
        hoverongaps: false,
    }];

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        title: { text: title, font: { size: 14 } },
        height: 350,
    };

    Plotly.newPlot(containerId, data, layout, { responsive: true, displayModeBar: false });
}

function plotLines(containerId, traces, title, xLabel, yLabel) {
    const colors = ['#60a5fa', '#4ade80', '#a78bfa', '#fb923c', '#f87171', '#22d3ee'];
    const data = traces.map((t, i) => ({
        x: t.x || Array.from({ length: t.y.length }, (_, i) => i),
        y: t.y,
        name: t.name,
        type: 'scatter',
        mode: t.mode || 'lines+markers',
        line: { color: colors[i % colors.length], width: 2 },
        marker: { size: 5 },
    }));

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        title: { text: title, font: { size: 14 } },
        xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis, title: xLabel },
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: yLabel },
        height: 350,
    };

    Plotly.newPlot(containerId, data, layout, { responsive: true, displayModeBar: false });
}

// ===== Tab Switching =====
function initTabs(groupId) {
    const group = document.getElementById(groupId);
    if (!group) return;

    const buttons = group.querySelectorAll('.tab-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            // Deactivate all
            buttons.forEach(b => b.classList.remove('active'));
            group.parentElement.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            // Activate target
            btn.classList.add('active');
            document.getElementById(target).classList.add('active');
            // Trigger resize for Plotly
            window.dispatchEvent(new Event('resize'));
        });
    });
}

// ===== Flow Diagram =====
function renderFlowDiagram(containerId, nodes, activeIdx = -1) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.className = 'flow-diagram';
    let html = '';

    nodes.forEach((node, idx) => {
        if (idx > 0) html += '<span class="flow-arrow">&#10132;</span>';
        const active = idx === activeIdx ? ' active' : '';
        html += `
            <div class="flow-node${active}">
                <span class="node-name">${node.name}</span>
                <span class="node-shape">${node.shape}</span>
            </div>
        `;
    });

    container.innerHTML = html;
}
