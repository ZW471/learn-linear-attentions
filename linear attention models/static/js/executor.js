/**
 * Step-by-step code execution engine.
 * Mimics the Transformer Explainer: highlights code lines,
 * shows tensor values flowing through, with animated transitions.
 */

// ===== Code Executor =====
class CodeExecutor {
    /**
     * @param {string} containerId - ID of container element
     * @param {object} config - {
     *   code: [{line, indent, comment, isBlank}],  // code lines
     *   steps: [{lineIdx, description, tensors, highlights}], // execution steps
     *   tensors: {name: {shape, initialValue, color}},  // tensor definitions
     * }
     */
    constructor(containerId, config) {
        this.container = document.getElementById(containerId);
        this.config = config;
        this.currentStep = -1;  // -1 = not started
        this.totalSteps = config.steps.length;
        this.isPlaying = false;
        this.playSpeed = 1500;
        this.playTimer = null;
        this.build();
    }

    build() {
        this.container.innerHTML = '';
        this.container.className = 'executor';

        // Main layout: code panel (left) + tensor panel (right)
        const layout = el('div', 'executor-layout');

        // Code panel
        const codePanel = el('div', 'executor-code-panel');
        const codeHeader = el('div', 'executor-panel-header');
        codeHeader.innerHTML = '<span class="panel-icon">&#9998;</span> Code Execution';
        codePanel.appendChild(codeHeader);

        this.codeBody = el('div', 'executor-code-body');
        this.config.code.forEach((line, idx) => {
            const lineEl = el('div', 'code-line');
            lineEl.dataset.lineIdx = idx;

            const lineNum = el('span', 'code-line-num');
            lineNum.textContent = line.isBlank ? '' : (idx + 1);

            const lineContent = el('span', 'code-line-content');
            const indent = '  '.repeat(line.indent || 0);
            lineContent.innerHTML = indent + highlightSyntax(line.line || '');

            const lineComment = el('span', 'code-line-comment');
            if (line.comment) lineComment.textContent = '  # ' + line.comment;

            lineEl.appendChild(lineNum);
            lineEl.appendChild(lineContent);
            lineEl.appendChild(lineComment);
            this.codeBody.appendChild(lineEl);
        });
        codePanel.appendChild(this.codeBody);
        layout.appendChild(codePanel);

        // Tensor panel
        const tensorPanel = el('div', 'executor-tensor-panel');
        const tensorHeader = el('div', 'executor-panel-header');
        tensorHeader.innerHTML = '<span class="panel-icon">&#9638;</span> Tensor Values';
        tensorPanel.appendChild(tensorHeader);

        this.tensorBody = el('div', 'executor-tensor-body');
        tensorPanel.appendChild(this.tensorBody);
        layout.appendChild(tensorPanel);

        this.container.appendChild(layout);

        // Description bar
        this.descBar = el('div', 'executor-desc');
        this.descBar.innerHTML = '<span class="desc-icon">&#9654;</span> Click <strong>Step</strong> or <strong>Play</strong> to begin execution';
        this.container.appendChild(this.descBar);

        // Controls
        const controls = el('div', 'executor-controls');

        this.prevBtn = makeBtn('&#9664; Prev', () => this.prev(), 'exec-btn');
        this.stepBtn = makeBtn('Step &#9654;', () => this.next(), 'exec-btn primary');
        this.playBtn = makeBtn('&#9654; Play', () => this.togglePlay(), 'exec-btn');
        this.resetBtn = makeBtn('&#8634; Reset', () => this.reset(), 'exec-btn');

        const speedGroup = el('div', 'exec-speed-group');
        speedGroup.innerHTML = '<label>Speed:</label>';
        this.speedSlider = document.createElement('input');
        this.speedSlider.type = 'range';
        this.speedSlider.min = 500;
        this.speedSlider.max = 3000;
        this.speedSlider.value = this.playSpeed;
        this.speedSlider.addEventListener('input', () => {
            this.playSpeed = parseInt(this.speedSlider.value);
        });
        speedGroup.appendChild(this.speedSlider);

        this.stepLabel = el('span', 'exec-step-label');
        this.stepLabel.textContent = `0 / ${this.totalSteps}`;

        // Progress bar
        this.progressBar = el('div', 'exec-progress-bar');
        this.progressFill = el('div', 'exec-progress-fill');
        this.progressBar.appendChild(this.progressFill);

        controls.appendChild(this.prevBtn);
        controls.appendChild(this.stepBtn);
        controls.appendChild(this.playBtn);
        controls.appendChild(this.resetBtn);
        controls.appendChild(speedGroup);
        controls.appendChild(this.stepLabel);

        this.container.appendChild(this.progressBar);
        this.container.appendChild(controls);

        this.updateUI();
    }

    next() {
        if (this.currentStep < this.totalSteps - 1) {
            this.currentStep++;
            this.updateUI();
        } else if (this.isPlaying) {
            this.togglePlay();
        }
    }

    prev() {
        if (this.currentStep > 0) {
            this.currentStep--;
            this.updateUI();
        }
    }

    reset() {
        this.currentStep = -1;
        if (this.isPlaying) this.togglePlay();
        this.updateUI();
    }

    togglePlay() {
        this.isPlaying = !this.isPlaying;
        this.playBtn.innerHTML = this.isPlaying ? '&#9724; Pause' : '&#9654; Play';

        if (this.isPlaying) {
            if (this.currentStep >= this.totalSteps - 1) {
                this.currentStep = -1;
            }
            this.playTimer = setInterval(() => this.next(), this.playSpeed);
        } else {
            clearInterval(this.playTimer);
        }
    }

    updateUI() {
        const step = this.currentStep >= 0 ? this.config.steps[this.currentStep] : null;

        // Update step label & progress
        this.stepLabel.textContent = `${this.currentStep + 1} / ${this.totalSteps}`;
        const pct = this.totalSteps > 0 ? ((this.currentStep + 1) / this.totalSteps * 100) : 0;
        this.progressFill.style.width = `${pct}%`;

        // Update buttons
        this.prevBtn.disabled = this.currentStep <= 0;

        // Highlight code lines
        this.codeBody.querySelectorAll('.code-line').forEach(line => {
            line.classList.remove('active', 'executed', 'next');
        });

        if (step) {
            // Mark all previously executed lines
            for (let i = 0; i < this.currentStep; i++) {
                const prevStep = this.config.steps[i];
                const prevLine = this.codeBody.querySelector(`[data-line-idx="${prevStep.lineIdx}"]`);
                if (prevLine) prevLine.classList.add('executed');
            }
            // Highlight current line
            const currentLine = this.codeBody.querySelector(`[data-line-idx="${step.lineIdx}"]`);
            if (currentLine) {
                currentLine.classList.add('active');
                currentLine.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
            // Show next line
            if (this.currentStep < this.totalSteps - 1) {
                const nextStep = this.config.steps[this.currentStep + 1];
                const nextLine = this.codeBody.querySelector(`[data-line-idx="${nextStep.lineIdx}"]`);
                if (nextLine) nextLine.classList.add('next');
            }
        }

        // Update description
        if (step) {
            this.descBar.innerHTML = `<span class="desc-step">Step ${this.currentStep + 1}:</span> ${step.description}`;
        } else {
            this.descBar.innerHTML = '<span class="desc-icon">&#9654;</span> Click <strong>Step</strong> or <strong>Play</strong> to begin execution';
        }

        // Update tensors
        this.renderTensors(step);
    }

    renderTensors(step) {
        this.tensorBody.innerHTML = '';

        if (!step || !step.tensors) {
            const placeholder = el('div', 'tensor-placeholder');
            placeholder.textContent = 'Tensors will appear here as code executes...';
            this.tensorBody.appendChild(placeholder);
            return;
        }

        // Group tensors by category
        const tensorEntries = Object.entries(step.tensors);
        tensorEntries.forEach(([name, info]) => {
            const card = el('div', 'tensor-card');
            if (info.highlight) card.classList.add('highlight');
            if (info.justComputed) card.classList.add('just-computed');

            // Header
            const header = el('div', 'tensor-card-header');
            const nameEl = el('span', 'tensor-name');
            nameEl.textContent = name;
            const shapeEl = el('span', 'tensor-shape-badge');
            shapeEl.textContent = info.shape || '';
            shapeEl.style.borderColor = info.color || 'rgba(96,165,250,0.3)';
            shapeEl.style.color = info.color || 'var(--accent-blue)';
            header.appendChild(nameEl);
            header.appendChild(shapeEl);
            card.appendChild(header);

            // Value display
            if (info.value !== undefined) {
                const valEl = el('div', 'tensor-value');
                if (Array.isArray(info.value)) {
                    if (Array.isArray(info.value[0])) {
                        // 2D matrix
                        valEl.appendChild(renderSmallMatrix(info.value, info.highlightCell, info.color));
                    } else {
                        // 1D vector
                        valEl.appendChild(renderSmallVector(info.value, info.highlightIdx, info.color));
                    }
                } else {
                    // Scalar
                    const scalarEl = el('span', 'tensor-scalar');
                    scalarEl.textContent = typeof info.value === 'number' ? info.value.toFixed(4) : String(info.value);
                    valEl.appendChild(scalarEl);
                }
                card.appendChild(valEl);
            }

            // Annotation
            if (info.annotation) {
                const annEl = el('div', 'tensor-annotation');
                annEl.innerHTML = info.annotation;
                card.appendChild(annEl);
            }

            this.tensorBody.appendChild(card);
        });
    }
}

// ===== Helpers =====
function el(tag, className) {
    const e = document.createElement(tag);
    if (className) e.className = className;
    return e;
}

function makeBtn(html, onclick, className) {
    const btn = document.createElement('button');
    btn.innerHTML = html;
    btn.onclick = onclick;
    btn.className = className || '';
    return btn;
}

function highlightSyntax(code) {
    return code
        .replace(/\b(def|class|for|in|if|else|return|import|from|as|None|True|False)\b/g, '<span class="hl-kw">$1</span>')
        .replace(/\b(torch|np|nn|F)\b/g, '<span class="hl-mod">$1</span>')
        .replace(/\b(zeros|ones|eye|exp|randn|softplus|sigmoid|outer|sum|matmul)\b/g, '<span class="hl-fn">$1</span>')
        .replace(/(#.*$)/gm, '<span class="hl-cm">$1</span>')
        .replace(/\b(\d+\.?\d*)\b/g, '<span class="hl-num">$1</span>')
        .replace(/(@)/g, '<span class="hl-op">$1</span>')
        .replace(/(".*?"|'.*?')/g, '<span class="hl-str">$1</span>');
}

function renderSmallMatrix(matrix, highlightCell, color) {
    const table = document.createElement('table');
    table.className = 'tensor-matrix';
    const rows = matrix.length;
    const cols = matrix[0].length;
    const flat = matrix.flat();
    const absMax = Math.max(...flat.map(Math.abs), 0.001);

    for (let i = 0; i < rows; i++) {
        const tr = document.createElement('tr');
        for (let j = 0; j < cols; j++) {
            const td = document.createElement('td');
            const val = matrix[i][j];
            const norm = val / absMax;
            const baseColor = color || '#60a5fa';
            if (norm >= 0) {
                td.style.background = `rgba(96,165,250,${0.05 + norm * 0.5})`;
            } else {
                td.style.background = `rgba(248,113,113,${0.05 + Math.abs(norm) * 0.5})`;
            }
            td.textContent = val.toFixed(2);
            td.title = `[${i},${j}] = ${val.toFixed(6)}`;
            if (highlightCell && highlightCell[0] === i && highlightCell[1] === j) {
                td.classList.add('cell-highlight');
            }
            tr.appendChild(td);
        }
        table.appendChild(tr);
    }
    return table;
}

function renderSmallVector(vec, highlightIdx, color) {
    const container = el('div', 'tensor-vector');
    vec.forEach((val, idx) => {
        const cell = el('span', 'vector-cell');
        const absMax = Math.max(...vec.map(Math.abs), 0.001);
        const norm = val / absMax;
        if (norm >= 0) {
            cell.style.background = `rgba(96,165,250,${0.05 + norm * 0.5})`;
        } else {
            cell.style.background = `rgba(248,113,113,${0.05 + Math.abs(norm) * 0.5})`;
        }
        cell.textContent = val.toFixed(3);
        cell.title = `[${idx}] = ${val.toFixed(6)}`;
        if (highlightIdx === idx) cell.classList.add('cell-highlight');
        container.appendChild(cell);
    });
    return container;
}

// ===== Flow Arrows =====
function renderTensorFlow(containerId, nodes) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.className = 'tensor-flow';
    let html = '';

    nodes.forEach((node, idx) => {
        if (idx > 0) {
            html += `<div class="tf-arrow">${node.arrowLabel || '&#10132;'}</div>`;
        }
        const activeClass = node.active ? ' tf-active' : '';
        const computedClass = node.computed ? ' tf-computed' : '';
        html += `
            <div class="tf-node${activeClass}${computedClass}">
                <div class="tf-name">${node.name}</div>
                <div class="tf-shape">${node.shape}</div>
                ${node.value ? `<div class="tf-value">${node.value}</div>` : ''}
                ${node.operation ? `<div class="tf-operation">${node.operation}</div>` : ''}
            </div>
        `;
    });

    container.innerHTML = html;
}
