// Configuration
const API_BASE = 'http://localhost:8080';
const POLL_INTERVAL = 500; // ms

// State
let experts = [];
let tickHistory = [];
let maxHistoryLength = 100;

// Elements
const connectionStatus = document.getElementById('connection-status');
const tickCounter = document.getElementById('tick-counter');
const kernelStatus = document.getElementById('kernel-status');
const eventCount = document.getElementById('event-count');
const merkleRoot = document.getElementById('merkle-root');
const expertGrid = document.getElementById('expert-grid');
const eventLog = document.getElementById('event-log');
const telemetryCanvas = document.getElementById('telemetry-canvas');
const localTime = document.getElementById('local-time');

// Telemetry Canvas Setup
const ctx = telemetryCanvas.getContext('2d');
let animationFrame;

function resizeCanvas() {
    const rect = telemetryCanvas.getBoundingClientRect();
    telemetryCanvas.width = rect.width;
    telemetryCanvas.height = rect.height;
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// API Calls
async function fetchStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();
        updateDashboard(data);
        setConnectionStatus('connected');
        return data;
    } catch (error) {
        console.error('Failed to fetch status:', error);
        setConnectionStatus('error');
        return null;
    }
}

async function fetchExperts() {
    try {
        const response = await fetch(`${API_BASE}/experts`);
        const data = await response.json();
        experts = data.experts || [];
        renderExpertGrid();
    } catch (error) {
        console.error('Failed to fetch experts:', error);
    }
}

// UI Updates
function setConnectionStatus(status) {
    connectionStatus.className = `status-indicator ${status}`;
    connectionStatus.querySelector('span:last-child').textContent = status.toUpperCase();
}

function updateDashboard(data) {
    // Update metrics
    tickCounter.textContent = data.tick || '0';
    kernelStatus.textContent = data.status || 'IDLE';
    eventCount.textContent = data.event_count || '0';

    // Update Merkle root
    const root = data.merkle_root || 'NONE';
    merkleRoot.textContent = root.length > 16 ? root.substring(0, 16) + '...' : root;
    merkleRoot.title = root;

    // Update expert activation
    updateExpertActivation(data.active_experts || []);

    // Add to telemetry history
    tickHistory.push({
        tick: data.tick || 0,
        activeCount: (data.active_experts || []).length,
        timestamp: Date.now()
    });

    if (tickHistory.length > maxHistoryLength) {
        tickHistory.shift();
    }

    // Log event if tick changed
    if (data.tick > 0 && (!window.lastTick || window.lastTick < data.tick)) {
        addLogEntry(`Tick ${data.tick}: ${data.status} | Active: ${(data.active_experts || []).join(', ')}`);
        window.lastTick = data.tick;
    }
}

function renderExpertGrid() {
    expertGrid.innerHTML = '';
    experts.forEach(expertName => {
        const card = document.createElement('div');
        card.className = 'expert-card';
        card.id = `expert-${expertName.replace(/\s+/g, '-')}`;
        card.innerHTML = `
            <div class="expert-name">${expertName}</div>
            <div class="expert-status">IDLE</div>
        `;
        expertGrid.appendChild(card);
    });
}

function updateExpertActivation(activeExperts) {
    // Reset all
    document.querySelectorAll('.expert-card').forEach(card => {
        card.classList.remove('active');
        card.querySelector('.expert-status').textContent = 'IDLE';
    });

    // Activate current
    activeExperts.forEach(expertName => {
        const card = document.getElementById(`expert-${expertName.replace(/\s+/g, '-')}`);
        if (card) {
            card.classList.add('active');
            card.querySelector('.expert-status').textContent = 'ACTIVE';
        }
    });
}

function addLogEntry(message) {
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];

    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
        <span class="log-time">${timeStr}</span>
        <span class="log-message">${message}</span>
    `;

    eventLog.insertBefore(entry, eventLog.firstChild);

    // Keep only last 50 entries
    while (eventLog.children.length > 50) {
        eventLog.removeChild(eventLog.lastChild);
    }
}

// Telemetry Visualization
function drawTelemetry() {
    const width = telemetryCanvas.width;
    const height = telemetryCanvas.height;

    // Clear
    ctx.fillStyle = '#162320';
    ctx.fillRect(0, 0, width, height);

    // Grid
    ctx.strokeStyle = '#1a3d35';
    ctx.lineWidth = 1;

    // Horizontal lines
    for (let i = 0; i <= 4; i++) {
        const y = (height / 4) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }

    // Vertical lines
    for (let i = 0; i <= 10; i++) {
        const x = (width / 10) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }

    if (tickHistory.length < 2) {
        animationFrame = requestAnimationFrame(drawTelemetry);
        return;
    }

    // Plot line graph
    ctx.strokeStyle = '#00d9a3';
    ctx.lineWidth = 2;
    ctx.beginPath();

    const xStep = width / (maxHistoryLength - 1);
    const maxActive = 8; // Assume max 8 experts

    tickHistory.forEach((point, index) => {
        const x = index * xStep;
        const y = height - (point.activeCount / maxActive) * height;

        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.stroke();

    // Draw points
    ctx.fillStyle = '#00ffcc';
    tickHistory.forEach((point, index) => {
        const x = index * xStep;
        const y = height - (point.activeCount / maxActive) * height;

        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
    });

    animationFrame = requestAnimationFrame(drawTelemetry);
}

// Update local time
function updateLocalTime() {
    const now = new Date();
    localTime.textContent = now.toTimeString().split(' ')[0];
}

// Initialization
async function init() {
    addLogEntry('Initializing Master Operator HUD...');
    await fetchExperts();
    renderExpertGrid();
    drawTelemetry();

    // Start polling
    setInterval(fetchStatus, POLL_INTERVAL);
    setInterval(updateLocalTime, 1000);

    addLogEntry('HUD ready. Monitoring kernel...');
}

// Start
init();
