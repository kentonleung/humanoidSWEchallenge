/* ═══════════════════════════════════════════════════════════
   LLM Agent Visualizer — Canvas Renderer & WebSocket Client
   ═══════════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────────
let ws = null;
let worldState = null;
let isRunning = false;

// Canvas setup
const canvas = document.getElementById('worldCanvas');
const ctx = canvas.getContext('2d');

// ── Color Palette for Tiles & Entities ───────────────────────
const COLORS = {
    // Tiles
    floor: '#1a1f2e',
    floorAlt: '#1e2438',   // Checkerboard
    wall: '#2d3436',
    wallTop: '#3d4446',
    water: '#2c6fb5',
    waterAlt: '#2563a0',

    // Agent
    agent: '#4ecdc4',
    agentGlow: 'rgba(78, 205, 196, 0.25)',
    agentDir: '#ffffff',

    // Entities
    key: '#fdcb6e',
    doorLocked: '#e17055',
    doorUnlocked: '#00b894',
    gem: '#e84393',
    gemRed: '#ff6b6b',
    gemBlue: '#74b9ff',
    gemGreen: '#00b894',
    gemGold: '#ffd700',
    sign: '#a29bfe',
    goal: '#ffd700',
    goalGlow: 'rgba(255, 215, 0, 0.2)',

    // Entity colors by color name
    entityColors: {
        red: '#ff6b6b',
        blue: '#74b9ff',
        green: '#00b894',
        gold: '#ffd700',
    },

    // Grid
    gridLine: 'rgba(255, 255, 255, 0.03)',
};

// Direction arrows for agent facing
const DIR_ARROWS = {
    north: '▲',
    south: '▼',
    east: '►',
    west: '◄',
};

// ── WebSocket Connection ─────────────────────────────────────
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        setStatus('ready', 'Connected');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };

    ws.onclose = () => {
        setStatus('error', 'Disconnected');
        // Retry connection after 2s
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = () => {
        setStatus('error', 'Connection Error');
    };
}

// ── Message Handling ─────────────────────────────────────────
function handleMessage(data) {
    switch (data.type) {
        case 'run_started':
            onRunStarted(data);
            break;
        case 'step_update':
            onStepUpdate(data);
            break;
        case 'run_complete':
            onRunComplete(data);
            break;
        case 'error':
            onError(data);
            break;
    }
}

function onRunStarted(data) {
    isRunning = true;
    setStatus('running', 'Agent Running...');
    document.getElementById('startBtn').disabled = true;
    if (window.audioEngine) window.audioEngine.startBGM();

    worldState = data.initial_state;
    document.getElementById('taskDescription').textContent = data.task_description;
    document.getElementById('worldName').textContent = worldState.world.name;
    document.getElementById('stepCounter').textContent = `Step 0/${worldState.max_steps}`;

    // Clear previous state
    document.getElementById('objectivesList').innerHTML = '';
    document.getElementById('thinkingContent').innerHTML = '<p class="placeholder">Agent is thinking...</p>';
    document.getElementById('actionLog').innerHTML = '';
    
    // Set initial inventory
    if (worldState.agent && worldState.agent.inventory) {
        updateInventory(worldState.agent.inventory);
    } else {
        document.getElementById('inventoryContent').innerHTML = '<span class="empty-inv">Empty</span>';
    }
    
    document.getElementById('completionBanner').style.display = 'none';

    renderWorld();
}

function onStepUpdate(data) {
    const { step_data, full_state } = data;
    worldState = full_state;

    // Update step counter
    document.getElementById('stepCounter').textContent =
        `Step ${full_state.step}/${full_state.max_steps}`;

    // Update thinking
    const thinkEl = document.getElementById('thinkingContent');
    thinkEl.textContent = step_data.thinking || '(no reasoning provided)';

    // Update action log
    addLogEntry(step_data);

    // Play SFX
    if (window.audioEngine) {
        const actionStr = step_data.action.toLowerCase();
        const resStr = step_data.result.toLowerCase();
        
        if (resStr.includes('dies with a terrible groan')) window.audioEngine.playCombat();
        else if (resStr.includes('you died')) window.audioEngine.playAgentDeath();
        else if (step_data.success) {
            if (actionStr.includes('move')) window.audioEngine.playFootstep();
            else if (actionStr.includes('pickup')) {
                if (resStr.includes('battle')) window.audioEngine.playWeaponPickup();
                else window.audioEngine.playPickup();
            }
            else if (actionStr.includes('unlock') || actionStr.includes('turn')) window.audioEngine.playUnlock();
        }
    }

    // Update objectives
    if (step_data.task_progress) {
        updateObjectives(step_data.task_progress);
    }

    // Update inventory
    if (step_data.inventory) {
        updateInventory(step_data.inventory);
    }

    // Render world
    renderWorld();
}

function onRunComplete(data) {
    isRunning = false;
    const summary = data.summary;
    const success = summary.completed;

    setStatus(success ? 'ready' : 'error', success ? 'Complete [PASS]' : 'Failed [FAIL]');
    document.getElementById('startBtn').disabled = false;
    
    if (window.audioEngine) {
        window.audioEngine.stopBGM();
        if (success) window.audioEngine.playSuccess();
        else window.audioEngine.playError();
    }

    // Show completion banner
    const banner = document.getElementById('completionBanner');
    banner.style.display = 'block';

    document.getElementById('completionIcon').textContent = success ? '[SUCCESS]' : '[FAIL]';
    document.getElementById('completionTitle').textContent =
        success ? 'Task Complete!' : 'Task Failed';
    document.getElementById('completionMessage').textContent =
        summary.completion_message || '';
    document.getElementById('statSteps').textContent =
        `${summary.steps_taken}/${summary.max_steps}`;
    document.getElementById('statTime').textContent =
        `${summary.elapsed_seconds}s`;
    document.getElementById('statTokens').textContent =
        (summary.tokens_used || 0).toLocaleString();
}

function onError(data) {
    isRunning = false;
    setStatus('error', 'Error');
    document.getElementById('startBtn').disabled = false;

    if (window.audioEngine) {
        window.audioEngine.stopBGM();
        window.audioEngine.playError();
    }

    const thinkEl = document.getElementById('thinkingContent');
    thinkEl.innerHTML = `<p style="color: var(--accent-red)">Error: ${data.message}</p>`;
}

// ── UI Updates ───────────────────────────────────────────────
function setStatus(state, text) {
    const badge = document.getElementById('statusBadge');
    badge.className = `status-badge ${state === 'running' ? 'running' : ''} ${state === 'error' ? 'error' : ''}`;
    badge.querySelector('.status-text').textContent = text;
}

function addLogEntry(stepData) {
    const log = document.getElementById('actionLog');

    // Remove empty message
    const empty = log.querySelector('.log-empty');
    if (empty) empty.remove();

    const entry = document.createElement('div');
    entry.className = 'log-entry';

    const successClass = stepData.success ? 'success' : 'failure';
    const icon = stepData.success ? '[PASS]' : '[FAIL]';

    entry.innerHTML = `
        <span class="log-step">${stepData.step}</span>
        <span class="log-action">${escapeHtml(stepData.action)}</span>
        <span class="log-result ${successClass}">${icon} ${escapeHtml(stepData.result.split('\n')[0])}</span>
    `;

    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

function updateObjectives(progress) {
    const container = document.getElementById('objectivesList');
    const objectives = progress.objectives || {};

    container.innerHTML = '';
    for (const [name, done] of Object.entries(objectives)) {
        const el = document.createElement('div');
        el.className = `objective ${done ? 'complete' : 'incomplete'}`;
        el.innerHTML = `
            <span class="objective-icon">${done ? '[PASS]' : '[ ]'}</span>
            <span>${escapeHtml(name)}</span>
        `;
        container.appendChild(el);
    }
}

function updateInventory(items) {
    const container = document.getElementById('inventoryContent');
    if (!items || items.length === 0) {
        container.innerHTML = '<span class="empty-inv">Empty</span>';
        return;
    }

    container.innerHTML = '';
    for (const item of items) {
        const el = document.createElement('span');
        el.className = 'inv-item';
        
        let iconHtml = '';
        if (item.includes('gem')) {
            const colorMatch = item.match(/red|blue|green/);
            const color = colorMatch ? colorMatch[0] : '#e84393';
            const hex = color === 'red' ? '#e74c3c' : (color === 'blue' ? '#3498db' : (color === 'green' ? '#2ecc71' : color));
            iconHtml = `<svg width="12" height="14" viewBox="0 0 12 14" style="vertical-align: middle; margin-right: 4px;"><path fill="${hex}" d="M6 0L12 7L6 14L0 7Z"/></svg>`;
        } else {
            const icon = item.includes('key') ? 'K' :
                         item.includes('sword') || item.includes('weapon') ? 'W' : 'I';
            iconHtml = `<span style="margin-right: 4px;">${icon}</span>`;
        }
        
        el.innerHTML = `${iconHtml}${escapeHtml(item)}`;
        container.appendChild(el);
    }
}

// ── World Rendering ──────────────────────────────────────────
function renderWorld() {
    if (!worldState) return;

    const world = worldState.world;
    const agent = worldState.agent;
    const fullGrid = worldState.full_grid;
    const entities = world.entities;

    const gridW = world.width;
    const gridH = world.height;

    // Calculate cell size to fit canvas
    const maxCellW = canvas.width / gridW;
    const maxCellH = canvas.height / gridH;
    const cellSize = Math.floor(Math.min(maxCellW, maxCellH, 40));

    const offsetX = Math.floor((canvas.width - gridW * cellSize) / 2);
    const offsetY = Math.floor((canvas.height - gridH * cellSize) / 2);

    // Clear
    ctx.fillStyle = '#050810';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw tiles
    for (let y = 0; y < gridH; y++) {
        for (let x = 0; x < gridW; x++) {
            const px = offsetX + x * cellSize;
            const py = offsetY + y * cellSize;
            const tile = world.grid[y][x];

            drawTile(px, py, cellSize, tile, x, y);
        }
    }

    // Draw rooms (subtle borders)
    for (const room of world.rooms) {
        const rx = offsetX + room.x1 * cellSize;
        const ry = offsetY + room.y1 * cellSize;
        const rw = (room.x2 - room.x1 + 1) * cellSize;
        const rh = (room.y2 - room.y1 + 1) * cellSize;

        ctx.strokeStyle = 'rgba(78, 205, 196, 0.08)';
        ctx.lineWidth = 1;
        ctx.strokeRect(rx, ry, rw, rh);

        // Room name
        ctx.fillStyle = 'rgba(78, 205, 196, 0.3)';
        ctx.font = `${Math.max(9, cellSize * 0.3)}px Inter`;
        ctx.textAlign = 'center';
        ctx.fillText(room.name, rx + rw / 2, ry + 12);
    }

    // Draw entities
    for (const entity of entities) {
        const px = offsetX + entity.x * cellSize;
        const py = offsetY + entity.y * cellSize;
        drawEntity(px, py, cellSize, entity);
    }

    // Draw agent
    const ax = offsetX + agent.position.x * cellSize;
    const ay = offsetY + agent.position.y * cellSize;
    drawAgent(ax, ay, cellSize, agent.facing);

    function hasLineOfSight(x0, y0, x1, y1) {
        if (!world || !world.grid) return true;
        let dx = Math.abs(x1 - x0);
        let dy = Math.abs(y1 - y0);
        let sx = (x0 < x1) ? 1 : -1;
        let sy = (y0 < y1) ? 1 : -1;
        let err = dx - dy;

        while (true) {
            if (x0 !== x1 || y0 !== y1) {
                if (world.grid[y0] && world.grid[y0][x0] === '#') {
                    return false;
                }
            }
            if (x0 === x1 && y0 === y1) break;
            let e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
        return true;
    }

    // Draw fog of war to simulate agent and monster vision
    ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
    const liveMonsters = entities.filter(e => e.type === 'monster' && e.alive);

    for (let y = 0; y < gridH; y++) {
        for (let x = 0; x < gridW; x++) {
            const distAgent = Math.abs(x - agent.position.x) + Math.abs(y - agent.position.y);
            const hasLOSAgent = hasLineOfSight(x, y, agent.position.x, agent.position.y);
            
            let distMonster = Infinity;
            let hasLOSMonster = false;
            for (const m of liveMonsters) {
                const d = Math.abs(x - m.x) + Math.abs(y - m.y);
                if (d <= 5 && hasLineOfSight(x, y, m.x, m.y)) {
                    if (d < distMonster) {
                        distMonster = d;
                        hasLOSMonster = true;
                    }
                }
            }

            const isVisible = (hasLOSAgent && distAgent <= 3) || (hasLOSMonster && distMonster <= 3);
            const isDim = (hasLOSAgent && distAgent <= 5) || (hasLOSMonster && distMonster <= 5);

            if (!isVisible) {
                const px = offsetX + x * cellSize;
                const py = offsetY + y * cellSize;
                if (!isDim) {
                    ctx.fillRect(px, py, cellSize, cellSize);
                } else {
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                    ctx.fillRect(px, py, cellSize, cellSize);
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
                }
            }
        }
    }
}

function drawTile(px, py, size, tileChar, gx, gy) {
    switch (tileChar) {
        case '#':
            // Wall — 3D effect
            ctx.fillStyle = COLORS.wall;
            ctx.fillRect(px, py, size, size);
            ctx.fillStyle = COLORS.wallTop;
            ctx.fillRect(px, py, size, size * 0.3);
            // Edge highlight
            ctx.fillStyle = 'rgba(255,255,255,0.03)';
            ctx.fillRect(px, py, size, 1);
            break;
        case '~':
            // Water — animated-looking
            ctx.fillStyle = (gx + gy) % 2 === 0 ? COLORS.water : COLORS.waterAlt;
            ctx.fillRect(px, py, size, size);
            // Shimmer
            ctx.fillStyle = 'rgba(255,255,255,0.05)';
            ctx.fillRect(px + 2, py + size * 0.4, size - 4, 2);
            break;
        default:
            // Floor — checkerboard
            ctx.fillStyle = (gx + gy) % 2 === 0 ? COLORS.floor : COLORS.floorAlt;
            ctx.fillRect(px, py, size, size);
            break;
    }

    // Grid line
    ctx.strokeStyle = COLORS.gridLine;
    ctx.lineWidth = 0.5;
    ctx.strokeRect(px, py, size, size);
}

function drawEntity(px, py, size, entity) {
    const cx = px + size / 2;
    const cy = py + size / 2;
    const r = size * 0.32;
    const color = entity.color ? COLORS.entityColors[entity.color] || '#fff' : '#fff';

    switch (entity.type) {
        case 'key':
            // Key — circle with a notch
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.25;
            ctx.beginPath();
            ctx.arc(cx, cy, r * 1.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = 1.0;

            ctx.fillStyle = color;
            ctx.font = `${size * 0.5}px serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('K', cx, cy);
            break;

        case 'door':
            // Door — rectangle with color
            const doorColor = entity.locked ? (entity.color ? color : COLORS.doorLocked) : COLORS.doorUnlocked;
            ctx.fillStyle = doorColor;
            ctx.fillRect(px + size * 0.15, py + size * 0.1, size * 0.7, size * 0.8);

            // Door handle
            ctx.fillStyle = entity.locked ? '#fff' : '#2d3436';
            ctx.beginPath();
            ctx.arc(px + size * 0.65, cy, size * 0.06, 0, Math.PI * 2);
            ctx.fill();

            // Lock icon
            if (entity.locked) {
                ctx.fillStyle = '#fff';
                ctx.font = `${size * 0.25}px serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('L', cx, cy - size * 0.15);
            }

            // Color stripe
            ctx.fillStyle = color;
            ctx.fillRect(px + size * 0.15, py + size * 0.1, size * 0.7, size * 0.08);
            break;

        case 'gem':
            // Gem — geometric diamond shape with glow
            ctx.fillStyle = `${color}30`;
            ctx.beginPath();
            ctx.arc(cx, cy, r * 1.6, 0, Math.PI * 2);
            ctx.fill();

            // Base color
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(cx, cy - r * 1.2);
            ctx.lineTo(cx + r * 0.9, cy);
            ctx.lineTo(cx, cy + r * 1.2);
            ctx.lineTo(cx - r * 0.9, cy);
            ctx.closePath();
            ctx.fill();

            // Specular highlight
            ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
            ctx.beginPath();
            ctx.moveTo(cx, cy - r * 1.2);
            ctx.lineTo(cx + r * 0.4, cy);
            ctx.lineTo(cx, cy);
            ctx.lineTo(cx - r * 0.4, cy);
            ctx.closePath();
            ctx.fill();
            break;



        case 'sign':
            ctx.fillStyle = COLORS.sign;
            ctx.font = `${size * 0.5}px serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('S', cx, cy);
            break;

        case 'goal_marker':
            // Dungeon Exit — swirling portal
            ctx.fillStyle = 'rgba(155, 89, 182, 0.4)';
            ctx.beginPath();
            ctx.arc(cx, cy, r * 2.2, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = '#fff';
            ctx.font = `${size * 0.65}px serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('G', cx, cy);
            break;

        case 'weapon':
            ctx.fillStyle = 'rgba(205, 164, 52, 0.2)';
            ctx.beginPath();
            ctx.arc(cx, cy, r * 1.5, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = '#fff';
            ctx.font = `${size * 0.5}px serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('W', cx, cy);
            break;

        case 'monster':
            ctx.fillStyle = entity.alive ? (entity.color ? color : 'rgba(179, 42, 42, 0.3)') : 'rgba(0,0,0,0.5)';
            if (entity.alive && entity.color) {
                ctx.globalAlpha = 0.35;
            }
            ctx.beginPath();
            ctx.arc(cx, cy, r * 1.8, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = 1.0;

            ctx.fillStyle = '#fff';
            ctx.font = `${size * 0.6}px serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(entity.alive ? 'M' : 'X', cx, cy);
            break;

        case 'potion':
            ctx.fillStyle = 'rgba(46, 204, 113, 0.3)';
            ctx.beginPath();
            ctx.arc(cx, cy, r * 1.2, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = '#fff';
            ctx.font = `${size * 0.5}px serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('P', cx, cy);
            break;
    }
}

function drawAgent(px, py, size, facing) {
    const cx = px + size / 2;
    const cy = py + size / 2;
    const r = size * 0.35;

    // Glow
    const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 2.5);
    gradient.addColorStop(0, COLORS.agentGlow);
    gradient.addColorStop(1, 'transparent');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(cx, cy, r * 2.5, 0, Math.PI * 2);
    ctx.fill();

    // Body
    ctx.fillStyle = COLORS.agent;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();

    // Shadow
    ctx.strokeStyle = 'rgba(0,0,0,0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Direction indicator
    ctx.fillStyle = COLORS.agentDir;
    ctx.font = `bold ${size * 0.35}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const arrow = DIR_ARROWS[facing] || '●';
    ctx.fillText(arrow, cx, cy);
}

// ── Start Run ────────────────────────────────────────────────
function startRun() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        alert('Not connected to server. Please wait...');
        return;
    }

    if (isRunning) return;

    const taskId = document.getElementById('taskSelect').value;
    const provider = document.getElementById('providerSelect').value;
    const model = document.getElementById('modelInput').value;
    const apiKey = document.getElementById('apiKeyInput').value;

    ws.send(JSON.stringify({
        type: 'start_run',
        task_id: taskId,
        provider: provider,
        model: model || undefined,
        api_key: apiKey || undefined,
    }));
}

// ── Utilities ────────────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function toggleSound() {
    if (window.audioEngine) {
        const isMuted = window.audioEngine.toggleMute();
        document.getElementById('soundBtn').innerHTML = isMuted ? 'Muted' : 'Sound On';
    }
}

// ── Initialize ───────────────────────────────────────────────
window.addEventListener('load', () => {
    connectWebSocket();

    // Draw empty canvas
    ctx.fillStyle = '#050810';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'rgba(78, 205, 196, 0.3)';
    ctx.font = '16px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Select a task and click Start to begin', canvas.width / 2, canvas.height / 2);

    // Provider change listener
    const providerSelect = document.getElementById('providerSelect');
    const modelInput = document.getElementById('modelInput');
    const apiKeyInput = document.getElementById('apiKeyInput');

    providerSelect.addEventListener('change', () => {
        const val = providerSelect.value;
        apiKeyInput.disabled = false;
        apiKeyInput.placeholder = 'sk-... or env variable';
        
        if (val === 'gemini') {
            modelInput.placeholder = 'e.g. gemini-3.5-flash';
            modelInput.value = 'gemini-3.5-flash';
        } else if (val === 'anthropic') {
            modelInput.placeholder = 'e.g. claude-4-8-opus-latest';
            modelInput.value = 'claude-4-8-opus-latest';
        } else if (val === 'openai') {
            modelInput.placeholder = 'e.g. gpt-4o-mini';
            modelInput.value = 'gpt-4o-mini';
        } else if (val === 'kimi') {
            modelInput.placeholder = 'e.g. kimi-k2.6';
            modelInput.value = 'kimi-k2.6';
        } else if (val === 'cerebras') {
            modelInput.placeholder = 'e.g. gpt-oss-120b';
            modelInput.value = 'gpt-oss-120b';
        } else if (val === 'replay') {
            modelInput.placeholder = '(replay from logs)';
            modelInput.value = '';
            apiKeyInput.disabled = true;
            apiKeyInput.placeholder = '(not needed)';
        }
    });

    // Call once to initialize default values
    providerSelect.dispatchEvent(new Event('change'));

    // Check for autoplay in URL query params
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('autoplay') === '1') {
        providerSelect.value = 'replay';
        providerSelect.dispatchEvent(new Event('change'));
        
        // Wait briefly for websocket connection
        setTimeout(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                startRun();
            } else if (ws) {
                ws.addEventListener('open', startRun, {once: true});
            }
        }, 300);
    }
});
