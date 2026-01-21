/**
 * Persona Steering Web UI
 * Vanilla JS frontend for persona management, mixing, and chat
 */

// ═══════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════

const state = {
    personas: [],
    traits: [],
    activePersonas: {}, // name -> weight (0-100)
    activeTraits: {},   // name -> weight (0-100)
    modelLoaded: false,
    creatingPersona: null,
    creatingTrait: null,
    ws: null,
    currentResponse: '',
    steeringScale: 0.1,
    currentTab: 'personas', // 'personas' or 'traits'
};

// ═══════════════════════════════════════════════════════════════════════════
// DOM ELEMENTS
// ═══════════════════════════════════════════════════════════════════════════

const elements = {
    // Status
    modelStatus: document.getElementById('model-status'),
    blendStatus: document.getElementById('blend-status'),
    
    // Personas
    personaList: document.getElementById('persona-list'),
    createPersonaBtn: document.getElementById('create-persona-btn'),
    
    // Traits
    traitList: document.getElementById('trait-list'),
    createTraitBtn: document.getElementById('create-trait-btn'),
    
    // Tabs
    subTabs: document.querySelectorAll('.sub-tab'),
    personasTab: document.getElementById('personas-tab'),
    traitsTab: document.getElementById('traits-tab'),
    
    // Chat
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    
    // Mixer
    mixerSliders: document.getElementById('mixer-sliders'),
    blendVisual: document.getElementById('blend-visual'),
    
    // Settings
    steeringScaleValue: document.getElementById('steering-scale-value'),
    scaleUpBtn: document.getElementById('scale-up'),
    scaleDownBtn: document.getElementById('scale-down'),
    
    // New Chat
    newChatBtn: document.getElementById('new-chat-btn'),
    
    // Persona Modal
    createModal: document.getElementById('create-modal'),
    personaName: document.getElementById('persona-name'),
    personaDescription: document.getElementById('persona-description'),
    personaPrompts: document.getElementById('persona-prompts'),
    personaQuestions: document.getElementById('persona-questions'),
    personaBatch: document.getElementById('persona-batch'),
    personaUseJudge: document.getElementById('persona-use-judge'),
    modalCancel: document.getElementById('modal-cancel'),
    modalCreate: document.getElementById('modal-create'),
    
    // Trait Modal
    createTraitModal: document.getElementById('create-trait-modal'),
    traitWord: document.getElementById('trait-word'),
    traitOpposite: document.getElementById('trait-opposite'),
    traitQuestions: document.getElementById('trait-questions'),
    traitBatch: document.getElementById('trait-batch'),
    traitModalCancel: document.getElementById('trait-modal-cancel'),
    traitModalCreate: document.getElementById('trait-modal-create'),
    
    // Loading
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingText: document.getElementById('loading-text'),
};

// ═══════════════════════════════════════════════════════════════════════════
// API FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

async function fetchStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        state.modelLoaded = data.model_loaded;
        state.creatingPersona = data.creating_persona;
        state.creatingTrait = data.creating_trait;
        state.steeringScale = data.steering_scale || 0.1;
        
        // Update status displays
        elements.modelStatus.textContent = data.model_loaded ? 'Ready' : 
            (data.model_loading ? 'Loading...' : 'Not Loaded');
        elements.modelStatus.className = 'status-value' + 
            (data.model_loaded ? ' ready' : '');
        
        // Update steering scale display
        updateSteeringScaleDisplay();
        
        // Show creating persona/trait status
        if (data.creating_persona) {
            showLoading(`Creating persona: ${data.creating_persona}...`);
        } else if (data.creating_trait) {
            showLoading(`Extracting trait: ${data.creating_trait}...`);
        }
        
        return data;
    } catch (error) {
        console.error('Failed to fetch status:', error);
        return null;
    }
}

function updateSteeringScaleDisplay() {
    if (elements.steeringScaleValue) {
        elements.steeringScaleValue.textContent = state.steeringScale.toFixed(2);
    }
}

async function setSteeringScale(scale) {
    try {
        const response = await fetch('/api/steering-scale', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scale }),
        });
        
        if (response.ok) {
            const data = await response.json();
            state.steeringScale = data.steering_scale;
            updateSteeringScaleDisplay();
        }
    } catch (error) {
        console.error('Failed to update steering scale:', error);
    }
}

async function fetchPersonas() {
    try {
        const response = await fetch('/api/personas');
        state.personas = await response.json();
        renderPersonaList();
    } catch (error) {
        console.error('Failed to fetch personas:', error);
    }
}

async function fetchTraits() {
    try {
        const response = await fetch('/api/traits');
        state.traits = await response.json();
        renderTraitList();
    } catch (error) {
        console.error('Failed to fetch traits:', error);
    }
}

async function createTrait(word, opposite, numQuestions, batchSize) {
    try {
        const response = await fetch('/api/traits', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                word,
                opposite: opposite || null,
                num_questions: numQuestions,
                batch_size: batchSize,
            }),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to create trait');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Failed to create trait:', error);
        throw error;
    }
}

async function deleteTrait(name) {
    try {
        const response = await fetch(`/api/traits/${name}`, { method: 'DELETE' });
        if (response.ok) {
            await fetchTraits();
            // Remove from active if present
            delete state.activeTraits[name];
            renderMixer();
        }
    } catch (error) {
        console.error('Failed to delete trait:', error);
    }
}

async function loadModel() {
    showLoading('Loading model... This may take a minute.');
    try {
        const response = await fetch('/api/load-model', { method: 'POST' });
        const data = await response.json();
        await fetchStatus();
        // Only hide loading if we're not in the middle of creating a persona
        if (!state.creatingPersona) {
            hideLoading();
        }
        return data.status === 'loaded' || data.status === 'already_loaded';
    } catch (error) {
        console.error('Failed to load model:', error);
        if (!state.creatingPersona) {
            hideLoading();
        }
        return false;
    }
}

async function createPersona(name, description, numPrompts, numQuestions, batchSize, useJudge) {
    try {
        const response = await fetch('/api/personas', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                name, 
                description, 
                num_prompts: numPrompts,
                num_questions: numQuestions,
                batch_size: batchSize,
                use_judge: useJudge,
            }),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to create persona');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Failed to create persona:', error);
        throw error;
    }
}

async function deletePersona(name) {
    try {
        const response = await fetch(`/api/personas/${name}`, { method: 'DELETE' });
        if (response.ok) {
            await fetchPersonas();
            // Remove from active if present
            delete state.activePersonas[name];
            renderMixer();
        }
    } catch (error) {
        console.error('Failed to delete persona:', error);
    }
}

async function sendMessage(message) {
    const personas = {};
    const traits = {};
    
    // Include all non-zero personas (positive or negative)
    for (const [name, weight] of Object.entries(state.activePersonas)) {
        if (weight !== 0) {
            personas[name] = weight / 100; // Convert to decimal (e.g., 100% -> 1.0, -50% -> -0.5)
        }
    }
    
    // Include all non-zero traits (positive or negative)
    for (const [name, weight] of Object.entries(state.activeTraits)) {
        if (weight !== 0) {
            traits[name] = weight / 100;
        }
    }
    
    // Use WebSocket if available
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({
            message,
            personas,
            traits,
        }));
        return;
    }
    
    // Fallback to REST API
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                personas,
                traits,
            }),
        });
        
        const data = await response.json();
        addMessage('assistant', data.response);
    } catch (error) {
        console.error('Failed to send message:', error);
        addMessage('assistant', 'Error: Failed to get response. Please try again.');
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// WEBSOCKET
// ═══════════════════════════════════════════════════════════════════════════

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/chat`;
    
    state.ws = new WebSocket(wsUrl);
    
    state.ws.onopen = () => {
        console.log('WebSocket connected');
    };
    
    state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
            case 'status':
                if (data.status === 'generating') {
                    state.currentResponse = '';
                    addMessage('assistant', '', true); // Add empty message for streaming
                }
                break;
                
            case 'chunk':
                state.currentResponse += data.content;
                updateStreamingMessage(state.currentResponse);
                break;
                
            case 'done':
                finalizeStreamingMessage();
                elements.sendBtn.disabled = false;
                elements.chatInput.disabled = false;
                break;
                
            case 'error':
                finalizeStreamingMessage();
                addMessage('assistant', `Error: ${data.error}`);
                elements.sendBtn.disabled = false;
                elements.chatInput.disabled = false;
                break;
        }
    };
    
    state.ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting...');
        setTimeout(connectWebSocket, 3000);
    };
    
    state.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// UI RENDERING
// ═══════════════════════════════════════════════════════════════════════════

function renderPersonaList() {
    if (state.personas.length === 0) {
        elements.personaList.innerHTML = `
            <p class="persona-empty">No personas available. Create one to get started.</p>
        `;
        return;
    }
    
    elements.personaList.innerHTML = state.personas.map(persona => `
        <div class="persona-card ${state.activePersonas[persona.name] !== undefined ? 'active' : ''}" 
             data-name="${persona.name}">
            <button class="persona-delete" data-name="${persona.name}" title="Delete persona">×</button>
            <div class="persona-name">${persona.name.replace(/_/g, ' ')}</div>
            <div class="persona-desc">${persona.description || 'No description'}</div>
            ${persona.traits.length > 0 ? `
                <div class="persona-traits">
                    ${persona.traits.slice(0, 3).map(t => `<span class="trait-tag">${t}</span>`).join('')}
                </div>
            ` : ''}
        </div>
    `).join('');
    
    // Add click handlers for cards
    elements.personaList.querySelectorAll('.persona-card').forEach(card => {
        card.addEventListener('click', (e) => {
            // Don't toggle if clicking delete button
            if (e.target.classList.contains('persona-delete')) return;
            togglePersona(card.dataset.name);
        });
    });
    
    // Add click handlers for delete buttons
    elements.personaList.querySelectorAll('.persona-delete').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const name = btn.dataset.name;
            if (confirm(`Delete persona "${name.replace(/_/g, ' ')}"?`)) {
                await deletePersona(name);
            }
        });
    });
}

function renderTraitList() {
    if (state.traits.length === 0) {
        elements.traitList.innerHTML = `
            <p class="trait-empty">No traits extracted. Create one to get started.</p>
        `;
        return;
    }
    
    elements.traitList.innerHTML = state.traits.map(trait => `
        <div class="trait-card ${state.activeTraits[trait.name] !== undefined ? 'active' : ''}" 
             data-name="${trait.name}">
            <button class="trait-delete" data-name="${trait.name}" title="Delete trait">×</button>
            <div class="trait-name">${trait.name}</div>
            ${trait.opposite ? `<div class="trait-opposite">${trait.opposite}</div>` : ''}
        </div>
    `).join('');
    
    // Add click handlers for cards
    elements.traitList.querySelectorAll('.trait-card').forEach(card => {
        card.addEventListener('click', (e) => {
            if (e.target.classList.contains('trait-delete')) return;
            toggleTrait(card.dataset.name);
        });
    });
    
    // Add click handlers for delete buttons
    elements.traitList.querySelectorAll('.trait-delete').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const name = btn.dataset.name;
            if (confirm(`Delete trait "${name}"?`)) {
                await deleteTrait(name);
            }
        });
    });
}

function renderMixer() {
    const activePersonaNames = Object.keys(state.activePersonas);
    const activeTraitNames = Object.keys(state.activeTraits);
    
    if (activePersonaNames.length === 0 && activeTraitNames.length === 0) {
        elements.mixerSliders.innerHTML = `
            <p class="mixer-empty">Click personas or traits to add them to the mix</p>
        `;
        elements.blendVisual.innerHTML = `
            <div class="blend-empty">No vectors selected</div>
        `;
        elements.blendStatus.textContent = 'None';
        return;
    }
    
    let html = '';
    
    // Render persona sliders
    if (activePersonaNames.length > 0) {
        html += `<div class="mixer-section-label personas">Personas</div>`;
        html += activePersonaNames.map(name => {
            const value = state.activePersonas[name];
            const absValue = Math.abs(value);
            const sign = value >= 0 ? '+' : '−';
            const signClass = value >= 0 ? 'positive' : 'negative';
            return `
            <div class="slider-group persona" data-name="${name}" data-type="persona">
                <div class="slider-header">
                    <span class="slider-name">${name.replace(/_/g, ' ')}</span>
                    <button class="sign-toggle ${signClass}" data-name="${name}" data-type="persona" title="Toggle push toward/away">${sign}</button>
                    <span class="slider-value">${absValue}%</span>
                    <button class="slider-remove" data-name="${name}" data-type="persona">×</button>
                </div>
                <input type="range" min="0" max="300" value="${absValue}" 
                       data-name="${name}" data-type="persona" />
            </div>
        `}).join('');
    }
    
    // Render trait sliders
    if (activeTraitNames.length > 0) {
        html += `<div class="mixer-section-label traits">Traits</div>`;
        html += activeTraitNames.map(name => {
            const value = state.activeTraits[name];
            const absValue = Math.abs(value);
            const sign = value >= 0 ? '+' : '−';
            const signClass = value >= 0 ? 'positive' : 'negative';
            return `
            <div class="slider-group trait" data-name="${name}" data-type="trait">
                <div class="slider-header">
                    <span class="slider-name">${name}</span>
                    <button class="sign-toggle ${signClass}" data-name="${name}" data-type="trait" title="Toggle push toward/away">${sign}</button>
                    <span class="slider-value">${absValue}%</span>
                    <button class="slider-remove" data-name="${name}" data-type="trait">×</button>
                </div>
                <input type="range" min="0" max="300" value="${absValue}" 
                       data-name="${name}" data-type="trait" />
            </div>
        `}).join('');
    }
    
    elements.mixerSliders.innerHTML = html;
    
    // Add slider handlers for both personas and traits
    elements.mixerSliders.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.addEventListener('input', (e) => {
            const name = e.target.dataset.name;
            const type = e.target.dataset.type;
            const absValue = parseInt(e.target.value);
            const stateObj = type === 'trait' ? state.activeTraits : state.activePersonas;
            const currentSign = stateObj[name] >= 0 ? 1 : -1;
            stateObj[name] = absValue * currentSign;
            e.target.parentElement.querySelector('.slider-value').textContent = `${absValue}%`;
            renderBlendVisual();
            updateBlendStatus();
        });
    });
    
    // Add sign toggle handlers
    elements.mixerSliders.querySelectorAll('.sign-toggle').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const name = btn.dataset.name;
            const type = btn.dataset.type;
            const stateObj = type === 'trait' ? state.activeTraits : state.activePersonas;
            stateObj[name] = -stateObj[name];
            renderMixer();
        });
    });
    
    // Add remove handlers
    elements.mixerSliders.querySelectorAll('.slider-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const name = btn.dataset.name;
            const type = btn.dataset.type;
            if (type === 'trait') {
                removeTraitFromMix(name);
            } else {
                removePersonaFromMix(name);
            }
        });
    });
    
    renderBlendVisual();
    updateBlendStatus();
}

function renderBlendVisual() {
    const activePersonaNames = Object.keys(state.activePersonas);
    const activeTraitNames = Object.keys(state.activeTraits);
    const hasActivePersonas = activePersonaNames.some(name => state.activePersonas[name] !== 0);
    const hasActiveTraits = activeTraitNames.some(name => state.activeTraits[name] !== 0);
    
    if (!hasActivePersonas && !hasActiveTraits) {
        elements.blendVisual.innerHTML = `
            <div class="blend-empty">Adjust sliders to see blend</div>
        `;
        return;
    }
    
    // Combine all active vectors for visualization
    const allPositive = [];
    const allNegative = [];
    
    // Add personas
    activePersonaNames.forEach(name => {
        const val = state.activePersonas[name];
        if (val > 0) allPositive.push({ name, val, type: 'persona' });
        else if (val < 0) allNegative.push({ name, val: Math.abs(val), type: 'persona' });
    });
    
    // Add traits
    activeTraitNames.forEach(name => {
        const val = state.activeTraits[name];
        if (val > 0) allPositive.push({ name, val, type: 'trait' });
        else if (val < 0) allNegative.push({ name, val: Math.abs(val), type: 'trait' });
    });
    
    let html = '';
    
    if (allPositive.length > 0) {
        const posTotal = allPositive.reduce((sum, item) => sum + item.val, 0);
        const posSegments = allPositive.map(item => {
            const percent = (item.val / posTotal * 100).toFixed(0);
            const colorClass = item.type === 'trait' ? 'style="background: var(--emerald-light)"' : '';
            return `<div class="blend-segment positive" ${colorClass} style="flex: ${item.val}">${percent}%</div>`;
        }).join('');
        html += `<div class="blend-bar positive">${posSegments}</div>`;
    }
    
    if (allNegative.length > 0) {
        const negTotal = allNegative.reduce((sum, item) => sum + item.val, 0);
        const negSegments = allNegative.map(item => {
            const percent = (item.val / negTotal * 100).toFixed(0);
            return `<div class="blend-segment negative" style="flex: ${item.val}">${percent}%</div>`;
        }).join('');
        html += `<div class="blend-label">Push away from:</div><div class="blend-bar negative">${negSegments}</div>`;
    }
    
    elements.blendVisual.innerHTML = html || '<div class="blend-empty">Adjust sliders to see blend</div>';
}

function updateBlendStatus() {
    const activePersonaNames = Object.keys(state.activePersonas).filter(name => state.activePersonas[name] !== 0);
    const activeTraitNames = Object.keys(state.activeTraits).filter(name => state.activeTraits[name] !== 0);
    const totalActive = activePersonaNames.length + activeTraitNames.length;
    
    if (totalActive === 0) {
        elements.blendStatus.textContent = 'None';
        return;
    }
    
    // Show single item if only one
    if (totalActive === 1) {
        if (activePersonaNames.length === 1) {
            const name = activePersonaNames[0];
            const value = state.activePersonas[name];
            const prefix = value < 0 ? '−' : '';
            elements.blendStatus.textContent = prefix + name.replace(/_/g, ' ');
        } else {
            const name = activeTraitNames[0];
            const value = state.activeTraits[name];
            const prefix = value < 0 ? '−' : '';
            elements.blendStatus.textContent = prefix + name + ' (trait)';
        }
    } else {
        // Count positive and negative across both
        let posCount = 0, negCount = 0;
        activePersonaNames.forEach(name => {
            if (state.activePersonas[name] > 0) posCount++;
            else negCount++;
        });
        activeTraitNames.forEach(name => {
            if (state.activeTraits[name] > 0) posCount++;
            else negCount++;
        });
        
        if (negCount > 0 && posCount > 0) {
            elements.blendStatus.textContent = `${posCount}+ / ${negCount}−`;
        } else {
            const parts = [];
            if (activePersonaNames.length > 0) parts.push(`${activePersonaNames.length}P`);
            if (activeTraitNames.length > 0) parts.push(`${activeTraitNames.length}T`);
            elements.blendStatus.textContent = parts.join(' + ');
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CHAT FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

function addMessage(role, content, isStreaming = false) {
    // Remove welcome message if present
    const welcome = elements.chatMessages.querySelector('.chat-welcome');
    if (welcome) {
        welcome.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    if (isStreaming) {
        messageDiv.classList.add('streaming');
    }
    
    messageDiv.innerHTML = `
        <div class="message-sender">${role === 'user' ? 'You' : 'Assistant'}</div>
        <div class="message-content">${formatMessage(content)}</div>
    `;
    
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function updateStreamingMessage(content) {
    const streamingMessage = elements.chatMessages.querySelector('.message.streaming');
    if (streamingMessage) {
        streamingMessage.querySelector('.message-content').innerHTML = formatMessage(content);
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }
}

function finalizeStreamingMessage() {
    const streamingMessage = elements.chatMessages.querySelector('.message.streaming');
    if (streamingMessage) {
        streamingMessage.classList.remove('streaming');
    }
}

function formatMessage(content) {
    // Basic markdown-like formatting
    return content
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>');
}

function clearChat() {
    elements.chatMessages.innerHTML = `
        <div class="chat-welcome">
            <div class="welcome-decoration"></div>
            <p>Welcome to Persona Steering</p>
            <p class="welcome-sub">Select personas from the left panel, adjust the mix on the right, then start chatting.</p>
            <div class="welcome-decoration"></div>
        </div>
    `;
}

// ═══════════════════════════════════════════════════════════════════════════
// PERSONA MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════

function togglePersona(name) {
    if (state.activePersonas[name] !== undefined) {
        // Already active - remove from mix
        delete state.activePersonas[name];
    } else {
        // Add to mix with default 50%
        state.activePersonas[name] = 50;
    }
    
    renderPersonaList();
    renderMixer();
}

function removePersonaFromMix(name) {
    delete state.activePersonas[name];
    renderPersonaList();
    renderMixer();
}

function toggleTrait(name) {
    if (state.activeTraits[name] !== undefined) {
        delete state.activeTraits[name];
    } else {
        state.activeTraits[name] = 50;
    }
    
    renderTraitList();
    renderMixer();
}

function removeTraitFromMix(name) {
    delete state.activeTraits[name];
    renderTraitList();
    renderMixer();
}

// ═══════════════════════════════════════════════════════════════════════════
// MODAL
// ═══════════════════════════════════════════════════════════════════════════

function showModal() {
    elements.createModal.classList.add('active');
    elements.personaName.value = '';
    elements.personaDescription.value = '';
    elements.personaPrompts.value = '5';
    elements.personaName.focus();
}

function hideModal() {
    elements.createModal.classList.remove('active');
}

function showTraitModal() {
    elements.createTraitModal.classList.add('active');
    elements.traitWord.value = '';
    elements.traitOpposite.value = '';
    elements.traitQuestions.value = '30';
    elements.traitWord.focus();
}

function hideTraitModal() {
    elements.createTraitModal.classList.remove('active');
}

async function handleCreateTrait() {
    const word = elements.traitWord.value.trim().toLowerCase();
    const opposite = elements.traitOpposite.value.trim().toLowerCase();
    const numQuestions = parseInt(elements.traitQuestions.value) || 30;
    const batchSize = parseInt(elements.traitBatch.value) || 8;
    
    if (!word) {
        alert('Please enter a trait word');
        return;
    }
    
    hideTraitModal();
    const oppositeText = opposite || 'auto-generated opposite';
    showLoading(`Extracting trait "${word}" ↔ "${oppositeText}"...\nUsing ${numQuestions} questions with contrastive prompts`);
    
    try {
        if (!state.modelLoaded) {
            elements.loadingText.textContent = 'Loading model first...';
            await loadModel();
        }
        
        elements.loadingText.textContent = `Extracting "${word}" trait...\nThis uses positive/negative prompts for contrastive extraction.`;
        
        await createTrait(word, opposite || null, numQuestions, batchSize);
        
        // Poll for completion
        const pollInterval = setInterval(async () => {
            const status = await fetchStatus();
            if (!status || !status.creating_trait) {
                clearInterval(pollInterval);
                state.creatingTrait = null;
                hideLoading();
                await fetchTraits();
            }
        }, 2000);
        
    } catch (error) {
        state.creatingTrait = null;
        hideLoading();
        alert(`Failed to extract trait: ${error.message}`);
    }
}

function switchTab(tabName) {
    state.currentTab = tabName;
    
    // Update tab buttons
    elements.subTabs.forEach(tab => {
        if (tab.dataset.tab === tabName) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });
    
    // Update tab content
    if (tabName === 'personas') {
        elements.personasTab.classList.add('active');
        elements.traitsTab.classList.remove('active');
    } else {
        elements.personasTab.classList.remove('active');
        elements.traitsTab.classList.add('active');
    }
}

async function handleCreatePersona() {
    const name = elements.personaName.value.trim().toLowerCase().replace(/\s+/g, '_');
    const description = elements.personaDescription.value.trim();
    const numPrompts = parseInt(elements.personaPrompts.value) || 5;
    const numQuestions = parseInt(elements.personaQuestions.value) || 40;
    const batchSize = parseInt(elements.personaBatch.value) || 8;
    const useJudge = elements.personaUseJudge.checked;
    
    if (!name) {
        alert('Please enter a persona name');
        return;
    }
    
    if (!description) {
        alert('Please enter a character description');
        return;
    }
    
    hideModal();
    const totalGenerations = numPrompts * numQuestions;
    const judgeText = useJudge ? ' with LLM judge' : '';
    showLoading(`Creating persona "${name}"${judgeText}...\n${totalGenerations} extractions (${numPrompts} prompts × ${numQuestions} questions)`);
    
    try {
        // First ensure model is loaded
        if (!state.modelLoaded) {
            elements.loadingText.textContent = 'Loading model first...';
            await loadModel();
        }
        
        elements.loadingText.textContent = `Creating "${name}"${judgeText}...\nExtracting from ${totalGenerations} prompt combinations.`;
        
        await createPersona(name, description, numPrompts, numQuestions, batchSize, useJudge);
        
        // Poll for completion
        const pollInterval = setInterval(async () => {
            const status = await fetchStatus();
            if (!status || !status.creating_persona) {
                clearInterval(pollInterval);
                state.creatingPersona = null;
                hideLoading();
                await fetchPersonas();
            }
        }, 2000);
        
    } catch (error) {
        state.creatingPersona = null;
        hideLoading();
        alert(`Failed to create persona: ${error.message}`);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LOADING OVERLAY
// ═══════════════════════════════════════════════════════════════════════════

function showLoading(text = 'Loading...') {
    elements.loadingText.textContent = text;
    elements.loadingOverlay.classList.add('active');
}

function hideLoading() {
    elements.loadingOverlay.classList.remove('active');
}

// ═══════════════════════════════════════════════════════════════════════════
// EVENT HANDLERS
// ═══════════════════════════════════════════════════════════════════════════

function setupEventListeners() {
    // Create persona button
    elements.createPersonaBtn.addEventListener('click', showModal);
    
    // Create trait button
    elements.createTraitBtn.addEventListener('click', showTraitModal);
    
    // Tab switching
    elements.subTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            switchTab(tab.dataset.tab);
        });
    });
    
    // New chat button
    elements.newChatBtn.addEventListener('click', clearChat);
    
    // Steering scale controls
    elements.scaleUpBtn.addEventListener('click', () => {
        const newScale = Math.min(2.0, state.steeringScale + 0.1);
        setSteeringScale(newScale);
    });
    elements.scaleDownBtn.addEventListener('click', () => {
        const newScale = Math.max(0.01, state.steeringScale - 0.1);
        setSteeringScale(newScale);
    });
    
    // Persona modal buttons
    elements.modalCancel.addEventListener('click', hideModal);
    elements.modalCreate.addEventListener('click', handleCreatePersona);
    elements.createModal.querySelector('.modal-backdrop').addEventListener('click', hideModal);
    
    // Trait modal buttons
    elements.traitModalCancel.addEventListener('click', hideTraitModal);
    elements.traitModalCreate.addEventListener('click', handleCreateTrait);
    elements.createTraitModal.querySelector('.modal-backdrop').addEventListener('click', hideTraitModal);
    
    // Chat input
    elements.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });
    
    elements.chatInput.addEventListener('input', () => {
        elements.sendBtn.disabled = !elements.chatInput.value.trim();
        
        // Auto-resize
        elements.chatInput.style.height = 'auto';
        elements.chatInput.style.height = Math.min(elements.chatInput.scrollHeight, 150) + 'px';
    });
    
    elements.sendBtn.addEventListener('click', handleSendMessage);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            hideModal();
            hideTraitModal();
        }
    });
}

async function handleSendMessage() {
    const message = elements.chatInput.value.trim();
    if (!message) return;
    
    // Check if model is loaded
    if (!state.modelLoaded) {
        const loaded = await loadModel();
        if (!loaded) {
            alert('Failed to load model. Please try again.');
            return;
        }
    }
    
    // Disable input while processing
    elements.sendBtn.disabled = true;
    elements.chatInput.disabled = true;
    
    // Add user message
    addMessage('user', message);
    elements.chatInput.value = '';
    elements.chatInput.style.height = 'auto';
    
    // Send message
    await sendMessage(message);
    
    // Re-enable if using REST API (WebSocket handles its own re-enable)
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
        elements.sendBtn.disabled = false;
        elements.chatInput.disabled = false;
    }
    
    elements.chatInput.focus();
}

// ═══════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

async function init() {
    setupEventListeners();
    
    // Fetch initial data
    await fetchStatus();
    await fetchPersonas();
    await fetchTraits();
    
    // Connect WebSocket
    connectWebSocket();
    
    // Poll for status updates
    setInterval(fetchStatus, 5000);
}

// Start the app
document.addEventListener('DOMContentLoaded', init);
