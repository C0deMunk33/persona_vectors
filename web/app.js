/**
 * Persona Steering Web UI
 * Vanilla JS frontend for persona management, mixing, and chat
 */

// ═══════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════

const state = {
    personas: [],
    activePersonas: {}, // name -> weight (0-100)
    modelLoaded: false,
    creatingPersona: null,
    ws: null,
    currentResponse: '',
    steeringScale: 0.1,
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
    
    // Modal
    createModal: document.getElementById('create-modal'),
    personaName: document.getElementById('persona-name'),
    personaDescription: document.getElementById('persona-description'),
    personaPrompts: document.getElementById('persona-prompts'),
    personaQuestions: document.getElementById('persona-questions'),
    personaBatch: document.getElementById('persona-batch'),
    personaUseJudge: document.getElementById('persona-use-judge'),
    modalCancel: document.getElementById('modal-cancel'),
    modalCreate: document.getElementById('modal-create'),
    
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
        state.steeringScale = data.steering_scale || 0.1;
        
        // Update status displays
        elements.modelStatus.textContent = data.model_loaded ? 'Ready' : 
            (data.model_loading ? 'Loading...' : 'Not Loaded');
        elements.modelStatus.className = 'status-value' + 
            (data.model_loaded ? ' ready' : '');
        
        // Update steering scale display
        updateSteeringScaleDisplay();
        
        // Show creating persona status
        if (data.creating_persona) {
            showLoading(`Creating persona: ${data.creating_persona}...`);
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
    
    // Include all non-zero personas (positive or negative)
    for (const [name, weight] of Object.entries(state.activePersonas)) {
        if (weight !== 0) {
            personas[name] = weight / 100; // Convert to decimal (e.g., 100% -> 1.0, -50% -> -0.5)
        }
    }
    
    // Use WebSocket if available
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({
            message,
            personas,
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

function renderMixer() {
    const activeNames = Object.keys(state.activePersonas);
    
    if (activeNames.length === 0) {
        elements.mixerSliders.innerHTML = `
            <p class="mixer-empty">Click personas to add them to the mix</p>
        `;
        elements.blendVisual.innerHTML = `
            <div class="blend-empty">No personas selected</div>
        `;
        elements.blendStatus.textContent = 'None';
        return;
    }
    
    // Render sliders with sign toggle
    elements.mixerSliders.innerHTML = activeNames.map(name => {
        const value = state.activePersonas[name];
        const absValue = Math.abs(value);
        const sign = value >= 0 ? '+' : '−';
        const signClass = value >= 0 ? 'positive' : 'negative';
        return `
        <div class="slider-group" data-name="${name}">
            <div class="slider-header">
                <span class="slider-name">${name.replace(/_/g, ' ')}</span>
                <button class="sign-toggle ${signClass}" data-name="${name}" title="Toggle push toward/away">${sign}</button>
                <span class="slider-value">${absValue}%</span>
                <button class="slider-remove" data-name="${name}">×</button>
            </div>
            <input type="range" min="0" max="300" value="${absValue}" 
                   data-name="${name}" />
        </div>
    `}).join('');
    
    // Add slider handlers
    elements.mixerSliders.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.addEventListener('input', (e) => {
            const name = e.target.dataset.name;
            const absValue = parseInt(e.target.value);
            const currentSign = state.activePersonas[name] >= 0 ? 1 : -1;
            state.activePersonas[name] = absValue * currentSign;
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
            // Flip the sign
            state.activePersonas[name] = -state.activePersonas[name];
            // Re-render to update toggle appearance
            renderMixer();
        });
    });
    
    // Add remove handlers
    elements.mixerSliders.querySelectorAll('.slider-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            removePersonaFromMix(btn.dataset.name);
        });
    });
    
    renderBlendVisual();
    updateBlendStatus();
}

function renderBlendVisual() {
    const activeNames = Object.keys(state.activePersonas);
    const hasActivePersonas = activeNames.some(name => state.activePersonas[name] !== 0);
    
    if (!hasActivePersonas) {
        elements.blendVisual.innerHTML = `
            <div class="blend-empty">Adjust sliders to see blend</div>
        `;
        return;
    }
    
    // Split into positive (push toward) and negative (push away)
    const positiveNames = activeNames.filter(name => state.activePersonas[name] > 0);
    const negativeNames = activeNames.filter(name => state.activePersonas[name] < 0);
    
    let html = '';
    
    if (positiveNames.length > 0) {
        const posTotal = positiveNames.reduce((sum, name) => sum + state.activePersonas[name], 0);
        const posSegments = positiveNames.map(name => {
            const percent = (state.activePersonas[name] / posTotal * 100).toFixed(0);
            return `<div class="blend-segment positive" style="flex: ${state.activePersonas[name]}">${percent}%</div>`;
        }).join('');
        html += `<div class="blend-bar positive">${posSegments}</div>`;
    }
    
    if (negativeNames.length > 0) {
        const negTotal = negativeNames.reduce((sum, name) => sum + Math.abs(state.activePersonas[name]), 0);
        const negSegments = negativeNames.map(name => {
            const percent = (Math.abs(state.activePersonas[name]) / negTotal * 100).toFixed(0);
            return `<div class="blend-segment negative" style="flex: ${Math.abs(state.activePersonas[name])}">${percent}%</div>`;
        }).join('');
        html += `<div class="blend-label">Push away from:</div><div class="blend-bar negative">${negSegments}</div>`;
    }
    
    elements.blendVisual.innerHTML = html || '<div class="blend-empty">Adjust sliders to see blend</div>';
}

function updateBlendStatus() {
    const activeNames = Object.keys(state.activePersonas).filter(name => state.activePersonas[name] !== 0);
    
    if (activeNames.length === 0) {
        elements.blendStatus.textContent = 'None';
        return;
    }
    
    // Show sign for single persona if negative
    if (activeNames.length === 1) {
        const name = activeNames[0];
        const value = state.activePersonas[name];
        const prefix = value < 0 ? '−' : '';
        elements.blendStatus.textContent = prefix + name.replace(/_/g, ' ');
    } else {
        const posCount = activeNames.filter(name => state.activePersonas[name] > 0).length;
        const negCount = activeNames.filter(name => state.activePersonas[name] < 0).length;
        if (negCount > 0 && posCount > 0) {
            elements.blendStatus.textContent = `${posCount}+ / ${negCount}−`;
        } else {
            elements.blendStatus.textContent = `${activeNames.length} personas`;
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
    
    // Modal buttons
    elements.modalCancel.addEventListener('click', hideModal);
    elements.modalCreate.addEventListener('click', handleCreatePersona);
    elements.createModal.querySelector('.modal-backdrop').addEventListener('click', hideModal);
    
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
    
    // Connect WebSocket
    connectWebSocket();
    
    // Poll for status updates
    setInterval(fetchStatus, 5000);
}

// Start the app
document.addEventListener('DOMContentLoaded', init);
