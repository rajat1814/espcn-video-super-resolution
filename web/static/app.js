/**
 * ESPCN Video Super-Resolution & Artifact Suppression Web Controller
 * Zero-dependency, pure vanilla ES6 application.
 */

// Global State
const state = {
  currentMode: 'photo', // 'photo' | 'video' | 'live' | 'benchmark'
  addCompression: false,
  compressionQuality: 30,
  comparisonView: 'sideBySide', // 'sideBySide' | 'bicubic'
  
  // Current data
  currentRawImage: null,
  currentModifiedImage: null,
  currentBicubicImage: null,
  currentRawBlob: null,
  currentSrBlob: null,
  
  // Video state
  activeVideoTaskId: null,
  videoPollInterval: null,
  
  // Live state
  isStreaming: false,
  webcamStream: null,
  liveSocket: null,
  liveFrameCounter: 0,
  liveTimer: null,
  
  // Event stream
  eventCount: 0,
};

// DOM Elements
const DOM = {
  // Navigation
  tabs: {
    photo: document.getElementById('tabPhoto'),
    video: document.getElementById('tabVideo'),
    live: document.getElementById('tabLive'),
    benchmark: document.getElementById('tabBenchmark'),
  },
  statusPulse: document.getElementById('statusPulse'),
  statusText: document.getElementById('statusText'),
  hardwareBadge: document.getElementById('hardwareBadge'),
  
  // Raw Box Elements
  rawCard: document.getElementById('rawCard'),
  rawViewport: document.getElementById('rawViewport'),
  rawEmptyState: document.getElementById('rawEmptyState'),
  rawImage: document.getElementById('rawImage'),
  rawVideo: document.getElementById('rawVideo'),
  rawResSubtitle: document.getElementById('rawResSubtitle'),
  rawTypeTag: document.getElementById('rawTypeTag'),
  liveCaptureCanvas: document.getElementById('liveCaptureCanvas'),
  
  // Modified Box Elements
  modifiedCard: document.getElementById('modifiedCard'),
  modifiedViewport: document.getElementById('modifiedViewport'),
  modifiedEmptyState: document.getElementById('modifiedEmptyState'),
  modifiedImage: document.getElementById('modifiedImage'),
  modifiedVideo: document.getElementById('modifiedVideo'),
  modifiedResSubtitle: document.getElementById('modifiedResSubtitle'),
  latencyTag: document.getElementById('latencyTag'),
  fpsTag: document.getElementById('fpsTag'),
  processingOverlay: document.getElementById('processingOverlay'),
  
  // Compression Controls
  addCompressionCheckbox: document.getElementById('addCompressionCheckbox'),
  compressionSliderPanel: document.getElementById('compressionSliderPanel'),
  compressionSlider: document.getElementById('compressionSlider'),
  compressionValLabel: document.getElementById('compressionValLabel'),
  presetChips: document.querySelectorAll('.preset-chip'),
  
  // Mode Action Bars
  photoActions: document.getElementById('photoActions'),
  videoActions: document.getElementById('videoActions'),
  liveActions: document.getElementById('liveActions'),
  
  // Photo Controls
  photoFileInput: document.getElementById('photoFileInput'),
  btnUploadPhoto: document.getElementById('btnUploadPhoto'),
  btnSamplePhoto: document.getElementById('btnSamplePhoto'),
  btnSavePhoto: document.getElementById('btnSavePhoto'),
  
  // Video Controls
  videoFileInput: document.getElementById('videoFileInput'),
  btnUploadVideo: document.getElementById('btnUploadVideo'),
  btnSaveVideo: document.getElementById('btnSaveVideo'),
  videoProgressWrap: document.getElementById('videoProgressWrap'),
  videoProgressBar: document.getElementById('videoProgressBar'),
  videoProgressText: document.getElementById('videoProgressText'),
  
  // Live Controls
  btnToggleWebcam: document.getElementById('btnToggleWebcam'),
  btnToggleSynthetic: document.getElementById('btnToggleSynthetic'),
  liveIndicator: document.getElementById('liveIndicator'),
  btnCaptureLiveFrame: document.getElementById('btnCaptureLiveFrame'),
  
  // Comparison buttons
  btnViewSideBySide: document.getElementById('btnViewSideBySide'),
  btnViewBicubic: document.getElementById('btnViewBicubic'),
  
  // Benchmark Modal
  benchmarkModal: document.getElementById('benchmarkModal'),
  btnCloseBenchmark: document.getElementById('btnCloseBenchmark'),
  btnRunBenchmark: document.getElementById('btnRunBenchmark'),
  benchmarkSpinner: document.getElementById('benchmarkSpinner'),
  benchmarkResultsWrap: document.getElementById('benchmarkResultsWrap'),
  benchmarkTableBody: document.getElementById('benchmarkTableBody'),
  
  // Telemetry Drawer
  drawerToggle: document.getElementById('drawerToggle'),
  drawerArrow: document.getElementById('drawerArrow'),
  drawerContent: document.getElementById('drawerContent'),
  logStream: document.getElementById('logStream'),
  eventCountBadge: document.getElementById('eventCountBadge'),
  modelMetaInfo: document.getElementById('modelMetaInfo'),
};

// ============================================================================
// Initialization & Mode Switching
// ============================================================================

function init() {
  setupNavigation();
  setupCompressionControls();
  setupPhotoHandlers();
  setupVideoHandlers();
  setupLiveHandlers();
  setupBenchmarkModal();
  setupTelemetryDrawer();
  setupDragAndDrop();
  initEventStream();
  fetchInitialStatus();
}

function switchMode(newMode) {
  if (state.currentMode === newMode) return;
  
  // If leaving live mode, shut down webcam
  if (state.currentMode === 'live' && state.isStreaming) {
    stopLiveStreaming();
  }
  
  state.currentMode = newMode;
  
  // Update Tab Styling
  Object.keys(DOM.tabs).forEach(m => {
    DOM.tabs[m].classList.toggle('active', m === newMode);
  });
  
  // Update Action Bar Visibility
  DOM.photoActions.classList.toggle('hidden', newMode !== 'photo');
  DOM.videoActions.classList.toggle('hidden', newMode !== 'video');
  DOM.liveActions.classList.toggle('hidden', newMode !== 'live');
  
  // Update Save Buttons Visibility
  DOM.btnSavePhoto.classList.toggle('hidden', newMode !== 'photo');
  DOM.btnSaveVideo.classList.toggle('hidden', newMode !== 'video');
  DOM.btnCaptureLiveFrame.classList.toggle('hidden', newMode !== 'live');
  
  resetViewports();
  appendLog(`Switched mode to: ${newMode.toUpperCase()}`, 'info');
}

function resetViewports() {
  DOM.rawEmptyState.classList.remove('hidden');
  DOM.modifiedEmptyState.classList.remove('hidden');
  DOM.rawImage.classList.add('hidden');
  DOM.rawVideo.classList.add('hidden');
  DOM.modifiedImage.classList.add('hidden');
  DOM.modifiedVideo.classList.add('hidden');
  DOM.processingOverlay.classList.add('hidden');
  
  DOM.rawResSubtitle.textContent = 'Awaiting input...';
  DOM.modifiedResSubtitle.textContent = '2× Super-Resolution';
  DOM.latencyTag.textContent = '-- ms';
  DOM.fpsTag.textContent = '-- FPS';
  
  DOM.btnSavePhoto.disabled = true;
  DOM.btnSaveVideo.disabled = true;
  DOM.btnCaptureLiveFrame.disabled = true;
}

// ============================================================================
// Compression Controls (Directly below Raw Box)
// ============================================================================

function setupCompressionControls() {
  DOM.addCompressionCheckbox.addEventListener('change', (e) => {
    state.addCompression = e.target.checked;
    DOM.compressionSliderPanel.classList.toggle('hidden', !state.addCompression);
    
    appendLog(
      state.addCompression
        ? `Network compression ENABLED (JPEG Q=${state.compressionQuality})`
        : 'Network compression DISABLED (Clean input)',
      state.addCompression ? 'warning' : 'info'
    );
    
    // Auto-reprocess photo if one is currently loaded
    if (state.currentMode === 'photo' && state.currentRawBlob) {
      processPhotoFile(state.currentRawBlob);
    }
  });

  DOM.compressionSlider.addEventListener('input', (e) => {
    state.compressionQuality = parseInt(e.target.value, 10);
    updateCompressionLabel();
  });

  DOM.compressionSlider.addEventListener('change', () => {
    if (state.currentMode === 'photo' && state.currentRawBlob) {
      processPhotoFile(state.currentRawBlob);
    }
  });

  DOM.presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      DOM.presetChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const q = parseInt(chip.dataset.q, 10);
      DOM.compressionSlider.value = q;
      state.compressionQuality = q;
      updateCompressionLabel();
      if (state.currentMode === 'photo' && state.currentRawBlob) {
        processPhotoFile(state.currentRawBlob);
      }
    });
  });
}

function updateCompressionLabel() {
  let descriptor = 'Mild';
  if (state.compressionQuality <= 15) descriptor = 'Extreme Packet Loss';
  else if (state.compressionQuality <= 30) descriptor = 'Heavy Streaming';
  else if (state.compressionQuality <= 60) descriptor = 'Moderate';
  DOM.compressionValLabel.textContent = `JPEG Q=${state.compressionQuality} (${descriptor})`;
}

// ============================================================================
// Photo Mode Handlers
// ============================================================================

function setupPhotoHandlers() {
  DOM.btnUploadPhoto.addEventListener('click', () => DOM.photoFileInput.click());
  DOM.photoFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      processPhotoFile(e.target.files[0]);
    }
  });

  DOM.btnSamplePhoto.addEventListener('click', loadSamplePhoto);

  DOM.btnSavePhoto.addEventListener('click', () => {
    if (!state.currentModifiedImage) return;
    downloadDataUrl(state.currentModifiedImage, 'espcn_2x_upscaled.jpg');
    appendLog('Saved upscaled photo to downloads.', 'success');
  });

  DOM.btnViewSideBySide.addEventListener('click', () => setComparisonView('sideBySide'));
  DOM.btnViewBicubic.addEventListener('click', () => setComparisonView('bicubic'));
}

function setComparisonView(view) {
  state.comparisonView = view;
  DOM.btnViewSideBySide.classList.toggle('active', view === 'sideBySide');
  DOM.btnViewBicubic.classList.toggle('active', view === 'bicubic');

  if (view === 'bicubic' && state.currentBicubicImage) {
    DOM.rawImage.src = state.currentBicubicImage;
    DOM.rawTypeTag.textContent = 'Bicubic 2× Baseline';
    DOM.rawTypeTag.style.color = '#fbbf24';
  } else if (state.currentRawImage) {
    DOM.rawImage.src = state.currentRawImage;
    DOM.rawTypeTag.textContent = state.addCompression ? 'Compressed Input' : 'Raw Input';
    DOM.rawTypeTag.style.color = '#cbd5e1';
  }
}

async function processPhotoFile(file) {
  state.currentRawBlob = file;
  DOM.processingOverlay.classList.remove('hidden');
  
  const formData = new FormData();
  formData.append('file', file);
  formData.append('add_compression', state.addCompression ? 'true' : 'false');
  formData.append('compression_quality', state.compressionQuality.toString());

  try {
    const res = await fetch('/api/photo/upscale', {
      method: 'POST',
      body: formData,
    });
    
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Photo upscaling failed');
    }
    
    const data = await res.json();
    
    // Store images
    state.currentRawImage = data.raw_image;
    state.currentModifiedImage = data.modified_image;
    state.currentBicubicImage = data.bicubic_image;
    
    // Render Raw Image
    DOM.rawEmptyState.classList.add('hidden');
    DOM.rawImage.classList.remove('hidden');
    DOM.rawImage.src = data.raw_image;
    DOM.rawResSubtitle.textContent = `Input: ${data.input_resolution}`;
    DOM.rawTypeTag.textContent = data.compression_applied
      ? `Compressed (Q=${data.compression_quality})`
      : 'Raw';

    // Render Modified Image
    DOM.modifiedEmptyState.classList.add('hidden');
    DOM.modifiedImage.classList.remove('hidden');
    DOM.modifiedImage.src = data.modified_image;
    DOM.modifiedResSubtitle.textContent = `ESPCN 2×: ${data.output_resolution}`;
    DOM.latencyTag.textContent = `${data.latency_ms} ms`;
    DOM.fpsTag.textContent = `${data.fps} FPS`;

    // Enable Save
    DOM.btnSavePhoto.disabled = false;
    
    appendLog(
      `Photo processed: ${data.input_resolution} -> ${data.output_resolution} in ${data.latency_ms}ms (${data.fps} FPS)`,
      'success'
    );
  } catch (err) {
    appendLog(`Error processing photo: ${err.message}`, 'error');
  } finally {
    DOM.processingOverlay.classList.add('hidden');
  }
}

async function loadSamplePhoto() {
  // Generate a high-contrast synthetic test face with text
  const canvas = document.createElement('canvas');
  canvas.width = 320;
  canvas.height = 180;
  const ctx = canvas.getContext('2d');

  // Background gradient
  const grad = ctx.createLinearGradient(0, 0, 320, 180);
  grad.addColorStop(0, '#1e293b');
  grad.addColorStop(1, '#0f172a');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 320, 180);

  // Draw face oval
  ctx.fillStyle = '#fbcfe8';
  ctx.beginPath();
  ctx.ellipse(160, 90, 50, 65, 0, 0, Math.PI * 2);
  ctx.fill();

  // Eyes & Features
  ctx.fillStyle = '#1e1b4b';
  ctx.beginPath();
  ctx.arc(142, 80, 5, 0, Math.PI * 2);
  ctx.arc(178, 80, 5, 0, Math.PI * 2);
  ctx.fill();

  // Smile
  ctx.strokeStyle = '#db2777';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(160, 95, 22, 0.2, Math.PI - 0.2);
  ctx.stroke();

  // High-frequency detail: Sharp text
  ctx.font = 'bold 12px Inter, sans-serif';
  ctx.fillStyle = '#38bdf8';
  ctx.fillText('ESPCN 21K TEST PATTERN', 80, 25);
  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px monospace';
  ctx.fillText('CRF=38 / Q=30 Codec Check', 85, 168);

  canvas.toBlob(blob => {
    processPhotoFile(blob);
  }, 'image/jpeg', 0.95);
}

// ============================================================================
// Video Mode Handlers
// ============================================================================

function setupVideoHandlers() {
  DOM.btnUploadVideo.addEventListener('click', () => DOM.videoFileInput.click());
  DOM.videoFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      uploadAndProcessVideo(e.target.files[0]);
    }
  });

  DOM.btnSaveVideo.addEventListener('click', () => {
    if (state.lastVideoDownloadUrl) {
      window.open(state.lastVideoDownloadUrl, '_blank');
      appendLog('Downloading processed video.', 'success');
    }
  });
}

async function uploadAndProcessVideo(file) {
  DOM.videoProgressWrap.classList.remove('hidden');
  DOM.videoProgressBar.style.width = '0%';
  DOM.videoProgressText.textContent = 'Uploading...';
  DOM.btnUploadVideo.disabled = true;

  // Show raw preview video
  const localUrl = URL.createObjectURL(file);
  DOM.rawEmptyState.classList.add('hidden');
  DOM.rawVideo.classList.remove('hidden');
  DOM.rawVideo.src = localUrl;
  DOM.rawVideo.loop = true;
  DOM.rawVideo.play().catch(() => {});

  const formData = new FormData();
  formData.append('file', file);
  formData.append('add_compression', state.addCompression ? 'true' : 'false');
  formData.append('compression_quality', state.compressionQuality.toString());

  try {
    const res = await fetch('/api/video/upload', {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Video upload failed');
    const data = await res.json();
    state.activeVideoTaskId = data.task_id;
    appendLog(`Video processing queued. Task ID: ${data.task_id}`, 'info');
    pollVideoProgress(data.task_id);
  } catch (err) {
    appendLog(`Video upload error: ${err.message}`, 'error');
    DOM.btnUploadVideo.disabled = false;
    DOM.videoProgressWrap.classList.add('hidden');
  }
}

function pollVideoProgress(taskId) {
  if (state.videoPollInterval) clearInterval(state.videoPollInterval);

  state.videoPollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/api/video/task/${taskId}`);
      if (!res.ok) return;
      const task = await res.json();

      DOM.videoProgressBar.style.width = `${task.progress}%`;
      DOM.videoProgressText.textContent = `${task.progress}% (${task.current_frame}/${task.total_frames})`;

      if (task.status === 'completed') {
        clearInterval(state.videoPollInterval);
        DOM.videoProgressText.textContent = '100% Done';
        DOM.btnUploadVideo.disabled = false;
        DOM.btnSaveVideo.disabled = false;
        state.lastVideoDownloadUrl = task.output_url;

        // Render modified video
        DOM.modifiedEmptyState.classList.add('hidden');
        DOM.modifiedVideo.classList.remove('hidden');
        DOM.modifiedVideo.src = task.output_url;
        DOM.modifiedVideo.loop = true;
        DOM.modifiedVideo.play().catch(() => {});

        appendLog('Video upscaling finished successfully!', 'success');
      } else if (task.status === 'failed') {
        clearInterval(state.videoPollInterval);
        DOM.btnUploadVideo.disabled = false;
        appendLog(`Video processing failed: ${task.error}`, 'error');
      }
    } catch (e) {
      console.error(e);
    }
  }, 600);
}

// ============================================================================
// Live Webcam Streaming Mode
// ============================================================================

function setupLiveHandlers() {
  DOM.btnToggleWebcam.addEventListener('click', () => {
    if (state.isStreaming) {
      stopLiveStreaming();
    } else {
      startLiveStreaming();
    }
  });

  DOM.btnToggleSynthetic.addEventListener('click', () => {
    if (state.isStreaming) {
      stopLiveStreaming();
    } else {
      startSyntheticStreaming();
    }
  });

  DOM.btnCaptureLiveFrame.addEventListener('click', captureLiveFrame);
}

let syntheticAnimId = null;
let syntheticFrameCounter = 0;

function startSyntheticStreaming() {
  state.isSynthetic = true;
  state.isStreaming = true;

  DOM.rawEmptyState.classList.add('hidden');
  DOM.rawVideo.classList.add('hidden');
  DOM.rawImage.classList.remove('hidden');

  DOM.modifiedEmptyState.classList.add('hidden');
  DOM.modifiedImage.classList.remove('hidden');

  DOM.btnToggleSynthetic.textContent = 'Stop Synthetic Stream';
  DOM.btnToggleSynthetic.classList.replace('secondary', 'primary');
  DOM.btnToggleWebcam.disabled = true;
  DOM.liveIndicator.classList.remove('hidden');
  DOM.btnCaptureLiveFrame.disabled = false;

  appendLog('Started Synthetic 30 FPS Stream (No physical webcam needed)', 'success');
  connectLiveWebSocket();
}

async function startLiveStreaming() {
  state.isSynthetic = false;
  try {
    appendLog('Requesting webcam permission...', 'info');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 320 }, height: { ideal: 180 }, frameRate: { ideal: 30 } },
      audio: false,
    });

    state.webcamStream = stream;
    DOM.rawVideo.srcObject = stream;
    DOM.rawEmptyState.classList.add('hidden');
    DOM.rawImage.classList.add('hidden');
    DOM.rawVideo.classList.remove('hidden');
    DOM.rawVideo.play();

    DOM.modifiedEmptyState.classList.add('hidden');
    DOM.modifiedImage.classList.remove('hidden');

    DOM.btnToggleWebcam.textContent = 'Stop Webcam';
    DOM.btnToggleWebcam.classList.replace('primary', 'secondary');
    DOM.btnToggleSynthetic.disabled = true;
    DOM.liveIndicator.classList.remove('hidden');
    DOM.btnCaptureLiveFrame.disabled = false;
    state.isStreaming = true;

    connectLiveWebSocket();
    appendLog('Webcam stream started at 320×180', 'success');
  } catch (err) {
    appendLog(`Webcam not available (${err.name}). You can use 'Synthetic Stream' instead!`, 'warning');
    // Prompt to use synthetic
    if (confirm("No physical webcam detected on this machine. Would you like to run the Synthetic 30 FPS Test Stream instead?")) {
      startSyntheticStreaming();
    }
  }
}

function stopLiveStreaming() {
  state.isStreaming = false;
  if (state.liveTimer) clearTimeout(state.liveTimer);
  if (syntheticAnimId) cancelAnimationFrame(syntheticAnimId);
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(t => t.stop());
    state.webcamStream = null;
  }
  if (state.liveSocket) {
    state.liveSocket.close();
    state.liveSocket = null;
  }

  DOM.rawVideo.srcObject = null;
  DOM.rawVideo.classList.add('hidden');
  DOM.rawImage.classList.add('hidden');
  DOM.rawEmptyState.classList.remove('hidden');

  DOM.btnToggleWebcam.textContent = 'Start Webcam';
  DOM.btnToggleWebcam.classList.replace('secondary', 'primary');
  DOM.btnToggleWebcam.disabled = false;

  DOM.btnToggleSynthetic.textContent = 'Synthetic Stream (No Cam)';
  DOM.btnToggleSynthetic.classList.replace('primary', 'secondary');
  DOM.btnToggleSynthetic.disabled = false;

  DOM.liveIndicator.classList.add('hidden');
  DOM.btnCaptureLiveFrame.disabled = true;

  appendLog('Live stream stopped.', 'info');
}

function connectLiveWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/live`;
  state.liveSocket = new WebSocket(wsUrl);

  state.liveSocket.onopen = () => {
    appendLog('Live stream WebSocket connected.', 'success');
    sendLiveFrameLoop();
  };

  state.liveSocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.sr_image) {
        DOM.modifiedImage.src = data.sr_image;
        DOM.latencyTag.textContent = `${data.latency_ms} ms`;
        DOM.fpsTag.textContent = `${data.fps} FPS`;
        DOM.rawResSubtitle.textContent = `Live In: ${data.in_res}`;
        DOM.modifiedResSubtitle.textContent = `ESPCN 2×: ${data.out_res}`;
      }
    } catch (e) {
      console.error(e);
    }
  };

  state.liveSocket.onerror = (err) => {
    appendLog('WebSocket error encountered', 'error');
  };

  state.liveSocket.onclose = () => {
    if (state.isStreaming) {
      appendLog('WebSocket connection closed.', 'warning');
    }
  };
}

function sendLiveFrameLoop() {
  if (!state.isStreaming || !state.liveSocket || state.liveSocket.readyState !== WebSocket.OPEN) {
    return;
  }

  const canvas = DOM.liveCaptureCanvas;
  const ctx = canvas.getContext('2d');
  canvas.width = 320;
  canvas.height = 180;

  if (state.isSynthetic) {
    // Generate animated synthetic talking-head frame
    syntheticFrameCounter++;
    const t = syntheticFrameCounter * 0.05;

    // Dark background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, 320, 180);

    // Grid lines to see spatial resolution
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let x = 0; x < 320; x += 20) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 180); ctx.stroke(); }
    for (let y = 0; y < 180; y += 20) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(320, y); ctx.stroke(); }

    // Head
    const headX = 160 + Math.sin(t * 0.7) * 12;
    const headY = 90 + Math.cos(t * 0.5) * 6;
    ctx.fillStyle = '#fbcfe8';
    ctx.beginPath();
    ctx.ellipse(headX, headY, 46, 58, 0, 0, Math.PI * 2);
    ctx.fill();

    // Eyes
    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.arc(headX - 16, headY - 10, 4, 0, Math.PI * 2);
    ctx.arc(headX + 16, headY - 10, 4, 0, Math.PI * 2);
    ctx.fill();

    // Animated mouth (speaking effect)
    const mouthHeight = Math.abs(Math.sin(t * 2)) * 14 + 2;
    ctx.fillStyle = '#db2777';
    ctx.beginPath();
    ctx.ellipse(headX, headY + 22, 12, mouthHeight, 0, 0, Math.PI * 2);
    ctx.fill();

    // Live clock and metadata
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.fillStyle = '#38bdf8';
    ctx.fillText('SYNTHETIC 30 FPS LIVE STREAM', 14, 22);

    ctx.font = '10px monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(`Frame: ${syntheticFrameCounter} | ${new Date().toISOString().slice(11, 23)}`, 14, 168);

    const frameData = canvas.toDataURL('image/jpeg', 0.85);
    DOM.rawImage.src = frameData;

    const payload = JSON.stringify({
      frame: frameData,
      add_compression: state.addCompression,
      quality: state.compressionQuality,
    });
    state.liveSocket.send(payload);
  } else {
    // Physical webcam extraction
    const video = DOM.rawVideo;
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      ctx.drawImage(video, 0, 0, 320, 180);
      const frameData = canvas.toDataURL('image/jpeg', 0.85);
      const payload = JSON.stringify({
        frame: frameData,
        add_compression: state.addCompression,
        quality: state.compressionQuality,
      });
      state.liveSocket.send(payload);
    }
  }

  // Schedule next frame (~30 FPS)
  state.liveTimer = setTimeout(() => {
    if (state.isStreaming) requestAnimationFrame(sendLiveFrameLoop);
  }, 33);
}

function captureLiveFrame() {
  if (DOM.modifiedImage.src) {
    downloadDataUrl(DOM.modifiedImage.src, `live_capture_${Date.now()}.jpg`);
    appendLog('Snapshot frame captured and downloaded.', 'success');
  }
}

// ============================================================================
// Benchmark Modal
// ============================================================================

function setupBenchmarkModal() {
  DOM.tabs.benchmark.addEventListener('click', () => {
    DOM.benchmarkModal.classList.remove('hidden');
  });

  DOM.btnCloseBenchmark.addEventListener('click', () => {
    DOM.benchmarkModal.classList.add('hidden');
  });

  DOM.benchmarkModal.addEventListener('click', (e) => {
    if (e.target === DOM.benchmarkModal) {
      DOM.benchmarkModal.classList.add('hidden');
    }
  });

  DOM.btnRunBenchmark.addEventListener('click', executeBenchmark);
}

async function executeBenchmark() {
  DOM.btnRunBenchmark.disabled = true;
  DOM.benchmarkSpinner.classList.remove('hidden');
  DOM.benchmarkResultsWrap.classList.add('hidden');
  appendLog('Running inference benchmark across hardware providers...', 'info');

  try {
    const res = await fetch('/api/benchmark', { method: 'POST' });
    const data = await res.json();

    DOM.benchmarkTableBody.innerHTML = '';
    data.results.forEach(item => {
      const tr = document.createElement('tr');
      if (item.success) {
        tr.innerHTML = `
          <td><strong>${item.target}</strong></td>
          <td><code>${item.provider}</code></td>
          <td style="color: var(--primary-accent);">${item.median_ms} ms</td>
          <td style="color: var(--success-accent); font-weight:600;">${item.fps} FPS</td>
          <td><span style="color:#10b981;">✓ Operational</span></td>
        `;
      } else {
        tr.innerHTML = `
          <td><strong>${item.target}</strong></td>
          <td><code>${item.provider}</code></td>
          <td colspan="2" style="color: var(--danger-accent);">${item.error}</td>
          <td><span style="color:#ef4444;">✗ Unavailable</span></td>
        `;
      }
      DOM.benchmarkTableBody.appendChild(tr);
    });

    DOM.benchmarkResultsWrap.classList.remove('hidden');
    appendLog('Benchmark completed successfully.', 'success');
  } catch (err) {
    appendLog(`Benchmark execution error: ${err.message}`, 'error');
  } finally {
    DOM.btnRunBenchmark.disabled = false;
    DOM.benchmarkSpinner.classList.add('hidden');
  }
}

// ============================================================================
// Telemetry Drawer & SSE Events
// ============================================================================

function setupTelemetryDrawer() {
  DOM.drawerToggle.addEventListener('click', () => {
    const isCollapsed = DOM.drawerContent.classList.toggle('collapsed');
    DOM.drawerArrow.classList.toggle('open', !isCollapsed);
  });
}

function appendLog(message, level = 'info') {
  state.eventCount++;
  DOM.eventCountBadge.textContent = `${state.eventCount} events`;

  const entry = document.createElement('div');
  entry.className = `log-entry ${level}`;

  const timeSpan = document.createElement('span');
  timeSpan.className = 'log-time';
  timeSpan.textContent = `[${new Date().toLocaleTimeString()}]`;

  const textSpan = document.createElement('span');
  textSpan.className = 'log-text';
  textSpan.textContent = message;

  entry.appendChild(timeSpan);
  entry.appendChild(textSpan);
  DOM.logStream.appendChild(entry);

  // Auto-scroll to bottom
  DOM.drawerContent.scrollTop = DOM.drawerContent.scrollHeight;
}

function initEventStream() {
  const eventSource = new EventSource('/api/events');

  eventSource.onopen = () => {
    DOM.statusPulse.classList.add('online');
    DOM.statusText.textContent = 'Server Online';
    appendLog('Real-time event stream connected.', 'success');
  };

  eventSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      appendLog(data.message, data.level || 'info');
      if (data.type === 'connected' && data.data) {
        DOM.hardwareBadge.textContent = data.data.active_provider.replace('ExecutionProvider', '');
        DOM.modelMetaInfo.textContent = `Model: ${data.data.model_name} (${data.data.precision})`;
      }
    } catch (err) {
      console.error(err);
    }
  };

  eventSource.onerror = () => {
    DOM.statusPulse.classList.remove('online');
    DOM.statusText.textContent = 'Reconnecting...';
  };
}

async function fetchInitialStatus() {
  try {
    const res = await fetch('/api/status');
    if (res.ok) {
      const data = await res.json();
      DOM.statusPulse.classList.add('online');
      DOM.statusText.textContent = 'Server Online';
      DOM.hardwareBadge.textContent = data.active_provider.replace('ExecutionProvider', '');
      DOM.modelMetaInfo.textContent = `Model: ${data.model_name} (${data.precision})`;
    }
  } catch (e) {
    // Handled by event source
  }
}

// ============================================================================
// Drag and Drop Helper
// ============================================================================

function setupDragAndDrop() {
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
    DOM.rawViewport.addEventListener(evt, preventDefaults, false);
    document.body.addEventListener(evt, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  DOM.rawViewport.addEventListener('dragover', () => {
    DOM.rawCard.style.borderColor = 'var(--primary-accent)';
  });

  DOM.rawViewport.addEventListener('dragleave', () => {
    DOM.rawCard.style.borderColor = '';
  });

  DOM.rawViewport.addEventListener('drop', (e) => {
    DOM.rawCard.style.borderColor = '';
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files[0]) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        switchMode('photo');
        processPhotoFile(file);
      } else if (file.type.startsWith('video/')) {
        switchMode('video');
        uploadAndProcessVideo(file);
      }
    }
  });
}

function setupNavigation() {
  Object.keys(DOM.tabs).forEach(mode => {
    DOM.tabs[mode].addEventListener('click', () => switchMode(mode));
  });
}

function downloadDataUrl(dataUrl, filename) {
  const link = document.createElement('a');
  link.href = dataUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Start application on DOM Ready
document.addEventListener('DOMContentLoaded', init);
