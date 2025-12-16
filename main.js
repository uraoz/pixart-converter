/**
 * Pixel Art Converter - Main Script
 * UI control and real-time adjustments
 */

// ============================================================================
// State
// ============================================================================

let originalImage = null;      // Original loaded image
let downscaledData = null;     // After downscale (base for adjustments)
let adjustedData = null;       // After brightness/contrast/saturation
let convertedData = null;      // After palette mapping
let worker = null;

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
  // File input
  dropZone: document.getElementById('dropZone'),
  fileInput: document.getElementById('fileInput'),

  // Canvases
  originalCanvas: document.getElementById('originalCanvas'),
  previewCanvas: document.getElementById('previewCanvas'),
  previewPlaceholder: document.getElementById('previewPlaceholder'),

  // Info
  originalInfo: document.getElementById('originalInfo'),
  previewInfo: document.getElementById('previewInfo'),

  // Controls
  pixelSize: document.getElementById('pixelSize'),
  pixelSizeValue: document.getElementById('pixelSizeValue'),
  palette: document.getElementById('palette'),
  customPaletteGroup: document.getElementById('customPaletteGroup'),
  customPalette: document.getElementById('customPalette'),
  palettePreview: document.getElementById('palettePreview'),
  useCiede2000: document.getElementById('useCiede2000'),
  scaleUp: document.getElementById('scaleUp'),
  preprocess: document.getElementById('preprocess'),
  gamma: document.getElementById('gamma'),
  gammaValue: document.getElementById('gammaValue'),
  brightness: document.getElementById('brightness'),
  brightnessValue: document.getElementById('brightnessValue'),
  contrast: document.getElementById('contrast'),
  contrastValue: document.getElementById('contrastValue'),
  saturation: document.getElementById('saturation'),
  saturationValue: document.getElementById('saturationValue'),
  resetAdjustments: document.getElementById('resetAdjustments'),
  adjustmentsSectionTitle: document.getElementById('adjustmentsSectionTitle'),

  // Actions
  convertBtn: document.getElementById('convertBtn'),
  downloadBtn: document.getElementById('downloadBtn'),

  // Progress
  progressSection: document.getElementById('progressSection'),
  progressFill: document.getElementById('progressFill'),
  progressText: document.getElementById('progressText'),
};

// ============================================================================
// Palettes (duplicated from worker.js for realtime preview)
// ============================================================================

const PALETTES = {
  pico8: [
    [0, 0, 0], [29, 43, 83], [126, 37, 83], [0, 135, 81],
    [171, 82, 54], [95, 87, 79], [194, 195, 199], [255, 241, 232],
    [255, 0, 77], [255, 163, 0], [255, 236, 39], [0, 228, 54],
    [41, 173, 255], [131, 118, 156], [255, 119, 168], [255, 204, 170],
  ],
  lost_century: [
    [209, 177, 135], [199, 123, 88], [174, 93, 64], [121, 68, 74],
    [75, 61, 68], [186, 145, 88], [146, 116, 65], [77, 69, 57],
    [119, 116, 59], [179, 165, 85], [210, 201, 165], [140, 171, 161],
    [75, 114, 110], [87, 72, 82], [132, 120, 117], [171, 155, 142],
  ],
  sunset8: [
    [255, 255, 120], [255, 214, 71], [255, 194, 71], [255, 169, 54],
    [255, 139, 111], [230, 117, 149], [154, 99, 144], [70, 70, 120],
  ],
  twilight5: [
    [251, 187, 173], [238, 134, 149], [74, 122, 150],
    [51, 63, 88], [41, 40, 49],
  ],
  hollow: [
    [15, 15, 27], [86, 90, 117], [198, 183, 190], [250, 251, 246],
  ],
  gameboy: [
    [15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15],
  ],
  endesga32: [
    [190, 74, 47], [215, 118, 67], [234, 212, 170], [228, 166, 114],
    [184, 111, 80], [115, 62, 57], [62, 39, 49], [162, 38, 51],
    [228, 59, 68], [247, 118, 34], [254, 174, 52], [254, 231, 97],
    [99, 199, 77], [62, 137, 72], [38, 92, 66], [25, 60, 62],
    [18, 78, 137], [0, 153, 219], [44, 232, 245], [192, 203, 220],
    [139, 155, 180], [90, 105, 136], [58, 68, 102], [38, 43, 68],
    [24, 20, 37], [255, 0, 68], [104, 56, 108], [181, 80, 136],
    [246, 117, 122], [232, 183, 150], [194, 133, 105], [140, 94, 88],
  ],
};

// Fast RGB-based palette mapping for realtime preview
function applyPaletteRgbFast(imageData, palette) {
  if (!palette || palette.length === 0) return imageData;

  const data = imageData.data;
  const numPixels = data.length / 4;

  for (let i = 0; i < numPixels; i++) {
    const idx = i * 4;
    const r = data[idx];
    const g = data[idx + 1];
    const b = data[idx + 2];

    // Find nearest palette color (RGB Euclidean)
    let minDist = Infinity;
    let nearest = palette[0];

    for (const color of palette) {
      const dist = (r - color[0]) ** 2 + (g - color[1]) ** 2 + (b - color[2]) ** 2;
      if (dist < minDist) {
        minDist = dist;
        nearest = color;
      }
    }

    data[idx] = nearest[0];
    data[idx + 1] = nearest[1];
    data[idx + 2] = nearest[2];
  }

  return imageData;
}

// Get current palette based on selection
function getCurrentPalette() {
  const paletteName = elements.palette.value;

  if (paletteName === 'none') {
    return null;
  } else if (paletteName === 'custom') {
    const colors = parseCustomPalette(elements.customPalette.value);
    return colors.length > 0 ? colors : null;
  } else if (PALETTES[paletteName]) {
    return PALETTES[paletteName];
  }

  return null;
}

// ============================================================================
// Custom Palette
// ============================================================================

function parseCustomPalette(text) {
  if (!text.trim()) return [];

  // Split by comma, space, newline, or any combination
  const tokens = text.split(/[\s,]+/).filter(t => t.trim());
  const colors = [];

  for (const token of tokens) {
    const color = parseColorCode(token.trim());
    if (color) {
      colors.push(color);
    }
  }

  return colors;
}

function parseColorCode(code) {
  // Remove # if present
  code = code.replace(/^#/, '');

  // Validate hex format
  if (!/^[0-9A-Fa-f]{6}$/.test(code)) {
    return null;
  }

  const r = parseInt(code.slice(0, 2), 16);
  const g = parseInt(code.slice(2, 4), 16);
  const b = parseInt(code.slice(4, 6), 16);

  return [r, g, b];
}

function updatePalettePreview() {
  const colors = parseCustomPalette(elements.customPalette.value);
  elements.palettePreview.innerHTML = '';

  for (const color of colors) {
    const div = document.createElement('div');
    div.className = 'palette-color';
    div.style.backgroundColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    div.title = `#${color.map(c => c.toString(16).padStart(2, '0')).join('').toUpperCase()}`;
    elements.palettePreview.appendChild(div);
  }
}

function toggleCustomPalette() {
  const isCustom = elements.palette.value === 'custom';
  elements.customPaletteGroup.hidden = !isCustom;

  if (isCustom) {
    updatePalettePreview();
  }
}

// ============================================================================
// Realtime Preview State
// ============================================================================

function isRealtimePreviewEnabled() {
  const useCiede2000 = elements.useCiede2000.checked;
  const preprocess = elements.preprocess.value;
  return !useCiede2000 && preprocess === 'none';
}

function updateRealtimeLabel() {
  const enabled = isRealtimePreviewEnabled();
  if (enabled) {
    elements.adjustmentsSectionTitle.textContent = '調整（リアルタイム）';
  } else {
    elements.adjustmentsSectionTitle.textContent = '調整（変換ボタンで反映）';
  }
}

// ============================================================================
// Worker Setup
// ============================================================================

function initWorker() {
  worker = new Worker('worker.js');

  worker.onmessage = function (e) {
    const { type, imageData, progress, error, message } = e.data;

    switch (type) {
      case 'progress':
        updateProgress(progress);
        break;

      case 'log':
        console.log('[Worker]', message);
        break;

      case 'complete':
        convertedData = imageData;
        displayConverted();
        finishConversion();
        break;

      case 'error':
        console.error('[Worker Error]', error);
        finishConversion();
        alert('変換エラー: ' + error);
        break;
    }
  };
}

// ============================================================================
// Image Loading
// ============================================================================

function loadImage(file) {
  const reader = new FileReader();

  reader.onload = function (e) {
    const img = new Image();

    img.onload = function () {
      originalImage = img;
      displayOriginal();
      processImage();

      elements.dropZone.classList.add('hidden');
      elements.previewPlaceholder.classList.add('hidden');
      elements.convertBtn.disabled = false;
    };

    img.src = e.target.result;
  };

  reader.readAsDataURL(file);
}

function displayOriginal() {
  const ctx = elements.originalCanvas.getContext('2d');

  // Fit canvas to image while respecting max size
  const maxSize = 400;
  let width = originalImage.width;
  let height = originalImage.height;

  if (width > maxSize || height > maxSize) {
    const scale = Math.min(maxSize / width, maxSize / height);
    width *= scale;
    height *= scale;
  }

  elements.originalCanvas.width = width;
  elements.originalCanvas.height = height;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(originalImage, 0, 0, width, height);

  elements.originalInfo.textContent = `${originalImage.width} × ${originalImage.height}`;
}

// ============================================================================
// Image Processing (Real-time)
// ============================================================================

function processImage() {
  if (!originalImage) return;

  const pixelSize = parseInt(elements.pixelSize.value);

  // Downscale
  const newW = Math.max(1, Math.floor(originalImage.width / pixelSize));
  const newH = Math.max(1, Math.floor(originalImage.height / pixelSize));

  // Create offscreen canvas for downscaling
  const offscreen = document.createElement('canvas');
  offscreen.width = newW;
  offscreen.height = newH;
  const offCtx = offscreen.getContext('2d');
  offCtx.imageSmoothingEnabled = true;
  offCtx.imageSmoothingQuality = 'high';
  offCtx.drawImage(originalImage, 0, 0, newW, newH);

  downscaledData = offCtx.getImageData(0, 0, newW, newH);

  // Apply adjustments
  applyAdjustments();
}

function applyAdjustments() {
  if (!downscaledData) return;

  // Check if realtime preview is enabled (heavy processing is disabled)
  const realtimeEnabled = isRealtimePreviewEnabled();

  // Clone downscaled data
  adjustedData = new ImageData(
    new Uint8ClampedArray(downscaledData.data),
    downscaledData.width,
    downscaledData.height
  );

  const brightness = parseInt(elements.brightness.value);
  const contrast = parseInt(elements.contrast.value);
  const saturation = parseInt(elements.saturation.value);

  const data = adjustedData.data;

  // Pre-calculate factors
  const brightFactor = brightness / 100 + 1;
  const contrastFactor = contrast / 100 + 1;
  const contrastOffset = 128 * (1 - contrastFactor);
  const satFactor = saturation / 100 + 1;

  for (let i = 0; i < data.length; i += 4) {
    let r = data[i];
    let g = data[i + 1];
    let b = data[i + 2];

    // Brightness
    if (brightness !== 0) {
      r *= brightFactor;
      g *= brightFactor;
      b *= brightFactor;
    }

    // Contrast
    if (contrast !== 0) {
      r = r * contrastFactor + contrastOffset;
      g = g * contrastFactor + contrastOffset;
      b = b * contrastFactor + contrastOffset;
    }

    // Saturation
    if (saturation !== 0) {
      const gray = 0.3086 * r + 0.6094 * g + 0.0820 * b;
      r = satFactor * r + (1 - satFactor) * gray;
      g = satFactor * g + (1 - satFactor) * gray;
      b = satFactor * b + (1 - satFactor) * gray;
    }

    data[i] = Math.max(0, Math.min(255, r));
    data[i + 1] = Math.max(0, Math.min(255, g));
    data[i + 2] = Math.max(0, Math.min(255, b));
  }

  // Display preview only if realtime preview is enabled
  if (realtimeEnabled) {
    // In realtime mode, also apply palette conversion
    const palette = getCurrentPalette();

    // Clone adjustedData for palette conversion
    convertedData = new ImageData(
      new Uint8ClampedArray(adjustedData.data),
      adjustedData.width,
      adjustedData.height
    );

    // Apply palette if selected
    if (palette) {
      applyPaletteRgbFast(convertedData, palette);
    }

    displayPreview(convertedData);
    elements.downloadBtn.disabled = false;
  } else {
    // Show message that real-time preview is disabled
    elements.previewInfo.textContent = `${adjustedData.width} × ${adjustedData.height} (変換ボタンで更新)`;
    // Clear converted data when adjustments change in non-realtime mode
    convertedData = null;
    elements.downloadBtn.disabled = true;
  }
}

function displayPreview(imageData) {
  const ctx = elements.previewCanvas.getContext('2d');

  // Scale up for display
  const scale = Math.min(
    400 / imageData.width,
    400 / imageData.height,
    8
  );

  const displayW = Math.floor(imageData.width * scale);
  const displayH = Math.floor(imageData.height * scale);

  elements.previewCanvas.width = displayW;
  elements.previewCanvas.height = displayH;

  // Draw scaled with nearest neighbor
  const offscreen = document.createElement('canvas');
  offscreen.width = imageData.width;
  offscreen.height = imageData.height;
  offscreen.getContext('2d').putImageData(imageData, 0, 0);

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, 0, 0, displayW, displayH);

  elements.previewInfo.textContent = `${imageData.width} × ${imageData.height}`;
}

function displayConverted() {
  if (!convertedData) return;

  displayPreview(convertedData);
  elements.downloadBtn.disabled = false;
}

// ============================================================================
// Conversion (Heavy Processing via Worker)
// ============================================================================

function startConversion() {
  if (!adjustedData || !worker) return;

  const paletteName = elements.palette.value;
  const useCiede2000 = elements.useCiede2000.checked;
  const preprocess = elements.preprocess.value;
  const gamma = parseFloat(elements.gamma.value);

  // Handle custom palette
  let customColors = null;
  if (paletteName === 'custom') {
    customColors = parseCustomPalette(elements.customPalette.value);
    if (customColors.length === 0) {
      alert('有効なカラーコードを入力してください');
      return;
    }
  }

  // Show progress
  elements.progressSection.hidden = false;
  elements.progressFill.style.width = '0%';
  elements.progressText.textContent = '処理中...';
  elements.convertBtn.classList.add('loading');
  elements.convertBtn.disabled = true;

  // Clone adjusted data for worker
  const dataToProcess = new ImageData(
    new Uint8ClampedArray(adjustedData.data),
    adjustedData.width,
    adjustedData.height
  );

  worker.postMessage({
    type: 'convert',
    imageData: dataToProcess,
    options: {
      paletteName: paletteName === 'none' ? null : paletteName,
      customColors,
      useCiede2000,
      preprocess,
      gamma,
    }
  });
}

function updateProgress(progress) {
  const percent = Math.round(progress * 100);
  elements.progressFill.style.width = percent + '%';
  elements.progressText.textContent = `処理中... ${percent}%`;
}

function finishConversion() {
  elements.progressSection.hidden = true;
  elements.convertBtn.classList.remove('loading');
  elements.convertBtn.disabled = false;
}

// ============================================================================
// Download
// ============================================================================

function downloadImage() {
  if (!convertedData) return;

  const scaleUp = elements.scaleUp.checked;
  let finalData = convertedData;

  // Scale up to original size if requested
  if (scaleUp && originalImage) {
    const offscreen = document.createElement('canvas');
    offscreen.width = convertedData.width;
    offscreen.height = convertedData.height;
    offscreen.getContext('2d').putImageData(convertedData, 0, 0);

    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = originalImage.width;
    finalCanvas.height = originalImage.height;
    const ctx = finalCanvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, 0, 0, originalImage.width, originalImage.height);

    finalCanvas.toBlob(function (blob) {
      downloadBlob(blob);
    }, 'image/png');
  } else {
    const canvas = document.createElement('canvas');
    canvas.width = convertedData.width;
    canvas.height = convertedData.height;
    canvas.getContext('2d').putImageData(convertedData, 0, 0);

    canvas.toBlob(function (blob) {
      downloadBlob(blob);
    }, 'image/png');
  }
}

function downloadBlob(blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'pixel_art.png';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ============================================================================
// Event Handlers
// ============================================================================

function setupEventListeners() {
  // File input
  elements.dropZone.addEventListener('click', () => elements.fileInput.click());
  elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) loadImage(e.target.files[0]);
  });

  // Drag and drop
  elements.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.dropZone.classList.add('dragover');
  });
  elements.dropZone.addEventListener('dragleave', () => {
    elements.dropZone.classList.remove('dragover');
  });
  elements.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.dropZone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) loadImage(e.dataTransfer.files[0]);
  });

  // Pixel size (triggers reprocess)
  elements.pixelSize.addEventListener('input', () => {
    elements.pixelSizeValue.textContent = elements.pixelSize.value;
    processImage();
  });

  // Palette selection
  elements.palette.addEventListener('change', () => {
    toggleCustomPalette();
    applyAdjustments(); // Update realtime preview with new palette
  });

  // Custom palette input
  elements.customPalette.addEventListener('input', () => {
    updatePalettePreview();
    applyAdjustments(); // Update realtime preview with custom palette changes
  });

  // Real-time adjustments
  elements.brightness.addEventListener('input', () => {
    elements.brightnessValue.textContent = elements.brightness.value;
    applyAdjustments();
  });
  elements.contrast.addEventListener('input', () => {
    elements.contrastValue.textContent = elements.contrast.value;
    applyAdjustments();
  });
  elements.saturation.addEventListener('input', () => {
    elements.saturationValue.textContent = elements.saturation.value;
    applyAdjustments();
  });

  // Gamma display update
  elements.gamma.addEventListener('input', () => {
    elements.gammaValue.textContent = elements.gamma.value;
  });

  // CIEDE2000 and preprocess changes - update realtime preview state
  elements.useCiede2000.addEventListener('change', () => {
    updateRealtimeLabel();
    applyAdjustments();
  });
  elements.preprocess.addEventListener('change', () => {
    updateRealtimeLabel();
    applyAdjustments();
  });

  // Reset adjustments
  elements.resetAdjustments.addEventListener('click', () => {
    elements.brightness.value = 0;
    elements.brightnessValue.textContent = '0';
    elements.contrast.value = 0;
    elements.contrastValue.textContent = '0';
    elements.saturation.value = 0;
    elements.saturationValue.textContent = '0';
    applyAdjustments();
  });

  // Convert button
  elements.convertBtn.addEventListener('click', startConversion);

  // Download button
  elements.downloadBtn.addEventListener('click', downloadImage);
}

// ============================================================================
// Initialization
// ============================================================================

function init() {
  initWorker();
  setupEventListeners();
  console.log('Pixel Art Converter initialized');
}

// Start
init();
