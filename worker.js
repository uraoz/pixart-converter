/**
 * Pixel Art Converter - Worker
 * Heavy processing: CIEDE2000, preprocessing, palette mapping
 */

// ============================================================================
// Palettes
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
  yaju: [
    [161, 115, 107], [62, 52, 63], [151, 159, 155], [201, 191 ,182],
  ],
};

// ============================================================================
// Color Space Conversion
// ============================================================================

function rgbToXyz(r, g, b) {
  // Normalize to 0-1
  let rn = r / 255;
  let gn = g / 255;
  let bn = b / 255;

  // sRGB to linear
  rn = rn > 0.04045 ? Math.pow((rn + 0.055) / 1.055, 2.4) : rn / 12.92;
  gn = gn > 0.04045 ? Math.pow((gn + 0.055) / 1.055, 2.4) : gn / 12.92;
  bn = bn > 0.04045 ? Math.pow((bn + 0.055) / 1.055, 2.4) : bn / 12.92;

  // Linear RGB to XYZ (D65)
  const x = (rn * 0.4124564 + gn * 0.3575761 + bn * 0.1804375) * 100;
  const y = (rn * 0.2126729 + gn * 0.7151522 + bn * 0.0721750) * 100;
  const z = (rn * 0.0193339 + gn * 0.1191920 + bn * 0.9503041) * 100;

  return [x, y, z];
}

function xyzToLab(x, y, z) {
  // D65 reference white
  const refX = 95.047;
  const refY = 100.000;
  const refZ = 108.883;

  let xn = x / refX;
  let yn = y / refY;
  let zn = z / refZ;

  const epsilon = 216 / 24389;
  const kappa = 24389 / 27;

  xn = xn > epsilon ? Math.cbrt(xn) : (kappa * xn + 16) / 116;
  yn = yn > epsilon ? Math.cbrt(yn) : (kappa * yn + 16) / 116;
  zn = zn > epsilon ? Math.cbrt(zn) : (kappa * zn + 16) / 116;

  const L = 116 * yn - 16;
  const a = 500 * (xn - yn);
  const b = 200 * (yn - zn);

  return [L, a, b];
}

function rgbToLab(r, g, b) {
  const [x, y, z] = rgbToXyz(r, g, b);
  return xyzToLab(x, y, z);
}

function labToXyz(L, a, b) {
  const refX = 95.047;
  const refY = 100.000;
  const refZ = 108.883;

  const fy = (L + 16) / 116;
  const fx = a / 500 + fy;
  const fz = fy - b / 200;

  const epsilon = 216 / 24389;
  const kappa = 24389 / 27;

  const xr = Math.pow(fx, 3) > epsilon ? Math.pow(fx, 3) : (116 * fx - 16) / kappa;
  const yr = L > kappa * epsilon ? Math.pow((L + 16) / 116, 3) : L / kappa;
  const zr = Math.pow(fz, 3) > epsilon ? Math.pow(fz, 3) : (116 * fz - 16) / kappa;

  return [xr * refX, yr * refY, zr * refZ];
}

function xyzToRgb(x, y, z) {
  // XYZ to linear RGB
  const xn = x / 100;
  const yn = y / 100;
  const zn = z / 100;

  let r = xn * 3.2404542 + yn * -1.5371385 + zn * -0.4985314;
  let g = xn * -0.9692660 + yn * 1.8760108 + zn * 0.0415560;
  let b = xn * 0.0556434 + yn * -0.2040259 + zn * 1.0572252;

  // Linear to sRGB
  r = r > 0.0031308 ? 1.055 * Math.pow(r, 1 / 2.4) - 0.055 : 12.92 * r;
  g = g > 0.0031308 ? 1.055 * Math.pow(g, 1 / 2.4) - 0.055 : 12.92 * g;
  b = b > 0.0031308 ? 1.055 * Math.pow(b, 1 / 2.4) - 0.055 : 12.92 * b;

  return [
    Math.max(0, Math.min(255, Math.round(r * 255))),
    Math.max(0, Math.min(255, Math.round(g * 255))),
    Math.max(0, Math.min(255, Math.round(b * 255))),
  ];
}

function labToRgb(L, a, b) {
  const [x, y, z] = labToXyz(L, a, b);
  return xyzToRgb(x, y, z);
}

// ============================================================================
// CIEDE2000 Color Difference
// ============================================================================

function ciede2000(lab1, lab2) {
  const [L1, a1, b1] = lab1;
  const [L2, a2, b2] = lab2;

  const C1 = Math.sqrt(a1 * a1 + b1 * b1);
  const C2 = Math.sqrt(a2 * a2 + b2 * b2);
  const Cbar = (C1 + C2) / 2;

  const Cbar7 = Math.pow(Cbar, 7);
  const G = 0.5 * (1 - Math.sqrt(Cbar7 / (Cbar7 + Math.pow(25, 7))));

  const a1p = a1 * (1 + G);
  const a2p = a2 * (1 + G);

  const C1p = Math.sqrt(a1p * a1p + b1 * b1);
  const C2p = Math.sqrt(a2p * a2p + b2 * b2);

  let h1p = Math.atan2(b1, a1p) * (180 / Math.PI);
  if (h1p < 0) h1p += 360;
  let h2p = Math.atan2(b2, a2p) * (180 / Math.PI);
  if (h2p < 0) h2p += 360;

  const dLp = L2 - L1;
  const dCp = C2p - C1p;

  let dhp;
  if (C1p * C2p === 0) {
    dhp = 0;
  } else {
    const hdiff = h2p - h1p;
    if (Math.abs(hdiff) <= 180) {
      dhp = hdiff;
    } else if (hdiff > 180) {
      dhp = hdiff - 360;
    } else {
      dhp = hdiff + 360;
    }
  }

  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((dhp / 2) * (Math.PI / 180));

  const Lbarp = (L1 + L2) / 2;
  const Cbarp = (C1p + C2p) / 2;

  let Hbarp;
  if (C1p * C2p === 0) {
    Hbarp = h1p + h2p;
  } else {
    const hsum = h1p + h2p;
    const habsdiff = Math.abs(h1p - h2p);
    if (habsdiff <= 180) {
      Hbarp = hsum / 2;
    } else if (hsum < 360) {
      Hbarp = (hsum + 360) / 2;
    } else {
      Hbarp = (hsum - 360) / 2;
    }
  }

  const T = 1
    - 0.17 * Math.cos((Hbarp - 30) * (Math.PI / 180))
    + 0.24 * Math.cos((2 * Hbarp) * (Math.PI / 180))
    + 0.32 * Math.cos((3 * Hbarp + 6) * (Math.PI / 180))
    - 0.20 * Math.cos((4 * Hbarp - 63) * (Math.PI / 180));

  const dTheta = 30 * Math.exp(-Math.pow((Hbarp - 275) / 25, 2));

  const Cbarp7 = Math.pow(Cbarp, 7);
  const RC = 2 * Math.sqrt(Cbarp7 / (Cbarp7 + Math.pow(25, 7)));

  const Lbarp50sq = Math.pow(Lbarp - 50, 2);
  const SL = 1 + (0.015 * Lbarp50sq) / Math.sqrt(20 + Lbarp50sq);
  const SC = 1 + 0.045 * Cbarp;
  const SH = 1 + 0.015 * Cbarp * T;

  const RT = -Math.sin((2 * dTheta) * (Math.PI / 180)) * RC;

  const dE = Math.sqrt(
    Math.pow(dLp / SL, 2) +
    Math.pow(dCp / SC, 2) +
    Math.pow(dHp / SH, 2) +
    RT * (dCp / SC) * (dHp / SH)
  );

  return dE;
}

// ============================================================================
// Preprocessing Functions
// ============================================================================

function autoLevels(imageData, clipPercent = 1.0) {
  const data = imageData.data;
  const pixels = [];
  
  for (let i = 0; i < data.length; i += 4) {
    pixels.push(data[i], data[i + 1], data[i + 2]);
  }
  
  pixels.sort((a, b) => a - b);
  const lowIdx = Math.floor(pixels.length * clipPercent / 100);
  const highIdx = Math.floor(pixels.length * (100 - clipPercent) / 100);
  
  const low = pixels[lowIdx];
  const high = pixels[highIdx];
  
  if (high - low < 1) return imageData;
  
  const scale = 255 / (high - low);
  
  for (let i = 0; i < data.length; i += 4) {
    data[i] = Math.max(0, Math.min(255, (data[i] - low) * scale));
    data[i + 1] = Math.max(0, Math.min(255, (data[i + 1] - low) * scale));
    data[i + 2] = Math.max(0, Math.min(255, (data[i + 2] - low) * scale));
  }
  
  return imageData;
}

function histogramEqualization(imageData) {
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  const numPixels = width * height;
  
  // Convert to LAB and extract L channel
  const labPixels = [];
  for (let i = 0; i < data.length; i += 4) {
    labPixels.push(rgbToLab(data[i], data[i + 1], data[i + 2]));
  }
  
  // Build histogram of L values
  const hist = new Array(256).fill(0);
  for (const lab of labPixels) {
    const bin = Math.floor(lab[0] * 255 / 100);
    hist[Math.max(0, Math.min(255, bin))]++;
  }
  
  // Build CDF
  const cdf = new Array(256);
  cdf[0] = hist[0];
  for (let i = 1; i < 256; i++) {
    cdf[i] = cdf[i - 1] + hist[i];
  }
  
  // Normalize CDF
  for (let i = 0; i < 256; i++) {
    cdf[i] = (cdf[i] / numPixels) * 100;
  }
  
  // Apply equalization
  for (let i = 0; i < labPixels.length; i++) {
    const bin = Math.floor(labPixels[i][0] * 255 / 100);
    labPixels[i][0] = cdf[Math.max(0, Math.min(255, bin))];
  }
  
  // Convert back to RGB
  for (let i = 0; i < labPixels.length; i++) {
    const [r, g, b] = labToRgb(labPixels[i][0], labPixels[i][1], labPixels[i][2]);
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
  }
  
  return imageData;
}

function clahe(imageData, clipLimit = 2.0, gridSize = 8) {
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  
  // Convert to LAB
  const labPixels = [];
  for (let i = 0; i < data.length; i += 4) {
    labPixels.push(rgbToLab(data[i], data[i + 1], data[i + 2]));
  }
  
  const tileH = Math.max(1, Math.floor(height / gridSize));
  const tileW = Math.max(1, Math.floor(width / gridSize));
  
  // Process each tile
  for (let ti = 0; ti < gridSize; ti++) {
    for (let tj = 0; tj < gridSize; tj++) {
      const y1 = ti * tileH;
      const y2 = Math.min((ti + 1) * tileH, height);
      const x1 = tj * tileW;
      const x2 = Math.min((tj + 1) * tileW, width);
      
      const tileSize = (y2 - y1) * (x2 - x1);
      if (tileSize === 0) continue;
      
      // Build histogram for tile
      const hist = new Array(256).fill(0);
      for (let y = y1; y < y2; y++) {
        for (let x = x1; x < x2; x++) {
          const idx = y * width + x;
          const bin = Math.floor(labPixels[idx][0] * 255 / 100);
          hist[Math.max(0, Math.min(255, bin))]++;
        }
      }
      
      // Clip histogram
      const clipThreshold = clipLimit * tileSize / 256;
      let excess = 0;
      for (let i = 0; i < 256; i++) {
        if (hist[i] > clipThreshold) {
          excess += hist[i] - clipThreshold;
          hist[i] = clipThreshold;
        }
      }
      const redistribute = excess / 256;
      for (let i = 0; i < 256; i++) {
        hist[i] += redistribute;
      }
      
      // Build CDF
      const cdf = new Array(256);
      cdf[0] = hist[0];
      for (let i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
      }
      
      // Normalize
      const cdfMax = cdf[255];
      if (cdfMax > 0) {
        for (let i = 0; i < 256; i++) {
          cdf[i] = (cdf[i] / cdfMax) * 100;
        }
      }
      
      // Apply to tile
      for (let y = y1; y < y2; y++) {
        for (let x = x1; x < x2; x++) {
          const idx = y * width + x;
          const bin = Math.floor(labPixels[idx][0] * 255 / 100);
          labPixels[idx][0] = cdf[Math.max(0, Math.min(255, bin))];
        }
      }
    }
  }
  
  // Convert back to RGB
  for (let i = 0; i < labPixels.length; i++) {
    const [r, g, b] = labToRgb(labPixels[i][0], labPixels[i][1], labPixels[i][2]);
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
  }
  
  return imageData;
}

function gammaCorrection(imageData, gamma) {
  if (gamma === 1.0) return imageData;
  
  const data = imageData.data;
  const invGamma = 1 / gamma;
  
  // Build lookup table
  const lut = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    lut[i] = Math.round(Math.pow(i / 255, invGamma) * 255);
  }
  
  for (let i = 0; i < data.length; i += 4) {
    data[i] = lut[data[i]];
    data[i + 1] = lut[data[i + 1]];
    data[i + 2] = lut[data[i + 2]];
  }
  
  return imageData;
}

// Binary search: find index of first element > target
function upperBound(sortedArr, target) {
  let lo = 0, hi = sortedArr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sortedArr[mid] <= target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

function matchPaletteLightness(imageData, palette) {
  const data = imageData.data;
  
  // Get palette L values
  const paletteLab = palette.map(c => rgbToLab(c[0], c[1], c[2]));
  const paletteL = paletteLab.map(lab => lab[0]).sort((a, b) => a - b);
  
  // Convert image to LAB
  const labPixels = [];
  for (let i = 0; i < data.length; i += 4) {
    labPixels.push(rgbToLab(data[i], data[i + 1], data[i + 2]));
  }
  
  // Build sorted L values for percentile lookup
  const Lvalues = labPixels.map(lab => lab[0]).sort((a, b) => a - b);
  const numPixels = Lvalues.length;
  
  // Map image percentiles to palette percentiles
  for (let i = 0; i < labPixels.length; i++) {
    const L = labPixels[i][0];
    
    // Binary search: O(log N) instead of O(N)
    const rank = upperBound(Lvalues, L);
    const percentile = rank / numPixels;
    
    // Map to palette
    const paletteIdx = Math.min(
      paletteL.length - 1,
      Math.floor(percentile * paletteL.length)
    );
    labPixels[i][0] = paletteL[paletteIdx];
  }
  
  // Convert back to RGB
  for (let i = 0; i < labPixels.length; i++) {
    const [r, g, b] = labToRgb(labPixels[i][0], labPixels[i][1], labPixels[i][2]);
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
  }
  
  return imageData;
}

function autoPreprocess(imageData, palette) {
  const data = imageData.data;
  
  // Analyze image
  let sumL = 0;
  const Lvalues = [];
  
  for (let i = 0; i < data.length; i += 4) {
    const [L] = rgbToLab(data[i], data[i + 1], data[i + 2]);
    Lvalues.push(L);
    sumL += L;
  }
  
  const meanL = sumL / Lvalues.length;
  let variance = 0;
  for (const L of Lvalues) {
    variance += (L - meanL) * (L - meanL);
  }
  const stdL = Math.sqrt(variance / Lvalues.length);
  
  self.postMessage({
    type: 'log',
    message: `Image analysis: mean_L=${meanL.toFixed(1)}, std_L=${stdL.toFixed(1)}`
  });
  
  if (stdL < 15) {
    self.postMessage({ type: 'log', message: 'Low contrast detected, using CLAHE' });
    return clahe(imageData, 3.0);
  } else if (meanL < 35) {
    self.postMessage({ type: 'log', message: 'Dark image detected, using gamma + palette matching' });
    imageData = gammaCorrection(imageData, 0.6);
    return matchPaletteLightness(imageData, palette);
  } else if (meanL > 70) {
    self.postMessage({ type: 'log', message: 'Bright image detected, using gamma + palette matching' });
    imageData = gammaCorrection(imageData, 1.5);
    return matchPaletteLightness(imageData, palette);
  } else {
    self.postMessage({ type: 'log', message: 'Normal image, using palette matching' });
    return matchPaletteLightness(imageData, palette);
  }
}

// ============================================================================
// Palette Mapping
// ============================================================================

function applyPaletteRgb(imageData, palette, progressCallback) {
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
    
    // Progress update every 1000 pixels
    if (i % 1000 === 0 && progressCallback) {
      progressCallback(i / numPixels);
    }
  }
  
  return imageData;
}

function applyPaletteCiede2000(imageData, palette, progressCallback) {
  const data = imageData.data;
  const numPixels = data.length / 4;
  
  // Pre-compute palette LAB values
  const paletteLab = palette.map(c => rgbToLab(c[0], c[1], c[2]));
  
  for (let i = 0; i < numPixels; i++) {
    const idx = i * 4;
    const lab = rgbToLab(data[idx], data[idx + 1], data[idx + 2]);
    
    // Find nearest palette color (CIEDE2000)
    let minDist = Infinity;
    let nearestIdx = 0;
    
    for (let j = 0; j < paletteLab.length; j++) {
      const dist = ciede2000(lab, paletteLab[j]);
      if (dist < minDist) {
        minDist = dist;
        nearestIdx = j;
      }
    }
    
    data[idx] = palette[nearestIdx][0];
    data[idx + 1] = palette[nearestIdx][1];
    data[idx + 2] = palette[nearestIdx][2];
    
    // Progress update every 500 pixels (CIEDE2000 is slower)
    if (i % 500 === 0 && progressCallback) {
      progressCallback(i / numPixels);
    }
  }
  
  return imageData;
}

// ============================================================================
// Message Handler
// ============================================================================

self.onmessage = function(e) {
  const { type, imageData, options } = e.data;
  
  if (type !== 'convert') return;
  
  const { paletteName, customColors, useCiede2000, preprocess, gamma } = options;
  
  // Determine palette: custom or preset
  let palette = null;
  if (customColors && customColors.length > 0) {
    palette = customColors;
    self.postMessage({ type: 'log', message: `Custom palette: ${palette.length} colors` });
  } else if (paletteName && PALETTES[paletteName]) {
    palette = PALETTES[paletteName];
  }
  
  let data = imageData;
  let stage = 0;
  const totalStages = (preprocess !== 'none' ? 1 : 0) + (gamma !== 1.0 ? 1 : 0) + (palette ? 1 : 0);
  
  const updateProgress = (stageProgress) => {
    const overall = (stage + stageProgress) / Math.max(1, totalStages);
    self.postMessage({ type: 'progress', progress: overall });
  };
  
  try {
    // Apply preprocessing
    if (preprocess && preprocess !== 'none' && palette) {
      self.postMessage({ type: 'log', message: `Preprocessing: ${preprocess}` });
      
      switch (preprocess) {
        case 'auto':
          data = autoPreprocess(data, palette);
          break;
        case 'auto_levels':
          data = autoLevels(data);
          break;
        case 'histogram':
          data = histogramEqualization(data);
          break;
        case 'clahe':
          data = clahe(data);
          break;
        case 'match_palette':
          data = matchPaletteLightness(data, palette);
          break;
      }
      stage++;
      updateProgress(0);
    }
    
    // Apply gamma
    if (gamma && gamma !== 1.0) {
      self.postMessage({ type: 'log', message: `Gamma: ${gamma}` });
      data = gammaCorrection(data, gamma);
      stage++;
      updateProgress(0);
    }
    
    // Apply palette
    if (palette) {
      const method = useCiede2000 ? 'CIEDE2000' : 'RGB';
      const paletteLbl = customColors ? `custom (${palette.length} colors)` : paletteName;
      self.postMessage({ type: 'log', message: `Palette mapping (${method}): ${paletteLbl}` });
      
      if (useCiede2000) {
        data = applyPaletteCiede2000(data, palette, updateProgress);
      } else {
        data = applyPaletteRgb(data, palette, updateProgress);
      }
    }
    
    self.postMessage({ type: 'complete', imageData: data });
    
  } catch (error) {
    self.postMessage({ type: 'error', error: error.message });
  }
};
