import { getCircleDistance, getGaussianWeight } from "./math.js";

/**
 * Sample the composited image (paint over base) within a circular brush region.
 * Returns a Float32Array buffer and its bounding box, used as the smudge carried buffer.
 */
function sampleComposite(baseCanvas, paintCanvas, centerX, centerY, radius) {
    const r = Math.ceil(radius);
    const left = Math.max(0, Math.round(centerX - r));
    const top = Math.max(0, Math.round(centerY - r));
    const right = Math.min(baseCanvas.width, Math.round(centerX + r));
    const bottom = Math.min(baseCanvas.height, Math.round(centerY + r));
    if (left >= right || top >= bottom) return null;

    const w = right - left;
    const h = bottom - top;

    const baseCtx = baseCanvas.getContext("2d", { willReadFrequently: true });
    const paintCtx = paintCanvas.getContext("2d", { willReadFrequently: true });
    const baseData = baseCtx.getImageData(left, top, w, h);
    const paintData = paintCtx.getImageData(left, top, w, h);

    // Composite: paint over base
    const composite = new Float32Array(w * h * 4);
    for (let i = 0; i < w * h * 4; i += 4) {
        const pa = paintData.data[i + 3] / 255;
        composite[i] = paintData.data[i] * pa + baseData.data[i] * (1 - pa);
        composite[i + 1] = paintData.data[i + 1] * pa + baseData.data[i + 1] * (1 - pa);
        composite[i + 2] = paintData.data[i + 2] * pa + baseData.data[i + 2] * (1 - pa);
        composite[i + 3] = Math.max(paintData.data[i + 3], baseData.data[i + 3]);
    }

    return {
        data: composite,
        width: w,
        height: h,
        offsetX: left,
        offsetY: top,
    };
}

/**
 * Stamp a single clone brush dab at (drawX, drawY).
 * Copies pixels from the composited (paint over base) source region to the paint canvas destination,
 * blending with a Gaussian falloff within the brush radius.
 */
function stampClone(baseCanvas, paintCanvas, cloneState, drawX, drawY, radius, opacity) {
    const r = Math.ceil(radius);
    const sigma = radius * 0.4;

    const srcX = cloneState.sampleX + (drawX - cloneState.strokeStartX);
    const srcY = cloneState.sampleY + (drawY - cloneState.strokeStartY);

    const srcL = Math.max(0, Math.round(srcX - r));
    const srcT = Math.max(0, Math.round(srcY - r));
    const srcR = Math.min(baseCanvas.width, Math.round(srcX + r));
    const srcB = Math.min(baseCanvas.height, Math.round(srcY + r));
    if (srcL >= srcR || srcT >= srcB) return;

    const w = srcR - srcL;
    const h = srcB - srcT;

    const dstL = Math.round(drawX - (srcX - srcL));
    const dstT = Math.round(drawY - (srcY - srcT));

    const baseCtx = baseCanvas.getContext("2d", { willReadFrequently: true });
    const paintCtx = paintCanvas.getContext("2d", { willReadFrequently: true });

    const baseData = baseCtx.getImageData(srcL, srcT, w, h);
    const paintSrcData = paintCtx.getImageData(srcL, srcT, w, h);
    const dstData = paintCtx.getImageData(dstL, dstT, w, h);

    for (let py = 0; py < h; py++) {
        for (let px = 0; px < w; px++) {
            const dist = getCircleDistance(srcL + px, srcT + py, srcX, srcY);
            if (dist > radius) continue;

            const weight = getGaussianWeight(dist, sigma);
            const i = (py * w + px) * 4;

            // Source pixel (normalized to 0-1)
            const pa = paintSrcData.data[i + 3] / 255;
            const sourceR = paintSrcData.data[i] * pa + baseData.data[i] * (1 - pa);
            const sourceG = paintSrcData.data[i + 1] * pa + baseData.data[i + 1] * (1 - pa);
            const sourceB = paintSrcData.data[i + 2] * pa + baseData.data[i + 2] * (1 - pa);
            const sourceA = Math.max(paintSrcData.data[i + 3], baseData.data[i + 3]) / 255;

            const sR = sourceR;
            const sG = sourceG;
            const sB = sourceB;
            const sA = sourceA * weight * opacity; // Apply brush falloff and opacity

            // Destination pixel
            const dR = dstData.data[i];
            const dG = dstData.data[i + 1];
            const dB = dstData.data[i + 2];
            const dA = dstData.data[i + 3] / 255;

            // Porter-Duff OVER alpha compositing
            // outA = srcA + dstA * (1 - srcA)
            const outA = sA + dA * (1 - sA);

            if (outA > 0) {
                // outRGB = (srcRGB * srcA + dstRGB * dstA * (1 - srcA)) / outA
                dstData.data[i] = Math.round((sR * sA + dR * dA * (1 - sA)) / outA);
                dstData.data[i + 1] = Math.round((sG * sA + dG * dA * (1 - sA)) / outA);
                dstData.data[i + 2] = Math.round((sB * sA + dB * dA * (1 - sA)) / outA);
                dstData.data[i + 3] = Math.round(outA * 255);
            }
        }
    }

    paintCtx.putImageData(dstData, dstL, dstT);
}

/**
 * Stamp a single smudge brush dab at (drawX, drawY).
 * Blends the carried buffer into the paint canvas using Gaussian falloff,
 * then progressively mixes the output back into the carried buffer to create
 * a trailing smear effect.
 */
function stampSmudge(baseCanvas, paintCanvas, carriedBuffer, drawX, drawY, radius, strength, opacity) {
    const r = Math.ceil(radius);
    const sigma = radius * 0.4;

    const dstL = Math.max(0, Math.round(drawX - r));
    const dstT = Math.max(0, Math.round(drawY - r));
    const dstR = Math.min(paintCanvas.width, Math.round(drawX + r));
    const dstB = Math.min(paintCanvas.height, Math.round(drawY + r));
    if (dstL >= dstR || dstT >= dstB) return;

    const w = dstR - dstL;
    const h = dstB - dstT;

    const baseCtx = baseCanvas.getContext("2d", { willReadFrequently: true });
    const paintCtx = paintCanvas.getContext("2d", { willReadFrequently: true });
    const baseData = baseCtx.getImageData(dstL, dstT, w, h);
    const paintData = paintCtx.getImageData(dstL, dstT, w, h);

    if (!carriedBuffer || !carriedBuffer.data) return;

    const cb = carriedBuffer.data;
    const cw = carriedBuffer.width;
    const ch = carriedBuffer.height;

    for (let py = 0; py < h; py++) {
        for (let px = 0; px < w; px++) {
            const canvasX = dstL + px;
            const canvasY = dstT + py;

            const dist = getCircleDistance(canvasX, canvasY, drawX, drawY);
            if (dist > radius) continue;

            const weight = getGaussianWeight(dist, sigma);
            const s = strength * weight * opacity;

            const di = (py * w + px) * 4;

            // Composite destination
            const pa = paintData.data[di + 3] / 255;
            const destR = paintData.data[di] * pa + baseData.data[di] * (1 - pa);
            const destG = paintData.data[di + 1] * pa + baseData.data[di + 1] * (1 - pa);
            const destB = paintData.data[di + 2] * pa + baseData.data[di + 2] * (1 - pa);
            const destA = Math.max(paintData.data[di + 3], baseData.data[di + 3]);

            // Brush-relative coordinates: index carried buffer from brush center
            // This keeps the mapping correct regardless of how far the brush has moved
            const cbx = Math.round(canvasX - drawX) + r;
            const cby = Math.round(canvasY - drawY) + r;

            if (cbx >= 0 && cbx < cw && cby >= 0 && cby < ch) {
                const ci = (cby * cw + cbx) * 4;

                // Blend: output = carried * s + dest * (1 - s)
                const outR = cb[ci] * s + destR * (1 - s);
                const outG = cb[ci + 1] * s + destG * (1 - s);
                const outB = cb[ci + 2] * s + destB * (1 - s);
                const outA = cb[ci + 3] * s + destA * (1 - s);

                paintData.data[di] = Math.round(outR);
                paintData.data[di + 1] = Math.round(outG);
                paintData.data[di + 2] = Math.round(outB);
                paintData.data[di + 3] = Math.round(outA);

                // Update carried (progressive mixing)
                cb[ci] = outR;
                cb[ci + 1] = outG;
                cb[ci + 2] = outB;
                cb[ci + 3] = outA;
            }
        }
    }

    paintCtx.putImageData(paintData, dstL, dstT);
}

export {
    sampleComposite,
    stampClone,
    stampSmudge,
};
