function getStrokeInterpolation(lastX, lastY, currentX, currentY, step) {
    const dist = Math.hypot(currentX - lastX, currentY - lastY);
    const steps = Math.ceil(dist / step);
    return { dist, steps };
}

function getLerpPoint(lastX, lastY, currentX, currentY, t) {
    return {
        x: lastX + (currentX - lastX) * t,
        y: lastY + (currentY - lastY) * t,
    };
}

function getGaussianWeight(dist, sigma) {
    if (sigma <= 0) return 0;
    return Math.exp(-(dist * dist) / (2 * sigma * sigma));
}

function getCircleDistance(x, y, centerX, centerY) {
    const dx = x - centerX;
    const dy = y - centerY;
    return Math.sqrt(dx * dx + dy * dy);
}

export {
    getStrokeInterpolation,
    getLerpPoint,
    getGaussianWeight,
    getCircleDistance,
};
