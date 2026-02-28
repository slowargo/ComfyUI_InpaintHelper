/**
 * 双线性插值
 */
function bilinearInterpolate(p0, p1, p2, p3, u, v) {
    const p01 = {
        x: p0.x + (p1.x - p0.x) * u,
        y: p0.y + (p1.y - p0.y) * u
    };
    const p32 = {
        x: p3.x + (p2.x - p3.x) * u,
        y: p3.y + (p2.y - p3.y) * u
    };
    return {
        x: p01.x + (p32.x - p01.x) * v,
        y: p01.y + (p32.y - p01.y) * v
    };
}

/**
 * 计算三角形仿射变换矩阵并应用到上下文
 */
function applyTriangleTransform(ctx, s0, s1, s2, d0, d1, d2) {
    // 计算源三角形的变换矩阵
    const denom = (s0.x * (s1.y - s2.y) - s1.x * (s0.y - s2.y) + s2.x * (s0.y - s1.y));
    if (Math.abs(denom) < 1e-10) return false;

    // 计算目标三角形的变换
    const m = new DOMMatrix();

    // 使用标准仿射变换: 3个点确定一个仿射变换
    const srcX0 = s0.x, srcY0 = s0.y;
    const srcX1 = s1.x, srcY1 = s1.y;
    const srcX2 = s2.x, srcY2 = s2.y;
    const dstX0 = d0.x, dstY0 = d0.y;
    const dstX1 = d1.x, dstY1 = d1.y;
    const dstX2 = d2.x, dstY2 = d2.y;

    // 计算从源到目标的变换矩阵
    const srcDet = (srcX0 - srcX2) * (srcY1 - srcY2) - (srcX1 - srcX2) * (srcY0 - srcY2);
    if (Math.abs(srcDet) < 1e-10) return false;

    const a11 = ((dstX0 - dstX2) * (srcY1 - srcY2) - (dstX1 - dstX2) * (srcY0 - srcY2)) / srcDet;
    const a12 = ((dstX1 - dstX2) * (srcX0 - srcX2) - (dstX0 - dstX2) * (srcX1 - srcX2)) / srcDet;
    const a21 = ((dstY0 - dstY2) * (srcY1 - srcY2) - (dstY1 - dstY2) * (srcY0 - srcY2)) / srcDet;
    const a22 = ((dstY1 - dstY2) * (srcX0 - srcX2) - (dstY0 - dstY2) * (srcX1 - srcX2)) / srcDet;
    const tx = dstX2 - a11 * srcX2 - a12 * srcY2;
    const ty = dstY2 - a21 * srcX2 - a22 * srcY2;

    m.a = a11; m.b = a21; m.c = a12; m.d = a22; m.e = tx; m.f = ty;
    ctx.setTransform(m);
    return true;
}

/**
 * 绘制透视四边形（网格细分法）
 * @param {CanvasRenderingContext2D} ctx - 目标画布上下文
 * @param {HTMLCanvasElement} srcCanvas - 源图像画布
 * @param {Array} srcQuad - 源四边形坐标 [{x,y}, {x,y}, {x,y}, {x,y}] (左上,右上,右下,左下)
 * @param {Array} dstQuad - 目标四边形坐标 [{x,y}, {x,y}, {x,y}, {x,y}] (左上,右上,右下,左下)
 * @param {number} gridSize - 网格细分数量（默认20）
 */
function drawPerspectiveQuad(ctx, srcCanvas, srcQuad, dstQuad, gridSize = 20) {
    if (!srcCanvas || !srcQuad || !dstQuad || srcQuad.length !== 4 || dstQuad.length !== 4) return;

    const srcW = srcCanvas.width;
    const srcH = srcCanvas.height;

    // 保存当前变换
    ctx.save();

    // 网格细分绘制
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            const u0 = i / gridSize;
            const u1 = (i + 1) / gridSize;
            const v0 = j / gridSize;
            const v1 = (j + 1) / gridSize;

            // 计算源网格的四个角（归一化坐标）
            const s00 = bilinearInterpolate(srcQuad[0], srcQuad[1], srcQuad[2], srcQuad[3], u0, v0);
            const s10 = bilinearInterpolate(srcQuad[0], srcQuad[1], srcQuad[2], srcQuad[3], u1, v0);
            const s11 = bilinearInterpolate(srcQuad[0], srcQuad[1], srcQuad[2], srcQuad[3], u1, v1);
            const s01 = bilinearInterpolate(srcQuad[0], srcQuad[1], srcQuad[2], srcQuad[3], u0, v1);

            // 计算目标网格的四个角
            const d00 = bilinearInterpolate(dstQuad[0], dstQuad[1], dstQuad[2], dstQuad[3], u0, v0);
            const d10 = bilinearInterpolate(dstQuad[0], dstQuad[1], dstQuad[2], dstQuad[3], u1, v0);
            const d11 = bilinearInterpolate(dstQuad[0], dstQuad[1], dstQuad[2], dstQuad[3], u1, v1);
            const d01 = bilinearInterpolate(dstQuad[0], dstQuad[1], dstQuad[2], dstQuad[3], u0, v1);

            // 将四边形分成两个三角形绘制
            // 三角形1: s00 -> s10 -> s11
            ctx.save();
            if (applyTriangleTransform(ctx, s00, s10, s11, d00, d10, d11)) {
                // 计算三角形在源图像中的包围盒
                const minX = Math.floor(Math.max(0, Math.min(s00.x, s10.x, s11.x)));
                const minY = Math.floor(Math.max(0, Math.min(s00.y, s10.y, s11.y)));
                const maxX = Math.ceil(Math.min(srcW, Math.max(s00.x, s10.x, s11.x) + 1));
                const maxY = Math.ceil(Math.min(srcH, Math.max(s00.y, s10.y, s11.y) + 1));
                const w = maxX - minX;
                const h = maxY - minY;
                if (w > 0 && h > 0) {
                    ctx.drawImage(srcCanvas, minX, minY, w, h, minX, minY, w, h);
                }
            }
            ctx.restore();

            // 三角形2: s00 -> s11 -> s01
            ctx.save();
            if (applyTriangleTransform(ctx, s00, s11, s01, d00, d11, d01)) {
                const minX = Math.floor(Math.max(0, Math.min(s00.x, s11.x, s01.x)));
                const minY = Math.floor(Math.max(0, Math.min(s00.y, s11.y, s01.y)));
                const maxX = Math.ceil(Math.min(srcW, Math.max(s00.x, s11.x, s01.x) + 1));
                const maxY = Math.ceil(Math.min(srcH, Math.max(s00.y, s11.y, s01.y) + 1));
                const w = maxX - minX;
                const h = maxY - minY;
                if (w > 0 && h > 0) {
                    ctx.drawImage(srcCanvas, minX, minY, w, h, minX, minY, w, h);
                }
            }
            ctx.restore();
        }
    }

    // 恢复变换
    ctx.restore();
}

export {
    bilinearInterpolate,
    applyTriangleTransform,
    drawPerspectiveQuad,
};
