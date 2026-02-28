let customCursorEl = null;

/**
 * 创建自定义光标元素
 */
function createCustomCursor() {
    if (customCursorEl) return;
    customCursorEl = document.createElement('div');
    customCursorEl.id = 'transform-tool-cursor';
    customCursorEl.style.cssText = `
        position: fixed;
        pointer-events: none;
        z-index: 99999;
        width: 32px;
        height: 32px;
        margin-left: -16px;
        margin-top: -16px;
        display: none;
    `;
    document.body.appendChild(customCursorEl);
}

/**
 * SVG cursor icons
 */
const cursorIcons = {
    // Crosshair with precision dot - for selection mode
    crosshair: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Outer circle -->
            <circle cx="16" cy="16" r="10" fill="none" stroke="#4a9eff" stroke-width="1.3" filter="url(#shadow)"/>
            <!-- Cross lines -->
            <line x1="16" y1="4" x2="16" y2="12" stroke="#4a9eff" stroke-width="1.3" filter="url(#shadow)"/>
            <line x1="16" y1="20" x2="16" y2="28" stroke="#4a9eff" stroke-width="1.3" filter="url(#shadow)"/>
            <line x1="4" y1="16" x2="12" y2="16" stroke="#4a9eff" stroke-width="1.3" filter="url(#shadow)"/>
            <line x1="20" y1="16" x2="28" y2="16" stroke="#4a9eff" stroke-width="1.3" filter="url(#shadow)"/>
            <!-- Center dot -->
            <circle cx="16" cy="16" r="1.5" fill="#4a9eff" filter="url(#shadow)"/>
        </svg>
    `,

    // Move - four-way arrows
    move: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Up arrow -->
            <polygon points="16,4 12,10 14,10 14,12 18,12 18,10 20,10" fill="#4a9eff" filter="url(#shadow)"/>
            <!-- Down arrow -->
            <polygon points="16,28 12,22 14,22 14,20 18,20 18,22 20,22" fill="#4a9eff" filter="url(#shadow)"/>
            <!-- Left arrow -->
            <polygon points="4,16 10,12 10,14 12,14 12,18 10,18 10,20" fill="#4a9eff" filter="url(#shadow)"/>
            <!-- Right arrow -->
            <polygon points="28,16 22,12 22,14 20,14 20,18 22,18 22,20" fill="#4a9eff" filter="url(#shadow)"/>
            <!-- Center dot -->
            <circle cx="16" cy="16" r="2" fill="#4a9eff" filter="url(#shadow)"/>
        </svg>
    `,

    // NW-SE resize (top-left to bottom-right)
    nwseResize: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Background circle -->
            <circle cx="16" cy="16" r="12" fill="rgba(255, 180, 60, 0.2)" stroke="#ffb43c" stroke-width="1.3" filter="url(#shadow)"/>
            <!-- Diagonal arrow (top-left to bottom-right) -->
            <line x1="8" y1="8" x2="24" y2="24" stroke="#ffb43c" stroke-width="2.1" stroke-linecap="round" filter="url(#shadow)"/>
            <polygon points="6,14 6,6 14,6" fill="#ffb43c" filter="url(#shadow)"/>
            <polygon points="26,18 26,26 18,26" fill="#ffb43c" filter="url(#shadow)"/>
        </svg>
    `,

    // NE-SW resize (top-right to bottom-left)
    neswResize: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Background circle -->
            <circle cx="16" cy="16" r="12" fill="rgba(255, 180, 60, 0.2)" stroke="#ffb43c" stroke-width="1.3" filter="url(#shadow)"/>
            <!-- Diagonal arrow (top-right to bottom-left) -->
            <line x1="24" y1="8" x2="8" y2="24" stroke="#ffb43c" stroke-width="2.1" stroke-linecap="round" filter="url(#shadow)"/>
            <polygon points="26,14 26,6 18,6" fill="#ffb43c" filter="url(#shadow)"/>
            <polygon points="6,18 6,26 14,26" fill="#ffb43c" filter="url(#shadow)"/>
        </svg>
    `,

    // NS resize (vertical)
    nsResize: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Background circle -->
            <circle cx="16" cy="16" r="12" fill="rgba(100, 220, 120, 0.2)" stroke="#64dc78" stroke-width="1.3" filter="url(#shadow)"/>
            <!-- Vertical arrows -->
            <line x1="16" y1="6" x2="16" y2="26" stroke="#64dc78" stroke-width="2.1" stroke-linecap="round" filter="url(#shadow)"/>
            <polygon points="16,4 11,10 21,10" fill="#64dc78" filter="url(#shadow)"/>
            <polygon points="16,28 11,22 21,22" fill="#64dc78" filter="url(#shadow)"/>
        </svg>
    `,

    // EW resize (horizontal)
    ewResize: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Background circle -->
            <circle cx="16" cy="16" r="12" fill="rgba(100, 220, 120, 0.2)" stroke="#64dc78" stroke-width="1.3" filter="url(#shadow)"/>
            <!-- Horizontal arrows -->
            <line x1="6" y1="16" x2="26" y2="16" stroke="#64dc78" stroke-width="2.1" stroke-linecap="round" filter="url(#shadow)"/>
            <polygon points="4,16 10,11 10,21" fill="#64dc78" filter="url(#shadow)"/>
            <polygon points="28,16 22,11 22,21" fill="#64dc78" filter="url(#shadow)"/>
        </svg>
    `,

    // Rotate (grab)
    rotate: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Background circle -->
            <circle cx="16" cy="16" r="12" fill="rgba(255, 100, 180, 0.2)" stroke="#ff64b4" stroke-width="1.3" filter="url(#shadow)"/>
            <!-- Rotation arrow arc -->
            <path d="M 10,16 A 6,6 0 1,1 22,16" fill="none" stroke="#ff64b4" stroke-width="2.1" stroke-linecap="round" filter="url(#shadow)"/>
            <!-- Arrow head -->
            <polygon points="22,12 26,18 18,18" fill="#ff64b4" filter="url(#shadow)"/>
            <!-- Center dot -->
            <circle cx="16" cy="16" r="2" fill="#ff64b4" filter="url(#shadow)"/>
        </svg>
    `,

    // Rotating (grabbing) - filled version
    rotating: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Filled background circle -->
            <circle cx="16" cy="16" r="12" fill="#ff64b4" stroke="#ff64b4" stroke-width="1.3" filter="url(#shadow)"/>
            <!-- White arc -->
            <path d="M 10,16 A 6,6 0 1,1 22,16" fill="none" stroke="white" stroke-width="2.1" stroke-linecap="round"/>
            <!-- White arrow head -->
            <polygon points="22,12 26,18 18,18" fill="white"/>
            <!-- Center dot -->
            <circle cx="16" cy="16" r="2" fill="white"/>
        </svg>
    `,

    // Default - simple pointer dot
    default: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <circle cx="16" cy="16" r="4" fill="#4a9eff" filter="url(#shadow)"/>
        </svg>
    `
};

/**
 * 销毁自定义光标元素
 */
function destroyCustomCursor() {
    if (customCursorEl) {
        customCursorEl.remove();
        customCursorEl = null;
    }
}

/**
 * 更新自定义光标位置和样式
 */
function updateCustomCursor(clientX, clientY, cursorType) {
    if (!customCursorEl) return;

    customCursorEl.style.left = clientX + 'px';
    customCursorEl.style.top = clientY + 'px';

    // 根据 cursorType 更新光标 SVG
    let svg = cursorIcons.default;
    switch (cursorType) {
        case 'crosshair':
            svg = cursorIcons.crosshair;
            break;
        case 'move':
            svg = cursorIcons.move;
            break;
        case 'nwse-resize':
            svg = cursorIcons.nwseResize;
            break;
        case 'nesw-resize':
            svg = cursorIcons.neswResize;
            break;
        case 'ns-resize':
            svg = cursorIcons.nsResize;
            break;
        case 'ew-resize':
            svg = cursorIcons.ewResize;
            break;
        case 'grab':
            svg = cursorIcons.rotate;
            break;
        case 'grabbing':
            svg = cursorIcons.rotating;
            break;
        default:
            svg = cursorIcons.default;
    }
    customCursorEl.innerHTML = svg;
}

/**
 * 显示/隐藏自定义光标
 */
function showCustomCursor(show) {
    if (customCursorEl) {
        customCursorEl.style.display = show ? 'block' : 'none';
    }
}

/**
 * 获取光标样式
 */
function getCursorForHandle(handle) {
    if (!handle) return "default";
    switch (handle.type) {
        case "corner":
            // 根据角点索引返回对应光标
            const cornerCursors = ["nwse-resize", "nesw-resize", "nwse-resize", "nesw-resize"];
            return cornerCursors[handle.index] || "nwse-resize";
        case "rotate":
            return "grab";
        case "edge":
            // 根据边的方向返回光标 (0=上,1=右,2=下,3=左)
            const edgeCursors = ["ns-resize", "ew-resize", "ns-resize", "ew-resize"];
            return edgeCursors[handle.index] || "move";
        case "move":
            return "move";
        default:
            return "default";
    }
}

export {
    createCustomCursor,
    destroyCustomCursor,
    updateCustomCursor,
    showCustomCursor,
    getCursorForHandle,
};
