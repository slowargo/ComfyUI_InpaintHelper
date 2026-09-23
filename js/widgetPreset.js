import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Widget Preset: snapshot widget values from connected nodes into named presets and write them back.
// Design notes and the reasoning behind each rule live in reports/WIDGET_PRESET_NODE_DESIGN.md.

const LOG_PREFIX = "[WidgetPreset]";
const TARGET_SLOT_PREFIX = "preset_target_";
const EMPTY_SLOT_LABEL = "+ connect";
const SHOW_DEBUG_PROPERTY = "showDebug";
const MAX_ENUMERATED_WIDGETS = 15;
const MANY_PRESETS_HINT = 12;
const NUMBER_EPSILON = 1e-9;
const MAX_PASS_THROUGH_HOPS = 32;
const UPDATE_DELAY_MS = 50;
const INPUT_SLOT = 1; // NodeSlotType.INPUT

// Widgets that can hold a plain string but are previews or UI, not parameters worth writing back.
// Includes types this frontend (1.37.2) does not have yet, so upgrading does not need a revisit.
const EXCLUDED_WIDGET_TYPES = new Set([
    "button", "image", "imageupload", "markdown", "galleria", "imagecompare",
    "chart", "preview", "painter", "asset",
]);

// Preset nodes currently in a graph, refreshed together whenever the change tracker reports a change
const livePresetNodes = new Set();
let graphChangedListenerRegistered = false;

// The legacy canvas hides on widget.hidden, the Vue node renderer on widget.options.hidden. The Vue side
// only re-reads widgets when the node's (shallowReactive) widgets array changes. Putting the same object
// back in place counts as no change there, so take it out and insert it again.
const setWidgetHidden = (node, widget, hidden) => {
    widget.hidden = hidden;
    if (widget.options) widget.options.hidden = hidden;
    const index = node.widgets?.indexOf(widget) ?? -1;
    if (index >= 0) {
        node.widgets.splice(index, 1);
        node.widgets.splice(index, 0, widget);
    }
};

const escapeRegExp = (text) => String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const isTargetSlot = (input) => typeof input?.name === "string" && input.name.startsWith(TARGET_SLOT_PREFIX);

const isPassThroughNode = (node) => {
    const type = String(node?.type ?? "");
    return type.includes("Reroute") || type.includes("Node Combiner") || type.includes("Node Collector");
};

// graph.links is a Map; the index form only still works through a deprecated Proxy
const getLink = (graph, linkId) => graph?.links?.get?.(linkId) ?? graph?.links?.[linkId];

// Follow a link upstream through Reroute-style pass-through nodes to the node that owns the widgets.
// Returns null when the chain ends somewhere that is not a regular node, e.g. a subgraph boundary.
const resolveLinkTarget = (graph, link) => {
    for (let hop = 0; link && hop < MAX_PASS_THROUGH_HOPS; hop++) {
        const origin = graph?.getNodeById(link.origin_id);
        if (!origin) return null;
        if (!isPassThroughNode(origin)) return origin;
        const upstream = origin.inputs?.find((input) => input.link != null);
        if (!upstream) return null;
        link = getLink(graph, upstream.link);
    }
    return null;
};

// null: not a parameter at all, "unsupported": a parameter we cannot snapshot yet, "ok": snapshot-able
const getWidgetStatus = (node, widget) => {
    if (!widget?.name) return null;
    if (EXCLUDED_WIDGET_TYPES.has(String(widget.type).toLowerCase())) return null;
    // Derived widgets (rebuilt from other widgets, not persisted on their own) must be left alone:
    // writing FloatSelector's float_N back would drag selected_index along with every write.
    if (widget.serialize === false || widget.options?.serialize === false) return null;
    // A widget whose input socket is linked takes its value from upstream
    if (node.inputs?.some((input) => input.widget?.name === widget.name && input.link != null)) return null;
    const valueType = typeof widget.value;
    return valueType === "string" || valueType === "number" || valueType === "boolean" ? "ok" : "unsupported";
};

const findSnapshotWidget = (node, name) =>
    node?.widgets?.find((widget) => widget.name === name && getWidgetStatus(node, widget) === "ok");

const valuesEqual = (a, b) =>
    typeof a === "number" && typeof b === "number" ? Math.abs(a - b) < NUMBER_EPSILON : a === b;

const comboAccepts = (widget, value) => {
    if (widget.type !== "combo") return true;
    let values = widget.options?.values;
    if (typeof values === "function") {
        try {
            values = values(widget, widget.node);
        } catch {
            return true;
        }
    }
    return Array.isArray(values) ? values.includes(value) : true;
};

const formatValue = (value) => {
    const text = typeof value === "string" ? JSON.stringify(value) : String(value);
    return text.length > 32 ? `${text.slice(0, 31)}…` : text;
};

// One regex per line. Blank lines and '#' lines are skipped, '-' lines exclude from the union of the rest.
const parseRules = (text) => {
    const include = [];
    const exclude = [];
    const lines = String(text ?? "").split(/\r?\n/);
    for (let index = 0; index < lines.length; index++) {
        const trimmed = lines[index].trim();
        if (!trimmed || trimmed.startsWith("#")) continue;
        const isExclude = trimmed.startsWith("-");
        const body = isExclude ? trimmed.slice(1).trim() : trimmed;
        if (!body) continue;
        try {
            (isExclude ? exclude : include).push({ source: trimmed, regex: new RegExp(body, "i") });
        } catch (error) {
            return { include, exclude, error: { line: index + 1, message: error.message } };
        }
    }
    return { include, exclude, error: null };
};

const classifyPath = (rules, path) => {
    if (!rules.include.some((rule) => rule.regex.test(path))) return { matched: false };
    const excludedBy = rules.exclude.find((rule) => rule.regex.test(path));
    return excludedBy ? { matched: false, excludedBy: excludedBy.source } : { matched: true };
};

const emptyPresets = () => ({ v: 1, items: [] });

// A broken JSON is reported rather than thrown away, so it can still be repaired by hand
const parsePresets = (text) => {
    if (!String(text ?? "").trim()) return { data: emptyPresets(), error: null };
    try {
        const data = JSON.parse(text);
        if (!data || !Array.isArray(data.items)) throw new Error("missing 'items' array");
        data.items = data.items.filter((item) => item && typeof item.name === "string");
        for (const item of data.items) {
            if (!item.nodes || typeof item.nodes !== "object") item.nodes = {};
        }
        return { data, error: null };
    } catch (error) {
        return { data: emptyPresets(), error: error.message };
    }
};

// Shared by Apply and the highlight, so both agree on which saved values can be reached right now
const locateWidget = (targets, nodeId, widgetName) => {
    const target = targets.get(String(nodeId));
    if (!target) return { reason: "node not connected" };
    const widget = findSnapshotWidget(target, widgetName);
    if (!widget) return { target, reason: "widget not found" };
    return { target, widget };
};

const isPresetActive = (item, targets) => {
    let located = 0;
    for (const [nodeId, record] of Object.entries(item.nodes)) {
        for (const [widgetName, value] of Object.entries(record?.widgets ?? {})) {
            const { widget } = locateWidget(targets, nodeId, widgetName);
            if (!widget) continue;
            located++;
            if (!valuesEqual(widget.value, value)) return false;
        }
    }
    return located > 0;
};

const withUndoTransaction = (app, fn) => {
    // canvas.emit* is what the change tracker listens to; graph.beforeChange() alone records nothing
    app.canvas?.emitBeforeChange?.();
    try {
        return fn();
    } finally {
        app.canvas?.emitAfterChange?.();
    }
};

const askName = (app, title, defaultValue, event, onName) => {
    // The canvas prompt mounts next to LGraphCanvas.active_canvas, which stays unset until the canvas has seen
    // a pointer event, so fall back to the browser prompt then. A missing event (Vue node buttons pass none)
    // only affects where the dialog is placed.
    if (app.canvas?.prompt && app.canvas.constructor?.active_canvas?.canvas) {
        app.canvas.prompt(title, defaultValue, (value) => onName(String(value ?? "").trim()), event);
    } else {
        const value = window.prompt(title, defaultValue);
        if (value != null) onName(value.trim());
    }
};

export function setupWidgetPreset(nodeType, nodeData, app) {
    if (!graphChangedListenerRegistered) {
        graphChangedListenerRegistered = true;
        // Fired on every captured change, which covers widget edits, queueing and undo/redo
        api.addEventListener("graphChanged", () => {
            for (const node of livePresetNodes) node.slowargoWidgetPreset?.scheduleUpdate();
        });
    }

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const result = origOnNodeCreated?.apply(this, arguments);
        const node = this;

        const filterWidget = node.widgets?.find((widget) => widget.name === "filter");
        const presetsWidget = node.widgets?.find((widget) => widget.name === "presets");
        if (!filterWidget || !presetsWidget) {
            console.warn(LOG_PREFIX, "filter/presets widgets not found, node left inactive");
            return result;
        }

        // Never sent to the backend; see the design doc for why this is safe
        node.isVirtualNode = true;
        node.properties ??= {};
        node.properties[SHOW_DEBUG_PROPERTY] ??= true;

        setWidgetHidden(node, presetsWidget, true);
        if (filterWidget.inputEl) {
            filterWidget.inputEl.placeholder = "One regex per line: node title/widget name";
        }

        const state = {
            presetButtons: [],
            lastApply: null,
            updateTimer: null,
            lastDebugText: null,
            // Set while restoring a saved node, whose stored size already includes every dynamic widget
            resizeSuppressed: false,
        };

        // Everything below is appended after the two declared widgets and never serialized, so the
        // positional widgets_values of filter/presets stay intact.
        const statusWidget = node.addWidget("button", "wp_status", null, () => {});
        statusWidget.serialize = false;
        statusWidget.disabled = true;

        const debugWidget = ComfyWidgets.STRING(node, "wp_debug", ["STRING", { multiline: true }], app).widget;
        debugWidget.serialize = false;
        debugWidget.label = "debug info";
        if (debugWidget.options) {
            debugWidget.options.serialize = false;
            // Read-only flag of the Vue textarea; the legacy one is handled through inputEl below
            debugWidget.options.read_only = true;
        }
        if (debugWidget.inputEl) {
            debugWidget.inputEl.readOnly = true;
            debugWidget.inputEl.wrap = "off";
            debugWidget.inputEl.placeholder = "";
            debugWidget.inputEl.style.fontFamily = "monospace";
        }

        const saveWidget = node.addWidget("button", "wp_save", null, (widget, canvas, n, pos, event) => {
            savePreset(event);
        });
        saveWidget.serialize = false;
        saveWidget.label = "💾 Save as Preset…";

        if (!node.inputs?.some(isTargetSlot)) {
            node.addInput(`${TARGET_SLOT_PREFIX}0`, "*", { label: EMPTY_SLOT_LABEL });
        }

        // Keep whatever height the user gave the node, only adding or removing what the change itself costs.
        // Skipped while collapsed, where computeSize() reports a bare title row.
        const resizeAround = (fn) => {
            if (node.flags?.collapsed || state.resizeSuppressed) {
                fn();
                return;
            }
            const before = node.computeSize()[1];
            fn();
            const after = node.computeSize()[1];
            if (after !== before) {
                node.setSize([node.size[0], Math.max(after, node.size[1] + after - before)]);
            }
        };

        const readPresets = () => parsePresets(presetsWidget.value);

        const writePresets = (data) => {
            withUndoTransaction(app, () => {
                presetsWidget.value = JSON.stringify(data);
                rebuildPresetButtons(data.items);
            });
            update();
        };

        // Connected nodes by id (as string, matching the JSON keys), in slot order
        const collectTargets = () => {
            const targets = new Map();
            let unresolved = 0;
            for (const input of node.inputs ?? []) {
                if (!isTargetSlot(input) || input.link == null) continue;
                const target = resolveLinkTarget(node.graph, getLink(node.graph, input.link));
                if (target) targets.set(String(target.id), target);
                else unresolved++;
            }
            return { targets, unresolved };
        };

        // One trailing empty slot to connect into, no empty slots in between. Slot numbers carry no
        // meaning, so shifting them is harmless. Names stay internal so the load-time reconcile in
        // litegraphService, which matches inputs by name, can never mistake one for filter/presets.
        const stabilizeSlots = () => {
            if (!node.graph || !node.inputs) return false;
            let changed = false;
            const targetIndexes = () => node.inputs.flatMap((input, index) => (isTargetSlot(input) ? [index] : []));

            let indexes = targetIndexes();
            for (let k = indexes.length - 2; k >= 0; k--) {
                if (node.inputs[indexes[k]].link == null) {
                    node.removeInput(indexes[k]);
                    changed = true;
                }
            }

            indexes = targetIndexes();
            const lastSlot = indexes.length ? node.inputs[indexes[indexes.length - 1]] : null;
            if (!lastSlot || lastSlot.link != null) {
                node.addInput(`${TARGET_SLOT_PREFIX}${indexes.length}`, "*", { label: EMPTY_SLOT_LABEL });
                changed = true;
            }

            targetIndexes().forEach((inputIndex, slotNumber) => {
                const input = node.inputs[inputIndex];
                const name = `${TARGET_SLOT_PREFIX}${slotNumber}`;
                if (input.name !== name) {
                    input.name = name;
                    changed = true;
                }
                const label = input.link == null
                    ? EMPTY_SLOT_LABEL
                    : resolveLinkTarget(node.graph, getLink(node.graph, input.link))?.title || "?";
                if (input.label !== label) {
                    input.label = label;
                    changed = true;
                }
            });
            return changed;
        };

        const rebuildPresetButtons = (items) => {
            const names = items.map((item) => item.name);
            const current = state.presetButtons.map((button) => button.slowargoPresetName);
            if (names.length === current.length && names.every((name, i) => name === current[i])) return;

            resizeAround(() => {
                for (const button of state.presetButtons) {
                    if (node.widgets?.includes(button)) node.removeWidget(button);
                }
                state.presetButtons = items.map((item, index) => {
                    const button = node.addWidget("button", `wp_preset_${index}`, null, () => applyPreset(item.name));
                    button.serialize = false;
                    button.slowargoPresetName = item.name;
                    button.label = `○ ${item.name}`;
                    return button;
                });
            });
        };

        const applyDebugVisibility = () => {
            const show = node.properties?.[SHOW_DEBUG_PROPERTY] !== false;
            if (Boolean(debugWidget.hidden) === !show) return;
            resizeAround(() => {
                setWidgetHidden(node, debugWidget, !show);
            });
            state.lastDebugText = null;
        };

        // Every widget on every connected node that is a parameter, with its rule verdict
        const collectRows = (targets, rules) => {
            const rows = [];
            for (const target of targets.values()) {
                for (const widget of target.widgets ?? []) {
                    const status = getWidgetStatus(target, widget);
                    if (!status) continue;
                    const path = `${target.title}/${widget.name}`;
                    const row = { path, target, widget, unsupported: status === "unsupported", matched: false };
                    if (!row.unsupported && !rules.error) Object.assign(row, classifyPath(rules, path));
                    rows.push(row);
                }
            }
            return rows;
        };

        const buildDebugText = ({ targets, unresolved }, rules, presets, rows) => {
            const lines = [];
            if (presets.error) {
                lines.push(`Preset data could not be parsed: ${presets.error}`, `Raw: ${presetsWidget.value}`, "");
            }
            if (rules.error) lines.push(`Rule line ${rules.error.line}: ${rules.error.message}`, "");
            if (unresolved) lines.push(`${unresolved} link(s) do not lead to a regular node (subgraph boundary?)`, "");
            if (!targets.size) lines.push("No nodes connected.");

            const pathWidth = Math.max(0, ...rows.map((row) => row.path.length));
            const idWidth = Math.max(0, ...rows.map((row) => String(row.target.id).length)) + 1;
            for (const row of rows) {
                const value = row.unsupported ? "—" : formatValue(row.widget.value);
                const verdict = row.unsupported
                    ? "unsupported (object value)"
                    : row.matched
                        ? "✓ matched"
                        : row.excludedBy
                            ? `excluded by ${row.excludedBy}`
                            : "";
                const id = `#${row.target.id}`.padEnd(idWidth);
                lines.push(`${row.path.padEnd(pathWidth)}  ${id}  ${value.padEnd(24)}  ${verdict}`.trimEnd());
            }

            if (state.lastApply?.skipped.length) {
                const { name, applied, total, skipped } = state.lastApply;
                lines.push("", `Last apply "${name}": ${applied}/${total} applied, skipped:`);
                for (const entry of skipped) lines.push(`  ${entry}`);
            }
            return lines.join("\n");
        };

        const update = () => {
            if (state.updateTimer) {
                clearTimeout(state.updateTimer);
                state.updateTimer = null;
            }
            if (!node.graph) return;

            let structureChanged = false;
            resizeAround(() => {
                structureChanged = stabilizeSlots();
            });

            const collected = collectTargets();
            const { targets } = collected;
            const rules = parseRules(filterWidget.value);
            const presets = readPresets();
            rebuildPresetButtons(presets.data.items);
            applyDebugVisibility();

            const rows = collectRows(targets, rules);
            const matchedCount = rows.filter((row) => row.matched).length;

            let status;
            if (presets.error) status = "⚠ Preset data unreadable, see debug info";
            else if (rules.error) status = `⚠ Rule line ${rules.error.line}: invalid regex`;
            else if (!targets.size) status = "No nodes connected";
            else if (!rules.include.length) status = `${targets.size} nodes · no rules`;
            else status = `${targets.size} nodes · ${matchedCount} widgets matched`;
            if (state.lastApply) status += ` · applied ${state.lastApply.applied}/${state.lastApply.total}`;
            if (presets.data.items.length > MANY_PRESETS_HINT) status += " · many presets";

            let needsRedraw = structureChanged;
            if (statusWidget.label !== status) {
                statusWidget.label = status;
                needsRedraw = true;
            }
            // Saving over unreadable data would silently discard it
            const saveDisabled = Boolean(presets.error || rules.error || matchedCount === 0);
            if (saveWidget.disabled !== saveDisabled) {
                saveWidget.disabled = saveDisabled;
                needsRedraw = true;
            }

            presets.data.items.forEach((item, index) => {
                const button = state.presetButtons[index];
                if (!button) return;
                const label = `${isPresetActive(item, targets) ? "●" : "○"} ${item.name}`;
                if (button.label !== label) {
                    button.label = label;
                    needsRedraw = true;
                }
            });

            if (!debugWidget.hidden) {
                const text = buildDebugText(collected, rules, presets, rows);
                if (text !== state.lastDebugText) {
                    debugWidget.value = text;
                    state.lastDebugText = text;
                }
            }

            // Only redraw on a real change: redrawing unconditionally from here would loop via graphChanged
            if (needsRedraw) node.setDirtyCanvas(true, true);
        };

        const scheduleUpdate = () => {
            // Deferred rather than a microtask so a paste has reconnected its links before slots are compacted
            if (state.updateTimer) return;
            state.updateTimer = setTimeout(update, UPDATE_DELAY_MS);
        };

        // Called for a genuinely new link only. Never touches existing lines; skips when a rule for this
        // title already exists, e.g. a second KSampler with the default title.
        const appendRuleFor = (target) => {
            const escapedTitle = escapeRegExp(target.title ?? "");
            const prefixes = [`^${escapedTitle}/`, `${escapedTitle}/`].map((prefix) => prefix.toLowerCase());
            const lines = String(filterWidget.value ?? "").split(/\r?\n/);
            const hasRule = lines.some((line) => {
                let trimmed = line.trim();
                if (trimmed.startsWith("-")) trimmed = trimmed.slice(1).trim();
                return prefixes.some((prefix) => trimmed.toLowerCase().startsWith(prefix));
            });
            if (hasRule) return;

            const names = (target.widgets ?? [])
                .filter((widget) => getWidgetStatus(target, widget) === "ok")
                .map((widget) => widget.name);
            if (!names.length) return;

            // Anchored: rules are partial matches, so an unanchored KSampler/steps would also hit "My KSampler"
            const body = names.length > MAX_ENUMERATED_WIDGETS ? ".*" : `(${names.map(escapeRegExp).join("|")})`;
            const rule = `^${escapedTitle}/${body}$`;
            const current = String(filterWidget.value ?? "").replace(/\s+$/, "");
            filterWidget.value = current ? `${current}\n${rule}` : rule;
        };

        const snapshotSelection = () => {
            const { targets } = collectTargets();
            const rows = collectRows(targets, parseRules(filterWidget.value));
            const nodes = {};
            for (const row of rows) {
                if (!row.matched) continue;
                const id = String(row.target.id);
                nodes[id] ??= { title: row.target.title, widgets: {} };
                nodes[id].widgets[row.widget.name] = row.widget.value;
            }
            return { nodes, targets };
        };

        // What an overwrite drops: saved values this snapshot no longer contains
        const describeRemovals = (previous, next, targets) => {
            const removals = [];
            for (const [id, record] of Object.entries(previous.nodes)) {
                const dropped = Object.keys(record?.widgets ?? {}).filter(
                    (name) => !(next[id] && name in next[id].widgets),
                );
                if (!dropped.length) continue;
                const reason = targets.has(id) ? "not selected by rules" : "node not connected";
                removals.push(`${record.title ?? "?"} #${id}: ${dropped.join(", ")} (${reason})`);
            }
            return removals;
        };

        const savePreset = (event) => {
            const presets = readPresets();
            if (presets.error) return;
            const rules = parseRules(filterWidget.value);
            if (rules.error) return;
            if (!Object.keys(snapshotSelection().nodes).length) return;

            const items = presets.data.items;
            const defaultName = `Preset ${items.length + 1}`;
            askName(app, "Preset name", defaultName, event, (name) => {
                if (!name) return;
                // Read again: the dialog may have stayed open while values changed
                const { nodes, targets } = snapshotSelection();
                if (!Object.keys(nodes).length) return;

                const existingIndex = items.findIndex((item) => item.name === name);
                if (existingIndex >= 0) {
                    const removals = describeRemovals(items[existingIndex], nodes, targets);
                    const details = removals.length
                        ? `\n\nThe following will be removed from it:\n${removals.slice(0, 12).join("\n")}`
                            + (removals.length > 12 ? `\n…and ${removals.length - 12} more` : "")
                        : "";
                    if (!window.confirm(`Overwrite preset "${name}"?${details}`)) return;
                    // Whole replacement, not a merge: the preset is exactly what the rules select now
                    items[existingIndex] = { name, nodes };
                } else {
                    items.push({ name, nodes });
                }
                state.lastApply = null;
                writePresets(presets.data);
            });
        };

        const applyPreset = (name) => {
            const presets = readPresets();
            const item = presets.data.items.find((entry) => entry.name === name);
            if (!item) return;

            const { targets } = collectTargets();
            let total = 0;
            let applied = 0;
            const skipped = [];

            withUndoTransaction(app, () => {
                for (const [nodeId, record] of Object.entries(item.nodes)) {
                    for (const [widgetName, value] of Object.entries(record?.widgets ?? {})) {
                        total++;
                        // Looked up per value: a callback may rebuild the node's widgets (FloatSelector does)
                        const { target, widget, reason } = locateWidget(targets, nodeId, widgetName);
                        if (!widget) {
                            skipped.push(`${record.title ?? "?"} #${nodeId}/${widgetName}: ${reason}`);
                            continue;
                        }
                        if (!comboAccepts(widget, value)) {
                            const shown = formatValue(value);
                            skipped.push(`${target.title} #${nodeId}/${widgetName}: ${shown} is not an option`);
                            continue;
                        }
                        widget.value = value;
                        widget.callback?.(value, app.canvas, target);
                        applied++;
                    }
                }
            });

            state.lastApply = { name, applied, total, skipped };
            if (skipped.length) console.warn(LOG_PREFIX, `apply "${name}" skipped:`, skipped);
            node.graph?.setDirtyCanvas(true, true);
            update();
        };

        const renamePreset = (oldName, event) => {
            askName(app, "Rename preset", oldName, event, (name) => {
                if (!name || name === oldName) return;
                const presets = readPresets();
                if (presets.error) return;
                if (presets.data.items.some((item) => item.name === name)) {
                    window.alert(`A preset named "${name}" already exists.`);
                    return;
                }
                const item = presets.data.items.find((entry) => entry.name === oldName);
                if (!item) return;
                item.name = name;
                writePresets(presets.data);
            });
        };

        const deletePreset = (name) => {
            if (!window.confirm(`Delete preset "${name}"?`)) return;
            const presets = readPresets();
            if (presets.error) return;
            presets.data.items = presets.data.items.filter((item) => item.name !== name);
            writePresets(presets.data);
        };

        const origFilterCallback = filterWidget.callback;
        filterWidget.callback = function () {
            const callbackResult = origFilterCallback?.apply(this, arguments);
            state.lastApply = null;
            scheduleUpdate();
            return callbackResult;
        };

        const origOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, connected, link) {
            const changeResult = origOnConnectionsChange?.apply(this, arguments);
            if (type === INPUT_SLOT && isTargetSlot(node.inputs?.[index])) {
                // Loading a workflow reports every existing link exactly like a new one, so only append
                // outside of graph configuration. A cloned node is configured with all links nulled.
                if (connected && link && !app.configuringGraph) {
                    const target = resolveLinkTarget(node.graph, link);
                    if (target) appendRuleFor(target);
                }
                state.lastApply = null;
                scheduleUpdate();
            }
            return changeResult;
        };

        const origOnConfigure = node.onConfigure;
        node.onConfigure = function () {
            const configureResult = origOnConfigure?.apply(this, arguments);
            // Rebuild right away so the restored size is read against the full widget list. Doing it in the
            // deferred update() instead would add the buttons' height on top of a size that already has it,
            // growing the node on every load.
            state.resizeSuppressed = true;
            try {
                rebuildPresetButtons(readPresets().data.items);
                applyDebugVisibility();
            } finally {
                state.resizeSuppressed = false;
            }
            state.lastDebugText = null;
            scheduleUpdate();
            return configureResult;
        };

        const origOnAfterGraphConfigured = node.onAfterGraphConfigured;
        node.onAfterGraphConfigured = function () {
            const configuredResult = origOnAfterGraphConfigured?.apply(this, arguments);
            scheduleUpdate();
            return configuredResult;
        };

        const origOnAdded = node.onAdded;
        node.onAdded = function () {
            const addedResult = origOnAdded?.apply(this, arguments);
            livePresetNodes.add(node);
            scheduleUpdate();
            return addedResult;
        };

        const origOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            const removedResult = origOnRemoved?.apply(this, arguments);
            livePresetNodes.delete(node);
            if (state.updateTimer) clearTimeout(state.updateTimer);
            state.updateTimer = null;
            return removedResult;
        };

        const origGetExtraMenuOptions = node.getExtraMenuOptions;
        node.getExtraMenuOptions = (canvas, options) => {
            // Same convention as FloatSelector: return our entries instead of pushing onto `options`
            const extraOptions = origGetExtraMenuOptions?.call(node, canvas, options);
            const entries = [];
            const presets = readPresets();
            const items = presets.error ? [] : presets.data.items;
            if (items.length) {
                entries.push({
                    content: "✏️ Rename preset",
                    submenu: {
                        options: items.map((item) => ({
                            content: item.name,
                            callback: (value, menuOptions, event) => renamePreset(item.name, event),
                        })),
                    },
                });
                entries.push({
                    content: "🗑️ Delete preset",
                    submenu: {
                        options: items.map((item) => ({
                            content: item.name,
                            callback: () => deletePreset(item.name),
                        })),
                    },
                });
            }
            // resizeAround() sits out while collapsed, so toggling there would leave a stale height
            if (!node.flags?.collapsed) {
                const shown = node.properties?.[SHOW_DEBUG_PROPERTY] !== false;
                entries.push({
                    content: shown ? "🐞 Hide debug info" : "🐞 Show debug info",
                    callback: () => {
                        node.properties[SHOW_DEBUG_PROPERTY] = !shown;
                        update();
                        node.setDirtyCanvas(true, true);
                    },
                });
            }
            return Array.isArray(extraOptions) ? [...extraOptions, ...entries] : entries;
        };

        node.slowargoWidgetPreset = { scheduleUpdate };
        statusWidget.label = "No nodes connected";
        applyDebugVisibility();
        return result;
    };
}
