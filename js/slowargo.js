import { ComfyApp } from "../../scripts/app.js";
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { $el } from "../../scripts/ui.js";
import { initTurboMode, performMaskSave } from "./turboMode.js";

app.registerExtension({
    name: "slowargo.js.extension",
    async setup() {
        // console.log("Hi from slowargo.js!");
        if (ComfyApp.maskeditor_is_opended == null || ComfyApp.maskeditor_is_opended == undefined) {
            // monkey patch ComfyApp.maskeditor_is_opended
            ComfyApp.maskeditor_is_opended = function() {
                if (document.querySelector("div.maskEditor_sidePanel")) {
                    return true;
                }
                return false;
            }
        }
        // Prevent accidentally undoing canvas textarea when pressing Ctrl+Z in mask editor
        window.addEventListener('keydown', function(e) {
            // Check if Ctrl+Z (Cmd+Z on Mac) is pressed in mask editor
            if ((e.ctrlKey || e.metaKey) && e.key === 'z' && document.querySelector("div.maskEditor_sidePanel")) {
                e.preventDefault(); // Prevent browser's default undo behavior (textarea undo)
                // Don't to this. It will break undoing in the mask editor
                //e.stopImmediatePropagation(); // Prevent other possible script handling.
                // console.log('[slowargo.js] preventDefault for Ctrl+Z ');
            }
        }, true); // true to ensure interception at the capture phase

        // Initialize Turbo Mode
        initTurboMode();

        // api.removeEventListener("executed", this._handleHotReload);
        // api.addEventListener("executed", async (event) => {
        //     console.log("executed", event)
        // })
        // api.addEventListener("slowargo.js.extension.FloatSwitch", async (event) => {
        //     console.log("slowargo.js.extension.FloatSwitch executed", event)
        // })

        // Auto open saved image in new tab
        api.addEventListener("slowargo.js.extension.SaveImageToFileName", async (event) => {
            // console.log("slowargo.js.extension.SaveImageToFileName executed", event)
            if (event?.detail?.results) {
                for (const result of event.detail.results) {
                    const subfolder = result.subfolder;
                    const imageName = result.filename;
                    const type = result.type;

                    const params = `${app.getPreviewFormatParam?.() || ""}${app.getRandParam?.() || ""}`;
                    const imgUrl = api.apiURL(`/view?filename=${encodeURIComponent(imageName)}${subfolder ? `&subfolder=${encodeURIComponent(subfolder)}` : ''}&type=output${params}`);
                    console.log("[slowargo.js] SaveImageToFileName executed", result, imgUrl);
                    window.open(imgUrl, '_blank');
                }
            }
        })

    },
    // async nodeCreated(node) {
    //     console.log("[slowargo.js] nodeCreated", node.id);
    // },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.python_module != "custom_nodes.Slowargo") return
        if (nodeType?.comfyClass == "FloatSwitch" ) {
            // console.log("[slowargo.js]", nodeType.prototype.onExecuted)
            //
            // const origOnExecuted = nodeType.prototype.onExecuted;
            // nodeType.prototype.onExecuted = function(output) {
            //     console.log("[slowargo.js] onExecuted", this, "output:", output);
            //     result = origOnExecuted?.apply(this, arguments);
            //     return result;
            // };
            //
            // const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            // nodeType.prototype.onNodeCreated = function() {
            //     const result = origOnNodeCreated?.apply(this, arguments);
            //     console.log("[slowargo.js] onNodeCreated", this);
            //     console.log("[slowargo.js] onNodeCreated onExecuted", this.onExecuted);
            //     return result;
            // }
            //

            // 目前切换 toggle 也会因 cached 没有实际执行，没有事件触发更新
            // const origonConfigure = nodeType.prototype.onConfigure;
            // // console.log("[slowargo.js]", origonConfigure)
            // nodeType.prototype.onConfigure = function(data) {
            //     const result = origonConfigure?.apply(this, arguments);
            //     // console.log("[slowargo.js] onConfigure", this.id);
            //     // console.log("[slowargo.js] onConfigure data", data.id);
            //     let myId = this.id;
            //     // api.addEventListener("executed", async (event) => {
            //     api.addEventListener("slowargo.js.extension.FloatSwitch", async (event) => {
            //         // console.log("event?.detail?.node", event?.detail?.node_id, myId);
            //         if (event?.detail?.node_id == myId) {
            //             // console.log("executed", event);
            //             let value = event.detail.selected_value;
            //             this.outputs[0].localized_name = value;
            //             console.log("executed", value, this);
            //             app.graph.setDirtyCanvas(true);
            //         }
            //     })
            //     return result;
            // }

        } else if (nodeType?.comfyClass == "LoadImageFromOutputsPlus") { // deprecated V3 extension
            console.log("[slowargo.js]", nodeData)
            console.log("[slowargo.js]", nodeType)

            const origOnNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                console.log("[slowargo.js] LoadImageFromOutputsPlus onNodeCreated")
                const result = origOnNodeCreated?.apply(this, arguments);
                // Image widget setup
                const imageWidget = this.widgets?.find(w => w.name === "image");
                if (imageWidget) {
                    // Store original methods
                    const origSetValue = imageWidget.setValue;
                    const origCallback = imageWidget.callback;

                    imageWidget.setValue = function(v, skip_callback) {
                        const result = origSetValue?.call(this, v, skip_callback);
                        if (v && !skip_callback) {
                            updateNodePreview(this.node, v);
                        }
                        return result;
                    };

                    // Update preview on value change
                    imageWidget.callback = function(value) {
                        if (origCallback) {
                            origCallback.call(this, value);
                        }
                        updateNodePreview(this.node, this.value);
                    };

                    imageWidget.callback.call(imageWidget);
                }

                // I borrow the idea from https://github.com/if-ai/ComfyUI_IF_AI_LoadImages/blob/main/web/js/IFLoadImagesNodeS.js
                // In the refresh button callback
                const refreshBtn = this.addWidget("button", "refresh_preview", "Refresh Previews 🔄", async () => {
                    try {
                        const inputPath = this.widgets.find(w => w.name === "image_folder")?.value;

                        const options = {
                            input_path: inputPath,
                        };

                        const response = await api.fetchApi("/slowargo_api/refresh_previews", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(options)
                        });

                        if (!response.ok) throw new Error(await response.text());
                        const result = await response.json();

                        if (!result.success) throw new Error(result.error);

                        // Update widgets
                        const imageWidget = this.widgets?.find(w => w.name === "image");

                        if (imageWidget && result.image_name?.length) {
                            imageWidget.options.values = result.image_name;
                            imageWidget.value = result.image_name[0];
                            imageWidget.callback.call(imageWidget);
                        }
                    } catch (error) {
                        console.error("Error refreshing previews:", error);
                    }
                });

                // Arrange widgets - move refresh button below image widget
                if (refreshBtn) {
                    const widgets = this.widgets.splice(-1); // Only remove one widget (refreshBtn)
                    this.widgets.splice(imageWidget ? this.widgets.indexOf(imageWidget) + 1 : 0, 0, ...widgets);
                }

                // Handle execution results
                this.onExecuted = function(output) {
                    console.log("[slowargo.js] onExecuted")
                    // if (output?.ui?.values) {
                    //     const imageWidget = this.widgets?.find(w => w.name === "image_name");
                    //     // const availableCountWidget = this.widgets?.find(w => w.name === "available_image_count");
                    //     // const maxImagesWidget = this.widgets?.find(w => w.name === "max_images");
                    //
                    //     if (imageWidget) {
                    //         // Store path mapping and image order
                    //         this.pathMapping = output.ui.values.path_mapping || {};
                    //         this.imageOrder = output.ui.values.image_order || {};
                    //
                    //         // Update widget options
                    //         if (output.ui.values.images) {
                    //             imageWidget.options.values = output.ui.values.images;
                    //
                    //             // Update available count and limits
                    //             // const count = output.ui.values.available_image_count;
                    //             // if (availableCountWidget) {
                    //             //     availableCountWidget.value = count;
                    //             // }
                    //             // if (maxImagesWidget) {
                    //             //     maxImagesWidget.options.max = count;
                    //             //     if (maxImagesWidget.value > count) {
                    //             //         maxImagesWidget.value = count;
                    //             //     }
                    //             // }
                    //         }
                    //
                    //         // Handle current selection
                    //         if (output.ui.values.current_thumbnails?.length > 0) {
                    //             const currentValue = imageWidget.value;
                    //             if (!this.pathMapping[currentValue]) {
                    //                 imageWidget.value = output.ui.values.current_thumbnails[0];
                    //             }
                    //         }
                    //
                    //         // Update preview
                    //         if (imageWidget.value) {
                    //             updateNodePreview(this, imageWidget.value);
                    //         }
                    //     }
                    // }
                };

                // Handle widget changes
                const origOnWidgetChanged = nodeType.prototype.onWidgetChanged;
                nodeType.prototype.onWidgetChanged = function (name, value) {
                    if (origOnWidgetChanged) {
                        origOnWidgetChanged.apply(this, arguments);
                    }

                    console.log("[slowargo.js] onWidgetChanged", name, value)
                    // Auto-refresh on certain changes
                    // if (["include_subfolders", "filter_type", "sort_method"].includes(name)) {
                    //     this.refreshPreviews();
                    // }
                };

                return result;
            }
        } else if (nodeType?.comfyClass == "LoadImageFromOutputPlusV1") {
            // TODO add sub_folder to remote route
            // console.log("[slowargo.js] LoadImageFromOutputPlusV1", nodeData, nodeType.prototype.constructor)
            // console.log("[slodole.log("[slowargo.js] onWidgetChanged", name, value)

            // Handle widget changes
            const origOnWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function (name, value) {
                if (origOnWidgetChanged) {
                    origOnWidgetChanged.apply(this, arguments);
                }

                console.log("[slowargo.js] onWidgetChanged", name, value)
                // Auto-refresh on certain changes
                // if (["include_subfolders", "filter_type", "sort_method"].includes(name)) {
                //     this.refreshPreviews();
                // }
            };

            const origOnNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                // console.log("[slowargo.js] LoadImageFromOutputPlusV1 onNodeCreated", this.widgets)
                const result = origOnNodeCreated?.apply(this, arguments);

                // Image widget setup
                const imageWidget = this.widgets?.find(w => w.name === "image");
                if (imageWidget) {
                    // Store original methods
                    // const origSetValue = imageWidget.setValue;
                    // const origCallback = imageWidget.callback;
                    //
                    // imageWidget.setValue = function(v, skip_callback) {
                    //     console.log("[slowargo.js] LoadImageFromOutputPlusV1 setValue", v)
                    //     const result = origSetValue?.call(this, v, skip_callback);
                    //     if (v && !skip_callback) {
                    //         updateNodePreview(this.node, v);
                    //     }
                    //     return result;
                    // };

                    // Update preview on value change
                    // imageWidget.callback = function(value) {
                    //     console.log(this.node)
                    //     console.log(value)
                    //     // 这里替换没用。call里会走 useImageUploadWidget.transform
                    //     // 在 formatPath 覆盖回去
                    //     // let node = this.node;
                    //     // if (node?.images?.length) {
                    //     //     for (let i = 0; i < node.images.length; i++) {
                    //     //         const image = node.images[i];
                    //     //         if (image.subfolder === 'clipspace' && image.filename) {
                    //     //             // Replace [output] suffix with [input] in filename
                    //     //             if (image.filename && image.filename.includes('[output]')) {
                    //     //                 image.filename = image.filename.replace(/\[output\]$/, '[input]');
                    //     //             }
                    //     //         }
                    //     //     }
                    //     // }
                    //     // console.log(this.node)
                    //     if (origCallback) {
                    //         // 会走到 useImageUploadWidget，不会用 第二个参数
                    //         // origCallback.call(this, processedValue);
                    //         origCallback.call(this);
                    //     }
                    //     // updateNodePreview(this.node, value);
                    // };

                    // imageWidget.callback.call(imageWidget);
                }
            };
        } else if (nodeType?.comfyClass == "LoadRecentImagePlusV1") {
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const node = this;

                const result = origOnNodeCreated?.apply(this, arguments);
                // console.log("[slowargo.js] LoadRecentImagePlusV1 onNodeCreated", this.widgets);
                // console.log("[slowargo.js] LoadRecentImagePlusV1 onNodeCreated", this);

                const imageWidget = this.widgets?.find(w => w.name === "image");
                if (imageWidget) {
                    const origCallback = imageWidget.callback;
                    imageWidget.callback = function(value) {
                        // console.log("[slowargo.js] LoadRecentImagePlusV1 callback", node);
                        // Make the node selected. Do what processSelect() does.
                        try {
                            // If another node has been selected, it becomes multiselection if we don't call this first.
                            app.canvas.deselectAll(node);
                            app.canvas.select(node);
                            // It's deprecated but we still need to call this to bring the toolbox up
                            app.canvas.onSelectionChange?.(app.canvas.selected_nodes)
                            app.canvas.setDirty(node);
                        } catch (e) {
                            console.error("Failed to select the node", e);
                        }
                        origCallback?.call(this);
                    };
                }
                // const refreshWidget = this.widgets?.find(w => w.name === "refresh");
                // if (refreshWidget) {
                //     refreshWidget.hidden = true;
                // }

                const refreshFn = async function(customWatchFolders) {
                    try {
                        // console.log("[slowargo.js] LoadRecentImagePlusV1 refresh",this, node);

                        const options = {
                            watch_folders: customWatchFolders || node.widgets.find(w => w.name === "watch_folders")?.value || "",
                        };

                        const response = await api.fetchApi("/slowargo_api/refresh_previews_recent", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(options)
                        });

                        if (!response.ok) throw new Error(await response.text());
                        const result = await response.json();

                        if (!result.success) throw new Error(result.error);

                        // Update widgets
                        const imageWidget = node.widgets?.find(w => w.name === "image");
                        // const availableCountWidget = this.widgets?.find(w => w.name === "available_image_count");

                        if (imageWidget && result.image_name?.length) {
                            // console.log("image_name", result.image_name)
                            imageWidget.options.values = result.image_name;
                            imageWidget.value = result.image_name[0];
                            imageWidget.callback.call(imageWidget);
                        }

                        // Hold shift and click refresh will open the mask editor after refreshing
                        if (app.shiftDown) {
                            ComfyApp.clipspace_return_node = node;
                            ComfyApp.open_maskeditor?.();
                        }

                    } catch (e) {
                        console.error("Error refreshing previews:", e);
                    }
                }

                // Expose refreshFn for external use
                node.refreshFn = refreshFn;

                // In the refresh button callback
                const refreshBtn = this.addWidget("button", "refresh", "",  () => {
                    this.refreshFn();
                });

                // Arrange widgets - move refresh button below watch_folders widget
                if (refreshBtn) {
                    const targetWidget = this.widgets?.find(w => w.name === "watch_folders");
                    const widgets = this.widgets.splice(-1); // Only remove one widget (refreshBtn)
                    this.widgets.splice(targetWidget ? this.widgets.indexOf(targetWidget) + 1 : 0, 0, ...widgets);
                }

                this.handleAction = async function(action) {
                    // console.log("[slowargo.js] handleAction", action);
                    if (action === "Refresh") {
                        await refreshFn();
                    }
                }

                this.constructor.exposedActions = ["Refresh"];

                return result;
            }
        } else if (nodeType?.comfyClass === "RefreshTriggerV1") {
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const node = this;
                const result = origOnNodeCreated?.apply(this, arguments);

                const refreshFn = async function() {
                    try {
                        // Find connected LoadRecentImagePlusV1 node via trigger input
                        const triggerInput = node.inputs?.find(i => i.name === "trigger");
                        if (!triggerInput || triggerInput.link == null) {
                            console.warn("[RefreshTriggerV1] No connected node. Connect trigger to a LoadRecentImagePlusV1 output.");
                            return;
                        }

                        const link = app.graph.links[triggerInput.link];
                        if (!link) return;

                        const targetNode = app.graph.getNodeById(link.origin_id);
                        if (!targetNode || targetNode.comfyClass !== "LoadRecentImagePlusV1") {
                            console.warn("[RefreshTriggerV1] Connected node is not LoadRecentImagePlusV1, got:", targetNode?.comfyClass);
                            return;
                        }

                        // Use this node's watch_folders value and call target's refreshFn
                        const watchFolders = node.widgets.find(w => w.name === "watch_folders")?.value ?? "";
                        await targetNode.refreshFn?.(watchFolders);

                    } catch (e) {
                        console.error("[RefreshTriggerV1] Error refreshing:", e);
                    }
                };

                const refreshBtn = this.addWidget("button", "refresh", "", refreshFn);

                // Arrange widgets - move refresh button below watch_folders widget
                if (refreshBtn) {
                    const targetWidget = this.widgets?.find(w => w.name === "watch_folders");
                    const widgets = this.widgets.splice(-1);
                    this.widgets.splice(targetWidget ? this.widgets.indexOf(targetWidget) + 1 : 0, 0, ...widgets);
                }

                this.handleAction = async function(action) {
                    if (action === "Refresh") {
                        await refreshFn();
                    }
                };

                this.constructor.exposedActions = ["Refresh"];

                return result;
            };
        } else if (nodeType?.comfyClass == "RunButtonNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // 找到我们定义的 trigger_count 挂件
                const widget = this.widgets.find((w) => w.name === "trigger_count");
                // widget.type = "hidden"; // 隐藏原始输入框
                widget.hidden = true;

                const runFn = async function () {
                    // 逻辑：增加计数器触发后端更新
                    widget.value += 1;

                    // 核心：手动触发整个工作流的执行
                    app.queuePrompt(0);
                }

                this.addWidget("button", "Run Prompt", null, runFn);

                this.handleAction = async function (action) {
                    if (action === "Run") {
                        await runFn();
                    }
                }

                this.constructor.exposedActions = ["Run"];

                return r;
            };
        } else if (nodeType?.comfyClass === "RememberStrings") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                this.addWidget("button", "View History 📂", null, () => {
                    this.showHistoryPopup();
                });

                // 弹出窗口逻辑
                this.showHistoryPopup = async () => {
                    const store_file = this.widgets.find(w => w.name === "store_file").value;

                    // 1. 请求后端获取数据
                    let entries = []; // Initialize entries outside try block
                    try {
                        const response = await api.fetchApi(`/slowargo_api/get_string_history?store_file=${encodeURIComponent(store_file)}`);
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        const data = await response.json();
                        entries = data.entries; // Directly assign as entries is guaranteed
                    } catch (error) {
                        console.error("[slowargo.js] Error fetching string history:", error);
                        alert("Error fetching history: " + error.message);
                    }
                    // console.log("[slowargo.js] get_string_history", entries);

                    // 2. 创建 Popup 内容
                    const content = $el("div", {
                        style: {
                            minWidth: "600px",
                            maxHeight: "500px",
                            overflowY: "auto",
                            display: "flex",
                            flexDirection: "column",
                            gap: "8px",
                            padding: "10px"
                        }
                    });

                    const renderList = (data) => {
                        content.innerHTML = ""; // Clear existing content
                        const fragment = document.createDocumentFragment(); // Create a document fragment
                        data.forEach(item => {
                            const row = $el("div", {
                                style: {
                                    display: "flex",
                                    alignItems: "center",
                                    padding: "5px",
                                    background: "#353535",
                                    borderRadius: "5px",
                                    // overflow: "hidden", // 确保子元素撑满时不会超出圆角
                                    gap: "2px"
                                }
                            });

                            // 点击内容回填并关闭
                            const text = $el("div", {
                                textContent: item.content,
                                style: {
                                    flex: 1,
                                    alignSelf: "stretch",    // 关键：高度撑满父容器
                                    display: "flex",        // 关键：为了内部文字垂直居中
                                    alignItems: "center",   // 关键：文字垂直居中
                                    padding: "5px 5px",    // 这里保留你需要的文字上下边距
                                    cursor: "pointer",
                                    fontSize: "10px",
                                    color: "#bbb",
                                    whiteSpace: "pre-wrap",
                                    transition: "background 0.2s"
                                },
                                onclick: () => {
                                    this.widgets.find(w => w.name === "string").value = item.content;
                                    popup.close(); // 选中后自动关闭
                                },
                                // 只在文本感应区显示悬停效果
                                // onmouseenter: (e) => e.target.style.background = "#454545",
                                // onmouseleave: (e) => e.target.style.background = "transparent"
                            });

                            const pinBtn = $el("button", {
                                textContent: item.pinned ? "📌" : "📍",
                                style: {
                                    background: "none",
                                    border: "none",
                                    cursor: "pointer",
                                    fontSize: "14px",
                                    opacity: item.pinned ? 1 : 0.3
                                },
                                onclick: async (e) => {
                                    e.stopPropagation();
                                    try {
                                        const res = await api.fetchApi("/slowargo_api/toggle_string_history_pin", {
                                            method: "POST",
                                            body: JSON.stringify({content: item.content, store_file})
                                        });
                                        if (!res.ok) {
                                            throw new Error(`HTTP error! status: ${res.status}`);
                                        }
                                        const nextData = await res.json();
                                        renderList(nextData.entries); // 局部刷新，直接使用 nextData.entries
                                    } catch (error) {
                                        console.error("[slowargo.js] Error toggling pin status:", error);
                                        alert("Error toggling pin status: " + error.message);
                                    }
                                }
                            });

                            const deleteBtn = $el("button", {
                                textContent: "🗑️",
                                style: {
                                    background: "none",
                                    border: "none",
                                    cursor: "pointer",
                                    fontSize: "14px",
                                    opacity: 0.6,
                                    // padding: "0 5px"
                                },
                                onclick: async (e) => {
                                    e.stopPropagation(); // 防止触发回填逻辑
                                    if (confirm("Delete entry " + item.content + " ?")) {
                                        try {
                                            const res = await api.fetchApi("/slowargo_api/delete_string_history", {
                                                method: "POST",
                                                body: JSON.stringify({content: item.content, store_file})
                                            });
                                            if (!res.ok) {
                                                throw new Error(`HTTP error! status: ${res.status}`);
                                            }
                                            const nextData = await res.json();
                                            renderList(nextData.entries); // 刷新列表，直接使用 nextData.entries
                                        } catch (error) {
                                            console.error("[slowargo.js] Error deleting history entry:", error);
                                            alert("Error deleting entry: " + error.message);
                                        }
                                    }
                                }
                            });

                            row.appendChild(text);
                            row.appendChild(pinBtn);
                            row.appendChild(deleteBtn);
                            fragment.appendChild(row); // Append to fragment instead of direct content
                        });
                        content.appendChild(fragment); // Append fragment to content once
                    };

                    renderList(entries);

                    // 3. 使用 ComfyUI 内部 Dialog 弹出
                    const popup = new (await import("../../../scripts/ui/dialog.js")).ComfyDialog();

                    // 手动 Esc 监听函数
                    const handleEsc = (e) => {
                        if (e.key === "Escape") {
                            popup.close();
                        }
                    };
                    window.addEventListener("keydown", handleEsc);

                    // 点击对话框外部区域关闭
                    const handleClickOutside = (e) => {
                        // popup.element 是整个弹窗的容器，包含遮罩层和内容
                        // 如果点击事件的目标不在 popup.element 内部，则关闭弹窗
                        if (!popup.element.contains(e.target)) {
                            popup.close();
                        }
                    };
                    // 延迟添加监听器，避免本次点击立即关闭
                    setTimeout(() => {
                        window.addEventListener("click", handleClickOutside);
                    }, 0);

                    // 修改 popup 的 close 方法，确保点击遮罩层关闭时也能移除监听
                    const originalClose = popup.close;
                    popup.close = () => {
                        window.removeEventListener("keydown", handleEsc);
                        window.removeEventListener("click", handleClickOutside); // 移除点击外部监听
                        originalClose.apply(popup);
                    };

                    popup.show(content);
                }
            }
        }

        // console.log("[slowargo.js] init done", nodeType?.comfyClass)
    },
    // async afterConfigureGraph(graph) {
    //     console.log("[slowargo.js] afterConfigureGraph", graph)
    // }
    commands: [
        {
            id: "slowargo.js.extension.open-external-link",
            label: "Open Image",
            icon: "pi pi-external-link",
            function: async () => {
                const selectedItems = app.canvas.selectedItems;
                if (!selectedItems || selectedItems.size === 0) {
                    console.warn("[slowargo.js] 没有选中任何节点");
                    return;
                }
                // 遍历所有选中的节点（支持多选！）
                selectedItems.forEach(node => {
                    // 图片查找逻辑
                    let imgUrl = null;
                    for (const wid of node.widgets || []) {
                        if (wid.constructor.name === "ImagePreviewWidget") {
                            if (wid.node?.imgs?.length > 0) {
                                imgUrl = wid.node.imgs[0].currentSrc;
                                break;
                            }
                        }
                    }

                    if (imgUrl) {
                        // console.log("[slowargo.js] 找到图片，新标签打开：", imgUrl, getComfyFilePathFromViewUrl(imgUrl));
                        window.open(imgUrl, '_blank');
                    }
                });
            }
        },
        {
            id: "slowargo.js.extension.maskeditor.mask",
            label: "Switch to Mask",
            function: async () => {
                // Switch to the Mask tool in the Mask Editor
                let ind = document.querySelectorAll("div.maskEditor_toolPanelIndicator")
                if (ind.length > 1) {
                    ind[0].click();
                }
            }
        },
        {
            id: "slowargo.js.extension.maskeditor.eyedropper",
            label: "Switch to Eye Dropper",
            function: async () => {
                // Switch to the Pen tool and enter color picking in the Mask Editor
                const colorInput = document.querySelector("div.maskEditor_sidePanel input[type=color]");
                if (!colorInput) {
                    return
                }
                // 1. 检查浏览器是否支持 EyeDropper API (Chrome/Edge 支持)
                if ('EyeDropper' in window) {
                    const eyeDropper = new EyeDropper();

                    try {
                        // 打开屏幕拾色器
                        const result = await eyeDropper.open();

                        // 获取颜色并赋值给 input
                        const hexColor = result.sRGBHex;
                        colorInput.value = hexColor;

                        // 触发 input 事件（如果你的业务逻辑依赖 input 事件）
                        colorInput.dispatchEvent(new Event('input', { bubbles: true }));

                        // console.log('拾取到的颜色:', hexColor);
                        // document.body.style.backgroundColor = hexColor; // 示例效果
                    } catch (e) {
                        // 用户按了 ESC 取消，或者其他错误
                        console.log('Color picking cancelled, or other error occurred', e);
                    }
                }
                // 2. 如果不支持 EyeDropper (如 Firefox/Safari)，回退到普通面板
                else {
                    console.log('Current browser does not support EyeDropper API, falling back to regular panel');
                    colorInput.click();
                }

                // Switch to pen tool
                let ind = document.querySelectorAll("div.maskEditor_toolPanelIndicator")
                if (ind.length > 1) {
                    ind[1].click();
                }
            }
        },
        {
            id: "slowargo.js.extension.maskeditor.save",
            label: "Save Mask",
            function: async () => {
                await performMaskSave();
            }
        },
    ],
    keybindings: [
        {
            combo: { key: "x", ctrl: true },
            commandId: "slowargo.js.extension.open-external-link"
        },
        {
            combo: { key: "m", ctrl: true, alt: true },
            commandId: "slowargo.js.extension.maskeditor.mask"
        },
        {
            combo: { key: "c", ctrl:true, alt: true },
            commandId: "slowargo.js.extension.maskeditor.eyedropper"
        },
        {
            combo: { key: "s", ctrl:true, alt: true },
            commandId: "slowargo.js.extension.maskeditor.save"
        }
    ],
    getSelectionToolboxCommands: (selectedItem) => {
        if (selectedItem?.widgets?.some(w => w.constructor.name === "ImagePreviewWidget")) {
            return ["slowargo.js.extension.open-external-link"];
        }
        return [];
    }
})

function getComfyFilePathFromViewUrl(url) {
    try {
        const urlObj = new URL(url);
        const params = urlObj.searchParams;

        const type = params.get('type') || 'output';           // 默认 output
        const filename = params.get('filename');
        let subfolder = params.get('subfolder') || '';         // 可能为空

        if (!filename) {
            throw new Error("URL 中缺少 filename 参数");
        }

        // 拼接，subfolder 为空时自动处理双斜杠
        const path = [type, subfolder, filename]
            .filter(part => part !== undefined && part !== null && part !== '')
            .join('/')
            .replace(/\/+/g, '/');  // 防止多余斜杠

        return path;
    } catch (err) {
        console.error("[getComfyFilePath] 解析失败:", err);
        return null;
    }
}

function updateNodePreview(node, imageName) {
    if (!imageName || !node) return;
    // 如果 value 是 clipspace，annotation 切成 input
    // 解析路径，如果父目录为 clipspace，替换文件名末尾的 [output] 为 [input]
    // let processedValue = imageName;
    // const pathParts = imageName.split('/');
    // if (pathParts.length >= 2 && pathParts[pathParts.length - 2] === "clipspace") {
    //     // 检查文件名是否以 [output] 结尾
    //     if (pathParts[pathParts.length - 1].endsWith("[output]")) {
    //         pathParts[pathParts.length - 1] = pathParts[pathParts.length - 1].replace("[output]", "[input]");
    //         processedValue = pathParts.join('/');
    //     }
    // }
    let { filename, subfolder } = parseFilePath(imageName)

    // 检查子文件夹是否为 "clipspace"
    if (subfolder === "clipspace") {
        // 检查文件名是否以 [output] 结尾
        if (filename.endsWith("[output]")) {
            // const newFilename = filename.replace("[output]", "[input]")
            // // 重新组合路径
            // processedValue =  subfolder ? `${subfolder}/${newFilename}` : newFilename
            filename = filename.replace("[output]", "[input]")
        }
    }

    console.log("[slowargo.js] updateNodePreview", imageName, subfolder, filename);
    imageName = filename;

    const img = new Image();
    img.onload = () => {
        node.imgs = [img];
        app.graph.setDirtyCanvas(true);
    };

    img.onerror = () => {
        console.warn(`Failed to load preview for ${imageName}`, node);
    };

    // Get the input directory path
    const inputPathWidget = node.widgets?.find(w => w.name === "input_path");
    const inputPath = inputPathWidget?.value || "";

    // Construct URL with proper path handling
    //const params = `&type=input${app.getPreviewFormatParam?.() || ""}${app.getRandParam?.() || ""}`;
    const params = `${app.getPreviewFormatParam?.() || ""}${app.getRandParam?.() || ""}`;

    // Use the thumbnail name directly since it's already in the input directory
    img.src = api.apiURL(`/view?filename=${encodeURIComponent(imageName)}${subfolder ? `&subfolder=${encodeURIComponent(subfolder)}` : ''}&type=output${params}`);
}

function parseFilePath(filepath) {
    if (!filepath?.trim()) return { filename: '', subfolder: '' }

    const normalizedPath = filepath
      .replace(/[\\/]+/g, '/') // Normalize path separators
      .replace(/^\//, '') // Remove leading slash
      .replace(/\/$/, '') // Remove trailing slash

    const lastSlashIndex = normalizedPath.lastIndexOf('/')

    if (lastSlashIndex === -1) {
        return {
            filename: normalizedPath,
            subfolder: ''
        }
    }

    return {
        filename: normalizedPath.slice(lastSlashIndex + 1),
        subfolder: normalizedPath.slice(0, lastSlashIndex)
    }
}