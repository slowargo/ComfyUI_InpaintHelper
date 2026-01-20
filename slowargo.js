import { ComfyApp } from "../../scripts/app.js";
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// const nodeImageCache = new WeakMap();

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
        window.addEventListener('keydown', function(e) {
            // 检查是否按下 Ctrl+Z (Cmd+Z on Mac)
            if ((e.ctrlKey || e.metaKey) && e.key === 'z' && document.querySelector("div.maskEditor_sidePanel")) {
                e.preventDefault(); // 阻止浏览器默认的撤销行为 (Undo textarea)
                //e.stopImmediatePropagation(); // 阻止其他可能的脚本处理
                console.log('[slowargo.js] preventDefault for Ctrl+Z ');
            }
        }, true); // 注意这里的 true，确保在捕获阶段第一时间拦截

        // api.removeEventListener("executed", this._handleHotReload);
        // api.addEventListener("executed", async (event) => {
        //     console.log("executed", event)
        // })
        // api.addEventListener("slowargo.js.extension.FloatSwitch", async (event) => {
        //     console.log("slowargo.js.extension.FloatSwitch executed", event)
        // })
        // const ComfyDialog = app.ui.dialog.constructor;
        // const oldcreateButtons = ComfyDialog.prototype.createButtons;
        //
        // ComfyDialog.prototype.createButtons = function(...args) {
        //     const res = oldcreateButtons.apply(this, args);
        //     console.log("[slowargo.js] createButtons", this.constructor.name, args)
        //     if (this.constructor.name === 'MaskEditorDialog') {
        //         console.log("[slowargo.js]", this)
        //         // if (useNewEditor) {
        //         //     this.addButton("mask_editor_button", "Mask Editor", () => {
        //         //         app.ui.maskEditor.show(this.node);
        //         //     });
        //         // }
        //     }
        // }

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

        } else if (nodeType?.comfyClass == "LoadImageFromOutputsPlus") {
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
                        // if (!inputPath) {
                        //     alert("Please select a folder first");
                        //     return;
                        // }

                        // // Get widget values
                        // const startIndexWidget = this.widgets.find(w => w.name === "start_index");
                        // const stopIndexWidget = this.widgets.find(w => w.name === "stop_index");
                        //
                        // if (stopIndexWidget.value <= startIndexWidget.value) {
                        //     alert("Stop index must be greater than start index");
                        //     return;
                        // }

                        const options = {
                            input_path: inputPath,
                            // include_subfolders: this.widgets.find(w => w.name === "include_subfolders")?.value ?? true,
                            // sort_method: this.widgets.find(w => w.name === "sort_method")?.value ?? "alphabetical",
                            // filter_type: this.widgets.find(w => w.name === "filter_type")?.value ?? "none",
                            // start_index: startIndexWidget.value,
                            // stop_index: stopIndexWidget.value,
                            // load_limit: parseInt(this.widgets.find(w => w.name === "load_limit")?.value || "1000")
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
                        // const availableCountWidget = this.widgets?.find(w => w.name === "available_image_count");

                        if (imageWidget && result.image_name?.length) {
                            // console.log("image_name", result.image_name)
                            imageWidget.options.values = result.image_name;
                            imageWidget.value = result.image_name[0];
                            imageWidget.callback.call(imageWidget);
                        }
                        // if (imageWidget && result.thumbnails?.length) {
                        //     imageWidget.options.values = result.thumbnails;
                        //     imageWidget.value = result.thumbnails[0];
                        //     this.imageOrder = result.image_order || {};
                        //     imageWidget.callback.call(imageWidget);
                        // }

                        // // Update the available count
                        // if (availableCountWidget) {
                        //     availableCountWidget.value = result.total_images;
                        // }
                        //
                        // // Update stop_index if needed
                        // if (result.stop_index !== stopIndexWidget.value) {
                        //     stopIndexWidget.value = result.stop_index;
                        // }

                    } catch (error) {
                        console.error("Error refreshing previews:", error);
                        // alert("Error refreshing previews: " + error.message);
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

                const refreshFn = async function() {
                    try {
                        // console.log("[slowargo.js] LoadRecentImagePlusV1 refresh",this, node);

                        const options = {
                            watch_folders: node.widgets.find(w => w.name === "watch_folders")?.value ?? "",
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

                // In the refresh button callback
                const refreshBtn = this.addWidget("button", "refresh", "", refreshFn);

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
        } 

        console.log("[slowargo.js] init done", nodeType?.comfyClass)
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
                // app.extensionManager.dialog.showAboutDialog();
                // const isMaskEditorOpen =
                //     (typeof ComfyApp.maskeditor_is_opended === 'function' && ComfyApp.maskeditor_is_opended()) ||
                //     (ComfyApp.clipspace_return_node !== undefined && ComfyApp.clipspace_return_node !== null);
                //
                // console.log("Hi", isMaskEditorOpen)
                // console.log("Hi", window.comfyAPI)

                // Switch to Mask tool
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
                        console.log('用户取消了拾色，或者其他错误', e);
                    }
                }
                // 2. 如果不支持 EyeDropper (如 Firefox/Safari)，回退到普通面板
                else {
                    console.log('当前浏览器不支持 EyeDropper API，回退到普通面板');
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
                let btn = document.querySelectorAll("#global-mask-editor button i.pi-check")
                if (btn.length > 0) {
                    btn[0].click();
                }
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