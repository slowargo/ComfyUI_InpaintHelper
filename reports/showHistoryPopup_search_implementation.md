# showHistoryPopup 添加本地增量搜索框实现方案

## 当前状态分析

`showHistoryPopup` 方法位于 `js/slowargo.js` 第 506-656 行，当前功能：
- 从后端获取历史记录列表
- 显示可滚动的弹窗，每条记录包含：内容、置顶按钮、删除按钮
- 支持点击内容回填、ESC 关闭、点击外部关闭

## UI 结构调整

```
┌─────────────────────────────┐
│ [🔍 Search...]              │  ← 新增搜索框
├─────────────────────────────┤
│ ┌─────────────────────────┐ │
│ │ 历史记录列表（可滚动）    │ │
│ │ - 条目 1                 │ │
│ │ - 条目 2                 │ │
│ └─────────────────────────┘ │
└─────────────────────────────┘
```

## 搜索逻辑

- **增量搜索**：输入时实时过滤，无需按回车
- **匹配方式**：不区分大小写的子串匹配
- **过滤范围**：仅对 `item.content` 字段进行匹配
- **空搜索**：显示全部记录

## 实现细节

### 数据管理
- 保留原始 `entries` 数组不变
- 搜索时过滤 `entries` 并重新渲染
- 删除/置顶操作后保持当前搜索状态

### 核心改动点

1. **创建搜索框和列表容器的包装结构**
   ```javascript
   const searchInput = $el("input", { ... });
   const listContainer = $el("div", { style: { ... } }); // 独立滚动容器
   const wrapper = $el("div", { style: { ... } }, [searchInput, listContainer]);
   ```

2. **修改 `renderList` 渲染到 `listContainer` 而非 `content`**
   ```javascript
   const renderList = (data) => {
       listContainer.innerHTML = "";
       // ... 渲染逻辑
   };
   ```

3. **添加搜索过滤逻辑**
   ```javascript
   searchInput.oninput = (e) => {
       const query = e.target.value.toLowerCase();
       const filtered = entries.filter(item => 
           item.content.toLowerCase().includes(query)
       );
       renderList(filtered);
   };
   ```

4. **焦点管理**
   ```javascript
   popup.show(wrapper);
   setTimeout(() => searchInput.focus(), 0);
   ```

## 完整代码实现

将 `showHistoryPopup` 方法（第 506-656 行）替换为以下代码：

```javascript
this.showHistoryPopup = async () => {
    const store_file = this.widgets.find(w => w.name === "store_file").value;

    let entries = [];
    try {
        const response = await api.fetchApi(`/slowargo_api/get_string_history?store_file=${encodeURIComponent(store_file)}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        entries = data.entries;
    } catch (error) {
        console.error("[slowargo.js] Error fetching string history:", error);
        alert("Error fetching history: " + error.message);
        return;
    }

    // 搜索框
    const searchInput = $el("input", {
        type: "text",
        placeholder: "🔍 Search...",
        style: {
            width: "100%",
            padding: "8px",
            fontSize: "12px",
            border: "1px solid #555",
            borderRadius: "5px",
            background: "#2a2a2a",
            color: "#ddd",
            marginBottom: "8px"
        }
    });

    // 列表容器（独立滚动）
    const listContainer = $el("div", {
        style: {
            maxHeight: "450px",
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: "8px"
        }
    });

    // 包装容器
    const wrapper = $el("div", {
        style: {
            minWidth: "600px",
            padding: "10px"
        }
    }, [searchInput, listContainer]);

    const renderList = (data) => {
        listContainer.innerHTML = "";
        const fragment = document.createDocumentFragment();
        data.forEach(item => {
            const row = $el("div", {
                style: {
                    display: "flex",
                    alignItems: "center",
                    padding: "5px",
                    background: "#353535",
                    borderRadius: "5px",
                    gap: "2px"
                }
            });

            const text = $el("div", {
                textContent: item.content,
                style: {
                    flex: 1,
                    alignSelf: "stretch",
                    display: "flex",
                    alignItems: "center",
                    padding: "5px 5px",
                    cursor: "pointer",
                    fontSize: "10px",
                    color: "#bbb",
                    whiteSpace: "pre-wrap",
                    transition: "background 0.2s"
                },
                onclick: () => {
                    this.widgets.find(w => w.name === "string").value = item.content;
                    popup.close();
                }
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
                        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                        const nextData = await res.json();
                        entries = nextData.entries;
                        renderList(searchInput.value ? entries.filter(i => 
                            i.content.toLowerCase().includes(searchInput.value.toLowerCase())
                        ) : entries);
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
                    opacity: 0.6
                },
                onclick: async (e) => {
                    e.stopPropagation();
                    if (confirm("Delete entry " + item.content + " ?")) {
                        try {
                            const res = await api.fetchApi("/slowargo_api/delete_string_history", {
                                method: "POST",
                                body: JSON.stringify({content: item.content, store_file})
                            });
                            if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                            const nextData = await res.json();
                            entries = nextData.entries;
                            renderList(searchInput.value ? entries.filter(i => 
                                i.content.toLowerCase().includes(searchInput.value.toLowerCase())
                            ) : entries);
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
            fragment.appendChild(row);
        });
        listContainer.appendChild(fragment);
    };

    // 搜索过滤
    searchInput.oninput = (e) => {
        const query = e.target.value.toLowerCase();
        const filtered = entries.filter(item => 
            item.content.toLowerCase().includes(query)
        );
        renderList(filtered);
    };

    renderList(entries);

    const popup = new (await import("../../../scripts/ui/dialog.js")).ComfyDialog();

    const handleEsc = (e) => {
        if (e.key === "Escape") popup.close();
    };
    window.addEventListener("keydown", handleEsc);

    const handleClickOutside = (e) => {
        if (!popup.element.contains(e.target)) popup.close();
    };
    setTimeout(() => window.addEventListener("click", handleClickOutside), 0);

    const originalClose = popup.close;
    popup.close = () => {
        window.removeEventListener("keydown", handleEsc);
        window.removeEventListener("click", handleClickOutside);
        originalClose.apply(popup);
    };

    popup.show(wrapper);
    setTimeout(() => searchInput.focus(), 0);
};
```

## 关键改进点

1. **保持搜索状态**：删除/置顶后保留当前搜索词，重新过滤
2. **焦点优化**：弹窗打开后自动聚焦搜索框
3. **布局优化**：搜索框固定，列表独立滚动
4. **最小改动**：复用现有逻辑，仅调整结构

## 代码量统计

- 原代码：约 150 行
- 新增核心功能：约 30 行
- 总代码：约 180 行

## 边界情况处理

- 搜索无结果时显示空列表
- 搜索框清空时恢复全部列表
- 删除/置顶操作后保持当前搜索状态

## 性能考虑

- 使用 `DocumentFragment` 优化渲染
- 搜索过滤在客户端进行，无需额外请求
- 对于大量数据（>1000 条），可考虑防抖（debounce）
