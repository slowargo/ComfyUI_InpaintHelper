# Widget Preset 节点设计方案

- 状态：已实现（`nodes_preset.py`、`js/widgetPreset.js`）
- 节点名：`WidgetPreset`，显示名 `Widget Preset`
- 对照的前端版本：ComfyUI_frontend 1.37.2（`run_cpu.bat` 通过 `--front-end-version` 实际加载的版本）。
  文中的前端行号都来自这个版本的原始源码，获取方法见附录 A

## 1. 要解决的问题

一个 workflow 里，常常要在几套参数之间来回切换，比如"快速预览"和"精细出图"两套 steps / cfg / denoise，
或者几组不同的提示词。这些参数散落在好几个节点上，每次切换都要逐个去改。

这个节点把它们打包成命名预设：

- 从预设节点拉线连到要管理的节点上，连线只用来指明"管哪些节点"，不传数据
- 用正则圈定要记录哪些参数
- 点 Save，把圈中参数的当前值存成一条预设
- 每条预设一个按钮，点一下就把值写回去
- 当前参数正好等于哪条预设，那条预设的按钮就高亮

预设只在当前 workflow 内使用，数据跟着 workflow 一起保存。

## 2. 现有方案调研

本地已装的 6 个插件包（rgthree / KJNodes / essentials / RES4LYF / ab-nodes / inpainteasy）和网上常见的
preset 类插件都看过，没有能直接用的，但需要的技术点都有先例可循。

"用连线定位目标节点、读它的 widget"是成熟做法：

| 方案 | 能做什么 | 不够的地方 |
|---|---|---|
| KJNodes `WidgetToString` | 读任意节点的一个 widget | 只读，一次一个 |
| GODMT `Get Widgets Values` | 顺着连线找到上游节点，读它的 `widgets_values` | 只读 |
| `LC Node Snapshot` | 连线或填标题/编号定位目标，读一个 widget | 只读，一次一个 |
| 本仓库 `RefreshTriggerV1` | `*` 输入连到任意输出，前端用 `origin_id` 拿到节点 | 只有一个目标 |

能写回 widget 的只有 Impact Pack 的 `ImpactSetWidgetValue`，一次写一个，而且是在执行期经由 PromptServer
同步，不是界面上的快照和回放。

preset 类插件的形态都不一样：

- `checkpoint_preset_manager`、`mittimiLoadPreset2`、`Apt_Preset`、`Model_preset_Pilot` 都是节点自己
  输出一组固定参数（steps / cfg / sampler 等），再手工连到下游。参数集是写死的，不能管任意节点的任意 widget
- `comfyui_workflow_state_presets` 形态最接近，也是按编号切换预设，但只切 bypass/mode，README 里写明参数
  快照还没做
- ComfyUI 自带的 Node Templates 会保存 widget 值，但它是插入新节点，不是改现有节点

所以自己做，不依赖其他插件。

## 3. 设计要点一览

| 项 | 结论 |
|---|---|
| 目标节点怎么定位 | 按 node id，而且这个节点必须是当前连在预设节点上的 |
| 参数路径 | `节点标题/widget名`，例如 `KSampler/steps`。同名节点由用户改标题区分 |
| 规则怎么写 | 多行文本，一行一条正则，结果取并集；`-` 开头的行表示排除；`#` 开头是注释 |
| 规则的默认内容 | 每连上一个新节点，自动追加一行，列出它的全部可记录参数 |
| 规则何时生效 | 只在 Save 时用来圈范围。已存的预设不受之后修改规则的影响 |
| 数据存在哪 | 预设节点自己的一个隐藏 widget 里，随 workflow 保存。没有后端存储 |
| 撤销 | 一次 Apply 对应一次 Ctrl+Z |
| 断线之后 | 预设和规则都保留；断线节点的数据跳过不写，重新连上就恢复。用当前状态覆盖某条预设时，这条预设里断线节点的数据随之去掉，其他预设不受影响 |
| 高亮 | 每次按当前值重新计算，不记"上次点了哪个" |
| 调试信息 | 节点上一块只读文本区，实时列出所有参数路径和命中情况 |
| 显示和隐藏 | 右键菜单里一个开关，同时显示或隐藏规则文本框和调试信息 |
| 提示信息 | 状态行的提示显示精简的调试信息，预设按钮的提示显示这条预设的内容，调试区隐藏时也能查看 |
| 对执行的影响 | 无。节点标记为虚拟节点，不会发给后端 |
| 预设管理 | 每条预设一个按钮；改名、删除放在节点右键菜单 |

几个取舍的理由：

**为什么不做后端存储。** 预设跟着 workflow 走，另存、复制、分享 workflow 时预设都在，也不会出现文件和
workflow 对不上的情况。代价是不能跨 workflow 共用，真需要时再加导入导出即可。

**为什么规则只在 Save 时生效。** 这样一个预设按钮不管什么时候点，效果都一样，不会因为你改了规则而变。
想只应用一部分参数，就另存一条范围更小的预设。

**为什么用 node id 定位。** node id 在 workflow 内是稳定的：存档读档、改标题、断线重连、调整连线顺序，
id 都不变。节点被删掉重建会换 id，对应的记录就找不到了，这时跳过并在调试区显示出来。
之所以要求"必须是当前连着的节点"，是为了防止写错：比如把预设节点连同目标节点一起复制，副本里的预设存的
还是原节点的 id，原节点也还在图里，只按 id 在全图里找就会改到原节点上。加上这条限制后，原节点没连在副本上，
就会跳过。

**为什么路径里不放连线序号。** 连线序号在断线、重连时会变，放进路径就会让规则悄悄指向别的节点。
node id 已经足以定位，路径只需要让人看得懂、写得出来，用标题加 widget 名就够了。

**为什么调试信息做成文本区，不做弹窗或输出端口。** 输出端口要等 workflow 跑完才有值，调正则时改一次就要
跑一次，太慢；而且节点一旦有输出、接到下游，就会被拉进执行流程。文本区可以实时刷新，实现上也最省：
rgthree 的 Display Any 用三行代码就做到了（`display_any.js:13-15`），而本仓库的历史记录弹窗
（`js/slowargo.js:1162`）有两百多行。

## 4. 规则怎么写

### 参数路径

每个可以记录的 widget 对应一条路径：

```
节点标题/widget名
```

- 节点标题取 `node.title`，也就是节点顶上显示的那行字
- widget 名取 `widget.name`，不是界面上显示的 `widget.label`。界面上的文字可能经过本地化，和 name 不一样，
  所以调试区会把 name 列出来，照着抄就行

两个节点标题相同时（比如两个都叫 `KSampler`），它们的路径也相同，同一条规则会同时命中两个。
这种情况下 Save 会把两个节点的参数分别记下来（各自按 id 存），不会混。只想选其中一个的话，
给节点改个不同的标题。

### 规则语法

`filter` 是多行文本，每行一条正则：

```
# 注释行，忽略
^KSampler/(steps|cfg|denoise)$
^Positive/text$
-/seed$
```

- 正则是部分匹配，不区分大小写。写 `steps` 就能命中所有路径里含 steps 的参数，需要精确匹配时自己加 `^` 和 `$`
- 普通行取并集，命中任意一行就选中
- `-` 开头的行是排除规则：先算出普通行的并集，再从中去掉命中排除行的。`-` 后面可以留一个空格
- 排除规则作用于所有节点。`-/seed$` 会去掉所有节点的 seed，只想去掉某个节点的，写成 `-^KSampler/seed$`
- 空行忽略；行首第一个非空白字符是 `#` 的行是注释
- 想匹配真的以 `#` 或 `-` 开头的内容，写成 `\#`、`\-`
- 标题里带正则特殊字符（比如 `Load Image (from Any Path)`）时，手写规则需要自己转义括号
- 规则为空时什么都不选，状态行提示"没有规则"

任何一行不是合法的正则，状态行会指出是第几行，Save 按钮变灰。不会跳过坏行继续保存，
否则容易存下一条自以为完整、实际漏了参数的预设。

### 连线时自动追加规则

用户新连上一个节点时，如果 `filter` 里还没有以这个节点标题开头的包含规则（排除行不算，因为排除行本身不选中任何参数），
就在末尾追加一行，
列出它当前所有可记录的参数：

```
^KSampler/(seed|control_after_generate|steps|cfg|sampler_name|scheduler|denoise)$
```

用户在这行里删掉不要的参数即可。细节：

- 生成的规则带 `^` 和 `$`。不加的话，因为是部分匹配，`KSampler/steps` 也会命中标题为 `My KSampler` 的节点
- 标题和 widget 名里的正则特殊字符会被转义
- 参数超过 15 个时，改为追加 `^标题/.*$`，免得一行太长没法看
- 只在连线的那一刻追加一次。之后用户删掉这行，不会再补回来
- 打开 workflow 时不追加。读档时 litegraph 会对每一根已有连线都调一次 `onConnectionsChange`
  （`LGraphNode.ts:820-827`），参数和真正的新连线（`:2916`）一样，分不出来。所以要靠 `app.configuringGraph`
  判断当前是否在加载 workflow（前端自己的 Vue 节点布局代码也是用它来判断的），加载期间一律不追加
- 断线时这行保留。系统除了上面这种追加，不会改动 `filter` 里的任何内容

追加的规则只是连上那一刻的参数列表。之后如果节点的参数变了（比如换了一个 widget 更多的采样器，
或者像本仓库 `FloatSelector` 那样参数数量会动态变化），新出现的参数不在规则里，也就不会被记录。
调试区会把没有命中任何规则的路径也列出来，这类遗漏一眼就能看到。

## 5. 数据结构

所有预设存成一个 JSON 字符串，放在隐藏的 `presets` widget 里。每条预设按节点分组，以 node id 为键：

```json
{
  "v": 1,
  "items": [
    {
      "name": "high detail",
      "nodes": {
        "12": { "title": "KSampler", "widgets": { "steps": 30, "cfg": 7.5 } },
        "7":  { "title": "Positive", "widgets": { "text": "..." } }
      }
    }
  ]
}
```

- `nodes` 的键是 node id，`widgets` 的键是 widget 名，两者合起来定位一个参数
- `title` 只用于在调试区显示，比如找不到节点时显示 `missing: KSampler #12`，不参与判断
- 每个参数独立处理，一个找不到不影响其他参数
- JSON 的键是字符串，读取时转回数字再和 `node.id` 比较
- JS 对象中像整数的键按数值从小到大遍历，所以 Apply 按 id 从小到大处理节点，而不是保存时的顺序。
  节点之间的写入顺序不影响结果；同一节点内的 widget 名不是数字，仍按保存时的顺序处理

JSON 解析失败时（被手工改坏、格式不兼容等），不报错也不清空，按"没有预设"处理，并在调试区显示原始内容和
错误信息，方便手工修复。`v` 字段留给以后做格式迁移。

## 6. 节点结构

### Python 部分（`nodes_preset.py`）

```python
class WidgetPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filter": ("STRING", {"multiline": True, "default": "", "tooltip": "..."}),
                "presets": ("STRING", {"default": ""}),
            },
            "optional": {},
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "Slowargo"

    def noop(self, filter, presets):
        return ()
```

Python 这边只是个空壳，功能全在前端。保留它是为了让节点正常出现在节点库的 Slowargo 分类下，
两个真实的 widget 也由它声明。

### widget 布局（从上到下）

| 顺序 | widget | 类型 | 是否随 workflow 保存 |
|---|---|---|---|
| 1 | `filter` | 多行文本 | 是 |
| 2 | `presets` | 文本，隐藏 | 是 |
| 3 | 状态行 | 前端动态添加 | 否 |
| 4 | 调试文本区（可隐藏） | 前端动态添加，只读多行文本 | 否 |
| 5 | `Save as Preset…` | 前端动态添加，按钮 | 否 |
| 6 起 | 每条预设一个按钮 | 前端动态添加，按钮 | 否 |

前端动态添加的 widget 一律设 `widget.serialize = false`，并且全部排在两个真实 widget 之后，原因见第 9 节。
隐藏 `presets` 沿用本仓库的做法 `widget.hidden = true`（`js/slowargo.js:245`）。

规则文本框 `filter` 和调试文本区一起显示或隐藏，由右键菜单里的一个开关控制（见 7.5），状态存在
`node.properties.showEditor`，随 workflow 保存，默认显示。隐藏时两者一起设为隐藏，并调整节点高度。
`filter` 隐藏后仍然随 workflow 保存，连线时自动追加规则也照常进行。

调整节点高度时要注意，`addInput` 和 `addWidget` 内部会调 `expandToFitContent()`（`LGraphNode.ts:1631`、`:1917`），
先把节点撑高。所以要在改动之前记下原来的高度，再按改动前后 `computeSize()` 的差值调整，否则同一段高度会加两次。
读档时节点尺寸已经包含所有动态 widget，这时不调整高度，并且要先恢复隐藏状态、再重建预设按钮，
否则重建按钮时的自动撑高会把隐藏的部分也算进去。

调试文本区照 rgthree Display Any 的写法：`ComfyWidgets.STRING` 建一个多行文本，`inputEl.readOnly = true`。
1.37.2 里这两样都还在（`widgets.ts:292`、`domWidget.ts:51`）。

### 去掉 `filter` 和 `presets` 的输入口

前端会给每个声明的 widget 配一个同名输入口。这两个输入口对本节点没有用（节点不执行，连进来的值不会被读取），
而且 widget 隐藏后会出问题：布局只更新可见 widget 的位置（`LGraphNode.ts:4096-4099`），给 widget 输入口定位时
却遍历所有 widget（`:4134-4140`），于是隐藏的 widget 的输入口停在它隐藏前的位置；悬停提示和连线落点又是按输入口
位置判断的，不看是否可见（`measureSlots.ts:17-28`）。结果是在那片区域悬停会弹出 `filter` 的提示，拖线松手也可能
落到这两个隐藏的输入口上。

节点定义里的 `socketless` 选项本可以避免创建输入口（`litegraphService.ts:180`），但 1.37.2 的字符串 widget 创建时
不会把它带进 `widget.options`，声明了也不生效。所以在 `onNodeCreated` 里直接移除这两个输入口；更早保存的 workflow
读档时会把它们作为额外输入恢复回来，`onConfigure` 里再移除一次。`removeInput` 会同步修正后面连线的槽位序号。

### 动态输入槽

照 rgthree `base_any_input_connected_node.js` 的 `stabilizeInputsOutputs()` 来做：

- 末尾始终保留一个空的 `*` 输入槽，用来连新节点
- 中间出现空槽就移除
- 在 `onConnectionsChange` 和 `onConfigure` 之后执行，用微任务合并同一时刻的多次触发

连线序号本身没有任何含义，移除空槽导致序号变化也无所谓。

槽的 `name` 用内部名 `preset_target_0`、`preset_target_1`……，目标节点的标题放在 `label` 上显示。
不能直接拿标题当 `name`：ComfyUI 读档时会按 name 把保存的输入和节点定义对账（`litegraphService.ts:313-340`），
名字和定义里的输入重名就会被当成定义里的那个。这个节点定义里有 `filter` 和 `presets` 两个输入，
要是连了一个标题恰好叫 `filter` 的节点，读档后这根线就会被挂到 `filter` 的输入口上。
用内部名就不会撞。对账时未声明的输入会原样保留（同一段代码的 `extraInputs`），所以动态槽的连线读档后还在。

### 复制节点

需要重写 `clone()`，参考 `base_any_input_connected_node.js:17-27`。

单独复制预设节点时，litegraph 不会带上连线。副本保留全部预设数据，动态槽整理成一个空槽，
状态行提示"没有连接节点"。保留数据是有用的：把副本重新连回原来那些节点，因为 id 没变，预设照样能用，
相当于多开一个控制面板。

把预设节点和目标节点一起复制时，粘贴出来的目标节点是新 id，而副本里的预设存的是原节点的 id，
所以副本上的预设不会生效。它也不会去改原节点，因为原节点没连在副本上。这是按 id 定位的已知代价。

## 7. 核心逻辑

### 7.1 找到目标节点

遍历所有有连线的动态槽，顺着连线找到上游节点，得到"当前连着的节点"集合。有三点要注意：

- 用 `node.graph`，不用 `app.graph`。`app.graph` 始终指向根图，预设节点放在 subgraph 里时，
  它的连线在 subgraph 里，从根图上查不到
- 用 `graph.links.get(id)`，不用 `graph.links[id]`。下标写法靠一个 Proxy 兼容（`LGraph.ts:294-298`），已标为废弃
- 要穿过 Reroute 这类转接节点。`link.origin_id` 只是直接上游，中间有 Reroute 的话拿到的就是 Reroute 本身。
  rgthree 用 `getConnectedInputNodesAndFilterPassThroughs` 处理（`base_any_input_connected_node.js:56`），
  判断条件是节点类型包含 `Reroute`、`Node Combiner` 或 `Node Collector`（`rgthree-comfy/web/comfyui/utils.js:318-320`），
  照着做

同一个节点被连了多根线，按 id 去重。穿透后拿到的是 subgraph 的输入输出节点时（用户把部分目标转成了 subgraph），
当作找不到处理，并在状态行提示。

### 7.2 哪些 widget 可以记录

要同时满足四个条件：

1. 类型不在排除列表里：`button`、`image`、`imageupload`、`markdown`、`galleria`、`imagecompare`、`chart`、
   `preview`、`painter`、`asset`
2. `widget.serialize !== false`，并且 `widget.options?.serialize !== false`
3. 没有被转成输入并连上线（节点的输入里没有 `input.widget?.name === widget.name` 且 `input.link != null` 的项）
4. 值是字符串、数字或布尔

各条的原因：

- 条件 1：这些 widget 的值也常常是字符串（比如图片路径），但它们不是参数，写回去等于伪造预览图。
  排除列表里有些类型 1.37.2 还没有（`painter`、`asset` 等），一并列上，以后升级前端不用再改
- 条件 2：用来排除"派生"的 widget，也就是由别的 widget 算出来、本身不单独保存的那些。
  不排除的话会改坏本仓库自己的 `FloatSelector`：它的 `float_1..float_N` 是按 `values` 动态生成的
  （`js/slowargo.js:553-572`），每个的回调最后都会把 `selected_index` 改成自己的序号。Apply 依次写这些 widget，
  写完 `selected_index` 就停在最后一个上，原来选的是哪个永远恢复不了。`float_N` 设了 `options.serialize = false`，
  这一条正好把它们排除掉，同时保留真正存状态的 `values` 和 `selected_index`
- 条件 3：已经连线的输入，值由上游决定，写了也没用。litegraph 自己判断 widget 是否转成输入也是这个写法
  （`LGraphNode.ts:850`）
- 条件 4：对象和数组类型的 widget（边框、曲线、多选之类）暂时不支持。1.37.2 自带的 widget 类型里还没有这些
  （`widgets.ts:288-302`），会受影响的主要是第三方插件自己加的 widget。它们会在调试区列出来并标注"暂不支持"，
  不会悄悄消失

### 7.3 Save

1. 弹出 `app.canvas.prompt` 输入预设名
2. 按规则圈出参数，逐个读取画布上的当前值。输入框不是模态的，输入名字期间参数、规则和预设本身
   （比如用右键菜单删除了一条）都可能变化，所以确认后要把这些全部重新读一遍再保存
3. 一个参数都没圈中时，不保存并提示
4. 和已有预设重名时，询问是否覆盖。覆盖就是更新预设的方式，用这次圈中的内容整条替换旧内容，
   不做合并。所以覆盖后，这条预设里原来记录、但这次没有圈中的参数都会去掉，包括断线节点的全部数据，
   以及还连着但被规则排除的参数。其他预设不受影响，它们里面断线节点的数据仍然保留，重新连上照常生效。
   确认框里列出这次会去掉的内容，例如"将移除 KSampler #15 的 3 个参数（节点未连接）"、
   "将移除 KSampler #12 的 denoise（未被规则选中）"，免得以为只是更新了数值
5. 写入 `presets`，重建按钮

### 7.4 Apply

整个过程包在一个撤销事务里，这样一次 Ctrl+Z 就能撤销：

```js
app.canvas.emitBeforeChange();
try {
  // 逐条写入
} finally {
  app.canvas.emitAfterChange();
}
```

要用的是画布上的这一对，不是 `graph.beforeChange()` / `graph.afterChange()`。ComfyUI 的撤销记录由
`changeTracker` 负责，它监听的是画布发出的 `litegraph:canvas` 事件（`changeTracker.ts:328-335`），
发这个事件的是 `canvas.emitBeforeChange()`（`LGraphCanvas.ts:3868-3879`）。`graph.beforeChange()` 只会去调
`LGraphCanvas.onBeforeChange`（`LGraph.ts:1234-1241`），而前端里没有任何地方给它赋过值。
画布拖拽节点的代码里能看到两者的关系：`emit*` 在外层，`graph.*` 嵌在里面（`LGraphCanvas.ts:3513-3519`）。

用错的话不只是撤销粒度不对，而是根本没有撤销记录：程序写入的值不会触发键盘、菜单、弹框这几个会记录历史的
时机（`changeTracker.ts:303-326`），结果是 Ctrl+Z 撤回到更早的状态、workflow 不显示未保存标记，
而且这次修改会被并进下一次无关操作的记录里。

Save、改名、删除这三处修改 `presets` 的地方也一样用这对事务包起来。它们目前碰巧能产生撤销记录，
因为 `canvas.prompt` 和右键菜单关闭时会被 `changeTracker` 捕获（`changeTracker.ts:303-326`），但不要依赖这个。

每条记录的处理：

1. 对预设里的每个 node id，在"当前连着的节点"里找对应节点。找不到就跳过这个节点的全部参数，
   包括断线、节点被删除、复制后 id 对不上这几种情况
2. 对该节点下的每个 widget 名，找 `name` 相同、并且满足 7.2 四个条件的 widget。找不到就跳过
3. 下拉框的值不在当前可选项里（比如模型文件已经换了），跳过，不强行写入
4. 通过 `widget.setValue` 写入：
   ```js
   widget.setValue(value, { e: undefined, node: targetNode, canvas: app.canvas });
   ```
   `setValue`（`BaseWidget.ts:313-333`）除了调用回调，还会同步 `options.property`（否则读档时会被旧的 property 值
   覆盖回去）、通知 `node.onWidgetChanged`、递增 `graph._version`。回调必须触发，本仓库的 `LoadRecentImagePlusV1`、
   `FloatSelector` 都靠回调同步界面和内部状态。没有 `setValue` 的 widget 退回直接赋值再调回调

每个参数都要当场重新查找 widget，不能事先查好存起来。有些回调会重建整组 widget，
比如 `FloatSelector` 改了 `slot_count` 会重建所有 `float_N`，事先存下的引用就失效了。

预设里没有记录的参数保持原样，不会被重置。完成后刷新画布，状态行显示 `applied 10/12`，
跳过的参数在调试区逐条列出原因。

### 7.5 右键菜单

在 `getExtraMenuOptions` 里加三项（1.37.2 仍会调用这个钩子，`LGraphCanvas.ts:8264`）：

- `Rename preset`、`Delete preset`：子菜单，列出所有预设名
- `Show filter & debug info` / `Hide filter & debug info`：同时切换规则文本框和调试文本区的显示

不在按钮上画删除图标，是因为 Vue 渲染模式下自己画的点击区域容易失效，原生菜单更可靠。

### 7.6 高亮

一条预设在以下情况高亮：它的记录里，能找到对应节点和 widget 的那些，当前值全部等于记录的值，而且至少有一条
能找到。

不记"上次点的是哪个"，每次都按当前值重新算，于是：

| 操作 | 结果 |
|---|---|
| 点某条预设 | 值都写进去了，这条自然高亮 |
| 手动改了其中一个值 | 不再相等，高亮消失 |
| 又手动改回去 | 重新相等，高亮回来 |
| 手动改成了另一条预设的值 | 高亮移到那一条 |
| 打开 workflow | 当前值正好等于某条预设的话直接高亮，能看出现在用的是哪套 |
| 两条预设内容完全一样 | 两条都高亮 |

找节点和 widget 的逻辑和 Apply 用同一个函数，否则两边标准一不一致，就会出现"点了却不高亮"。
找不到的记录两边都跳过。

数字用 `Math.abs(a - b) < 1e-9` 比较，字符串和布尔直接比。

**什么时候重新计算。** 不能放在 `onDrawForeground` 里。Vue 渲染模式下 `drawNode` 会提前返回
（`LGraphCanvas.ts:5262-5270`），走不到 `onDrawForeground`（`:5349`），高亮就会一直停在最初的状态，
显示一个错误的结果。Vue 模式在 1.37.2 里已经存在，默认关闭，但菜单里可以打开。

改为在这些时机重新计算，并用一个标记避免重复：

- Apply、Save、改名、删除之后
- 本节点的 `onConnectionsChange`、`onAfterGraphConfigured`
- `api` 的 `graphChanged` 事件（`changeTracker.ts:111`，每次记录历史时都会发）

`graphChanged` 的监听是全局的一个，遍历所有预设节点。清空根图时只有根图里的节点会收到 `onRemoved`，
subgraph 里的节点不会，所以遍历时要把已经不在当前 workflow 里的节点剔除掉：节点所在的图必须是根图，
或者是根图 `subgraphs` 表里登记的那张图，并且这张图里按 id 找到的就是它自己。

只在结果真的变了时才刷新画布。在绘制过程中改按钮文字再触发重绘，会变成无限循环。

**显示方式。** 按钮文字前加标记：`● 名称` 表示高亮，`○ 名称` 表示未高亮。不自己画背景色，
因为那同样依赖绘制钩子，Vue 模式下不生效。

Vue 模式下按钮文字能不能跟着更新，取决于 `WidgetButton.vue` 拿到的 widget 对象是不是原对象的响应式代理，
这点还没确认，见第 10 节。

### 7.7 调试文本区

节点上一块只读文本，和规则文本框一起通过右键菜单显示或隐藏，和高亮在同样的时机刷新。隐藏时不计算内容。内容是所有连着的节点上的全部参数，每行一个：

```
  KSampler/steps     #12   30        ✓ 命中
  KSampler/cfg       #12   7.5       ✓ 命中
  KSampler/seed      #12   12345       被 -/seed$ 排除
  KSampler/steps     #15   20        ✓ 命中
  Positive/text      #7    "a cat…"  ✓ 命中
  Positive/mask      #7    —           暂不支持（对象类型）
```

它有三个用途：确认规则圈中了什么、提供可以照抄的路径、说明某个参数为什么没被选中。
带上 node id 是为了区分同名节点。Apply 之后在末尾追加跳过参数的明细。

### 7.8 提示信息（tooltip）

调试区隐藏时也要能看到关键信息，所以另外在两处的悬停提示里显示：

- **状态行**：精简版调试信息。只列命中的（`✓ 路径 = 值`）、被排除的（`− 路径 (排除规则)`）、不支持的
  （`× 路径 (unsupported)`），以及规则错误、无法解析的连线、上一次 Apply 跳过的内容；没命中的路径不列，
  完整列表看调试区
- **预设按钮**：这条预设记录的内容，按节点分组列出 `widget = 值`。当前找不到的节点或参数（断线、节点被删除、
  widget 不存在、下拉框没有这个选项），也就是 Apply 时会跳过的，前面标 `×`。不和当前值对比

提示最多 30 行，超出的显示 `… N more`。原因是提示框鼠标一动就消失，不能滚动也不能复制，只限制了最大宽度
（30vw），没有最大高度。

提示内容在每次刷新时写进 `widget.tooltip`，画布在悬停时读取（`NodeTooltip.vue:107-115`，包括禁用的 widget）。
提示框用的不是等宽字体，所以不做列对齐。Vue 渲染模式下前端不会把动态 widget 的 `tooltip` 传给界面，
这两处提示只在 legacy 画布下有效。

### 7.9 其他

- 一个 workflow 里可以放多个预设节点，各管各的。多个预设节点管同一个节点时，谁最后点谁生效，不做冲突检测
- 状态行平时显示 `3 nodes · 12 widgets matched`，Apply 之后显示 `applied 10/12`
- 预设按钮数量不设上限，超过 12 条时状态行提示

## 8. 对执行的影响

预设节点在前端设 `isVirtualNode = true`，提交 workflow 时会被直接跳过，不会发给后端（`executionUtil.ts:75-83`）。

其实不设这个标记也不会影响执行：后端 `validate_prompt` 只从输出节点往上检查和执行，这个节点没有输出，
本来就不会被检查或执行，连线也不会把上游节点拉进执行。设这个标记只是让发给后端的数据干净一些。

以后如果要给这个节点加输出，需要重新考虑：虚拟节点的输出会走另一套解析逻辑
（`ExecutableNodeDTO.ts` 的 `resolveOutput`），行为会变。

## 9. 序列化的两个注意点

**`widget.serialize` 和 `widget.options.serialize` 是两回事。** 前者决定 widget 的值是否存进 workflow
（`LGraphNode.ts:909-916` 保存、`:862-869` 读取），后者决定是否发给后端（`executionUtil.ts:90`）。
动态添加的 widget 要设的是前者。

本仓库 `FloatSelector` 设的是 `options.serialize = false`（`js/slowargo.js:569-572`），所以那些动态 widget
其实还是会被存进 workflow。它们排在最后，读档后又会整体重建，所以目前没出问题。这次不改它，
但 7.2 的条件 2 正是依赖这个字段来识别派生 widget，以后动它时要记得这里。

**动态 widget 要排在最后。** 保存时按 widget 在数组里的位置写入，跳过的位置留空；读取时却是跳过之后按顺序
依次赋值（`LGraphNode.ts:909-916` 对比 `:862-869`）。不保存的 widget 夹在中间，读档后后面的值就全部错位了。
都排在最后就没这个问题。

## 10. 实测结果和遗留问题

实现后在 1.37.2 上实测（legacy 和 Vue 两种渲染模式都测过）：

| 项 | 结果 |
|---|---|
| Vue 模式下按钮文字能否更新 | 能。改动目标参数后，高亮标记在 Vue 模式下也会跟着变 |
| 状态行用什么 widget | 用一个禁用的按钮显示文字。禁用后不响应点击，画面上以半透明显示，两种模式下都是只读的 |
| 隐藏 `presets` | legacy 画布看 `widget.hidden`，Vue 渲染层看 `widget.options.hidden`（`NodeWidgets.vue:28`），两个都要设。Vue 层只在 widgets 数组变化时才重新读取，所以运行时切换显示后要把 widget 从数组里取出再插回 |
| 调试文本区在 Vue 模式下 | Vue 用自己的文本框渲染，`inputEl` 上的只读和等宽字体不生效。只读改用 `options.read_only`；等宽字体在 Vue 模式下没有，列对不齐，只影响观感 |
| 输入槽加 label | 读档、撤销、重做后连线和 label 都正常 |
| 状态行和预设按钮的提示 | 在调试区隐藏的状态下，悬停状态行显示精简调试信息，悬停预设按钮显示预设内容，找不到的节点前标 `×` |
| 隐藏后的悬停提示 | 隐藏 `filter` 后，在它原来输入口的位置悬停会弹出 `filter` 的提示。移除两个 widget 输入口后不再出现；带着这两个输入口保存的旧 workflow 读档后会被去掉，原有连线保留 |
| `control_after_generate` | 前端给它设了 `options.serialize = false`，被 7.2 的条件 2 自动排除，所以 Apply 不会改动随机策略。代价是 seed 写进去后，如果随机策略是 randomize，下次运行 seed 会再被改掉 |
| 按钮回调没有事件 / 画布未激活 | `canvas.prompt` 依赖 `LGraphCanvas.active_canvas`，画布上没有过鼠标操作时它为空，调用会报错。此时改用浏览器自带的 `prompt` |
| 自动追加规则和撤销 | 未测。连线触发的规则追加是否和连线在同一条撤销记录里还不确定 |

其他验证过的行为：一次 Ctrl+Z 撤销整次 Apply；反复读档后节点尺寸不变；节点不会出现在提交给后端的数据里；
经 Reroute 连接能找到真实节点；断线后空槽收拢；覆盖预设时确认框列出被移除的内容，其他预设不受影响；
单独复制节点保留预设数据、只剩一个空槽；非法正则时保存按钮变灰。

## 11. 实现步骤

1. 新建 `nodes_preset.py`，写 `WidgetPreset` 空壳
2. 在 `__init__.py` 导入并注册到两张映射表
3. 新建 `js/widgetPreset.js`，导出 `setupWidgetPreset(nodeType, nodeData, app)`。不单独注册扩展，
   由 `slowargo.js` 的 `beforeRegisterNodeDef` 调用，和 `maskEditorTurbo.js` 一样
4. 在 `slowargo.js` 里加 `WidgetPreset` 的分支
5. 在 `CLAUDE.md` 的节点列表里补一行

建议顺序：先测 Vue 模式下按钮文字能否更新 → 动态输入槽和目标收集（含 Reroute）→ 调试文本区和状态行
→ 规则解析和自动追加 → Save → Apply → 高亮 → 右键菜单 → `clone()`。

调试文本区放在前面，是因为后面每一步都要靠它观察结果。

## 12. 这一期不做

- 弹窗式的预设管理（预设多了再做，到时节点上只留下拉框和一个 Apply 按钮）
- 对象、数组类型的 widget
- 提供给 rgthree Fast Actions Button 调用的动作
- 跨 workflow 共用预设、导入导出
- 记录节点的 bypass / mode 状态
- 只应用预设中的一部分

## 参考

外部插件：

- [comfyui_workflow_state_presets](https://github.com/CarlMarkswx/comfyui_workflow_state_presets)
- [comfyui_checkpoint_preset_manager](https://github.com/TakkunRed/comfyui_checkpoint_preset_manager)
- [ComfyUI_mittimiLoadPreset2](https://github.com/mittimi/ComfyUI_mittimiLoadPreset2)
- [ComfyUI-Apt_Preset](https://github.com/cardenluo/ComfyUI-Apt_Preset)
- [ComfyUI-Model_preset_Pilot](https://github.com/NewLouwa/ComfyUI-Model_preset_Pilot)
- [ImpactSetWidgetValue](https://www.runcomfy.com/comfyui-nodes/ComfyUI-Impact-Pack/ImpactSetWidgetValue)
- [LC Node Snapshot](https://comfy.icu/node/LCNodeSnapshot)
- [Get Widgets Values (GODMT)](https://comfy.icu/node/GODMT_GetWidgetsValues)

本地代码：

- `rgthree-comfy/web/comfyui/base_any_input_connected_node.js`：动态输入槽、`clone()`、穿过转接节点
- `rgthree-comfy/web/comfyui/display_any.js:13-15`：节点上的只读文本区
- `rgthree-comfy/web/comfyui/utils.js:318-320`：转接节点的判断
- `Slowargo/js/slowargo.js:928`：`RefreshTriggerV1` 通过连线拿到节点
- `Slowargo/js/slowargo.js:548`：`FloatSelector` 的动态 widget
- 前端 `executionUtil.ts:78`：提交时跳过虚拟节点
- 前端 `LGraphCanvas.ts:3868`：撤销事务
- 前端 `litegraphService.ts:313`：读档时按名字对账输入

## 附录 A：如何核对 1.37.2 的源码

实际运行的前端在 `ComfyUI/web_custom_versions/Comfy-Org_ComfyUI_frontend/1.37.2/`，里面只有打包后的 js，
但每个 js 都带 `.js.map`，其中的 `sourcesContent` 就是完整的原始源码，用脚本导出来即可。

几个容易弄错的地方：

- `python_embeded` 里装的 `comfyui_frontend_package` 1.45.20 并没有被使用，启动脚本用 `--front-end-version`
  指定了 1.37.2
- 仓库旁边的 `ComfyUI_frontend` 源码是 1.47.6，行号和 1.37.2 不同
- 不要在打包后的 js 里搜注释文字，注释在打包时已被去掉，搜不到不代表代码不存在

## 附录 B：修订记录

2026-09-22 至 09-23 讨论中的修改，按时间顺序：

1. **去掉后端存储。** 最初设想用后端 JSON 文件保存预设。确定只在 workflow 内使用后，改为存进节点自身，
   后端只剩空壳
2. **规则只在 Save 时生效。** 讨论过 Apply 时是否也用当前规则过滤，决定不过滤，保证同一个按钮效果不变
3. **增加高亮，并改为按当前值计算。** 最初只打算高亮最后点击的按钮，后来改为每次按当前值比较，
   手动改值会自动取消或转移高亮
4. **调试信息的形式。** 考虑过弹窗和输出端口，输出端口因为要执行才有值、且会让节点进入执行流程而放弃，
   最后选择实现最简单的只读文本区
5. **规则从单行改为多行。** 单行一条正则改为一行一条、取并集，加入注释和 `-` 排除行
6. **修正撤销的写法。** 初稿用的是 `graph.beforeChange/afterChange`，评审发现它不会产生任何撤销记录，
   改为 `canvas.emitBeforeChange/emitAfterChange`
7. **修正高亮的计算时机。** 初稿放在 `onDrawForeground` 里，评审发现 Vue 模式下它不会被调用，改为事件驱动；
   同时放弃了自己画背景色的方案
8. **补充可记录 widget 的条件。** 评审发现只排除按钮不够，而且会改坏 `FloatSelector`，补上排除列表和
   `serialize` 条件
9. **补充转接节点、复制节点、subgraph 的处理。** 评审发现这三处都没考虑
10. **动态输入槽改用内部名。** 评审发现用标题做槽名会在读档时和 `filter` / `presets` 撞名
11. **更正对照的前端版本。** 起初按 1.47.6 源码核对，之后误以为运行的是 1.45.20，最后确认实际是 1.37.2，
    所有结论在 1.37.2 源码上重新核对过
12. **目标节点的定位方式，前后改了三次。** 先是"标题优先、编号兜底"，评审指出互换标题会把值写到错的节点、
    同名节点删了重建会失效；于是改为按连线序号定位；接着发现连线序号在断线时会变，规则会指向错的节点。
    最后改为按 node id 定位，并要求节点必须当前连着预设节点，同时从路径里去掉连线序号
13. **参数路径格式。** 依次是 `节点标题.widget名`、`连线序号/节点标题/widget名`，最后定为 `节点标题/widget名`。
    分隔符从 `.` 换成 `/`，因为 `.` 在正则里是特殊字符，照抄路径容易出错
14. **增加连线时自动追加规则。** 追加的规则列出节点的全部参数，方便删减；只在新连线时追加一次，加载 workflow
    时不追加；规则的默认值因此从 `.*` 改为空
15. **预设数据改为按节点分组。** 原来每个参数一条平铺记录，各自带 id 和标题，是定位 key 还是组合键时留下的写法。
    改为以 node id 为键分组，id 和标题每个节点只存一次
16. **调试文本区改由右键菜单控制显示。** 原来设想在节点上直接收起，改为右键菜单开关，状态随 workflow 保存
17. **明确覆盖预设的语义。** 覆盖同名预设是整条替换而不是合并，断线节点的数据因此在这条预设里去掉，
    其他预设不动；确认框列出会被去掉的内容
18. **实现后补充实测结果。** 第 10 节从待测清单改为实测结果；补充了 Vue 模式下隐藏 widget 的做法、
    画布未激活时输入名字改用浏览器 `prompt`
19. **规则文本框和调试信息共用一个显示开关。** 原来只有调试信息可以隐藏，改为右键菜单一个开关同时控制两者，
    属性名从 `showDebug` 改为 `showEditor`
20. **根据代码评审修正。** 调整高度时在改动前记下原高度，避免重复增高；读档时先恢复隐藏状态再重建按钮；
    清理残留在 subgraph 里的失效节点；Apply 改用 `widget.setValue`；保存时重新读取预设数据；
    判断是否已有规则时不再把排除行算在内
21. **去掉 `filter` 和 `presets` 的输入口。** 隐藏后它们的输入口停留在旧位置，会误弹提示、误接连线；
    `socketless` 在 1.37.2 的字符串 widget 上不生效，改为在前端移除
22. **增加状态行和预设按钮的提示。** 调试区隐藏时也能看到信息：状态行显示精简调试信息，预设按钮显示预设内容，
    无效项标 `×`
