# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI_InpaintHelper is a ComfyUI extension that provides utility nodes for enhancing inpainting workflows. The extension includes both Python backend nodes and JavaScript frontend enhancements.

## Architecture

### Core Components

- **`__init__.py`**: Package entry point only — relative imports, `WEB_DIRECTORY`, and the two
  `NODE_*_MAPPINGS` tables. Node implementations live in the `nodes_*.py` modules below.
- **`nodes_color.py`**: Lab colour correction (`MaskedColorMatch`, `InpaintRegionColorFix`) plus the
  shared tensor helpers. Depends only on torch and kornia, so it can be tested without ComfyUI.
- **`nodes_sampling.py`**: In-loop inpaint drift guard (`InpaintX0DriftGuard`): a
  `sampler_post_cfg_function` that anchors the masked region's x0 low frequencies to the original
  every step. Depends only on torch.
- **`nodes_image_io.py`**: Image loading, recent-file enumeration, and the preview refresh routes
- **`nodes_save.py`**: Image saving, server-side file transfer, and the transfer route
- **`nodes_strings.py`**: String memory and its history routes
- **`nodes_util.py`**: Float switch/selector, triggers, history clearing, SSIM comparison
- **`nodes_preset.py`**: Declaration of the Widget Preset node; all of its logic lives in `js/widgetPreset.js`
- **`js/slowargo.js`**: Main frontend extension for node UI enhancements, keyboard shortcuts, and general ComfyUI integration
- **`js/widgetPreset.js`**: Widget Preset frontend (regex-selected widget snapshots, apply, highlight). Imported by
  `slowargo.js`; design in `reports/WIDGET_PRESET_NODE_DESIGN.md`
- **`js/maskEditorTurbo.js`**: Mask editor integration module with Fast Forward Mode and clipspace reload features
- **Node Categories**: All nodes are categorized under "Slowargo" in ComfyUI

### Node Architecture Patterns

The extension supports both V1 (legacy) and V3 (modern) ComfyUI node APIs:

- **V1 Nodes**: Traditional ComfyUI nodes with `INPUT_TYPES()`, `RETURN_TYPES`, etc.
- **V3 Nodes**: Modern nodes using `io.ComfyNode` with `define_schema()` and `execute()` methods

### Key Node Types

1. **Image Loading Nodes**: Load images from various sources (output directory, any path, recent files)
2. **Image Saving Nodes**: Save images with custom filenames and metadata preservation
3. **Utility Nodes**: Float switching, path extraction, string memory
4. **UI Enhancement Nodes**: Button triggers, history management

### Frontend Integration

The extension integrates with ComfyUI through two main JavaScript modules:

**`js/slowargo.js`**:
- Enhances node UI via `beforeRegisterNodeDef` hook
- Adds keyboard shortcuts for common operations (Ctrl+X, Ctrl+Alt+M, etc.)
- Implements custom node behaviors (e.g., SaveImageToFileName auto-open)
- Provides integration with rgthree actions

**`js/maskEditorTurbo.js`**:
- Manages mask editor workflows with "Fast Forward Mode" (press Enter to save & run, auto-refresh)
- Provides Reload buttons in mask editor toolbar (split into "Mask" and "All" layers)
- Implements Ctrl+L shortcut to load clipspace content (base, mask, paint layers)
- **Critical**: Properly clears GPU textures and Canvas state before reloading (prevents visual artifacts from GPU data)
- Preserves brush color across Fast Forward cycles using localStorage
- Handles editor open/close lifecycle and execution monitoring

## Development Commands

### Testing Changes
```bash
# No build process required - changes are loaded directly when ComfyUI reloads
# Python: Restart ComfyUI server
# JavaScript: Clear browser cache + reload page, or restart ComfyUI
```

### Debugging
- Python nodes: Check ComfyUI server logs for errors
- Frontend: Open browser DevTools Console (F12) for JavaScript errors and console logs
- GPU issues: Check for "GPU" warnings in console when debugging mask editor operations

### Common Tasks

**Adding a new node**:
1. Define the node class in whichever `nodes_*.py` module fits (see patterns in "Key Patterns")
2. Import it in `__init__.py` and add it to both `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
3. If it needs UI enhancements, modify `js/slowargo.js` using `beforeRegisterNodeDef`

**Adding a new API route**:
1. Add decorator and handler in the `nodes_*.py` module that owns the code the route serves,
   using `@PromptServer.instance.routes.method()`
2. Route pattern: `/slowargo_api/endpoint_name`
3. Can be called from frontend via `api.fetchApi()`
4. Routes register as an *import side effect*. If the module is not already imported by
   `__init__.py`, add the import — and never drop an existing one because it looks unused.

**Updating mask editor features**:
1. Modify `js/maskEditorTurbo.js` for Fast Forward Mode and reload functionality
2. Ensure GPU cleanup happens before loading new mask data
3. Test GPU texture clearing by checking for visual artifacts when reloading

## Key Patterns

### Adding New Nodes

1. **V1 Style Node**:
```python
class NewNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {...}}

    RETURN_TYPES = (...)
    FUNCTION = "execute_function"
    CATEGORY = "Slowargo"
```

2. **V3 Style Node**:
```python
class NewNodeV3(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(...)

    @classmethod
    def execute(cls, ...) -> io.NodeOutput:
        return io.NodeOutput(...)
```

### Image Processing Pattern
Use the shared `process_image_to_tensor()` function for consistent image handling across all image loading nodes.

### API Routes Pattern
Custom API routes follow the pattern `/slowargo_api/endpoint_name` and are registered using `@PromptServer.instance.routes.method()`.
Each route lives next to the code it serves rather than in a central routes module, so the only
thing that registers it is `__init__.py` importing that module.

### Source Width
Keep lines at 120 characters or less. Long user-facing prose (`tooltip`, `DESCRIPTION`) wraps at
100 using parenthesised implicit string concatenation — when rewrapping one, check the joined value
is unchanged rather than eyeballing the spaces at the seams.

### Widget Order
ComfyUI serialises widget values *by position*, so a new widget goes at the END of `required`.
Inserting one in the middle silently breaks every saved workflow that uses the node.

### Frontend Node Enhancement
Enhance nodes in `js/slowargo.js` using the `beforeRegisterNodeDef` hook, checking `nodeType?.comfyClass` to target specific nodes.

## File Watching and Recent Files

The extension uses a sophisticated file watching system in `get_recent_image_files()` that supports:
- Multiple directory monitoring with labels
- Configurable file counts per directory
- Global sorting by modification time
- Support for both input and output directories

## Keyboard Shortcuts

- `Ctrl+X`: Open selected node's image in browser
- `Ctrl+Alt+M`: Switch to mask tool in mask editor
- `Ctrl+Alt+C`: Switch to eye dropper tool in mask editor
- `Ctrl+Alt+S`: Save mask and exit mask editor
- `Ctrl+L` (in mask editor): Load most recent clipspace content (base, mask, paint layers)
- `Enter` (in mask editor, with Fast Forward Mode enabled): Save mask → execute workflow → auto-refresh

## API Endpoints

- `/slowargo_api/refresh_previews`: Refresh image previews for output loading
- `/slowargo_api/refresh_previews_recent`: Refresh recent image lists
- `/slowargo_api/get_string_history`: Get remembered strings history
- `/slowargo_api/toggle_string_history_pin`: Toggle pin status for strings
- `/slowargo_api/delete_string_history`: Delete string from history

## Metadata Handling

The extension preserves PNG metadata when loading/saving images:
- Loads metadata using `process_image_to_tensor()`
- Saves metadata in `SaveImageToFileName` with format-specific handling
- Supports ICC profiles and text metadata for PNG format

## Fast Forward Mode & Mask Editor Workflow

**Fast Forward Mode** (`maskEditorTurbo.js`):
- **Activation**: Toggle button in mask editor topbar (defaults to enabled)
- **Workflow**: Press Enter → saves mask → executes workflow → auto-refreshes image
- **Color Memory**: Brush color persists via localStorage across cycles
- **Efficiency**: Eliminates manual save/execute/refresh steps

**Mask Reload Operations**:
- **"Reload Mask" button**: Loads mask layer from latest clipspace (preserves base/paint layers)
- **"Reload All" button**: Loads base + mask + paint layers from clipspace (full restore)
- **Ctrl+L shortcut**: Same as "Reload All" button
- **GPU Cleanup**: Clicking reload buttons first triggers Clear (GPU + Canvas cleanup) to prevent visual artifacts from stale GPU texture data

## Important Notes for Development

### GPU Resource Management
The mask editor integrates with ComfyUI_frontend's GPU-based rendering. When reloading mask data:
1. **Always clear GPU textures before loading** - Click the Clear button or call its handler
2. **Wait for GPU operations** - Allow 200ms for GPU cleanup to complete
3. **Avoid stale texture artifacts** - GPU textures can persist across operations if not properly cleared

### ComfyUI_frontend Integration Points
- Mask editor is a Vue 3 app in ComfyUI_frontend (separate repository)
- This extension coordinates with mask editor via:
  - DOM selectors for button triggering (e.g., Clear button: `#global-mask-editor > div.flex.items-center > div > button:nth-child(4)`)
  - Image loading from clipspace directory (`clipspace [N][input]`)
  - Color input widget updates via DOM events

### Event Handling
- Fast Forward Mode monitors `execution_success`, `execution_error`, `execution_interrupted` events
- Mask editor opening detected via `MutationObserver` on document.body
- Timeout protection for execution monitoring (120 seconds default)

### Others
- 前端代码位置在../ComfyUI_frontend
