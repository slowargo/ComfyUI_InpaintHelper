# Repository Guidelines

## Project Structure & Module Organization
This repository is a ComfyUI custom node extension with a Python backend and browser-side JavaScript.

- `__init__.py`: main backend module; defines node classes (V1 and V3 style), utility functions, and `/slowargo_api/*` routes.
- `js/`: frontend extension code and styles.
- `js/slowargo.js`: node UI hooks, shortcuts, and extension registration.
- `js/maskEditorTurbo.js` and `js/maskEditorBrushTools*.js`: mask editor workflow/tools.
- `reports/`: implementation/design notes and archived analysis docs.
- `README.md` and `README_CN.md`: user-facing usage docs.

## Build, Test, and Development Commands
There is no standalone build step; ComfyUI loads this extension directly.

- `python -m py_compile __init__.py`: quick Python syntax validation.
- `node --check js/slowargo.js && node --check js/maskEditorTurbo.js`: quick JS syntax check (if Node.js is available).
- `rg "pattern" __init__.py js/`: fast code search during development.

Run/develop flow:
1. Place repo under `ComfyUI/custom_nodes/ComfyUI_InpaintHelper`.
2. Restart ComfyUI after backend changes.
3. Hard-refresh browser (or clear cache) after frontend changes.

## Coding Style & Naming Conventions
- Python: 4-space indentation, snake_case for functions, PascalCase for node classes, constants in UPPER_CASE when appropriate.
- JavaScript: ES modules, 4-space indentation, camelCase for functions/variables.
- Keep API endpoints under `/slowargo_api/<action>`.
- Keep node category labels consistent (`"Slowargo"`).
- Follow existing file naming patterns (e.g., `maskEditorBrushToolsTransform.js`).

## Testing Guidelines
No dedicated automated test suite is present in this repository today. Use targeted manual validation:

- Node behavior in ComfyUI graph (widget values, outputs, refresh behavior).
- Mask editor flows (Fast Forward, reload mask/all, keyboard shortcuts).
- Browser console and ComfyUI server logs for regressions/errors.

Document exact reproduction and verification steps in PRs.

## Commit & Pull Request Guidelines
Git history follows Conventional Commit style with scopes:

- Examples: `feat(maskEditorTurbo): ...`, `fix(maskEditorBrushTools): ...`, `refactor(brush-tools): ...`.

For PRs, include:
- clear summary of behavior changes,
- linked issue/ticket (if any),
- test steps and expected results,
- screenshots or short recordings for UI/mask-editor changes.

## Any other things
- 前端代码位置在../ComfyUI_frontend