# InpaintHelper Extension for ComfyUI

A collection of utility nodes for ComfyUI that enhance inpainting workflows.

## Features

### Node Types

#### Float Switch
- **FloatSwitch**: A versatile float value selector that allows switching between two float values based on a boolean toggle.
  - Provides two input floats with slider controls
  - Toggle switch to select between the two values
  - Optional override input that takes priority when greater than 0
  - Usage example: Set values 0.32 and 0.55, and connect to the denoise parameter of ksampler. 0.32 corresponds to light inpainting (fixing minor flaws), while 0.55 corresponds to deep inpainting (changing content based on prompt). Click the switch to toggle between the two. Generally, it's recommended to perform a light inpainting after deep inpainting to make the image more natural.

#### Image Loading Nodes
- **Load Image (from Outputs) Plus V1**: Enhanced image loading from output directories with recent file previews
  - Similar to the official Load Image from Outputs, but also includes clipspace files from the input directory (up to 10 most recent output files and 3 clipspace files). Clipspace files are automatically generated after editing masks, and sometimes ComfyUI may lose editing results. This design facilitates retrieving previous editing results easily.

- **Load Recent Image**: Loads images from configurable watched folders with flexible filtering
  - Allows specification of multiple folders to monitor (currently only supports input and output directories)
  - Configurable number of recent files to show per folder
  - For example `[10][output]; [5][input]; clipspace [6][input]` means refreshing checks the output directory, input directory and input/clipspace directory, showing the 10 most recent image files in the output directory, 5 most recent image files in the input directory, and 6 most recent image files in the input/clipspace directory
  - Need to click the refresh button to refresh the list after changing watch_folders
  - When performing inpainting on different positions of an image, you can first mask edit one position and generate multiple images. As long as you don't refresh, the mask will remain effective. After getting satisfactory results, click refresh and then mask edit the next position.
  - This node also exports rgthree actions (Refresh action) that can be used with the Fast Actions node from rgthree-comfy.

- **Load Image (from Any Path)**: Loads images from any specified file path
  - Load images from any ComfyUI path
  - 

#### Image Saving Nodes
- **Save Image to Specified File Name**: Enhanced saving functionality with custom filenames
  - Customizable output filename
  - Support for various image formats (PNG, JPEG, WEBP) automatically determined based on filename, or forcibly specified
  - Metadata preservation (input comes from Load Image (from Any Path) node)
  - Subfolder support
  - Allows automatic opening of the image in the browser after saving, convenient for carefully inspecting output results

#### Utility Nodes
- **Extract Sub Folder**: Extracts subdirectory path from a given file path
  - Configurable extraction level (number of folder levels)
  - Useful for organizing output files

#### Other Enhancements
- Shortcuts
  - **Open Image**: When the node has an image widget, this shortcut opens the image in the browser. Equivalent to the right-click menu's open image
  - **Switch to Mask**: Switch to the mask tool in the Mask Editor
  - **Switch to Eye Dropper**: Switch to the brush tool in the Mask Editor and activate the eye dropper color picker
  - **Save Mask**: Save the mask and exit in the Mask Editor
- Selected node's toolbox buttons
  - **Open Image**: Adds an open image button, with the same effect as the shortcut and right-click menu
- Other improvements
  - When using Ctrl+Z to undo changes in the Mask Editor in ComfyUI, it may accidentally undo node edits or text edits outside. This extension attempts to fix this issue.

## Installation

1. Clone or copy this repository to your ComfyUI custom nodes directory
2. Restart ComfyUI
3. The nodes will appear under the "Slowargo" category in the node menu

## Categories

Nodes in this extension are categorized under "Slowargo" in the ComfyUI node menu.