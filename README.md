# InpaintHelper Extension for ComfyUI

A collection of utility nodes for ComfyUI that enhance workflow flexibility and image handling capabilities.

## Features

### Node Types

#### Float Switch
- **FloatSwitch**: A versatile float value selector that allows switching between two float values based on a boolean toggle.
  - Provides two input floats with slider controls
  - Toggle switch to select between the two values
  - Optional override input that takes priority when greater than 0

#### Image Loading Nodes
- **Load Image (from Outputs) Plus**: Enhanced image loading from output directories
  - Automatically shows recently created images from output folder
  - Includes metadata extraction support

- **Load Recent Image**: Loads images from configurable watched folders with flexible filtering
  - Allows specification of multiple folders to monitor
  - Configurable number of recent files to show per folder
  - Supports both input and output directories

- **Load Image (from Any Path)**: Loads images from any specified file path
  - Direct file path input
  - Supports metadata extraction

#### Image Saving Nodes
- **Save Image to Specified File Name**: Enhanced saving functionality with custom filenames
  - Customizable output filename
  - Support for various image formats (PNG, JPEG, WEBP)
  - Metadata preservation
  - Subfolder support

#### Utility Nodes
- **Extract Sub Folder**: Extracts subdirectory path from a given file path
  - Configurable extraction level (number of folder levels)
  - Useful for organizing output files

## Installation

1. Clone or copy this repository to your ComfyUI custom nodes directory
2. Restart ComfyUI
3. The nodes will appear under the "Slowargo" category in the node menu

## Categories

Nodes in this extension are categorized under "Slowargo" in the ComfyUI node menu.