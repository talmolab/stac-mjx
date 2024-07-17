#!/bin/bash

# Update package list and install system packages with sudo
echo "Installing system packages with apt-get..."
sudo apt-get update
sudo apt-get install -y libglfw3 libglew2.0 libgl1-mesa-glx libosmesa6

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda could not be found. Please install conda and try again."
    exit 1
fi

# Install conda packages
echo "Installing conda packages..."
conda install -c conda-forge glew -y
conda install -c conda-forge mesalib -y
conda install -c anaconda mesa-libgl-cos6-x86_64 -y
conda install -c menpo glfw3 -y

echo "Installation complete."
