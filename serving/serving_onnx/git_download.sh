#!/bin/bash

# Exit on any error
set -e

# Install Git LFS (for Debian/Ubuntu-based systems)
echo "Installing Git LFS..."
sudo apt-get install git-lfs -y

# Initialize Git LFS
echo "Initializing Git LFS..."
git lfs install

# Pull LFS-tracked files (assumes you're inside a git repo)
echo "Pulling LFS-tracked files..."
git lfs pull

echo "Done."