#!/usr/bin/env bash

set -e

# Get script directory and workspace root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# LLVM version to download
LLVM_VERSION="20.1.8"
EXTERNAL_DEPS_DIR="${WORKSPACE_ROOT}/externalDeps"

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*)
            echo "macos"
            ;;
        Linux*)
            echo "linux"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            echo "windows"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Detect architecture
detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64)
            echo "x86_64"
            ;;
        arm64|aarch64)
            echo "arm64"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

OS=$(detect_os)
ARCH=$(detect_arch)

echo "Detected OS: $OS"
echo "Detected Architecture: $ARCH"

# Source tarball is OS-independent
TARBALL_NAME="llvm-project-${LLVM_VERSION}.src.tar.xz"
DOWNLOAD_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/${TARBALL_NAME}"

# Create externalDeps directory if it doesn't exist
mkdir -p "$EXTERNAL_DEPS_DIR"

cd "$EXTERNAL_DEPS_DIR"

# Check if already downloaded
if [ -f "$TARBALL_NAME" ]; then
    echo "Tarball already exists: $TARBALL_NAME"
    read -p "Re-download? (y/N): " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Skipping download."
    else
        rm "$TARBALL_NAME"
        echo "Downloading LLVM $LLVM_VERSION source tarball..."
        curl -L -O "$DOWNLOAD_URL"
    fi
else
    echo "Downloading LLVM $LLVM_VERSION source tarball..."
    curl -L -O "$DOWNLOAD_URL"
fi

# Check if already unpacked
if [ -d "llvm-project-${LLVM_VERSION}.src" ]; then
    echo "LLVM source already unpacked: llvm-project-${LLVM_VERSION}.src"
    read -p "Re-extract? (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf "llvm-project-${LLVM_VERSION}.src"
        echo "Extracting tarball..."
        tar -xf "$TARBALL_NAME"
    else
        echo "Skipping extraction."
    fi
else
    echo "Extracting tarball..."
    tar -xf "$TARBALL_NAME"
fi

echo "Done! LLVM ${LLVM_VERSION} source is now in ${EXTERNAL_DEPS_DIR}/llvm-project-${LLVM_VERSION}.src"

# Clean up tarball
echo "Cleaning up tarball..."
rm -f "$TARBALL_NAME"
echo "Tarball removed."
