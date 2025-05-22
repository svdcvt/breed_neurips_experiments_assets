#! /bin/bash

set -ex

# =====================================================================================
# This script automates the installation and setup process for the MELISSA project.
# It removes old build and installation files, installs Python dependencies,
# builds the project using CMake, and compiles example executables.

# Variables:
# MELISSA_DIR: Melissa repository path
# MELISSA_INSTALL_PREFIX: Directory where MELISSA will be installed.
# MELISSA_BIN_DIR: Directory where MELISSA binaries will be built.

# Steps:
# 1. Remove old build and installation directories, and environment setup script.
# 2. Install Python dependencies in the specified installation directory.
# 3. Configure the project with CMake and specify the installation prefix.
# 4. Build the project using the generated Makefiles.
# 5. Install the built project to the specified installation directory.
# 6. Source the environment setup script to set necessary environment variables.
# =====================================================================================


MELISSA_DIR="${1:-$PWD}"
echo $MELISSA_DIR
MELISSA_INSTALL_PREFIX="$MELISSA_DIR/install"
MELISSA_BIN_DIR="$MELISSA_DIR/build"

# remove the old files
rm -rf $MELISSA_BIN_DIR $MELISSA_INSTALL_PREFIX $MELISSA_DIR/melissa_set_env.sh

python3 -m pip install --target=$MELISSA_INSTALL_PREFIX --no-deps "$MELISSA_DIR[dl]"
cmake -DCMAKE_INSTALL_PREFIX=$MELISSA_INSTALL_PREFIX -DINSTALL_ZMQ=ON -B $MELISSA_BIN_DIR -S $MELISSA_DIR
make -C $MELISSA_BIN_DIR
make -C $MELISSA_BIN_DIR install
ls -l $MELISSA_INSTALL_PREFIX/bin

source $MELISSA_DIR/melissa_set_env.sh