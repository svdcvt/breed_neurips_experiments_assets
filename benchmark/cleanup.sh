#!/bin/bash

echo "Starting cleanup..."

d="$1"

find $d -type d -name "client_scripts" -exec rm -rf {} +
find $d -type d -name "checkpoints" -exec rm -rf {} +
find $d -type d -name "stdout" -exec rm -rf {} +

echo "Cleanup complete."

