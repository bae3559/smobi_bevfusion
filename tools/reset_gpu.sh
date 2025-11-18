#!/bin/bash
# Reset GPU by killing all Python processes using GPU

echo "Current GPU usage:"
nvidia-smi

echo ""
echo "Killing all Python processes..."
pkill -9 python
pkill -9 python3

echo ""
echo "Waiting 2 seconds..."
sleep 2

echo ""
echo "GPU usage after reset:"
nvidia-smi

echo ""
echo "GPU reset complete!"
