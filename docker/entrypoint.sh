#!/usr/bin/env bash
set -e
export PYTHONPATH="/home/bevfusion:/home/mmdet3d:${PYTHONPATH}"
exec "$@"