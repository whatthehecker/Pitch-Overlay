#!/bin/sh

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 MODEL_SIZE" >&2
  exit 1
fi

python save_keras_model.py --size "$1"
python -m tf2onnx.convert --saved-model "outputs/keras_assets" --output "outputs/crepe-$1.onnx"