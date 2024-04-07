#!/bin/bash
echo "Simplifying $1"
echo "Output file: $2"
python -m onnxsim $1 $2 --dynamic-input-shape --input-shape input:1,1,512,640
