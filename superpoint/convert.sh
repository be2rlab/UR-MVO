#!/bin/bash
pt_file=$1
pt_file_name=$(basename $pt_file)
pt_file_name="${pt_file_name%.*}"
output_path=$2
echo "Converting $pt_file_name to onnx"
python SP/convert_superpoint_to_onnx.py --weight_file $pt_file --output_dir $output_path
bash SP/simplify_superpoint.sh $output_path/$pt_file_name.onnx $output_path/${pt_file_name}_sim.onnx
python SP/convert_int32.py --model_path $output_path/${pt_file_name}_sim.onnx --output_path $output_path/${pt_file_name}_sim_int32.onnx
    
