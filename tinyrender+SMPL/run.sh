#!/bin/bash

# 1. 生成 SMPL 模型 (.obj)
echo "Generating SMPL model..."
python3 SMPL_/export_smpl.py

if [ $? -ne 0 ]; then
    echo "Error generating SMPL model."
    exit 1
fi

# 2. 编译渲染器 (确保是最新版本)
echo "Compiling renderer..."
cd myrender_/07Shader
g++ main.cpp tgaimage.cpp model.cpp geometry.cpp our_gl.cpp -o tinyrender

if [ $? -ne 0 ]; then
    echo "Error compiling renderer."
    exit 1
fi

# 3. 运行渲染器
echo "Rendering..."
./tinyrender ../../smpl_output.obj

if [ $? -ne 0 ]; then
    echo "Error rendering."
    exit 1
fi

echo "Done! Result saved to myrender_/07Shader/output.tga"
