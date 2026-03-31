# TinyRender + SMPL Project

This project integrates the SMPL body model generation with a custom C++ renderer (TinyRender).

## How to Run

I have created a helper script `run.sh` to automate the process.

### 1. One-Click Run
Simply execute the following command in the terminal:
```bash
./run.sh
```
This script will:
1.  Run the Python script to generate a new SMPL model (`smpl_output.obj`).
2.  Compile the C++ renderer.
3.  Render the model to `myrender_/07Shader/output.tga`.

### 2. Manual Steps

**Step 1: Generate SMPL Model**
```bash
python3 SMPL_/export_smpl.py
```
This creates `smpl_output.obj` in the project root.
*Note: You can edit `SMPL_/export_smpl.py` to change the pose parameters (beta, pose).*

**Step 2: Compile & Run Renderer**
```bash
cd myrender_/07Shader
g++ main.cpp tgaimage.cpp model.cpp geometry.cpp our_gl.cpp -o tinyrender
./tinyrender ../../smpl_output.obj
```

## Output
The rendered image is saved at:
`myrender_/07Shader/output.tga`
