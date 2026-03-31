from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import os
import subprocess
import sys
from PIL import Image
import io
import time

# ==============================================================================
# 1. 环境配置与路径设置
# ==============================================================================
# 将 SMPL 目录添加到系统路径，以便导入 web_smpl_test 模块
script_dir = os.path.dirname(os.path.abspath(__file__))
smpl_dir = os.path.join(script_dir, 'SMPL_')
sys.path.append(smpl_dir)

# 尝试导入 SMPL 模型类
try:
    from web_smpl_test import SMPLModel
except ImportError:
    print("Could not import web_smpl_test. Please ensure SMPL_ directory exists.")
    sys.exit(1)

app = Flask(__name__)

# ==============================================================================
# 2. 初始化 SMPL 模型
# ==============================================================================
# 加载预训练的 SMPL 模型参数 (.npy 文件)
MODEL_FILE = 'basicModel_m_lbs_10_207_0_v1.0.0.npy'
MODEL_PATH = os.path.join(smpl_dir, MODEL_FILE)

smpl = None
try:
    if os.path.exists(MODEL_PATH):
        smpl = SMPLModel(MODEL_PATH)
        print(f"SMPL Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading SMPL model: {e}")

# ==============================================================================
# 3. 渲染器路径配置
# ==============================================================================
# 渲染器工作目录
RENDERER_DIR = os.path.join(script_dir, 'myrender_', '07Shader')
# 渲染器可执行文件路径
RENDERER_EXE = os.path.join(RENDERER_DIR, 'tinyrender')
# 渲染输出的 TGA 图片路径 (tinyrender 默认输出到其工作目录)
TGA_PATH = os.path.join(RENDERER_DIR, 'output.tga')
# 中间 OBJ 文件路径 (SMPL 生成的模型)
OBJ_PATH = os.path.join(script_dir, 'smpl_output.obj')

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/render', methods=['POST'])
def render():
    """处理渲染请求的核心接口"""
    if smpl is None:
        return jsonify({'error': 'SMPL model not loaded'}), 500

    try:
        data = request.json
        # print(f"Received render request. Beta: {data.get('betas', [])[:3]}...") # 打印日志
        
        # ----------------------------------------------------------------------
        # 第一步：根据前端参数更新 SMPL 模型
        # ----------------------------------------------------------------------
        pose = np.zeros((24, 3))
        beta = np.zeros(10)

        # 解析 Betas 参数 (体型系数)
        if 'betas' in data:
            for i, val in enumerate(data['betas']):
                if i < 10:
                    beta[i] = float(val)

        # 解析 Pose 参数 (姿态系数，目前前端未启用，保留接口)
        # 默认为零姿态 (T-pose)
        
        # 更新 SMPL 参数并导出为 OBJ 文件
        smpl.set_params(pose=pose, beta=beta)
        smpl.save_to_obj(OBJ_PATH)
        # print(f"Saved OBJ to {OBJ_PATH}")

        # ----------------------------------------------------------------------
        # 第二步：调用 C++ 渲染器 (TinyRender)
        # ----------------------------------------------------------------------
        # 计算 OBJ 文件的相对路径，因为是在 RENDERER_DIR 目录下执行命令
        rel_obj_path = os.path.relpath(OBJ_PATH, RENDERER_DIR)
        
        if not os.path.exists(RENDERER_EXE):
             return jsonify({'error': f'Renderer executable not found at {RENDERER_EXE}'}), 500

        # 确保可执行文件有权限
        os.chmod(RENDERER_EXE, 0o755)

        # 使用 subprocess 调用外部 C++ 程序
        # 相当于在终端执行：./tinyrender ../../smpl_output.obj
        result = subprocess.run(
            [RENDERER_EXE, rel_obj_path],
            cwd=RENDERER_DIR,          # 切换到渲染器目录执行
            capture_output=True,       # 捕获输出以便调试
            text=True
        )

        if result.returncode != 0:
            print(f"Renderer failed with return code {result.returncode}")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            return jsonify({'error': 'Rendering failed', 'details': result.stderr}), 500

        if not os.path.exists(TGA_PATH):
            return jsonify({'error': 'Output TGA file not found'}), 500

        # ----------------------------------------------------------------------
        # 第三步：图像格式转换 (TGA -> PNG -> Base64)
        # ----------------------------------------------------------------------
        # 关键修复：手动解析二进制 TGA 文件
        # 背景：TinyRender 输出的 TGA 文件可能包含 RLE 压缩或特殊头部，
        # Python 的 Pillow 库在某些情况下读取会报 "buffer overrun" 错误。
        # 解决方案：在 C++ 端关闭 RLE 压缩，在 Python 端手动读取二进制像素数据。
        
        try:
            # 尝试使用 Pillow 标准库读取
            image = Image.open(TGA_PATH)
            
            # 强制加载数据，这一步通常会触发 Pillow 的潜在 Bug
            image.load() 
            
            # 将图像垂直翻转（如果 C++ 端没有翻转，或者坐标系不一致）
            # image = image.transpose(Image.FLIP_TOP_BOTTOM) 
            
        except Exception as pillow_error:
            # print(f"Pillow failed to read TGA: {pillow_error}. Falling back to manual parsing.")
            
            # 备用方案：手动解析 TGA 二进制文件
            # TGA 文件头结构 (18字节):
            # ID length (1), Colormap type (1), Image type (1), Colormap spec (5), 
            # X origin (2), Y origin (2), Width (2), Height (2), Pixel depth (1), Descriptor (1)
            
            import struct
            
            with open(TGA_PATH, 'rb') as f:
                header = f.read(18)
                # 解析宽和高 (小端序 short)
                width = struct.unpack('<H', header[12:14])[0]
                height = struct.unpack('<H', header[14:16])[0]
                depth = struct.unpack('B', header[16:17])[0]
                
                # print(f"Manual parsing: {width}x{height}, {depth} bits")
                
                # 计算像素数据大小
                # 假设是 RGB (24位) 或 RGBA (32位)
                pixel_bytes = depth // 8
                data_size = width * height * pixel_bytes
                
                # 读取像素数据
                raw_data = f.read(data_size)
                
                # 创建图像对象
                mode = 'RGB' if depth == 24 else 'RGBA'
                image = Image.frombytes(mode, (width, height), raw_data)
                
                # 手动解析的 TGA 通常是倒置的，因为 TGA 默认原点在左下角
                # 如果 C++ 代码里已经 flip_vertically() 了，这里可能不需要
                # 但根据观察，这里通常需要根据实际情况调整
                # image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # 将图像转换为 PNG 格式的字节流
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        
        import base64
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        return jsonify({'image': img_base64})

    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 启动 Flask 服务器
    print("Starting Flask server on port 5002...")
    app.run(debug=True, port=5002)
