import numpy as np
from web_smpl_test import SMPLModel
import os

# ==============================================================================
# SMPL 模型导出工具
# 功能：加载 SMPL 参数文件，生成指定姿态和体型的人体网格，并导出为 OBJ 文件
# ==============================================================================

def main():
    # 1. 路径设置
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # SMPL 模型参数文件路径 (预训练的权重)
    model_path = os.path.join(script_dir, 'basicModel_m_lbs_10_207_0_v1.0.0.npy')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    # 2. 初始化 SMPL 模型
    try:
        smpl = SMPLModel(model_path)
        print("SMPL Model loaded successfully.")
    except Exception as e:
        print(f"Error loading SMPL model: {e}")
        return

    # 3. 设置人体参数
    # Pose: 关节旋转参数，shape=(24, 3)，表示24个关节的轴角旋转向量
    pose = np.zeros((24, 3))
    
    # Beta: 体型参数，shape=(10,)，控制胖瘦、高矮等
    # 例如：beta[0] 通常控制高矮，beta[1] 控制胖瘦
    beta = np.zeros(10)
    
    # 示例：设置左腿微屈 (仅作测试用，实际参数通常由外部传入)
    # pose[1] = [0, 0, 0.5] 
    
    # 4. 更新模型并计算顶点
    # set_params 会触发 SMPL 内部计算：
    # Shape Blend Shapes -> Pose Blend Shapes -> Forward Kinematics -> Skinning
    smpl.set_params(beta=beta, pose=pose)
    
    # 5. 导出为 OBJ 文件
    # 导出路径：项目根目录下的 smpl_output.obj
    output_path = os.path.join(script_dir, '../smpl_output.obj')
    smpl.save_to_obj(output_path)
    print(f"Saved SMPL mesh to {output_path}")

if __name__ == '__main__':
    main()
