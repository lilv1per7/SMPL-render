import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import scipy

#封装 SMPL 人体模型的所有参数和计算逻辑
class SMPLModel():
    def __init__(self, model_path):
        """
        SMPL model.

        Parameter:
        ---------
        model_path: Path to the SMPL model parameters, pre-processed by
        `preprocess.py`.

        """
        with open(model_path, 'rb') as f:
            # 兼容不同numpy版本的加载方式
            try:
                params = np.load(model_path, allow_pickle=True, encoding='latin1')[()]
            except:
                params = np.load(model_path, allow_pickle=True)[()]

            self.J_regressor = params['J_regressor']
            self.weights = params['weights']
            self.posedirs = params['posedirs']
            self.v_template = params['v_template']
            self.shapedirs = params['shapedirs']
            self.faces = params['f']
            self.kintree_table = params['kintree_table']

        id_to_col = {
            self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
        }
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }

        #定义三组参数的标准形状：
        self.pose_shape = [24, 3]#pose 是 24 个关节 × 3 个轴角分量 = 72 个自由度；
        self.beta_shape = [10]#beta 是 10 个体型 PCA 系数；
        self.trans_shape = [3]#trans 是 3 维全局平移（x, y, z）
        #将姿态、体型、平移全部初始化为零向量。零姿态表示 T-pose（双臂伸展站立），零体型表示平均身材。
        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)
        #vetrs将储存最终计算出的6890个顶点的坐标
        #R将记录24个关节的旋转矩阵
        self.verts = None
        self.R = None

        """
        调用一次核心管线，基于初始参数（全零）计算出默认人体T-pose的顶点和旋转矩阵。
        这样创建对象后立刻就有可用的网格数据。
        """
        self.update()
    
    #参数设置方法
    def set_params(self, pose=None, beta=None, trans=None):
        """
        设置 SMPL 模型的姿势、形状和/或平移参数。模型的顶点
        将被更新并返回.
        
        参数:
        pose: [24, 3] 关节旋转参数 (轴角表示)
        beta: [10] 体型参数
        trans: [3] 全局位移
        """
        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        #只有传入非None参数才改变属性，重新跑一边管线，返回新的顶点坐标
        self.update()
        return self.verts
    
    #SMPL管线第一步Shpe Blend Shapes得到体型修正后的顶点坐标
    def shape_blend_shape(self):
        """
        1. Shape Blend Shapes (体型混合形状):
        计算形状系数(beta)对模板顶点的线性偏移。
        原理: 每个人体形状都是基础模板加上若干个主成分(PCA)的线性组合。
        公式: v_shaped = v_template + shapedirs * beta
        """
        # shapedirs shape: [6890, 3, 10] (每个顶点在每个 beta 分量方向上的偏移量)
        # beta shape: [10] (10个体型系数)
        # output: [6890, 3] (体型修正后的顶点位置)
        v_shaped = self.v_template + self.shapedirs.dot(self.beta)
        return v_shaped
    
    #SMPL管线第三步Pose Blend Shapes得到姿态偏移修正后的顶点坐标
    def pose_blend_shape(self, v_shaped):
        """
        3. Pose Blend Shapes (姿态混合形状):
        计算姿态旋转对身体形状的非线性修正（比如弯曲手肘时肌肉的隆起）。
        原理: 简单的骨骼蒙皮会导致关节处体积塌陷，SMPL 引入姿态相关的形状修正来模拟肌肉变形。
        公式: v_posed = v_shaped + posedirs * (R - R_rest)
        """
        # self.R 是旋转矩阵 [24, 3, 3]
        # 我们需要忽略根关节的旋转 (索引0)，只关注剩下的23个关节
        # 将旋转矩阵展平为 [207] (23 * 9)
        # posedirs shape: [6890, 3, 207] (每个顶点在每个旋转分量上的偏移量)
        
        # 获取除了根节点以外的旋转矩阵，并减去单位矩阵（静止姿态）
        r_rot = (self.R[1:] - np.eye(3)).reshape(-1) 
        
        # 计算姿态修正偏移
        pose_offsets = self.posedirs.dot(r_rot)
        
        v_posed = v_shaped + pose_offsets
        return v_posed
    
    #SMPL管线第四步前向运动学 得到关节变换的齐次矩阵
    def cal_joints(self, rest_J):
        """
        4. Calculate Global Transformations (Forward Kinematics / 正向运动学):
        计算每个关节在世界坐标系下的变换矩阵 G。
        """
        # 1. 为每个关节构建局部变换矩阵 (相对于父关节)
        # [R | J] 
        # [0 | 1]
        # 注意：这里是相对变换，需要先移动到关节位置，旋转，再移回来
        
        # 构建局部旋转平移矩阵 [24, 4, 4]
        # 根关节的位置直接就是 rest_J[0]
        G = np.zeros([self.R.shape[0], 4, 4])
        G[0] = self.with_zeros(np.hstack((self.R[0], rest_J[0].reshape(3, 1))))

        # 遍历运动学树计算全局变换 (从父节点到子节点)
        for i in range(1, self.kintree_table.shape[1]):
            # 当前关节的旋转矩阵 [3, 3]
            curr_R = self.R[i]
            # 当前关节相对于父关节的偏移位置
            j_offset = rest_J[i] - rest_J[self.parent[i]]
            
            # 构建局部变换矩阵 [4, 4]
            local_transform = self.with_zeros(np.hstack((curr_R, j_offset.reshape(3, 1))))
            
            # 全局变换 = 父关节全局变换 * 局部变换
            G[i] = np.matmul(G[self.parent[i]], local_transform)

        # 这一步非常关键：
        # SMPL的蒙皮公式需要去掉静止姿态下的关节位置影响
        # 蒙皮时，顶点是相对于静止姿态定义的，所以变换矩阵应该是：
        # G_final = G_world * G_rest_inverse
        # 其中 G_rest_inverse 相当于将关节平移回原点的矩阵 T(-J)
        """
        SMPL 的蒙皮公式中，顶点 V是在**静止姿态（rest pose）**下定义的。
        如果直接用 𝐺𝑖去变换顶点，由于 𝐺𝑖中已经包含了"从原点移动到关节位置"的平移分量，
        会导致顶点被多移动了一个 𝐽的距离。
        """
        G_prime = G.copy()
        for i in range(G.shape[0]):
            rel_J = np.zeros((4, 4))
            rel_J[:3, 3] = -rest_J[i]
            rel_J[0, 0] = rel_J[1, 1] = rel_J[2, 2] = rel_J[3, 3] = 1
            G_prime[i] = np.matmul(G[i], rel_J)

        return G_prime
    #SMPL管线第五步，线性混合蒙皮 返回最终顶点位置
    def skinning(self, v_posed, G):
        """
        5. Linear Blend Skinning (LBS / 线性混合蒙皮):
        根据蒙皮权重，将骨骼变换应用到顶点上。
        原理: 每个顶点受多个骨骼影响，最终位置是各骨骼变换结果的加权平均。
        公式: v = sum(weight_k * G_k * v_posed)
        """
        # G shape: [24, 4, 4] (24个关节的全局变换矩阵)
        # weights shape: [6890, 24] (每个顶点受24个关节影响的权重，大部分权重为0)
        
        # 计算每个顶点的加权变换矩阵 T [6890, 4, 4]
        # 这是一个张量点积: T = weights dot G
        # 结果 T[i] 是第 i 个顶点的最终变换矩阵
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        
        # 将 v_posed 转换为齐次坐标 [6890, 4] (x, y, z, 1)
        v_homo = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        
        # 应用变换: v_new = T * v_old
        # 使用 einsum 进行高效的批量矩阵乘法
        # 'nij,nj->ni': 
        #   n: 顶点索引 (0~6889)
        #   i, j: 矩阵维度 (0~3)
        #   对每个顶点 n，计算矩阵 T[n] (4x4) 乘以 向量 v_homo[n] (4x1)
        v_prime = np.einsum('nij,nj->ni', T, v_homo)
        
        # 取前3个分量 (x, y, z)
        return v_prime[:, :3]
    
    def update(self):
        """
        SMPL 核心管线：当参数更新时自动调用
        """
        # 0. 预计算旋转矩阵 (Rodrigues公式: 轴角 -> 旋转矩阵)
        # pose [24, 3] -> R [24, 3, 3]
        self.R = self.rodrigues(self.pose.reshape(-1, 1, 3))

        # 1. Shape Blend Shapes: 计算体型对顶点的影响
        v_shaped = self.shape_blend_shape()
        
        # 2. Joint Regressor: 根据新的体型推断关节位置 (Rest Pose Joints)
        rest_J = self.J_regressor.dot(v_shaped)
        
        # 3. Pose Blend Shapes: 计算姿态对形状的修正 (肌肉变形)
        v_posed = self.pose_blend_shape(v_shaped)
        
        # 4. Forward Kinematics: 计算所有关节的全局变换矩阵
        G = self.cal_joints(rest_J)
        
        # 5. Linear Blend Skinning: 蒙皮，计算最终顶点位置
        self.verts = self.skinning(v_posed, G)
        
        # 应用全局位移
        self.verts += self.trans.reshape([1, 3])

    def rodrigues(self, r):
        """
        Rodrigues' rotation formula.
        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(r.dtype).eps)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        """
        Append a [0, 0, 0, 1] vector to a [3, 4] matrix.
        """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x):
        """
        Append zero matrices of shape [4, 3] to vectors of [4, 1] shape.
        """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_to_obj(self, path):
        """
        Save the SMPL model into .obj file.
        """
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the SMPL model
# 请确保该文件在当前目录下
try:
    smpl = SMPLModel('basicModel_m_lbs_10_207_0_v1.0.0.npy')
    print("SMPL Model loaded successfully.")
except Exception as e:
    print(f"Error loading SMPL model: {e}")
    # 创建一个伪造的类以防止服务器启动失败（仅用于调试）
    smpl = None

def get_shift_param(body_vertices):
    v_body_arr = np.array(body_vertices)
    if v_body_arr.size == 0: return 0.0
    min_y = (min(v_body_arr[:, 1]))
    if min_y < 0:
        return abs(min_y)
    return 0.0

@app.route('/generate', methods=['POST'])
def generate_body():
    if smpl is None:
        return jsonify({'status': 'error', 'message': 'SMPL model not loaded.'}), 500
        
    try:
        # Extract beta parameters from the POST request
        data = request.json
        
        # 处理输入，确保是numpy array且长度正确
        beta_input = data.get('beta', [])
        if not beta_input:
            beta_params = np.zeros(10)
        else:
            beta_params = np.array(beta_input)[:10]
            # 补齐到10维
            if len(beta_params) < 10:
                beta_params = np.pad(beta_params, (0, 10 - len(beta_params)), 'constant')
        
        print(f"Received beta params: {beta_params}")
        
        # Use a default neutral pose
        pose = np.zeros(smpl.pose_shape)  
        
        smpl.set_params(beta=beta_params, pose=pose, trans=None)
        
        vertices = smpl.verts
        faces = smpl.faces
        shift_Z = get_shift_param(vertices)  # Find the lowest z value (code implies Y is up/down)
        vertices[:, 1] += shift_Z
        
        return jsonify({'body': {'vertices': vertices.tolist(), 'faces': faces.tolist()}})
    except Exception as e:
        print(f"Error in generate: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
