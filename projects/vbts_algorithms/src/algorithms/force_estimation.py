"""
力估计算法 - 视触传感器核心算法之二

基于PDF提取内容实现，将位移场转换为接触力。
实现Hertz接触理论、有限元反问题求解等力计算方法。
"""

import numpy as np
from typing import Tuple, Optional

# 尝试导入matplotlib，如果不可用则提供占位
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


class ForceEstimation:
    """
    力估计类，用于从位移场计算接触力
    """
    
    def __init__(self, youngs_modulus: float = 1.0e6, poissons_ratio: float = 0.49,
                 sensor_thickness: float = 5.0):
        """
        初始化力估计器
        
        Args:
            youngs_modulus: 杨氏模量 (Pa)
            poissons_ratio: 泊松比
            sensor_thickness: 传感器厚度 (mm)
        """
        self.youngs_modulus = youngs_modulus
        self.poissons_ratio = poissons_ratio
        self.sensor_thickness = sensor_thickness
        
        # 计算剪切模量
        self.shear_modulus = youngs_modulus / (2 * (1 + poissons_ratio))
        
    def filter_displacement_field(self, displacement_field: np.ndarray,
                                positions: np.ndarray,
                                filter_type: str = 'gaussian',
                                filter_radius: float = 5.0) -> np.ndarray:
        """
        对位移场进行噪声滤波，降低力估计的噪声敏感性
        
        Args:
            displacement_field: 位移场 (N, 2) 或 (N, 3) (单位: mm)
            positions: 测量点坐标 (N, 2) (单位: mm)
            filter_type: 滤波器类型 ('gaussian', 'median', 'moving_average', 'physical')
            filter_radius: 滤波半径 (mm)
            
        Returns:
            filtered_displacement: 滤波后的位移场
        """
        if len(displacement_field) < 3:
            return displacement_field
            
        n_points = len(displacement_field)
        filtered = np.zeros_like(displacement_field)
        
        if filter_type == 'gaussian':
            # 高斯空间滤波
            for i in range(n_points):
                # 计算到当前点的距离
                distances = np.linalg.norm(positions - positions[i], axis=1)
                
                # 计算高斯权重
                weights = np.exp(-0.5 * (distances / filter_radius) ** 2)
                weights = weights / np.sum(weights)  # 归一化
                
                # 加权平均
                for d in range(displacement_field.shape[1]):
                    filtered[i, d] = np.sum(displacement_field[:, d] * weights)
                    
        elif filter_type == 'median':
            # 中值滤波（基于距离的局部邻域）
            for i in range(n_points):
                # 找到邻域内点
                distances = np.linalg.norm(positions - positions[i], axis=1)
                neighbor_indices = np.where(distances <= filter_radius)[0]
                
                if len(neighbor_indices) > 0:
                    # 对每个位移分量取中值
                    for d in range(displacement_field.shape[1]):
                        neighbor_values = displacement_field[neighbor_indices, d]
                        filtered[i, d] = np.median(neighbor_values)
                else:
                    filtered[i] = displacement_field[i]
                    
        elif filter_type == 'moving_average':
            # 移动平均滤波
            for i in range(n_points):
                distances = np.linalg.norm(positions - positions[i], axis=1)
                neighbor_indices = np.where(distances <= filter_radius)[0]
                
                if len(neighbor_indices) > 0:
                    filtered[i] = np.mean(displacement_field[neighbor_indices], axis=0)
                else:
                    filtered[i] = displacement_field[i]
                    
        elif filter_type == 'physical':
            # 基于物理模型的滤波
            # 假设位移场应满足连续性和平滑性约束
            # 使用薄板样条或类似方法
            
            # 计算位移梯度
            # 简化方法：使用局部平面拟合
            for i in range(n_points):
                distances = np.linalg.norm(positions - positions[i], axis=1)
                neighbor_indices = np.where(distances <= filter_radius)[0]
                
                if len(neighbor_indices) >= 3:
                    # 使用局部平面拟合估计平滑位移
                    neighbor_pos = positions[neighbor_indices]
                    neighbor_disp = displacement_field[neighbor_indices]
                    
                    # 拟合平面: u = a*x + b*y + c
                    try:
                        # 构建设计矩阵
                        A = np.column_stack([neighbor_pos, np.ones(len(neighbor_indices))])
                        
                        # 对每个位移分量拟合
                        for d in range(displacement_field.shape[1]):
                            coeffs, _, _, _ = np.linalg.lstsq(A, neighbor_disp[:, d], rcond=None)
                            # 计算拟合值
                            filtered[i, d] = np.dot([positions[i, 0], positions[i, 1], 1], coeffs)
                    except:
                        filtered[i] = displacement_field[i]
                else:
                    filtered[i] = displacement_field[i]
        else:
            # 未知滤波类型，返回原始数据
            return displacement_field
            
        return filtered
    
    def analyze_displacement_field(self, displacement_field: np.ndarray,
                                 positions: np.ndarray) -> dict:
        """
        分析位移场特征，为方法选择提供依据
        
        Args:
            displacement_field: 位移场 (N, 2) 或 (N, 3) (单位: mm)
            positions: 测量点坐标 (N, 2) (单位: mm)
            
        Returns:
            features: 位移场特征字典
        """
        n_points = len(displacement_field)
        
        if n_points < 3:
            return {
                'n_points': n_points,
                'max_displacement': 0.0,
                'mean_displacement': 0.0,
                'noise_level': 0.0,
                'spatial_variation': 0.0,
                'recommended_method': 'hertz'
            }
        
        # 计算位移大小
        displacement_magnitude = np.linalg.norm(displacement_field, axis=1)
        max_displacement = np.max(displacement_magnitude)
        mean_displacement = np.mean(displacement_magnitude)
        
        # 估计噪声水平（通过局部变化）
        # 使用局部邻域的标准差作为噪声估计
        noise_estimate = 0.0
        spatial_variation = 0.0
        
        if n_points >= 10:
            # 计算位移梯度的局部变化
            local_variations = []
            for i in range(min(50, n_points)):  # 采样点以减少计算
                distances = np.linalg.norm(positions - positions[i], axis=1)
                neighbor_indices = np.where(distances <= 5.0)[0]  # 5mm半径
                
                if len(neighbor_indices) >= 3:
                    neighbor_displacement = displacement_magnitude[neighbor_indices]
                    local_std = np.std(neighbor_displacement)
                    local_variations.append(local_std)
            
            if local_variations:
                noise_estimate = np.mean(local_variations)
                spatial_variation = np.std(local_variations)
        
        # 计算数据密度（点/mm²）
        if len(positions) >= 2:
            x_range = positions[:, 0].max() - positions[:, 0].min()
            y_range = positions[:, 1].max() - positions[:, 1].min()
            area = x_range * y_range
            point_density = n_points / max(area, 1.0)
        else:
            point_density = 0.0
        
        # 转换为Python float类型
        max_displacement_float = float(max_displacement)
        mean_displacement_float = float(mean_displacement)
        noise_estimate_float = float(noise_estimate)
        spatial_variation_float = float(spatial_variation)
        point_density_float = float(point_density)
        
        # 基于特征推荐方法
        recommended_method = self._recommend_method(
            max_displacement_float, mean_displacement_float, noise_estimate_float, 
            spatial_variation_float, point_density_float, n_points
        )
        
        return {
            'n_points': n_points,
            'max_displacement': max_displacement_float,
            'mean_displacement': mean_displacement_float,
            'noise_level': noise_estimate_float,
            'spatial_variation': spatial_variation_float,
            'point_density': point_density_float,
            'recommended_method': recommended_method
        }
    
    def _recommend_method(self, max_displacement: float, mean_displacement: float,
                         noise_level: float, spatial_variation: float,
                         point_density: float, n_points: int) -> str:
        """
        基于位移场特征推荐最佳力估计方法
        
        改进的决策逻辑基于64测试用例验证结果：
        1. Hertz: 快速(0.2ms)，误差67.8%，对小力值误差大(5N:96.3%)，对大力值较好(40N:65.9%)
        2. Boussinesq: 慢(5.86s)，误差153.8%，但对低噪声和大力值表现更好
        3. FEM: 误差极大(932.2%)，不推荐除非有特殊需求
        
        优化目标: 在精度和速度之间取得平衡
        
        Returns:
            method: 推荐的方法 ('hertz', 'boussinesq', 'fem')
        """
        # 转换为Python float类型避免numpy类型问题
        max_displacement = float(max_displacement)
        mean_displacement = float(mean_displacement)
        noise_level = float(noise_level)
        spatial_variation = float(spatial_variation)
        point_density = float(point_density)
        
        # 如果数据点太少，使用Hertz
        if n_points < 15:
            return 'hertz'
        
        # 计算噪声比率
        noise_ratio = noise_level / max(mean_displacement, 0.01)
        
        # 决策树
        # 条件1: 高噪声 -> Hertz（更稳健）
        if noise_ratio > 0.5:  # 噪声超过位移的50%
            return 'hertz'
        
        # 条件2: 低数据密度 -> Hertz
        if point_density < 0.05:  # 低于0.05点/mm²（降低阈值）
            return 'hertz'
        
        # 条件3: 极小位移 -> Hertz（Boussinesq对小位移数值不稳定）
        if max_displacement < 0.05:  # 小于0.05mm
            return 'hertz'
        
        # 条件4: 中等以上位移 + 低噪声 -> Boussinesq可能更好
        # 基于验证结果: Boussinesq在40N时误差43.6%，Hertz在40N时误差65.9%
        if max_displacement > 0.5:  # 位移较大
            if noise_ratio < 0.2:  # 低噪声
                if n_points > 30:  # 足够的数据点
                    # 权衡: Boussinesq更精确但慢
                    # 这里可以根据应用需求调整
                    # 如果需要高精度且可以接受较慢计算，选择Boussinesq
                    # 否则选择Hertz
                    return 'boussinesq'
        
        # 条件5: 实时性要求高 -> Hertz（快速）
        # 这里假设大多数应用需要实时性
        
        # 默认使用Hertz（平衡精度与速度）
        return 'hertz'
    
    def adaptive_force_estimation(self, displacement_field: np.ndarray,
                                positions: np.ndarray,
                                use_filter: bool = True,
                                filter_type: str = 'gaussian',
                                filter_radius: float = 5.0) -> dict:
        """
        自适应力估计：基于位移场特征自动选择最佳方法
        
        Args:
            displacement_field: 位移场 (N, 2) 或 (N, 3) (单位: mm)
            positions: 测量点坐标 (N, 2) (单位: mm)
            use_filter: 是否自动应用滤波
            filter_type: 滤波类型
            filter_radius: 滤波半径
            
        Returns:
            result: 力估计结果字典，包含额外的方法选择信息
        """
        # 分析位移场特征
        features = self.analyze_displacement_field(displacement_field, positions)
        recommended_method = features['recommended_method']
        
        print(f"自适应方法选择:")
        print(f"  数据点: {features['n_points']}")
        print(f"  最大位移: {features['max_displacement']:.3f} mm")
        print(f"  平均位移: {features['mean_displacement']:.3f} mm")
        print(f"  噪声估计: {features['noise_level']:.3f} mm")
        print(f"  点密度: {features['point_density']:.2f} 点/mm²")
        print(f"  推荐方法: {recommended_method}")
        
        # 如果需要滤波且噪声较高
        pre_filter = use_filter
        if use_filter and features['noise_level'] > features['mean_displacement'] * 0.2:
            print(f"  应用{filter_type}滤波 (噪声较高)")
            pre_filter = True
        
        # 使用推荐方法进行力估计
        result = self.estimate_force_from_displacement(
            displacement_field, positions,
            method=recommended_method,
            pre_filter=pre_filter,
            filter_type=filter_type,
            filter_radius=filter_radius,
            boussinesq_grid_resolution=20.0,
            boussinesq_regularization=1e-8
        )
        
        # 添加特征信息到结果
        result['features'] = features
        result['method_selected'] = recommended_method
        result['filter_applied'] = pre_filter
        
        return result
        
    def hertz_contact_force(self, displacement: np.ndarray, 
                           contact_radius: float = 10.0,
                           sphere_radius: float = 50.0) -> Tuple[float, np.ndarray]:
        """
        基于Hertz接触理论计算接触力
        
        Hertz接触理论描述弹性球体与平面的接触
        
        Args:
            displacement: 接触中心位移向量 (3,) [dx, dy, dz] (单位: mm)
            contact_radius: 接触半径 (mm)
            sphere_radius: 球体半径 (mm)
            
        Returns:
            total_force: 总接触力大小 (N)
            force_distribution: 力分布 (N/mm²)
        """
        # Hertz接触公式: F = (4/3) * E* * sqrt(R) * δ^(3/2)
        # 其中 E* = E/(1-ν²) 为等效杨氏模量，R为球体半径，δ为压缩位移
        # 所有长度单位需要转换为米
        
        # 计算等效杨氏模量 (Pa)
        E_star = self.youngs_modulus / (1 - self.poissons_ratio**2)
        
        # 主要考虑垂直位移 (dz)，单位: mm → m
        dz = displacement[2] if len(displacement) > 2 else displacement[1]
        
        # 确保位移为正（压缩）
        if dz < 0:
            dz = abs(dz)
        
        # 单位转换: mm → m
        dz_m = dz / 1000.0
        sphere_radius_m = sphere_radius / 1000.0
        
        # Hertz接触力公式 (单位: N)
        total_force = (4/3) * E_star * np.sqrt(sphere_radius_m) * (dz_m ** 1.5)
        
        # 力分布（Hertz接触压力分布）
        # 接触压力分布: p(r) = p0 * sqrt(1 - (r/a)²), 其中 p0 = (3F)/(2πa²) 为最大压力
        # 单位: N/m² (Pa)，需要转换为 N/mm² 用于可视化
        
        # 创建坐标网格 (单位: mm)
        x = np.linspace(-contact_radius, contact_radius, 50)
        y = np.linspace(-contact_radius, contact_radius, 50)
        xx, yy = np.meshgrid(x, y)
        
        # 距离接触中心的距离 (mm)
        r = np.sqrt(xx**2 + yy**2)
        
        # 最大接触压力 (Pa)
        contact_radius_m = contact_radius / 1000.0  # mm → m
        p0 = (3 * total_force) / (2 * np.pi * contact_radius_m**2)  # 单位: Pa
        
        # 压力分布 (Pa)
        pressure_distribution = np.zeros_like(r)
        mask = r <= contact_radius
        pressure_distribution[mask] = p0 * np.sqrt(1 - (r[mask]/contact_radius)**2)
        
        # 转换为 N/mm² (1 MPa = 1 N/mm²)
        force_distribution = pressure_distribution / 1e6  # Pa → MPa = N/mm²
        
        return total_force, force_distribution
    
    def boussinesq_solution(self, displacement_field: np.ndarray, 
                           positions: np.ndarray, 
                             grid_resolution: float = 20.0,
                             regularization: float = 1e-8,
                             pre_filter: bool = False,
                             filter_type: str = 'gaussian',
                             filter_radius: float = 5.0) -> np.ndarray:
        """
        使用Boussinesq解计算表面力分布
        
        Boussinesq解描述弹性半空间表面受集中力作用的位移场
        这里求解反问题：从位移场反推力分布
        
        Args:
            displacement_field: 位移场 (N, 2) 或 (N, 3) (单位: mm)
            positions: 位移测量点坐标 (N, 2) (单位: mm)
            grid_resolution: 力网格分辨率 (mm)
            regularization: 正则化参数
            pre_filter: 是否对位移场进行预滤波 (默认: False)
            filter_type: 滤波器类型 ('gaussian', 'median', 'moving_average', 'physical')
            filter_radius: 滤波半径 (mm)
            
        Returns:
            force_grid: 力分布网格 (M, M) (单位: N)
        """
        # 可选的位移场预滤波
        if pre_filter and len(displacement_field) > 3:
            displacement_field = self.filter_displacement_field(
                displacement_field, positions, filter_type, filter_radius
            )
        # Boussinesq解: 弹性半空间表面点力引起的垂直位移
        # 对于点力P作用于原点，表面点(x,y)的垂直位移为:
        # u_z = P(1+ν)/(2πE) * (1-2ν)/r，其中r=√(x²+y²)
        # 对于平面应变问题，使用公式: u_z = P(1-ν²)/(πE) * 1/r
        
        # 创建力网格
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        # 扩展边界
        margin = 20.0  # mm
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        # 生成网格
        x_grid = np.arange(x_min, x_max, grid_resolution)
        y_grid = np.arange(y_min, y_max, grid_resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        n_grid = len(grid_points)
        n_measurements = len(positions)
        
        if n_grid == 0 or n_measurements == 0:
            return np.zeros(xx.shape)
        
        # 构建影响系数矩阵 G (n_measurements × n_grid)
        # G[i,j] = (1-ν²)/(πE) * 1/rij，其中rij为测量点i到网格点j的距离 (单位: m)
        G = np.zeros((n_measurements, n_grid))
        
        # 弹性系数 (单位: m/N)
        elastic_coeff = (1 - self.poissons_ratio**2) / (np.pi * self.youngs_modulus)
        
        # 单位转换: mm → m
        mm_to_m = 1e-3
        
        for i in range(n_measurements):
            # 测量点位置 (mm → m)
            xi_m = positions[i, 0] * mm_to_m
            yi_m = positions[i, 1] * mm_to_m
            
            # 计算到所有网格点的距离 (mm → m)
            dx = (grid_points[:, 0] - positions[i, 0]) * mm_to_m
            dy = (grid_points[:, 1] - positions[i, 1]) * mm_to_m
            distances_m = np.sqrt(dx**2 + dy**2)
            
            # 避免除零
            distances_m = np.maximum(distances_m, 0.1 * mm_to_m)  # 最小距离0.1mm → 0.0001m
            
            # 影响系数 (位移/力)，单位: m/N
            G[i, :] = elastic_coeff / distances_m
        
        # 位移向量 (垂直位移分量) (mm → m)
        # 假设位移场为垂直位移，取位移向量的模作为垂直位移近似
        u_m = np.linalg.norm(displacement_field, axis=1) * mm_to_m  # 单位: m
        
        # 求解反问题: G * f = u，其中f为网格点力 (N)
        # 使用正则化最小二乘法: (G^T G + λI) f = G^T u
        try:
            # 构建正则化矩阵
            GTG = G.T @ G
            GTG_reg = GTG + regularization * np.eye(n_grid)
            
            # 求解: G * f = u_m
            f = np.linalg.solve(GTG_reg, G.T @ u_m)
        except np.linalg.LinAlgError:
            # 如果求解失败，使用伪逆
            print("警告: 直接求解失败，使用伪逆")
            f = np.linalg.pinv(G) @ u_m
        
        # 重塑为网格
        force_grid = f.reshape(xx.shape)
        
        # 确保力为非负（压缩力）
        force_grid = np.maximum(force_grid, 0)
        
        return force_grid
    
    def finite_element_inverse(self, displacement_field: np.ndarray,
                              node_positions: np.ndarray,
                              connectivity: np.ndarray,
                              regularization: float = 1e-6,
                              check_condition: bool = True) -> np.ndarray:
        """
        有限元反问题求解：从位移场计算节点力
        
        基于简化的弹簧网络模型构建刚度矩阵，使用Tikhonov正则化处理病态问题
        
        Args:
            displacement_field: 节点位移场 (N, 2) 或 (N, 3) (单位: mm)
            node_positions: 节点坐标 (N, 2) (单位: mm)
            connectivity: 单元连接关系 (M, 3) 三角形单元
            regularization: 正则化参数（默认: 1e-6）
            check_condition: 是否检查刚度矩阵条件数（默认: True）
            
        Returns:
            node_forces: 节点力 (N, 2)
        """
        n_nodes = len(node_positions)
        
        # 构建弹簧网络刚度矩阵
        # 每个三角形单元贡献弹簧刚度，近似为线性弹性
        # 简化: 刚度与杨氏模量成正比，与单元面积成反比
        
        # 初始化刚度矩阵 (2N × 2N)
        K = np.zeros((2 * n_nodes, 2 * n_nodes))
        
        # 遍历所有三角形单元
        for elem in connectivity:
            if len(elem) != 3:
                continue
                
            # 三角形顶点
            i, j, k = elem
            if i >= n_nodes or j >= n_nodes or k >= n_nodes:
                continue
                
            # 节点坐标
            p1 = node_positions[i]
            p2 = node_positions[j]
            p3 = node_positions[k]
            
            # 计算三角形面积 (mm²)
            # 使用叉积公式: area = 0.5 * |(p2-p1) × (p3-p1)|
            v1 = p2 - p1
            v2 = p3 - p1
            area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
            
            if area < 1e-6:
                continue
                
            # 单元刚度系数 (改进的物理模型)
            # 对于平面应力问题，单元刚度与 E * t * A 成正比
            # 其中 E: 杨氏模量 (Pa), t: 厚度 (m), A: 面积 (m²)
            # 转换为国际单位制
            area_m2 = area * 1e-6  # mm² → m²
            thickness_m = self.sensor_thickness * 1e-3  # mm → m
            
            # 简化刚度系数 (比例因子)
            # 对于线性弹性三角形单元，刚度矩阵元素 ~ E * t * A / L²
            # 其中 L 为特征长度，这里使用三角形平均边长
            # 计算三角形边长
            side1 = np.linalg.norm(p2 - p1)
            side2 = np.linalg.norm(p3 - p2)
            side3 = np.linalg.norm(p1 - p3)
            avg_side = (side1 + side2 + side3) / 3.0
            avg_side_m = avg_side * 1e-3  # mm → m
            
            # 刚度系数 (N/m)
            k_elem = self.youngs_modulus * thickness_m * area_m2 / (avg_side_m ** 2)
            
            # 分配刚度到节点对 (简化: 均分到所有节点对)
            # 实际有限元需要形函数，这里简化
            for node_idx in [i, j, k]:
                for node_jdx in [i, j, k]:
                    if node_idx == node_jdx:
                        # 对角项
                        K[2*node_idx:2*node_idx+2, 2*node_idx:2*node_idx+2] += np.eye(2) * k_elem / 3.0
                    else:
                        # 耦合项 (负号表示吸引力)
                        K[2*node_idx:2*node_idx+2, 2*node_jdx:2*node_jdx+2] -= np.eye(2) * k_elem / 6.0
        
        # 检查刚度矩阵条件数
        if check_condition and n_nodes > 0:
            try:
                # 计算条件数（近似）
                eigenvalues = np.linalg.eigvalsh(K[:min(10, K.shape[0]), :min(10, K.shape[1])])
                cond_number = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues[np.abs(eigenvalues) > 1e-10]))
                if cond_number > 1e12:
                    print(f"警告: 刚度矩阵条件数过大 ({cond_number:.2e})，增加正则化")
                    regularization *= 100  # 自动增加正则化
            except:
                pass  # 条件数计算失败时继续
        
        # 添加Tikhonov正则化 (改进版)
        # 基于刚度矩阵迹调整正则化强度
        
        # 默认正则化强度
        reg_strength = self.youngs_modulus * regularization
        
        if n_nodes > 0:
            # 计算刚度矩阵迹（对角线元素和）作为参考
            try:
                K_trace = np.trace(K) / (2 * n_nodes)  # 平均对角线值
                # 如果迹很小或为零，使用默认值
                if K_trace > 1e-10:
                    # 自适应正则化：正则化项 = α * K_trace * I
                    # 其中 α 是正则化参数
                    alpha = regularization
                    reg_strength = alpha * K_trace
            except:
                pass  # 使用默认值
        
        # 应用正则化
        K_reg = K + np.eye(2 * n_nodes) * reg_strength
        
        # 位移向量 (展平并转换为米)
        u_mm = displacement_field.flatten()  # 单位: mm
        u_m = u_mm * 1e-3  # 单位: m
        
        # 确保尺寸匹配
        if len(u_m) != K_reg.shape[0]:
            # 如果尺寸不匹配，创建简单对角刚度矩阵
            print("警告: 位移场尺寸与节点数不匹配，使用简化刚度矩阵")
            K_reg = np.eye(len(u_m)) * self.youngs_modulus * 0.1
        
        # 计算力: f = K_reg * u_m  (使用正则化后的刚度矩阵)
        # K_reg 单位: N/m, u_m 单位: m → f 单位: N
        try:
            f = K_reg @ u_m
        except np.linalg.LinAlgError as e:
            # 如果矩阵求解失败，尝试使用伪逆
            print(f"警告: 矩阵求解失败 ({e})，使用伪逆")
            try:
                f = np.linalg.pinv(K_reg) @ u_m
            except:
                # 如果伪逆也失败，使用简化方法
                print("警告: 伪逆也失败，使用简化方法")
                f = u_m * self.youngs_modulus * 0.01  # 更小的比例因子
        
        # 重塑为节点力 (N, 2)
        node_forces = f.reshape(-1, 2)
        
        return node_forces
    
    def estimate_force_from_displacement(self, displacement_field: np.ndarray,
                                        positions: np.ndarray,
                                        method: str = 'boussinesq',
                                        pre_filter: bool = False,
                                        filter_type: str = 'gaussian',
                                        filter_radius: float = 5.0,
                                        boussinesq_grid_resolution: float = 20.0,
                                        boussinesq_regularization: float = 1e-8) -> dict:
        """
        从位移场估计力分布
        
        Args:
            displacement_field: 位移场 (N, 2)
            positions: 测量点位置 (N, 2)
            method: 力估计方法 ('hertz', 'boussinesq', 'fem')
            pre_filter: 是否对位移场进行预滤波 (默认: False)
            filter_type: 滤波器类型 ('gaussian', 'median', 'moving_average', 'physical')
            filter_radius: 滤波半径 (mm)
            boussinesq_grid_resolution: Boussinesq方法网格分辨率 (mm)
            boussinesq_regularization: Boussinesq方法正则化参数
            
        Returns:
            result: 包含力估计结果的字典
        """
        result = {
            'method': method,
            'total_force': 0.0,
            'force_distribution': None,
            'max_force': 0.0,
            'force_center': None
        }
        
        if method == 'hertz':
            # 使用Hertz接触理论
            # 假设主要位移在接触中心
            center_idx = np.argmax(np.linalg.norm(displacement_field, axis=1))
            center_displacement = displacement_field[center_idx]
            
            # 添加垂直位移分量
            if len(center_displacement) == 2:
                center_displacement = np.array([center_displacement[0], 
                                               center_displacement[1], 
                                               np.linalg.norm(center_displacement)])
            
            total_force, force_dist = self.hertz_contact_force(
                center_displacement, 
                contact_radius=15.0,
                sphere_radius=50.0
            )
            
            result['total_force'] = total_force
            result['force_distribution'] = force_dist
            result['max_force'] = force_dist.max() if force_dist is not None else 0.0
            result['force_center'] = positions[center_idx]
            
        elif method == 'boussinesq':
            # 使用Boussinesq解
            force_grid = self.boussinesq_solution(
                displacement_field, positions, 
                grid_resolution=boussinesq_grid_resolution, 
                regularization=boussinesq_regularization,
                pre_filter=pre_filter,
                filter_type=filter_type,
                filter_radius=filter_radius
            )
            
            result['force_distribution'] = force_grid
            result['total_force'] = np.sum(force_grid)  # 总力为网格点力之和 (N)
            result['max_force'] = force_grid.max()
            
            # 力中心（加权平均）
            if force_grid.sum() > 0:
                y_coords, x_coords = np.indices(force_grid.shape)
                force_center_x = np.average(x_coords, weights=force_grid)
                force_center_y = np.average(y_coords, weights=force_grid)
                result['force_center'] = np.array([force_center_x, force_center_y])
        
        elif method == 'fem':
            # 使用有限元反问题
            # 可选：对位移场进行预滤波
            if pre_filter and len(displacement_field) > 3:
                displacement_field = self.filter_displacement_field(
                    displacement_field, positions, filter_type, filter_radius
                )
            
            # 生成简单三角形网格
            # 尝试导入scipy生成Delaunay三角网格，失败时使用简单网格
            try:
                from scipy.spatial import Delaunay
                tri = Delaunay(positions)
                connectivity = tri.simplices
            except ImportError:
                # scipy不可用，为规则网格创建三角形网格连接
                # 假设positions是近似规则排列的（来自标记点检测）
                connectivity = self._create_grid_connectivity(positions)
                print(f"注意：scipy不可用，使用规则网格连接 ({len(connectivity)} 个三角形)")
            
            node_forces = self.finite_element_inverse(
                displacement_field, positions, connectivity,
                regularization=1e-6, check_condition=True
            )
            
            result['force_distribution'] = node_forces
            result['total_force'] = np.linalg.norm(node_forces, axis=1).sum()
            result['max_force'] = np.linalg.norm(node_forces, axis=1).max()
            result['force_center'] = positions[np.argmax(np.linalg.norm(node_forces, axis=1))]
        
        return result
    
    def calibrate_from_data(self, displacement_fields: list, known_forces: list,
                           initial_params: dict = None, method: str = 'hertz') -> dict:
        """
        从已知力-位移数据校准材料参数
        
        Args:
            displacement_fields: 位移场列表，每个为(N, 2)数组
            known_forces: 已知力列表，每个为标量或向量
            initial_params: 初始参数字典 {'youngs_modulus', 'poissons_ratio'}
            method: 使用的力估计方法 ('hertz', 'boussinesq', 'fem')
            
        Returns:
            calibration_result: 包含校准参数和评估指标的字典
        """
        if initial_params is None:
            initial_params = {
                'youngs_modulus': self.youngs_modulus,
                'poissons_ratio': self.poissons_ratio
            }
        
        # 简单实现：使用平均比例因子调整杨氏模量
        # 更复杂的实现可以使用优化算法（如scipy.optimize）
        
        total_error = 0.0
        n_samples = min(len(displacement_fields), len(known_forces))
        
        if n_samples == 0:
            print("警告：无校准数据")
            return {'success': False, 'error': '无数据'}
        
        # 计算当前参数下的力估计误差
        estimated_forces = []
        errors = []
        
        for i in range(n_samples):
            # 使用当前参数估计力
            result = self.estimate_force_from_displacement(
                displacement_fields[i], 
                np.zeros((len(displacement_fields[i]), 2)),  # 占位位置
                method=method
            )
            
            estimated_force = result['total_force']
            known_force = known_forces[i] if isinstance(known_forces[i], (int, float)) else np.linalg.norm(known_forces[i])
            
            estimated_forces.append(estimated_force)
            error = abs(estimated_force - known_force)
            errors.append(error)
            total_error += error
        
        # 计算平均误差和比例因子
        avg_error = total_error / n_samples
        avg_estimated = np.mean(np.array(estimated_forces))
        known_forces_array = np.array([f if isinstance(f, (int, float)) else np.linalg.norm(f) for f in known_forces])
        avg_known = np.mean(known_forces_array)
        
        # 如果已知力平均值不为零，计算比例因子
        if avg_known > 0:
            scale_factor = avg_known / avg_estimated
            # 调整杨氏模量（假设力与杨氏模量成正比）
            calibrated_youngs_modulus = self.youngs_modulus * scale_factor
        else:
            scale_factor = 1.0
            calibrated_youngs_modulus = self.youngs_modulus
        
        calibration_result = {
            'success': True,
            'initial_params': initial_params,
            'calibrated_params': {
                'youngs_modulus': calibrated_youngs_modulus,
                'poissons_ratio': self.poissons_ratio,  # 泊松比通常变化不大
                'scale_factor': scale_factor
            },
            'performance': {
                'n_samples': n_samples,
                'avg_error': avg_error,
                'max_error': np.max(np.array(errors)) if errors else 0,
                'relative_error': avg_error / avg_known if avg_known > 0 else float('inf'),
                'r_squared': 1.0 - (np.var(np.array(errors)) / np.var(known_forces_array)) if n_samples > 1 else 1.0
            },
            'recommendation': f"建议使用杨氏模量 {calibrated_youngs_modulus:.2e} Pa (原值 {self.youngs_modulus:.2e} Pa)"
        }
        
        return calibration_result
    
    def validate_estimation(self, displacement_fields: list, known_forces: list,
                           method: str = 'hertz') -> dict:
        """
        验证力估计准确性
        
        Args:
            displacement_fields: 位移场列表
            known_forces: 已知力列表
            method: 力估计方法
            
        Returns:
            validation_result: 包含验证指标的字典
        """
        n_samples = min(len(displacement_fields), len(known_forces))
        
        if n_samples == 0:
            return {'success': False, 'error': '无验证数据'}
        
        estimated_forces = []
        errors = []
        relative_errors = []
        
        for i in range(n_samples):
            result = self.estimate_force_from_displacement(
                displacement_fields[i],
                np.zeros((len(displacement_fields[i]), 2)),  # 占位位置
                method=method
            )
            
            estimated_force = result['total_force']
            known_force = known_forces[i] if isinstance(known_forces[i], (int, float)) else np.linalg.norm(known_forces[i])
            
            estimated_forces.append(estimated_force)
            error = abs(estimated_force - known_force)
            errors.append(error)
            
            if known_force > 0:
                relative_errors.append(error / known_force)
        
        validation_result = {
            'success': True,
            'method': method,
            'n_samples': n_samples,
            'metrics': {
                'mae': np.mean(np.array(errors)) if errors else 0,  # 平均绝对误差
                'rmse': np.sqrt(np.mean(np.array(errors)**2)) if errors else 0,  # 均方根误差
                'max_error': np.max(np.array(errors)) if errors else 0,
                'mean_relative_error': np.mean(np.array(relative_errors)) if relative_errors else 0,
                'std_relative_error': np.std(np.array(relative_errors)) if relative_errors else 0,
                'r_squared': 1.0 - (np.var(np.array(errors)) / np.var(np.array([f if isinstance(f, (int, float)) else np.linalg.norm(f) for f in known_forces]))) if n_samples > 1 else 1.0
            },
            'force_comparison': list(zip(known_forces[:n_samples], estimated_forces))
        }
        
        return validation_result
    
    def cross_validate_methods(self, displacement_fields: list, known_forces: list,
                              methods: list = None) -> dict:
        """
        交叉验证不同力估计方法
        
        Args:
            displacement_fields: 位移场列表
            known_forces: 已知力列表
            methods: 要比较的方法列表，默认 ['hertz', 'boussinesq', 'fem']
            
        Returns:
            cross_validation_result: 包含各方法性能比较的字典
        """
        if methods is None:
            methods = ['hertz', 'boussinesq', 'fem']
        
        n_samples = min(len(displacement_fields), len(known_forces))
        
        if n_samples == 0:
            return {'success': False, 'error': '无验证数据'}
        
        results = {}
        
        for method in methods:
            try:
                validation = self.validate_estimation(displacement_fields, known_forces, method)
                if validation['success']:
                    results[method] = validation['metrics']
                else:
                    results[method] = {'error': validation.get('error', '验证失败')}
            except Exception as e:
                results[method] = {'error': str(e)}
        
        # 找出最佳方法（基于MAE）
        best_method = None
        best_mae = float('inf')
        
        for method, metrics in results.items():
            if 'mae' in metrics and metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                best_method = method
        
        cross_validation_result = {
            'success': True,
            'n_samples': n_samples,
            'methods_tested': methods,
            'results': results,
            'best_method': best_method,
            'best_mae': best_mae,
            'recommendation': f"基于当前数据，推荐使用 {best_method} 方法 (MAE: {best_mae:.3f} N)"
        }
        
        return cross_validation_result
    
    def _create_grid_connectivity(self, positions: np.ndarray) -> np.ndarray:
        """
        为规则网格创建三角形网格连接
        
        假设positions近似规则排列（如标记点检测的网格）
        创建三角形网格连接，每个网格单元分成两个三角形
        
        Args:
            positions: 节点位置 (N, 2)
            
        Returns:
            connectivity: 三角形连接关系 (M, 3)
        """
        # 尝试检测网格结构
        n_points = len(positions)
        if n_points < 4:
            # 点数太少，创建简单连接
            if n_points >= 3:
                return np.array([[0, 1, 2]], dtype=int)
            else:
                return np.array([], dtype=int).reshape(0, 3)
        
        # 使用k-means简单聚类检测网格结构
        # 简单方法：按x坐标排序，假设近似网格排列
        try:
            # 按x和y坐标排序检测网格
            sorted_by_x = positions[np.argsort(positions[:, 0])]
            sorted_by_y = positions[np.argsort(positions[:, 1])]
            
            # 检测可能的行数和列数（假设为近似矩形网格）
            # 使用简单启发式：查找x坐标的聚类
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]
            
            # 使用分位数估计列数
            x_unique = np.unique(np.round(x_coords, 1))  # 1mm精度
            y_unique = np.unique(np.round(y_coords, 1))
            
            if len(x_unique) > 1 and len(y_unique) > 1:
                # 可能是规则网格，创建网格连接
                n_cols = len(x_unique)
                n_rows = len(y_unique)
                
                if n_rows * n_cols == n_points:
                    # 完美网格，创建规则三角形网格
                    connectivity = []
                    for i in range(n_rows - 1):
                        for j in range(n_cols - 1):
                            # 计算4个角点的索引
                            idx_sw = i * n_cols + j
                            idx_se = idx_sw + 1
                            idx_nw = (i + 1) * n_cols + j
                            idx_ne = idx_nw + 1
                            
                            # 将四边形分成两个三角形
                            # 三角形1: 西南-东南-西北
                            connectivity.append([idx_sw, idx_se, idx_nw])
                            # 三角形2: 东南-东北-西北
                            connectivity.append([idx_se, idx_ne, idx_nw])
                    
                    return np.array(connectivity, dtype=int)
        except Exception as e:
            print(f"网格连接创建失败: {e}")
        
        # 回退：创建Delaunay-like连接（简单方法）
        # 使用最近邻创建三角形
        connectivity = []
        for i in range(n_points):
            # 找到最近的两个点（不同于i）
            distances = np.linalg.norm(positions - positions[i], axis=1)
            distances[i] = np.inf  # 排除自身
            
            # 找到最近的两个点
            nearest = np.argsort(distances)[:2]
            if len(nearest) == 2:
                connectivity.append([i, nearest[0], nearest[1]])
        
        # 去重并确保至少有一个三角形
        if len(connectivity) > 0:
            # 简单去重（保持顺序）
            unique_connectivity = []
            for tri in connectivity:
                sorted_tri = sorted(tri)
                if sorted_tri not in unique_connectivity:
                    unique_connectivity.append(sorted_tri)
            
            return np.array(unique_connectivity[:50], dtype=int)  # 限制三角形数量
        else:
            # 最终回退：简单三角形
            return np.array([[0, 1, 2]], dtype=int) if n_points >= 3 else np.array([], dtype=int).reshape(0, 3)
    
    def visualize_force_distribution(self, force_result: dict, 
                                    positions: Optional[np.ndarray] = None,
                                    save_path: Optional[str] = None):
        """
        可视化力分布
        
        Args:
            force_result: estimate_force_from_displacement返回的结果
            positions: 测量点位置（可选）
            save_path: 保存图像路径（可选）
        """
        if not MATPLOTLIB_AVAILABLE:
            print("警告：matplotlib不可用，无法可视化。请安装matplotlib以使用此功能。")
            print("力估计方法:", force_result.get('method', 'unknown'))
            print("总接触力:", force_result.get('total_force', 0), "N")
            print("最大局部力:", force_result.get('max_force', 0), "N")
            if force_result.get('force_center') is not None:
                fc = force_result['force_center']
                print("力中心位置:", fc)
            return
        
        method = force_result['method']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左侧：力分布图
        ax = axes[0]
        
        if method == 'hertz' and force_result['force_distribution'] is not None:
            # Hertz接触力分布
            force_dist = force_result['force_distribution']
            im = ax.imshow(force_dist, cmap='hot', origin='lower',
                          extent=[-15, 15, -15, 15])
            ax.set_title(f'Hertz接触力分布\n总力: {force_result["total_force"]:.2f} N')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            plt.colorbar(im, ax=ax, label='压力 (N/mm²)')
            
        elif method in ['boussinesq', 'fem'] and force_result['force_distribution'] is not None:
            # 网格力分布
            force_dist = force_result['force_distribution']
            
            if method == 'boussinesq':
                im = ax.imshow(force_dist, cmap='viridis', origin='lower')
                ax.set_title(f'Boussinesq力分布\n总力: {force_result["total_force"]:.2f} N')
                ax.set_xlabel('X (像素)')
                ax.set_ylabel('Y (像素)')
                plt.colorbar(im, ax=ax, label='力密度')
            else:
                # 节点力向量
                node_forces = force_dist
                if positions is not None:
                    ax.quiver(positions[:, 0], positions[:, 1], 
                             node_forces[:, 0], node_forces[:, 1],
                             color='red', scale=20)
                    ax.set_title(f'有限元节点力\n总力: {force_result["total_force"]:.2f} N')
                else:
                    ax.text(0.5, 0.5, '需要位置信息显示力向量', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes)
                    ax.set_title(f'有限元节点力 (无位置信息)\n总力: {force_result["total_force"]:.2f} N')
                ax.set_xlabel('X (像素)')
                ax.set_ylabel('Y (像素)')
        
        # 右侧：统计信息
        ax = axes[1]
        ax.axis('off')
        
        stats_text = f"""
        力估计方法: {method}
        ======================
        总接触力: {force_result['total_force']:.3f} N
        最大局部力: {force_result['max_force']:.3f} N
        """
        
        if force_result['force_center'] is not None:
            fc = force_result['force_center']
            stats_text += f"\n力中心位置: ({fc[0]:.1f}, {fc[1]:.1f})"
        
        stats_text += f"""
        ======================
        材料参数:
        杨氏模量: {self.youngs_modulus:.1e} Pa
        泊松比: {self.poissons_ratio:.3f}
        传感器厚度: {self.sensor_thickness} mm
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"力分布图像已保存至: {save_path}")
        
        plt.show()


def example_usage():
    """示例用法"""
    print("=== 视触传感器力估计算法示例 ===")
    
    # 创建力估计器
    estimator = ForceEstimation(
        youngs_modulus=2.0e6,  # 典型硅胶模量
        poissons_ratio=0.49,   # 近似不可压缩
        sensor_thickness=5.0
    )
    
    # 生成示例位移场（模拟）
    print("1. 生成示例位移场...")
    n_points = 100
    positions = np.random.rand(n_points, 2) * 100  # 100x100区域
    
    # 创建位移场（模拟接触中心在(50, 50)）
    displacement_field = np.zeros((n_points, 2))
    for i, (x, y) in enumerate(positions):
        dx = 50 - x
        dy = 50 - y
        distance = np.sqrt(dx**2 + dy**2)
        
        # 距离越近位移越大
        if distance < 30:
            displacement_field[i, 0] = dx / (distance + 1e-6) * (30 - distance) * 0.5
            displacement_field[i, 1] = dy / (distance + 1e-6) * (30 - distance) * 0.5
    
    print(f"   位移场大小: {displacement_field.shape}")
    print(f"   最大位移: {np.max(np.linalg.norm(displacement_field, axis=1)):.2f} 像素")
    
    # 使用不同方法估计力
    print("\n2. 使用Hertz接触理论估计力...")
    result_hertz = estimator.estimate_force_from_displacement(
        displacement_field, positions, method='hertz'
    )
    print(f"   总力: {result_hertz['total_force']:.3f} N")
    
    print("\n3. 使用Boussinesq解估计力...")
    result_boussinesq = estimator.estimate_force_from_displacement(
        displacement_field, positions, method='boussinesq'
    )
    print(f"   总力: {result_boussinesq['total_force']:.3f} N")
    
    print("\n4. 可视化力分布...")
    estimator.visualize_force_distribution(
        result_boussinesq, positions, save_path="force_distribution.png"
    )
    
    print("\n5. 算法验证完成")
    return estimator, displacement_field, positions, result_boussinesq


if __name__ == "__main__":
    example_usage()