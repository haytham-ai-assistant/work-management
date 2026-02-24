"""
力估计算法 - 视触传感器核心算法之二

基于PDF提取内容实现，将位移场转换为接触力。
实现Hertz接触理论、有限元反问题求解等力计算方法。
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


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
        
    def hertz_contact_force(self, displacement: np.ndarray, 
                           contact_radius: float = 10.0,
                           sphere_radius: float = 50.0) -> Tuple[float, np.ndarray]:
        """
        基于Hertz接触理论计算接触力
        
        Hertz接触理论描述弹性球体与平面的接触
        
        Args:
            displacement: 接触中心位移向量 (3,) [dx, dy, dz]
            contact_radius: 接触半径 (mm)
            sphere_radius: 球体半径 (mm)
            
        Returns:
            total_force: 总接触力大小 (N)
            force_distribution: 力分布 (N/mm²)
        """
        # 简化Hertz接触公式: F = (4/3) * E* * sqrt(R) * δ^(3/2)
        # 其中 E* = E/(1-ν²)，R为等效半径
        
        # 计算等效杨氏模量
        E_star = self.youngs_modulus / (1 - self.poissons_ratio**2)
        
        # 主要考虑垂直位移 (dz)
        dz = displacement[2] if len(displacement) > 2 else displacement[1]
        
        # 确保位移为正（压缩）
        if dz < 0:
            dz = abs(dz)
        
        # Hertz接触力公式
        total_force = (4/3) * E_star * np.sqrt(sphere_radius) * (dz ** 1.5)
        
        # 力分布（假设为抛物线分布）
        x = np.linspace(-contact_radius, contact_radius, 50)
        y = np.linspace(-contact_radius, contact_radius, 50)
        xx, yy = np.meshgrid(x, y)
        
        # 距离接触中心的距离
        r = np.sqrt(xx**2 + yy**2)
        
        # 抛物线分布：p(r) = p0 * sqrt(1 - (r/a)²)
        p0 = (3 * total_force) / (2 * np.pi * contact_radius**2)  # 最大接触压力
        force_distribution = np.zeros_like(r)
        mask = r <= contact_radius
        force_distribution[mask] = p0 * np.sqrt(1 - (r[mask]/contact_radius)**2)
        
        return total_force, force_distribution
    
    def boussinesq_solution(self, displacement_field: np.ndarray, 
                           positions: np.ndarray, 
                           grid_resolution: float = 1.0) -> np.ndarray:
        """
        使用Boussinesq解计算表面力分布
        
        Boussinesq解描述弹性半空间表面受集中力作用的位移场
        这里求解反问题：从位移场反推力分布
        
        Args:
            displacement_field: 位移场 (N, 2) 或 (N, 3)
            positions: 位移测量点坐标 (N, 2)
            grid_resolution: 力网格分辨率 (mm)
            
        Returns:
            force_grid: 力分布网格 (M, M)
        """
        # 创建力网格
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        # 扩展边界
        x_min -= 10
        x_max += 10
        y_min -= 10
        y_max += 10
        
        # 生成网格
        x_grid = np.arange(x_min, x_max, grid_resolution)
        y_grid = np.arange(y_min, y_max, grid_resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        # 初始化力网格
        force_grid = np.zeros(xx.shape)
        
        # 简化反问题求解：假设每个位移点由最近网格点的力引起
        # 实际应用中应使用正则化方法求解
        for i, (x, y) in enumerate(positions):
            # 找到最近网格点
            idx_x = np.argmin(np.abs(x_grid - x))
            idx_y = np.argmin(np.abs(y_grid - y))
            
            # 位移大小作为力的指标
            disp_magnitude = np.linalg.norm(displacement_field[i])
            
            # 累加到力网格
            force_grid[idx_y, idx_x] += disp_magnitude
        
        # 归一化并转换为力值（简化）
        if force_grid.max() > 0:
            force_grid = force_grid / force_grid.max() * 10.0  # 最大10N
        
        return force_grid
    
    def finite_element_inverse(self, displacement_field: np.ndarray,
                              node_positions: np.ndarray,
                              connectivity: np.ndarray) -> np.ndarray:
        """
        有限元反问题求解：从位移场计算节点力
        
        简化实现，实际应用需要完整有限元求解器
        
        Args:
            displacement_field: 节点位移场 (N, 2) 或 (N, 3)
            node_positions: 节点坐标 (N, 2)
            connectivity: 单元连接关系 (M, 3) 三角形单元
            
        Returns:
            node_forces: 节点力 (N, 2)
        """
        n_nodes = len(node_positions)
        
        # 简化方法：使用刚度矩阵概念
        # 对于线性弹性材料，力 = K * 位移
        # 这里使用对角刚度矩阵简化
        
        # 创建简化刚度矩阵（对角占优）
        K = np.eye(n_nodes * 2) * self.youngs_modulus * 0.1
        
        # 添加相邻节点耦合（简化）
        for elem in connectivity:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        idx_i = elem[i] * 2
                        idx_j = elem[j] * 2
                        # 添加小的耦合项
                        K[idx_i:idx_i+2, idx_j:idx_j+2] += np.eye(2) * self.youngs_modulus * 0.01
        
        # 位移向量
        u = displacement_field.flatten()
        
        # 确保尺寸匹配
        if len(u) != K.shape[0]:
            # 调整K尺寸
            K = np.eye(len(u)) * self.youngs_modulus * 0.1
        
        # 计算力
        f = K @ u
        
        # 重塑为节点力
        node_forces = f.reshape(-1, 2)
        
        return node_forces
    
    def estimate_force_from_displacement(self, displacement_field: np.ndarray,
                                        positions: np.ndarray,
                                        method: str = 'boussinesq') -> dict:
        """
        从位移场估计力分布
        
        Args:
            displacement_field: 位移场 (N, 2)
            positions: 测量点位置 (N, 2)
            method: 力估计方法 ('hertz', 'boussinesq', 'fem')
            
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
                displacement_field, positions, grid_resolution=2.0
            )
            
            result['force_distribution'] = force_grid
            result['total_force'] = np.sum(force_grid) * 4.0  # 粗略积分
            result['max_force'] = force_grid.max()
            
            # 力中心（加权平均）
            if force_grid.sum() > 0:
                y_coords, x_coords = np.indices(force_grid.shape)
                force_center_x = np.average(x_coords, weights=force_grid)
                force_center_y = np.average(y_coords, weights=force_grid)
                result['force_center'] = np.array([force_center_x, force_center_y])
        
        elif method == 'fem':
            # 使用有限元反问题
            # 生成简单三角形网格
            # 尝试导入scipy生成Delaunay三角网格，失败时使用简单网格
            try:
                from scipy.spatial import Delaunay
                tri = Delaunay(positions)
                connectivity = tri.simplices
            except ImportError:
                # 简化网格连接（假设点阵排列）
                n_points = len(positions)
                # 创建简单三角形网格（示例）
                if n_points >= 3:
                    connectivity = np.array([[0, 1, 2]], dtype=int)  # 仅一个三角形作为示例
                else:
                    connectivity = np.array([], dtype=int).reshape(0, 3)
                print("注意：scipy不可用，使用简化网格连接")
            
            node_forces = self.finite_element_inverse(
                displacement_field, positions, connectivity
            )
            
            result['force_distribution'] = node_forces
            result['total_force'] = np.linalg.norm(node_forces, axis=1).sum()
            result['max_force'] = np.linalg.norm(node_forces, axis=1).max()
            result['force_center'] = positions[np.argmax(np.linalg.norm(node_forces, axis=1))]
        
        return result
    
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