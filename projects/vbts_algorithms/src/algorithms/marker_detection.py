"""
标记点检测算法 - 视触传感器核心算法之一

基于PDF提取内容实现，模拟视触传感器中的标记点检测过程。
由于实际图像处理需要OpenCV，本模块提供合成数据生成和基础算法框架。
"""

import numpy as np
from typing import List, Tuple, Optional

# 尝试导入matplotlib，如果不可用则提供占位
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


class MarkerDetection:
    """
    标记点检测类，用于检测和追踪视触传感器表面的标记点
    """
    
    def __init__(self, marker_radius: float = 5.0, grid_spacing: float = 20.0):
        """
        初始化标记点检测器
        
        Args:
            marker_radius: 标记点半径（像素）
            grid_spacing: 标记点网格间距（像素）
        """
        self.marker_radius = marker_radius
        self.grid_spacing = grid_spacing
        self.detected_markers = []
        
    def generate_synthetic_markers(self, image_shape: Tuple[int, int] = (480, 640), 
                                   grid_offset: Tuple[float, float] = (10, 10)) -> np.ndarray:
        """
        生成合成标记点网格，模拟视触传感器表面的标记阵列
        
        Args:
            image_shape: 图像尺寸 (height, width)
            grid_offset: 网格偏移量 (x_offset, y_offset)
            
        Returns:
            markers: (N, 2)数组，标记点中心坐标 [x, y]
        """
        height, width = image_shape
        x_offset, y_offset = grid_offset
        
        # 生成网格点
        x_coords = np.arange(x_offset, width - x_offset, self.grid_spacing)
        y_coords = np.arange(y_offset, height - y_offset, self.grid_spacing)
        
        # 创建网格
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # 展平为坐标列表
        markers = np.column_stack([xx.ravel(), yy.ravel()])
        
        # 添加微小随机偏移，模拟制造误差
        markers += np.random.normal(0, 0.5, markers.shape)
        
        self.detected_markers = markers
        return markers
    
    def simulate_deformation(self, markers: np.ndarray, 
                            force_center: Tuple[float, float] = (320, 240),
                            force_magnitude: float = 10.0,
                            deformation_radius: float = 100.0) -> np.ndarray:
        """
        模拟外力作用下的标记点位移
        
        基于Hertz接触理论简化模型，距离力中心越近的点位移越大
        
        Args:
            markers: 原始标记点坐标 (N, 2)
            force_center: 力作用中心 (x, y)
            force_magnitude: 力大小（影响位移幅度）
            deformation_radius: 变形影响半径
            
        Returns:
            deformed_markers: 变形后的标记点坐标 (N, 2)
        """
        deformed_markers = markers.copy()
        fx, fy = force_center
        
        for i, (x, y) in enumerate(markers):
            # 计算到力中心的距离
            dx = x - fx
            dy = y - fy
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < deformation_radius:
                # 基于距离的位移衰减函数（高斯函数）
                displacement_factor = np.exp(-(distance**2) / (2 * (deformation_radius/3)**2))
                
                # 位移方向：从力中心向外（假设为垂直向下压）
                # 实际传感器中，位移方向与表面法向相关
                displacement_x = dx / (distance + 1e-6) * displacement_factor * force_magnitude * 0.1
                displacement_y = dy / (distance + 1e-6) * displacement_factor * force_magnitude * 0.1
                
                # 添加垂直位移（主要变形方向）
                displacement_z = displacement_factor * force_magnitude
                
                # 更新坐标（这里只考虑二维位移，实际为三维）
                deformed_markers[i, 0] = x + displacement_x
                deformed_markers[i, 1] = y + displacement_y
        
        return deformed_markers
    
    def calculate_displacement_field(self, original_markers: np.ndarray, 
                                    deformed_markers: np.ndarray) -> np.ndarray:
        """
        计算位移场
        
        Args:
            original_markers: 原始标记点坐标 (N, 2)
            deformed_markers: 变形后标记点坐标 (N, 2)
            
        Returns:
            displacement_field: 位移向量场 (N, 2) [dx, dy]
        """
        if len(original_markers) != len(deformed_markers):
            raise ValueError("标记点数量不一致")
        
        displacement_field = deformed_markers - original_markers
        return displacement_field
    
    def generate_synthetic_image(self, markers: Optional[np.ndarray] = None,
                                image_shape: Tuple[int, int] = (480, 640),
                                marker_intensity: float = 255.0,
                                background_intensity: float = 50.0,
                                marker_std: float = 2.0) -> np.ndarray:
        """
        生成合成图像，包含标记点

        在没有真实传感器图像时，用于算法测试和验证

        Args:
            markers: 标记点坐标 (N, 2)，如果为None则自动生成
            image_shape: 图像尺寸 (height, width)
            marker_intensity: 标记点亮度 (0-255)
            background_intensity: 背景亮度 (0-255)
            marker_std: 标记点高斯模糊标准差

        Returns:
            image: 合成图像 (height, width)
        """
        if markers is None:
            markers = self.generate_synthetic_markers(image_shape)

        height, width = image_shape
        image = np.ones((height, width)) * background_intensity

        # 创建网格用于高斯分布
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        for x, y in markers:
            # 计算每个像素到标记点的距离
            distance_sq = (x_coords - x)**2 + (y_coords - y)**2
            
            # 添加高斯斑点
            intensity = marker_intensity * np.exp(-distance_sq / (2 * marker_std**2))
            image += intensity

        # 裁剪到有效范围
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def detect_markers_from_image(self, image: np.ndarray,
                                 intensity_threshold: float = 200.0,
                                 min_distance: float = 10.0) -> np.ndarray:
        """
        从图像中检测标记点

        在没有OpenCV时使用简化方法：阈值化 + 局部极大值检测

        Args:
            image: 输入图像 (2D numpy array)
            intensity_threshold: 强度阈值 (0-255)
            min_distance: 标记点之间的最小距离 (像素)

        Returns:
            markers: 检测到的标记点坐标 (N, 2)
        """
        # 检查图像维度
        if len(image.shape) != 2:
            if len(image.shape) == 3:
                # 转换为灰度图（简单平均）
                image = np.mean(image, axis=2)
            else:
                raise ValueError(f"不支持的图像维度: {image.shape}")

        # 阈值化
        binary = image > intensity_threshold

        # 如果没有足够像素超过阈值，返回空数组
        if np.sum(binary) < 10:
            print(f"警告：只有 {np.sum(binary)} 个像素超过阈值 {intensity_threshold}，可能无标记点")
            return np.array([])

        # 找到连通区域（简化方法：使用局部极大值）
        height, width = image.shape
        markers = []

        # 创建距离掩码以避免重复检测
        visited = np.zeros((height, width), dtype=bool)

        # 按强度排序的像素位置
        bright_pixels = np.column_stack(np.where(image > intensity_threshold))
        intensities = image[image > intensity_threshold]

        if len(bright_pixels) == 0:
            return np.array([])

        # 按强度降序排序
        sorted_indices = np.argsort(-intensities)
        bright_pixels = bright_pixels[sorted_indices]

        for y, x in bright_pixels:
            # 如果已访问过附近区域，跳过
            if visited[y, x]:
                continue

            # 标记为已访问
            y_min = max(0, int(y - min_distance/2))
            y_max = min(height, int(y + min_distance/2))
            x_min = max(0, int(x - min_distance/2))
            x_max = min(width, int(x + min_distance/2))
            visited[y_min:y_max, x_min:x_max] = True

            # 添加标记点
            markers.append([x, y])

        if len(markers) == 0:
            return np.array([])

        markers_array = np.array(markers)

        # 后处理：移除离群点（如果标记点应大致呈网格分布）
        if len(markers_array) > 10:
            try:
                # 计算标记点之间的平均距离
                from scipy.spatial import distance_matrix
                distances = distance_matrix(markers_array, markers_array)
                np.fill_diagonal(distances, np.inf)
                min_distances = np.min(distances, axis=1)
                
                # 移除距离异常的点（太近或太远）
                median_dist = np.median(min_distances)
                valid_mask = (min_distances > median_dist * 0.3) & (min_distances < median_dist * 3)
                markers_array = markers_array[valid_mask]
            except ImportError:
                # scipy不可用，跳过后处理
                pass

        self.detected_markers = markers_array
        return markers_array
    
    def visualize_markers(self, original_markers: np.ndarray, 
                         deformed_markers: Optional[np.ndarray] = None,
                         displacement_field: Optional[np.ndarray] = None,
                         save_path: Optional[str] = None):
        """
        可视化标记点和位移场
        
        Args:
            original_markers: 原始标记点
            deformed_markers: 变形后标记点（可选）
            displacement_field: 位移场（可选）
            save_path: 保存图像路径（可选）
        """
        if not MATPLOTLIB_AVAILABLE:
            print("警告：matplotlib不可用，无法可视化。请安装matplotlib以使用此功能。")
            print("原始标记点数量:", len(original_markers))
            if deformed_markers is not None:
                print("变形后标记点数量:", len(deformed_markers))
            if displacement_field is not None:
                avg_displacement = np.mean(np.linalg.norm(displacement_field, axis=1))
                print("平均位移:", avg_displacement, "像素")
            return
        
        fig, axes = plt.subplots(1, 2 if deformed_markers is not None else 1, 
                                figsize=(12, 5))
        
        if deformed_markers is None:
            axes = [axes]
        
        # 绘制原始标记点
        ax = axes[0]
        ax.scatter(original_markers[:, 0], original_markers[:, 1], 
                  c='blue', s=20, label='原始标记点')
        ax.set_title('原始标记点分布')
        ax.set_xlabel('X (像素)')
        ax.set_ylabel('Y (像素)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')
        
        # 如果提供变形后标记点，绘制对比
        if deformed_markers is not None:
            ax = axes[1]
            ax.scatter(original_markers[:, 0], original_markers[:, 1], 
                      c='blue', s=20, alpha=0.5, label='原始')
            ax.scatter(deformed_markers[:, 0], deformed_markers[:, 1], 
                      c='red', s=20, alpha=0.7, label='变形后')
            
            # 绘制位移向量
            if displacement_field is not None:
                for i in range(len(original_markers)):
                    ax.arrow(original_markers[i, 0], original_markers[i, 1],
                            displacement_field[i, 0], displacement_field[i, 1],
                            head_width=2, head_length=3, fc='green', ec='green', alpha=0.5)
            
            ax.set_title('标记点变形与位移场')
            ax.set_xlabel('X (像素)')
            ax.set_ylabel('Y (像素)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存至: {save_path}")
        
        plt.show()


def example_usage():
    """示例用法"""
    print("=== 视触传感器标记点检测算法示例 ===")
    
    # 创建检测器
    detector = MarkerDetection(marker_radius=6.0, grid_spacing=25.0)
    
    # 生成合成标记点
    print("1. 生成合成标记点网格...")
    original_markers = detector.generate_synthetic_markers(image_shape=(480, 640), 
                                                          grid_offset=(20, 20))
    print(f"   生成 {len(original_markers)} 个标记点")
    
    # 模拟变形
    print("2. 模拟外力作用下的变形...")
    deformed_markers = detector.simulate_deformation(
        original_markers, 
        force_center=(320, 240),  # 图像中心偏右
        force_magnitude=15.0,
        deformation_radius=120.0
    )
    
    # 计算位移场
    print("3. 计算位移场...")
    displacement_field = detector.calculate_displacement_field(original_markers, deformed_markers)
    
    # 计算平均位移
    avg_displacement = np.mean(np.sqrt(np.sum(displacement_field**2, axis=1)))
    print(f"   平均位移: {avg_displacement:.2f} 像素")
    
    # 可视化
    print("4. 可视化结果...")
    detector.visualize_markers(
        original_markers, 
        deformed_markers, 
        displacement_field,
        save_path="marker_deformation.png"
    )
    
    print("5. 算法验证完成")
    return detector, original_markers, deformed_markers, displacement_field


if __name__ == "__main__":
    example_usage()