#!/usr/bin/env python3
"""
视触传感器完整算法流程演示

从标记点检测到力估计的完整流程展示，使用合成数据模拟视触传感器工作原理。
"""

import sys
import os
import numpy as np

# 添加src目录到Python路径
src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
if os.path.exists(src_dir):
    sys.path.insert(0, src_dir)

# 导入自定义算法模块
try:
    from algorithms.marker_detection import MarkerDetection
    from algorithms.force_estimation import ForceEstimation
    import_success = True
except ImportError as e:
    print(f"导入算法模块失败: {e}")
    print("尝试从当前目录导入...")
    # 尝试直接导入（如果脚本在项目根目录运行）
    try:
        import algorithms.marker_detection as marker_detection
        import algorithms.force_estimation as force_estimation
        MarkerDetection = marker_detection.MarkerDetection
        ForceEstimation = force_estimation.ForceEstimation
        import_success = True
    except ImportError:
        print("无法导入算法模块，请确保在正确目录运行。")
        import_success = False

# 检查matplotlib可用性
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    print("警告: matplotlib不可用，可视化功能将受限")
    matplotlib_available = False


class VBTSPipelineDemo:
    """
    视触传感器完整算法流程演示类
    """
    
    def __init__(self):
        """初始化演示管道"""
        if not import_success:
            raise ImportError("无法导入必要的算法模块")
        
        print("=" * 60)
        print("视触传感器算法流程演示")
        print("=" * 60)
        
        # 初始化算法模块
        self.marker_detector = MarkerDetection(
            marker_radius=6.0,
            grid_spacing=25.0
        )
        
        self.force_estimator = ForceEstimation(
            youngs_modulus=2.0e6,  # 硅胶典型模量 2MPa
            poissons_ratio=0.49,
            sensor_thickness=5.0
        )
        
        # 存储中间结果
        self.original_markers = None
        self.deformed_markers = None
        self.displacement_field = None
        self.force_results = {}
        
    def generate_synthetic_data(self):
        """步骤1：生成合成数据"""
        print("\n1. 生成合成数据")
        print("-" * 40)
        
        # 生成标记点网格
        self.original_markers = self.marker_detector.generate_synthetic_markers(
            image_shape=(480, 640),
            grid_offset=(30, 30)
        )
        
        print(f"   ✓ 生成 {len(self.original_markers)} 个标记点")
        print(f"   ✓ 图像尺寸: 640x480 像素")
        print(f"   ✓ 网格间距: {self.marker_detector.grid_spacing} 像素")
        
        return self.original_markers
    
    def simulate_contact_deformation(self):
        """步骤2：模拟接触变形"""
        print("\n2. 模拟接触变形")
        print("-" * 40)
        
        if self.original_markers is None:
            raise ValueError("请先生成合成数据")
        
        # 定义接触参数
        force_center = (320, 240)  # 图像中心
        force_magnitude = 20.0     # 力大小
        deformation_radius = 150.0 # 变形影响半径
        
        # 模拟变形
        self.deformed_markers = self.marker_detector.simulate_deformation(
            self.original_markers,
            force_center=force_center,
            force_magnitude=force_magnitude,
            deformation_radius=deformation_radius
        )
        
        # 计算位移场
        self.displacement_field = self.marker_detector.calculate_displacement_field(
            self.original_markers,
            self.deformed_markers
        )
        
        # 计算统计信息
        disp_magnitudes = np.linalg.norm(self.displacement_field, axis=1)
        max_disp = np.max(disp_magnitudes)
        avg_disp = np.mean(disp_magnitudes)
        
        print(f"   ✓ 力作用中心: ({force_center[0]}, {force_center[1]})")
        print(f"   ✓ 力大小: {force_magnitude} N")
        print(f"   ✓ 最大位移: {max_disp:.2f} 像素")
        print(f"   ✓ 平均位移: {avg_disp:.2f} 像素")
        
        return self.deformed_markers, self.displacement_field
    
    def estimate_contact_force(self):
        """步骤3：估计接触力"""
        print("\n3. 估计接触力")
        print("-" * 40)
        
        if self.displacement_field is None or self.original_markers is None:
            raise ValueError("请先完成变形模拟")
        
        # 使用不同方法估计力
        methods = ['hertz', 'boussinesq']
        
        for method in methods:
            print(f"\n   使用 {method} 方法:")
            
            result = self.force_estimator.estimate_force_from_displacement(
                self.displacement_field,
                self.original_markers[:, :2],  # 仅使用x,y坐标
                method=method
            )
            
            self.force_results[method] = result
            
            print(f"     ✓ 总接触力: {result['total_force']:.3f} N")
            print(f"     ✓ 最大局部力: {result['max_force']:.3f} N")
            
            if result['force_center'] is not None:
                fc = result['force_center']
                print(f"     ✓ 力中心位置: ({fc[0]:.1f}, {fc[1]:.1f})")
        
        return self.force_results
    
    def visualize_results(self, save_dir="output"):
        """步骤4：可视化结果"""
        print("\n4. 可视化结果")
        print("-" * 40)
        
        # 检查数据是否可用
        if self.original_markers is None or self.deformed_markers is None or self.displacement_field is None:
            print("   ⚠  数据不可用，请先运行前序步骤")
            return
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        if not matplotlib_available:
            print("   ⚠  matplotlib不可用，跳过可视化")
            print("   ✓ 结果数据已计算完成，可以查看self.force_results")
            # 保存数据
            data_path = os.path.join(save_dir, "pipeline_data.npz")
            np.savez(
                data_path,
                original_markers=self.original_markers,
                deformed_markers=self.deformed_markers,
                displacement_field=self.displacement_field
            )
            print(f"   ✓ 数据已保存至: {data_path}")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # 创建综合可视化图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('视触传感器算法流程演示', fontsize=16, fontweight='bold')
            
            # 1. 标记点变形图
            ax = axes[0, 0]
            ax.scatter(self.original_markers[:, 0], self.original_markers[:, 1], 
                      c='blue', s=10, alpha=0.6, label='原始')
            ax.scatter(self.deformed_markers[:, 0], self.deformed_markers[:, 1], 
                      c='red', s=10, alpha=0.6, label='变形后')
            
            # 绘制位移向量（每10个点绘制一个）
            for i in range(0, len(self.original_markers), 10):
                ax.arrow(self.original_markers[i, 0], self.original_markers[i, 1],
                        self.displacement_field[i, 0], self.displacement_field[i, 1],
                        head_width=3, head_length=4, fc='green', ec='green', alpha=0.5)
            
            ax.set_title('标记点变形与位移场')
            ax.set_xlabel('X (像素)')
            ax.set_ylabel('Y (像素)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            # 2. 位移场幅值分布
            ax = axes[0, 1]
            disp_magnitudes = np.linalg.norm(self.displacement_field, axis=1)
            scatter = ax.scatter(self.original_markers[:, 0], self.original_markers[:, 1],
                               c=disp_magnitudes, s=20, cmap='hot', alpha=0.7)
            ax.set_title('位移场幅值分布')
            ax.set_xlabel('X (像素)')
            ax.set_ylabel('Y (像素)')
            plt.colorbar(scatter, ax=ax, label='位移大小 (像素)')
            ax.grid(True, alpha=0.3)
            
            # 3. Hertz接触力分布
            ax = axes[1, 0]
            if 'hertz' in self.force_results and self.force_results['hertz']['force_distribution'] is not None:
                force_dist = self.force_results['hertz']['force_distribution']
                im = ax.imshow(force_dist, cmap='hot', origin='lower',
                              extent=[-15, 15, -15, 15])
                ax.set_title(f'Hertz接触力分布\n总力: {self.force_results["hertz"]["total_force"]:.2f} N')
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                plt.colorbar(im, ax=ax, label='压力 (N/mm²)')
            else:
                ax.text(0.5, 0.5, 'Hertz力分布不可用', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
                ax.set_title('Hertz接触力分布')
            
            # 4. Boussinesq力分布
            ax = axes[1, 1]
            if 'boussinesq' in self.force_results and self.force_results['boussinesq']['force_distribution'] is not None:
                force_dist = self.force_results['boussinesq']['force_distribution']
                im = ax.imshow(force_dist, cmap='viridis', origin='lower')
                ax.set_title(f'Boussinesq力分布\n总力: {self.force_results["boussinesq"]["total_force"]:.2f} N')
                ax.set_xlabel('X (像素)')
                ax.set_ylabel('Y (像素)')
                plt.colorbar(im, ax=ax, label='力密度')
            else:
                ax.text(0.5, 0.5, 'Boussinesq力分布不可用', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
                ax.set_title('Boussinesq力分布')
            
            plt.tight_layout()
            
            # 保存图像
            output_path = os.path.join(save_dir, "vbts_pipeline_demo.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ 可视化结果已保存至: {output_path}")
            
            plt.show()
            
        except ImportError:
            print("   ⚠  matplotlib不可用，跳过可视化")
            print("   ✓ 结果数据已计算完成，可以查看self.force_results")
        
        # 保存数据
        data_path = os.path.join(save_dir, "pipeline_data.npz")
        np.savez(
            data_path,
            original_markers=self.original_markers,
            deformed_markers=self.deformed_markers,
            displacement_field=self.displacement_field
        )
        print(f"   ✓ 数据已保存至: {data_path}")
    
    def run_full_pipeline(self):
        """运行完整流程"""
        print("\n" + "=" * 60)
        print("开始运行完整视触传感器算法流程")
        print("=" * 60)
        
        try:
            # 步骤1: 生成合成数据
            self.generate_synthetic_data()
            
            # 步骤2: 模拟接触变形
            self.simulate_contact_deformation()
            
            # 步骤3: 估计接触力
            self.estimate_contact_force()
            
            # 步骤4: 可视化结果
            self.visualize_results()
            
            print("\n" + "=" * 60)
            print("算法流程演示完成!")
            print("=" * 60)
            
            # 打印总结
            print("\n总结:")
            print("-" * 40)
            if self.original_markers is not None:
                print(f"标记点数量: {len(self.original_markers)}")
            if self.displacement_field is not None:
                disp_magnitudes = np.linalg.norm(self.displacement_field, axis=1)
                print(f"最大位移: {np.max(disp_magnitudes):.2f} 像素")
            
            if self.force_results:
                for method, result in self.force_results.items():
                    print(f"{method}方法估计总力: {result['total_force']:.3f} N")
            else:
                print("力估计结果不可用")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 流程执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    try:
        demo = VBTSPipelineDemo()
        success = demo.run_full_pipeline()
        
        if success:
            print("\n✅ 演示成功完成!")
            print("\n生成的文件:")
            print("  - examples/output/vbts_pipeline_demo.png (可视化结果)")
            print("  - examples/output/pipeline_data.npz (原始数据)")
            print("\n下一步建议:")
            print("  1. 使用真实图像测试标记点检测算法")
            print("  2. 校准传感器材料参数")
            print("  3. 集成到实际硬件系统")
        else:
            print("\n❌ 演示失败，请检查错误信息")
            
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())