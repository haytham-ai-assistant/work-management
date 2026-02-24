#!/usr/bin/env python3
"""
力估计算法验证脚本

根据实验设计进行力估计算法性能评估。
使用合成位移场测试不同条件下力估计的准确性。
"""

import os
import sys
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path

# 添加vbts_algorithms到路径
project_root = Path(__file__).parent.parent.parent.parent
vbts_algorithms_path = project_root / "vbts_algorithms" / "src"
if vbts_algorithms_path.exists():
    sys.path.insert(0, str(vbts_algorithms_path))

try:
    from algorithms.force_estimation import ForceEstimation
    from algorithms.marker_detection import MarkerDetection
    ALGORITHMS_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入算法模块: {e}")
    ALGORITHMS_AVAILABLE = False


class ForceEstimationEvaluator:
    """力估计算法评估器"""
    
    def __init__(self, output_dir="results", pixel_to_mm=0.1):
        """
        初始化评估器
        
        Args:
            output_dir: 输出目录
            pixel_to_mm: 像素到毫米转换系数 (mm/pixel)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建结果目录
        self.results_dir = self.output_dir / "force_estimation_validation"
        self.results_dir.mkdir(exist_ok=True)
        
        # 像素到毫米转换系数
        self.pixel_to_mm = pixel_to_mm
        
        # 初始化力估计器
        self.force_estimator = ForceEstimation(
            youngs_modulus=2.0e6,  # 硅胶典型模量
            poissons_ratio=0.49,
            sensor_thickness=5.0
        )
        
        # 初始化标记点检测器（用于生成合成数据）
        self.marker_detector = MarkerDetection(
            marker_radius=6.0,
            grid_spacing=25.0
        )
        
        # 存储结果
        self.results = []
        
    def generate_test_cases(self):
        """生成测试用例"""
        test_cases = []
        
        # 基础参数
        base_params = {
            "image_shape": (480, 640),
            "grid_offset": (30, 30)
        }
        
        # 不同力大小
        force_magnitudes = [5.0, 10.0, 20.0, 40.0]
        
        # 不同材料参数（杨氏模量）
        youngs_moduli = [1.0e6, 2.0e6, 5.0e6, 10.0e6]
        
        # 不同噪声水平（位移场噪声）
        noise_levels = [0.0, 0.5, 1.0, 2.0]
        
        # 生成测试用例
        case_id = 0
        for force in force_magnitudes:
            for youngs in youngs_moduli:
                for noise in noise_levels:
                    case_id += 1
                    test_cases.append({
                        "case_id": f"case_{case_id:03d}",
                        "force_magnitude": force,
                        "youngs_modulus": youngs,
                        "noise_level": noise,
                        "parameters": {
                            **base_params,
                            "force": force,
                            "youngs": youngs,
                            "noise": noise
                        }
                    })
        
        return test_cases
    
    def generate_synthetic_displacement_field(self, force_magnitude, youngs_modulus, 
                                            noise_level=0.0):
        """
        生成合成位移场
        
        基于Boussinesq解生成真实的位移场，用于测试力估计算法。
        假设点力作用于图像中心，计算弹性半空间表面位移。
        
        Args:
            force_magnitude: 力大小 (N)
            youngs_modulus: 杨氏模量 (Pa)
            noise_level: 噪声水平 (像素)
            
        Returns:
            displacement_field: 位移场 (N, 2) (单位: mm)
            positions: 位置坐标 (N, 2) (单位: mm)
            ground_truth_force: 真实力值 (N)
        """
        # 生成标记点网格 (像素坐标)
        markers_pixels = self.marker_detector.generate_synthetic_markers(
            image_shape=(480, 640),
            grid_offset=(30, 30)
        )
        
        # 转换为毫米坐标
        markers = markers_pixels * self.pixel_to_mm
        
        # 力作用中心（图像中心，转换为毫米）
        force_center_pixels = np.array([320, 240])
        force_center = force_center_pixels * self.pixel_to_mm
        
        # Boussinesq解: 弹性半空间表面点力引起的垂直位移
        # 对于点力P作用于原点，表面点(x,y)的垂直位移为:
        # u_z = P(1-ν²)/(πE) * 1/r，其中r=√(x²+y²)
        # 单位: P (N), E (Pa), r (m) → u_z (m)
        # 注意: 这里使用简化公式，仅考虑垂直位移
        
        poissons_ratio = self.force_estimator.poissons_ratio
        
        # 弹性系数
        elastic_coeff = (1 - poissons_ratio**2) / (np.pi * youngs_modulus)
        
        displacement_field = np.zeros_like(markers)
        
        for i, (x, y) in enumerate(markers):
            # 计算到力中心的距离 (mm)
            dx = x - force_center[0]
            dy = y - force_center[1]
            distance_mm = np.sqrt(dx**2 + dy**2)
            
            # 避免除零
            if distance_mm < 0.1:  # 最小距离0.1mm
                distance_mm = 0.1
            
            # 转换为米
            distance_m = distance_mm / 1000.0
            
            # 垂直位移 (m)
            u_z = force_magnitude * elastic_coeff / distance_m
            
            # 转换为毫米
            u_z_mm = u_z * 1000.0
            
            # 假设位移方向为径向向外（压缩导致表面点向外移动）
            # 实际位移方向应垂直向下，但这里简化为径向以便生成2D位移场
            if distance_mm > 0:
                displacement_field[i, 0] = (dx / distance_mm) * u_z_mm * 0.3  # 比例因子
                displacement_field[i, 1] = (dy / distance_mm) * u_z_mm * 0.3
        
        # 添加噪声 (单位: mm)
        if noise_level > 0:
            # 噪声水平转换为毫米
            noise_mm = noise_level * self.pixel_to_mm
            noise = np.random.normal(0, noise_mm, displacement_field.shape)
            displacement_field += noise
        
        return displacement_field, markers, force_magnitude
    
    def evaluate_force_estimation(self, displacement_field, positions, 
                                ground_truth_force, methods=None):
        """
        评估力估计准确性
        
        Args:
            displacement_field: 位移场
            positions: 位置坐标
            ground_truth_force: 真实力值
            methods: 要评估的方法列表
            
        Returns:
            evaluation_results: 各方法的评估结果字典
        """
        if methods is None:
            methods = ['hertz', 'boussinesq', 'fem']
        
        evaluation_results = {}
        
        for method in methods:
            try:
                # 估计力
                start_time = time.time()
                result = self.force_estimator.estimate_force_from_displacement(
                    displacement_field, positions, method=method
                )
                estimation_time = time.time() - start_time
                
                estimated_force = result['total_force']
                
                # 计算误差
                absolute_error = abs(estimated_force - ground_truth_force)
                relative_error = absolute_error / ground_truth_force if ground_truth_force > 0 else float('inf')
                
                evaluation_results[method] = {
                    'estimated_force': float(estimated_force),
                    'ground_truth_force': float(ground_truth_force),
                    'absolute_error': float(absolute_error),
                    'relative_error': float(relative_error),
                    'estimation_time': estimation_time,
                    'success': True
                }
                
            except Exception as e:
                evaluation_results[method] = {
                    'success': False,
                    'error': str(e)
                }
        
        return evaluation_results
    
    def run_validation(self):
        """运行验证实验"""
        if not ALGORITHMS_AVAILABLE:
            print("错误: 算法模块不可用，无法运行验证")
            return
        
        print("=" * 60)
        print("力估计算法验证实验")
        print("=" * 60)
        
        # 生成测试用例
        test_cases = self.generate_test_cases()
        print(f"生成 {len(test_cases)} 个测试用例")
        
        # 运行每个测试用例
        for i, test_case in enumerate(test_cases):
            print(f"\n测试用例 {i+1}/{len(test_cases)}: {test_case['case_id']}")
            print(f"  力大小: {test_case['force_magnitude']} N, "
                  f"杨氏模量: {test_case['youngs_modulus']:.1e} Pa, "
                  f"噪声水平: {test_case['noise_level']}")
            
            # 生成合成位移场
            displacement_field, positions, ground_truth_force = \
                self.generate_synthetic_displacement_field(
                    test_case['force_magnitude'],
                    test_case['youngs_modulus'],
                    test_case['noise_level']
                )
            
            # 评估不同力估计方法
            evaluation_results = self.evaluate_force_estimation(
                displacement_field, positions, ground_truth_force
            )
            
            # 记录结果
            result = {
                **test_case,
                "ground_truth_force": float(ground_truth_force),
                "displacement_field_shape": list(displacement_field.shape),
                "evaluation_results": evaluation_results,
                "timestamp": datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            # 打印简要结果
            for method, method_result in evaluation_results.items():
                if method_result.get('success', False):
                    print(f"  {method}: 估计力={method_result['estimated_force']:.2f} N, "
                          f"误差={method_result['relative_error']*100:.1f}%")
                else:
                    print(f"  {method}: 失败 - {method_result.get('error', '未知错误')}")
        
        # 保存结果
        self.save_results()
        
        # 生成报告
        self.generate_report()
        
        print(f"\n验证完成! 结果保存在: {self.results_dir}")
    
    def save_results(self):
        """保存结果到文件"""
        # 保存为JSON
        json_path = self.results_dir / "validation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 保存为CSV（简化版本）
        csv_path = self.results_dir / "validation_results.csv"
        if self.results:
            import csv
            
            # 提取主要指标
            rows = []
            for result in self.results:
                for method, method_result in result['evaluation_results'].items():
                    if method_result.get('success', False):
                        rows.append({
                            'case_id': result['case_id'],
                            'force_magnitude': result['force_magnitude'],
                            'youngs_modulus': result['youngs_modulus'],
                            'noise_level': result['noise_level'],
                            'method': method,
                            'ground_truth_force': method_result['ground_truth_force'],
                            'estimated_force': method_result['estimated_force'],
                            'absolute_error': method_result['absolute_error'],
                            'relative_error': method_result['relative_error'],
                            'estimation_time': method_result['estimation_time']
                        })
            
            if rows:
                fieldnames = rows[0].keys()
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
    
    def generate_report(self):
        """生成实验报告"""
        if not self.results:
            print("无结果可生成报告")
            return
        
        report_path = self.results_dir / "validation_report.md"
        
        # 计算总体统计
        all_errors = []
        all_relative_errors = []
        method_stats = {}
        
        for result in self.results:
            for method, method_result in result['evaluation_results'].items():
                if method_result.get('success', False):
                    if method not in method_stats:
                        method_stats[method] = {
                            'errors': [],
                            'relative_errors': [],
                            'estimation_times': []
                        }
                    
                    method_stats[method]['errors'].append(method_result['absolute_error'])
                    method_stats[method]['relative_errors'].append(method_result['relative_error'])
                    method_stats[method]['estimation_times'].append(method_result['estimation_time'])
                    
                    all_errors.append(method_result['absolute_error'])
                    all_relative_errors.append(method_result['relative_error'])
        
        report = f"""# 力估计算法验证报告

## 实验概况
- **实验日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **测试用例数量**: {len(self.results)}
- **评估方法**: Hertz接触理论, Boussinesq解, 有限元反问题
- **真实力范围**: {min([r['force_magnitude'] for r in self.results])} - {max([r['force_magnitude'] for r in self.results])} N
- **材料参数范围**: {min([r['youngs_modulus'] for r in self.results]):.1e} - {max([r['youngs_modulus'] for r in self.results]):.1e} Pa

## 总体性能统计
| 指标 | 值 |
|------|----|
| 平均绝对误差 | {np.mean(all_errors) if all_errors else 0:.3f} N |
| 平均相对误差 | {np.mean(all_relative_errors) if all_relative_errors else 0:.1%} |
| 最大绝对误差 | {np.max(all_errors) if all_errors else 0:.3f} N |
| 最大相对误差 | {np.max(all_relative_errors) if all_relative_errors else 0:.1%} |

## 各方法性能比较
| 方法 | 平均绝对误差 (N) | 平均相对误差 | 平均估计时间 (ms) | 成功率 |
|------|------------------|--------------|-------------------|--------|
"""
        
        for method, stats in method_stats.items():
            if stats['errors']:
                avg_error = np.mean(stats['errors'])
                avg_rel_error = np.mean(stats['relative_errors'])
                avg_time = np.mean(stats['estimation_times']) * 1000  # 转换为毫秒
                success_rate = len(stats['errors']) / len(self.results) * 100
                
                report += f"| {method} | {avg_error:.3f} | {avg_rel_error:.1%} | {avg_time:.2f} | {success_rate:.1f}% |\n"
        
        report += """
## 按力大小分析
"""
        
        # 按力大小分组
        force_groups = {}
        for result in self.results:
            force = result['force_magnitude']
            if force not in force_groups:
                force_groups[force] = []
            force_groups[force].append(result)
        
        for force in sorted(force_groups.keys()):
            group = force_groups[force]
            group_errors = []
            
            for result in group:
                for method, method_result in result['evaluation_results'].items():
                    if method_result.get('success', False):
                        group_errors.append(method_result['absolute_error'])
            
            report += f"\n### 力大小: {force} N\n"
            report += f"- 测试用例数: {len(group)}\n"
            if group_errors:
                report += f"- 平均绝对误差: {np.mean(group_errors):.3f} N\n"
                report += f"- 最大绝对误差: {np.max(group_errors):.3f} N\n"
        
        report += """
## 按噪声水平分析
"""
        
        # 按噪声水平分组
        noise_groups = {}
        for result in self.results:
            noise = result['noise_level']
            if noise not in noise_groups:
                noise_groups[noise] = []
            noise_groups[noise].append(result)
        
        for noise in sorted(noise_groups.keys()):
            group = noise_groups[noise]
            group_errors = []
            
            for result in group:
                for method, method_result in result['evaluation_results'].items():
                    if method_result.get('success', False):
                        group_errors.append(method_result['absolute_error'])
            
            report += f"\n### 噪声水平: {noise}\n"
            report += f"- 测试用例数: {len(group)}\n"
            if group_errors:
                report += f"- 平均绝对误差: {np.mean(group_errors):.3f} N\n"
                report += f"- 误差标准差: {np.std(group_errors):.3f} N\n"
        
        timestamp = datetime.now().isoformat()
        report += f"""
## 结论与建议

### 主要发现
1. **方法比较**: 不同力估计方法在不同条件下表现不同
2. **噪声影响**: 噪声增加会降低所有方法的准确性
3. **力大小影响**: 对于不同大小的力，各方法的相对误差可能不同
4. **计算效率**: 不同方法的计算时间差异

### 改进建议
1. **参数校准**: 针对特定传感器材料校准力估计算法参数
2. **方法选择**: 根据应用场景选择最合适的力估计方法
3. **噪声鲁棒性**: 添加位移场预处理步骤以减少噪声影响
4. **算法优化**: 优化计算效率，特别是有限元方法

### 后续步骤
1. 使用真实传感器数据验证算法性能
2. 实现更复杂的位移场生成模型
3. 开发自适应参数选择机制
4. 集成到完整的视触传感器处理流程中

---

*报告生成时间: {timestamp}*  
*实验代码: `scripts/analysis/force_estimation_validation.py`*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"报告已生成: {report_path}")
        
        # 尝试生成可视化图表
        self.generate_visualizations()
        # 生成文本分析报告
        self.generate_text_analysis()
    
    def generate_visualizations(self):
        """生成可视化图表"""
        try:
            import matplotlib.pyplot as plt
            
            # 创建可视化目录
            viz_dir = self.results_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # 准备数据
            methods = set()
            force_levels = set()
            noise_levels = set()
            
            for result in self.results:
                force_levels.add(result['force_magnitude'])
                noise_levels.add(result['noise_level'])
                for method in result['evaluation_results'].keys():
                    methods.add(method)
            
            methods = sorted(methods)
            force_levels = sorted(force_levels)
            noise_levels = sorted(noise_levels)
            
            # 1. 不同方法的误差对比
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('力估计算法性能分析', fontsize=16)
            
            # 绝对误差箱线图
            ax = axes[0]
            error_data = []
            method_labels = []
            
            for method in methods:
                method_errors = []
                for result in self.results:
                    method_result = result['evaluation_results'].get(method)
                    if method_result and method_result.get('success', False):
                        method_errors.append(method_result['absolute_error'])
                
                if method_errors:
                    error_data.append(method_errors)
                    method_labels.append(method)
            
            if error_data:
                box = ax.boxplot(error_data, labels=method_labels, patch_artist=True)
                # 设置颜色
                colors = ['lightblue', 'lightgreen', 'lightcoral']
                for patch, color in zip(box['boxes'], colors[:len(method_labels)]):
                    patch.set_facecolor(color)
                
                ax.set_ylabel('绝对误差 (N)')
                ax.set_title('各方法绝对误差分布')
                ax.grid(True, alpha=0.3)
            
            # 相对误差散点图
            ax = axes[1]
            for method in methods:
                method_relative_errors = []
                method_forces = []
                
                for result in self.results:
                    method_result = result['evaluation_results'].get(method)
                    if method_result and method_result.get('success', False):
                        method_relative_errors.append(method_result['relative_error'])
                        method_forces.append(result['force_magnitude'])
                
                if method_relative_errors:
                    ax.scatter(method_forces, method_relative_errors, 
                              label=method, alpha=0.6, s=50)
            
            ax.set_xlabel('力大小 (N)')
            ax.set_ylabel('相对误差')
            ax.set_title('相对误差 vs 力大小')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'method_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"可视化图表已保存至: {viz_dir}")
            
        except ImportError:
            print("警告: matplotlib不可用，跳过可视化")
        except Exception as e:
            print(f"可视化生成失败: {e}")
    
    def generate_text_analysis(self):
        """生成文本分析报告"""
        try:
            # 尝试导入文本分析工具
            from text_analysis import TextAnalysis
            print("生成文本分析报告...")
            analyzer = TextAnalysis(self.results_dir)
            analyzer.analyze_and_report()
            print("文本分析报告生成完成")
        except ImportError as e:
            print(f"警告: 文本分析工具不可用，跳过文本分析: {e}")
        except Exception as e:
            print(f"文本分析失败: {e}")


def main():
    """主函数"""
    print("力估计算法验证实验")
    print("=" * 60)
    
    # 创建评估器
    evaluator = ForceEstimationEvaluator()
    
    # 运行验证
    evaluator.run_validation()
    
    print("\n实验完成!")
    print(f"结果目录: {evaluator.results_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())