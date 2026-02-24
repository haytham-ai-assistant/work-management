#!/usr/bin/env python3
"""
文本分析工具 - 用于在没有可视化库的环境中分析实验结果

提供纯文本输出，不依赖matplotlib、pandas等库。
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


class TextAnalysis:
    """文本分析工具类"""
    
    def __init__(self, results_dir="results"):
        """
        初始化分析工具
        
        Args:
            results_dir: 结果目录路径
        """
        self.results_dir = Path(results_dir)
    
    def load_validation_results(self, filename="validation_results.json"):
        """
        加载验证结果
        
        Args:
            filename: 结果文件名
            
        Returns:
            results: 加载的结果列表，如果失败返回None
        """
        file_path = self.results_dir / "force_estimation_validation" / filename
        if not file_path.exists():
            # 尝试直接路径
            file_path = self.results_dir / filename
        
        if not file_path.exists():
            print(f"错误: 找不到结果文件 {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"成功加载 {len(results)} 条结果")
            return results
        except Exception as e:
            print(f"加载结果文件失败: {e}")
            return None
    
    def analyze_force_estimation(self, results):
        """
        分析力估计算法性能
        
        Args:
            results: 验证结果列表
            
        Returns:
            analysis: 分析结果字典
        """
        if not results:
            return {"error": "无结果数据"}
        
        # 按方法分组
        method_stats = {}
        
        for result in results:
            for method, method_result in result.get('evaluation_results', {}).items():
                if not method_result.get('success', False):
                    continue
                
                if method not in method_stats:
                    method_stats[method] = {
                        'absolute_errors': [],
                        'relative_errors': [],
                        'estimation_times': [],
                        'force_magnitudes': [],
                        'youngs_moduli': [],
                        'noise_levels': []
                    }
                
                method_stats[method]['absolute_errors'].append(method_result['absolute_error'])
                method_stats[method]['relative_errors'].append(method_result['relative_error'])
                method_stats[method]['estimation_times'].append(method_result['estimation_time'])
                method_stats[method]['force_magnitudes'].append(result['force_magnitude'])
                method_stats[method]['youngs_moduli'].append(result['youngs_modulus'])
                method_stats[method]['noise_levels'].append(result['noise_level'])
        
        # 计算统计指标
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(results),
            'methods': {}
        }
        
        for method, stats in method_stats.items():
            abs_errors = np.array(stats['absolute_errors'])
            rel_errors = np.array(stats['relative_errors'])
            times = np.array(stats['estimation_times'])
            
            analysis['methods'][method] = {
                'samples': len(abs_errors),
                'mean_absolute_error': float(np.mean(abs_errors)),
                'std_absolute_error': float(np.std(abs_errors)),
                'max_absolute_error': float(np.max(abs_errors)),
                'mean_relative_error': float(np.mean(rel_errors)),
                'std_relative_error': float(np.std(rel_errors)),
                'max_relative_error': float(np.max(rel_errors)),
                'mean_time_ms': float(np.mean(times) * 1000),
                'std_time_ms': float(np.std(times) * 1000)
            }
        
        # 找出最佳方法 (基于平均绝对误差)
        best_method = None
        best_mae = float('inf')
        
        for method, stats in analysis['methods'].items():
            if stats['mean_absolute_error'] < best_mae:
                best_mae = stats['mean_absolute_error']
                best_method = method
        
        analysis['best_method'] = best_method
        analysis['best_mae'] = best_mae
        
        return analysis
    
    def print_analysis_summary(self, analysis):
        """
        打印分析摘要
        
        Args:
            analysis: 分析结果字典
        """
        print("=" * 70)
        print("力估计算法性能分析报告")
        print("=" * 70)
        print(f"分析时间: {analysis['timestamp']}")
        print(f"总测试用例: {analysis['total_samples']}")
        print(f"评估方法数量: {len(analysis['methods'])}")
        print()
        
        # 打印各方法性能
        print("各方法性能比较:")
        print("-" * 70)
        print(f"{'方法':<12} {'样本数':<8} {'MAE(N)':<10} {'MRE(%)':<10} {'时间(ms)':<10}")
        print("-" * 70)
        
        for method, stats in analysis['methods'].items():
            mae = stats['mean_absolute_error']
            mre = stats['mean_relative_error'] * 100
            time_ms = stats['mean_time_ms']
            print(f"{method:<12} {stats['samples']:<8} {mae:<10.3f} {mre:<10.1f} {time_ms:<10.2f}")
        
        print()
        print(f"最佳方法: {analysis['best_method']} (MAE: {analysis['best_mae']:.3f} N)")
        print()
        
        # 按力大小分析
        print("按力大小分析:")
        print("-" * 70)
        
        # 简单分组示例（实际需要更复杂的分组逻辑）
        if 'methods' in analysis and analysis['methods']:
            # 使用第一个方法的数据
            first_method = list(analysis['methods'].keys())[0]
            print(f"注: 使用 {first_method} 方法的数据进行分组分析")
        
        print()
    
    def generate_text_report(self, analysis, output_file=None):
        """
        生成文本报告
        
        Args:
            analysis: 分析结果字典
            output_file: 输出文件路径 (可选)
            
        Returns:
            report_text: 报告文本
        """
        report = f"""力估计算法性能分析报告
==================================================
分析时间: {analysis['timestamp']}
总测试用例: {analysis['total_samples']}
评估方法数量: {len(analysis['methods'])}

各方法性能比较:
--------------------------------------------------
"""
        
        # 添加表格
        report += f"{'方法':<12} {'样本数':<8} {'MAE(N)':<10} {'MRE(%)':<10} {'时间(ms)':<10}\n"
        report += "-" * 50 + "\n"
        
        for method, stats in analysis['methods'].items():
            mae = stats['mean_absolute_error']
            mre = stats['mean_relative_error'] * 100
            time_ms = stats['mean_time_ms']
            report += f"{method:<12} {stats['samples']:<8} {mae:<10.3f} {mre:<10.1f} {time_ms:<10.2f}\n"
        
        report += f"\n最佳方法: {analysis['best_method']} (平均绝对误差: {analysis['best_mae']:.3f} N)\n\n"
        
        # 添加详细统计
        report += "详细统计:\n"
        report += "--------------------------------------------------\n"
        
        for method, stats in analysis['methods'].items():
            report += f"\n{method} 方法:\n"
            report += f"  样本数: {stats['samples']}\n"
            report += f"  平均绝对误差: {stats['mean_absolute_error']:.3f} ± {stats['std_absolute_error']:.3f} N\n"
            report += f"  最大绝对误差: {stats['max_absolute_error']:.3f} N\n"
            report += f"  平均相对误差: {stats['mean_relative_error']*100:.1f} ± {stats['std_relative_error']*100:.1f}%\n"
            report += f"  最大相对误差: {stats['max_relative_error']*100:.1f}%\n"
            report += f"  平均估计时间: {stats['mean_time_ms']:.2f} ± {stats['std_time_ms']:.2f} ms\n"
        
        report += "\n==================================================\n"
        report += "说明:\n"
        report += "- MAE: 平均绝对误差 (Mean Absolute Error)\n"
        report += "- MRE: 平均相对误差 (Mean Relative Error)\n"
        report += "- 所有力单位: 牛顿 (N)\n"
        report += "- 时间单位: 毫秒 (ms)\n"
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"报告已保存至: {output_path}")
        
        return report
    
    def analyze_and_report(self, results_file="validation_results.json"):
        """
        完整的分析报告流程
        
        Args:
            results_file: 结果文件名
        """
        print("开始分析验证结果...")
        
        # 加载结果
        results = self.load_validation_results(results_file)
        if not results:
            return
        
        # 分析性能
        analysis = self.analyze_force_estimation(results)
        
        # 打印摘要
        self.print_analysis_summary(analysis)
        
        # 生成详细报告
        report_dir = self.results_dir / "force_estimation_validation"
        report_file = report_dir / "text_analysis_report.txt"
        
        report_text = self.generate_text_report(analysis, report_file)
        
        print(f"分析完成! 详细报告: {report_file}")
        
        return analysis


def main():
    """主函数"""
    import sys
    
    # 默认结果目录
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "results"
    
    print("文本分析工具 - 力估计算法性能分析")
    print("=" * 60)
    
    analyzer = TextAnalysis(results_dir)
    analyzer.analyze_and_report()
    
    print("\n工具执行完成!")


if __name__ == "__main__":
    main()