#!/usr/bin/env python3
"""
数据集验证脚本

本脚本验证处理后的数据集的质量和一致性，包括:
1. 数据完整性检查
2. 格式合规性检查
3. 物理合理性检查
4. 生成验证报告

使用方法:
    python validate_dataset.py --experiment_id exp_001 --data_dir data/processed
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import csv
from datetime import datetime


class DatasetValidator:
    """数据集验证器"""
    
    def __init__(self, experiment_id, data_dir="data/processed"):
        """
        初始化验证器
        
        Args:
            experiment_id: 实验ID (如 "exp_001")
            data_dir: 数据目录 (包含处理后的数据)
        """
        self.experiment_id = experiment_id
        self.data_dir = Path(data_dir) / experiment_id
        
        # 验证目录是否存在
        if not self.data_dir.exists():
            print(f"错误: 数据目录不存在: {self.data_dir}")
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 验证结果存储
        self.validation_results = {
            "experiment_id": experiment_id,
            "validation_time": datetime.now().isoformat(),
            "checks": {},
            "summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "warnings": 0
            },
            "issues": []
        }
    
    def check_directory_structure(self):
        """检查目录结构"""
        check_name = "directory_structure"
        print(f"检查: {check_name}")
        
        required_dirs = [
            "displacement_fields",
            "marker_positions",
            "force_estimates",
            "validation_results"
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            result = {
                "status": "FAILED",
                "message": f"缺失目录: {', '.join(missing_dirs)}",
                "missing_dirs": missing_dirs
            }
            self.validation_results["issues"].append({
                "type": "missing_directory",
                "directories": missing_dirs,
                "severity": "high"
            })
        else:
            result = {
                "status": "PASSED",
                "message": "目录结构完整"
            }
        
        self.validation_results["checks"][check_name] = result
        return result
    
    def check_displacement_fields(self):
        """检查位移场数据"""
        check_name = "displacement_fields"
        print(f"检查: {check_name}")
        
        disp_dir = self.data_dir / "displacement_fields"
        if not disp_dir.exists():
            result = {
                "status": "FAILED",
                "message": "位移场目录不存在",
                "files_found": 0
            }
            self.validation_results["checks"][check_name] = result
            return result
        
        # 查找位移场文件
        disp_files = list(disp_dir.glob("disp_*.npy"))
        
        if not disp_files:
            result = {
                "status": "FAILED",
                "message": "未找到位移场文件",
                "files_found": 0
            }
            self.validation_results["issues"].append({
                "type": "no_displacement_files",
                "severity": "high"
            })
        else:
            # 检查文件格式和内容
            valid_files = 0
            file_stats = []
            
            for i, file_path in enumerate(disp_files[:5]):  # 检查前5个文件
                try:
                    disp = np.load(str(file_path))
                    
                    # 检查数组形状和类型
                    if disp.ndim == 2 and disp.shape[1] == 2:
                        stats = {
                            "file": file_path.name,
                            "shape": disp.shape,
                            "dtype": str(disp.dtype),
                            "min": float(np.min(disp)),
                            "max": float(np.max(disp)),
                            "mean": float(np.mean(disp)),
                            "std": float(np.std(disp))
                        }
                        file_stats.append(stats)
                        valid_files += 1
                    else:
                        self.validation_results["issues"].append({
                            "type": "invalid_displacement_shape",
                            "file": file_path.name,
                            "shape": disp.shape,
                            "expected_shape": "(N, 2)",
                            "severity": "medium"
                        })
                        
                except Exception as e:
                    self.validation_results["issues"].append({
                        "type": "displacement_file_error",
                        "file": file_path.name,
                        "error": str(e),
                        "severity": "medium"
                    })
            
            # 检查位移值的物理合理性 (单位: mm)
            # 典型位移应该在0-5mm范围内
            physical_issues = 0
            for stats in file_stats:
                max_abs = max(abs(stats["min"]), abs(stats["max"]))
                if max_abs > 10.0:  # 位移过大警告
                    self.validation_results["issues"].append({
                        "type": "large_displacement",
                        "file": stats["file"],
                        "max_abs_displacement_mm": max_abs,
                        "threshold_mm": 10.0,
                        "severity": "low"
                    })
                    physical_issues += 1
            
            result = {
                "status": "PASSED" if valid_files > 0 else "FAILED",
                "message": f"找到 {len(disp_files)} 个位移场文件，{valid_files} 个有效",
                "files_found": len(disp_files),
                "valid_files": valid_files,
                "sample_stats": file_stats[:3] if file_stats else [],
                "physical_issues": physical_issues
            }
        
        self.validation_results["checks"][check_name] = result
        return result
    
    def check_marker_positions(self):
        """检查标记点位置数据"""
        check_name = "marker_positions"
        print(f"检查: {check_name}")
        
        markers_dir = self.data_dir / "marker_positions"
        if not markers_dir.exists():
            result = {
                "status": "FAILED",
                "message": "标记点位置目录不存在",
                "files_found": 0
            }
            self.validation_results["checks"][check_name] = result
            return result
        
        # 查找标记点文件
        marker_files = list(markers_dir.glob("*.npy"))
        
        # 检查参考标记点文件
        ref_marker_file = markers_dir / "reference_markers.npy"
        has_reference = ref_marker_file.exists()
        
        if not marker_files:
            result = {
                "status": "FAILED",
                "message": "未找到标记点文件",
                "files_found": 0,
                "has_reference": has_reference
            }
            self.validation_results["issues"].append({
                "type": "no_marker_files",
                "severity": "high"
            })
        else:
            valid_files = 0
            file_stats = []
            
            # 检查参考标记点
            if has_reference:
                try:
                    ref_markers = np.load(str(ref_marker_file))
                    if ref_markers.ndim == 2 and ref_markers.shape[1] == 2:
                        ref_stats = {
                            "file": "reference_markers.npy",
                            "shape": ref_markers.shape,
                            "num_markers": ref_markers.shape[0],
                            "x_range": [float(np.min(ref_markers[:, 0])), 
                                       float(np.max(ref_markers[:, 0]))],
                            "y_range": [float(np.min(ref_markers[:, 1])), 
                                       float(np.max(ref_markers[:, 1]))]
                        }
                        file_stats.append(ref_stats)
                        valid_files += 1
                        
                        # 检查标记点间距是否合理
                        if ref_markers.shape[0] > 1:
                            from scipy.spatial import distance_matrix
                            try:
                                dists = distance_matrix(ref_markers, ref_markers)
                                # 获取最小非零距离
                                dists[dists == 0] = np.inf
                                min_dist = np.min(dists)
                                if min_dist < 5.0:  # 标记点间距过小
                                    self.validation_results["issues"].append({
                                        "type": "small_marker_spacing",
                                        "file": "reference_markers.npy",
                                        "min_spacing_px": min_dist,
                                        "threshold_px": 5.0,
                                        "severity": "medium"
                                    })
                            except:
                                pass  # 跳过距离计算错误
                    else:
                        self.validation_results["issues"].append({
                            "type": "invalid_reference_markers",
                            "file": "reference_markers.npy",
                            "shape": ref_markers.shape,
                            "expected_shape": "(N, 2)",
                            "severity": "high"
                        })
                        
                except Exception as e:
                    self.validation_results["issues"].append({
                        "type": "reference_marker_error",
                        "file": "reference_markers.npy",
                        "error": str(e),
                        "severity": "high"
                    })
            
            # 检查其他标记点文件
            for file_path in marker_files:
                if file_path.name == "reference_markers.npy":
                    continue
                
                try:
                    markers = np.load(str(file_path))
                    if markers.ndim == 2 and markers.shape[1] == 2:
                        valid_files += 1
                except Exception as e:
                    self.validation_results["issues"].append({
                        "type": "marker_file_error",
                        "file": file_path.name,
                        "error": str(e),
                        "severity": "medium"
                    })
            
            result = {
                "status": "PASSED" if valid_files > 0 else "FAILED",
                "message": f"找到 {len(marker_files)} 个标记点文件，{valid_files} 个有效",
                "files_found": len(marker_files),
                "valid_files": valid_files,
                "has_reference": has_reference,
                "reference_stats": file_stats[0] if file_stats else None
            }
        
        self.validation_results["checks"][check_name] = result
        return result
    
    def check_force_estimates(self):
        """检查力估计数据"""
        check_name = "force_estimates"
        print(f"检查: {check_name}")
        
        force_dir = self.data_dir / "force_estimates"
        if not force_dir.exists():
            result = {
                "status": "FAILED",
                "message": "力估计目录不存在",
                "files_found": 0
            }
            self.validation_results["checks"][check_name] = result
            return result
        
        # 检查CSV文件
        csv_file = force_dir / "force_estimates.csv"
        json_file = force_dir / "force_estimates.json"
        
        files_exist = {
            "csv": csv_file.exists(),
            "json": json_file.exists()
        }
        
        if not files_exist["csv"] and not files_exist["json"]:
            result = {
                "status": "FAILED",
                "message": "未找到力估计文件",
                "files_found": 0
            }
            self.validation_results["issues"].append({
                "type": "no_force_estimate_files",
                "severity": "medium"
            })
            self.validation_results["checks"][check_name] = result
            return result
        
        # 检查CSV文件内容
        csv_data = None
        if files_exist["csv"]:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                    if rows:
                        # 检查列名
                        expected_cols = ["frame_id", "force_x_n", "force_y_n", "force_z_n"]
                        actual_cols = list(rows[0].keys())
                        missing_cols = [col for col in expected_cols if col not in actual_cols]
                        
                        if missing_cols:
                            self.validation_results["issues"].append({
                                "type": "missing_force_columns",
                                "file": "force_estimates.csv",
                                "missing_columns": missing_cols,
                                "severity": "medium"
                            })
                        
                        # 检查数据值
                        force_values = []
                        for row in rows:
                            try:
                                fz = float(row.get("force_z_n", 0))
                                force_values.append(fz)
                            except:
                                pass
                        
                        if force_values:
                            stats = {
                                "num_records": len(rows),
                                "force_z_stats": {
                                    "min": float(np.min(force_values)),
                                    "max": float(np.max(force_values)),
                                    "mean": float(np.mean(force_values)),
                                    "std": float(np.std(force_values))
                                }
                            }
                            csv_data = stats
                            
                            # 检查力值的物理合理性 (单位: N)
                            # 典型接触力应该在0-50N范围内
                            max_abs = max(abs(stats["force_z_stats"]["min"]), 
                                         abs(stats["force_z_stats"]["max"]))
                            if max_abs > 100.0:  # 力值过大警告
                                self.validation_results["issues"].append({
                                    "type": "large_force_value",
                                    "file": "force_estimates.csv",
                                    "max_abs_force_n": max_abs,
                                    "threshold_n": 100.0,
                                    "severity": "low"
                                })
                            
                    else:
                        self.validation_results["issues"].append({
                            "type": "empty_force_csv",
                            "file": "force_estimates.csv",
                            "severity": "medium"
                        })
                        
            except Exception as e:
                self.validation_results["issues"].append({
                    "type": "force_csv_error",
                    "file": "force_estimates.csv",
                    "error": str(e),
                    "severity": "medium"
                })
        
        result = {
            "status": "PASSED" if (files_exist["csv"] or files_exist["json"]) else "FAILED",
            "message": f"力估计文件: CSV={files_exist['csv']}, JSON={files_exist['json']}",
            "files_exist": files_exist,
            "csv_stats": csv_data
        }
        
        self.validation_results["checks"][check_name] = result
        return result
    
    def check_validation_results(self):
        """检查验证结果"""
        check_name = "validation_results"
        print(f"检查: {check_name}")
        
        validation_dir = self.data_dir / "validation_results"
        if not validation_dir.exists():
            result = {
                "status": "WARNING",
                "message": "验证结果目录不存在",
                "files_found": 0
            }
            self.validation_results["checks"][check_name] = result
            return result
        
        # 查找验证报告
        report_files = list(validation_dir.glob("*.md")) + list(validation_dir.glob("*.json"))
        
        if not report_files:
            result = {
                "status": "WARNING",
                "message": "未找到验证报告文件",
                "files_found": 0
            }
            self.validation_results["issues"].append({
                "type": "no_validation_reports",
                "severity": "low"
            })
        else:
            result = {
                "status": "PASSED",
                "message": f"找到 {len(report_files)} 个验证报告文件",
                "files_found": len(report_files),
                "file_list": [f.name for f in report_files]
            }
        
        self.validation_results["checks"][check_name] = result
        return result
    
    def check_data_consistency(self):
        """检查数据一致性"""
        check_name = "data_consistency"
        print(f"检查: {check_name}")
        
        issues = []
        
        # 检查位移场和标记点文件数量是否匹配
        disp_dir = self.data_dir / "displacement_fields"
        markers_dir = self.data_dir / "marker_positions"
        
        if disp_dir.exists() and markers_dir.exists():
            disp_files = list(disp_dir.glob("disp_*.npy"))
            marker_files = [f for f in markers_dir.glob("*.npy") 
                           if f.name != "reference_markers.npy"]
            
            if disp_files and marker_files:
                num_disp = len(disp_files)
                num_markers = len(marker_files)
                
                if num_disp != num_markers:
                    issues.append({
                        "type": "file_count_mismatch",
                        "displacement_files": num_disp,
                        "marker_files": num_markers,
                        "difference": abs(num_disp - num_markers),
                        "severity": "medium"
                    })
                
                # 检查文件名对应关系
                disp_indices = []
                for f in disp_files:
                    try:
                        idx = int(f.stem.split('_')[1])
                        disp_indices.append(idx)
                    except:
                        pass
                
                marker_indices = []
                for f in marker_files:
                    try:
                        idx = int(f.stem.split('_')[1])
                        marker_indices.append(idx)
                    except:
                        pass
                
                if disp_indices and marker_indices:
                    missing_in_markers = set(disp_indices) - set(marker_indices)
                    missing_in_disp = set(marker_indices) - set(disp_indices)
                    
                    if missing_in_markers:
                        issues.append({
                            "type": "missing_marker_files",
                            "missing_indices": list(missing_in_markers),
                            "severity": "medium"
                        })
                    
                    if missing_in_disp:
                        issues.append({
                            "type": "missing_displacement_files",
                            "missing_indices": list(missing_in_disp),
                            "severity": "medium"
                        })
        
        # 检查位移场和力估计的帧数匹配
        force_csv = self.data_dir / "force_estimates" / "force_estimates.csv"
        if disp_dir.exists() and force_csv.exists():
            try:
                disp_files = list(disp_dir.glob("disp_*.npy"))
                with open(force_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    force_rows = list(reader)
                
                if disp_files and force_rows:
                    if len(disp_files) != len(force_rows):
                        issues.append({
                            "type": "displacement_force_count_mismatch",
                            "displacement_files": len(disp_files),
                            "force_records": len(force_rows),
                            "difference": abs(len(disp_files) - len(force_rows)),
                            "severity": "medium"
                        })
            except:
                pass  # 跳过错误
        
        result = {
            "status": "PASSED" if not issues else "WARNING",
            "message": f"数据一致性检查: 发现 {len(issues)} 个问题",
            "issues_found": len(issues),
            "issues": issues
        }
        
        # 添加问题到总列表
        for issue in issues:
            self.validation_results["issues"].append(issue)
        
        self.validation_results["checks"][check_name] = result
        return result
    
    def run_all_checks(self):
        """运行所有检查"""
        print(f"=== 数据集验证开始 ===")
        print(f"实验ID: {self.experiment_id}")
        print(f"数据目录: {self.data_dir}")
        print()
        
        checks = [
            self.check_directory_structure,
            self.check_displacement_fields,
            self.check_marker_positions,
            self.check_force_estimates,
            self.check_validation_results,
            self.check_data_consistency
        ]
        
        for check_func in checks:
            try:
                result = check_func()
                
                # 更新摘要统计
                self.validation_results["summary"]["total_checks"] += 1
                status = result.get("status", "UNKNOWN")
                
                if status == "PASSED":
                    self.validation_results["summary"]["passed_checks"] += 1
                elif status == "FAILED":
                    self.validation_results["summary"]["failed_checks"] += 1
                elif status == "WARNING":
                    self.validation_results["summary"]["warnings"] += 1
                
                print(f"  {status}: {result.get('message', '')}")
                
            except Exception as e:
                print(f"  检查失败: {e}")
                self.validation_results["summary"]["total_checks"] += 1
                self.validation_results["summary"]["failed_checks"] += 1
        
        # 生成总体状态
        total = self.validation_results["summary"]["total_checks"]
        passed = self.validation_results["summary"]["passed_checks"]
        failed = self.validation_results["summary"]["failed_checks"]
        
        if failed == 0:
            overall_status = "PASSED"
        elif failed / total < 0.3:  # 少于30%失败
            overall_status = "WARNING"
        else:
            overall_status = "FAILED"
        
        self.validation_results["overall_status"] = overall_status
        
        print(f"\n=== 验证完成 ===")
        print(f"总体状态: {overall_status}")
        print(f"检查统计: {passed}/{total} 通过, {failed} 失败, "
              f"{self.validation_results['summary']['warnings']} 警告")
        
        if self.validation_results["issues"]:
            print(f"发现的问题: {len(self.validation_results['issues'])}")
            for i, issue in enumerate(self.validation_results["issues"][:5]):  # 显示前5个
                print(f"  {i+1}. [{issue.get('severity', 'unknown')}] {issue.get('type', 'unknown')}")
        
        return overall_status
    
    def generate_report(self, output_dir=None):
        """生成验证报告"""
        if output_dir is None:
            output_dir = self.data_dir / "validation_results"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 生成Markdown报告
        report_path = output_dir / "dataset_validation_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 数据集验证报告\n\n")
            
            # 基本信息
            f.write("## 基本信息\n")
            f.write(f"- **实验ID**: {self.validation_results['experiment_id']}\n")
            f.write(f"- **验证时间**: {self.validation_results['validation_time']}\n")
            f.write(f"- **数据目录**: {self.data_dir}\n")
            f.write(f"- **总体状态**: **{self.validation_results['overall_status']}**\n\n")
            
            # 检查统计
            f.write("## 检查统计\n")
            summary = self.validation_results["summary"]
            f.write(f"- **总检查数**: {summary['total_checks']}\n")
            f.write(f"- **通过**: {summary['passed_checks']}\n")
            f.write(f"- **失败**: {summary['failed_checks']}\n")
            f.write(f"- **警告**: {summary['warnings']}\n\n")
            
            # 详细检查结果
            f.write("## 详细检查结果\n")
            for check_name, result in self.validation_results["checks"].items():
                status = result.get("status", "UNKNOWN")
                status_emoji = "✅" if status == "PASSED" else "⚠️" if status == "WARNING" else "❌"
                f.write(f"\n### {check_name.replace('_', ' ').title()}\n")
                f.write(f"{status_emoji} **状态**: {status}\n")
                f.write(f"**消息**: {result.get('message', '')}\n")
                
                # 添加详细信息
                for key, value in result.items():
                    if key not in ["status", "message"] and value:
                        f.write(f"- **{key}**: {value}\n")
            
            # 问题列表
            if self.validation_results["issues"]:
                f.write("\n## 发现的问题\n")
                f.write(f"共发现 {len(self.validation_results['issues'])} 个问题:\n\n")
                
                for i, issue in enumerate(self.validation_results["issues"]):
                    severity = issue.get("severity", "unknown")
                    severity_emoji = {
                        "high": "🔴", "medium": "🟡", "low": "🟢"
                    }.get(severity, "⚪")
                    
                    f.write(f"{i+1}. {severity_emoji} **[{severity.upper()}] {issue.get('type', 'unknown')}**\n")
                    
                    for key, value in issue.items():
                        if key not in ["type", "severity"]:
                            f.write(f"   - {key}: {value}\n")
            
            # 建议
            f.write("\n## 建议与下一步\n")
            
            if self.validation_results["overall_status"] == "PASSED":
                f.write("数据集验证通过，可以用于算法训练和验证。\n")
            elif self.validation_results["overall_status"] == "WARNING":
                f.write("数据集有警告，建议检查并修复问题后再使用。\n")
            else:
                f.write("数据集验证失败，需要修复关键问题后才能使用。\n")
            
            f.write("\n### 下一步操作\n")
            f.write("1. 根据问题列表修复数据问题\n")
            f.write("2. 重新运行数据收集和处理流程\n")
            f.write("3. 重新运行本验证脚本\n")
            f.write("4. 使用验证通过的数据集进行算法训练和测试\n")
        
        print(f"验证报告已生成: {report_path}")
        
        # 保存JSON格式的详细结果
        json_path = output_dir / "dataset_validation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        print(f"详细结果已保存: {json_path}")
        
        return report_path, json_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据集验证脚本")
    parser.add_argument("--experiment_id", type=str, default="exp_001",
                       help="实验ID (默认: exp_001)")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="数据目录 (默认: data/processed)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录 (默认: <data_dir>/validation_results)")
    parser.add_argument("--skip_report", action="store_true",
                       help="跳过报告生成")
    
    args = parser.parse_args()
    
    try:
        # 创建验证器
        validator = DatasetValidator(
            experiment_id=args.experiment_id,
            data_dir=args.data_dir
        )
        
        # 运行所有检查
        overall_status = validator.run_all_checks()
        
        # 生成报告
        if not args.skip_report:
            print("\n生成验证报告...")
            report_path, json_path = validator.generate_report(args.output_dir)
            print(f"报告文件: {report_path}")
            print(f"JSON结果: {json_path}")
        
        # 返回退出码
        if overall_status == "FAILED":
            print("\n数据集验证失败，请检查并修复问题。")
            return 1
        elif overall_status == "WARNING":
            print("\n数据集验证有警告，建议检查问题。")
            return 0
        else:
            print("\n数据集验证通过。")
            return 0
            
    except Exception as e:
        print(f"验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())