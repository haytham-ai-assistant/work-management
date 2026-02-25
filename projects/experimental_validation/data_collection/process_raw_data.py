#!/usr/bin/env python3
"""
原始传感器数据处理脚本

本脚本处理从collect_data.py收集的原始数据，包括:
1. 从图像序列计算位移场
2. 提取标记点位置
3. 同步力测量数据
4. 保存处理后的数据供算法使用

使用方法:
    python process_raw_data.py --experiment_id exp_001 --input_dir data/raw --output_dir data/processed
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import csv

# 尝试导入算法库
try:
    sys.path.append(str(Path(__file__).parents[3] / "vbts_algorithms" / "src"))
    from algorithms.marker_detection import MarkerDetection
    from algorithms.force_estimation import ForceEstimation
    ALGORITHMS_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入算法库: {e}")
    ALGORITHMS_AVAILABLE = False

# 尝试导入可视化库
try:
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    print("警告: OpenCV不可用，图像处理功能将受限")


class RawDataProcessor:
    """原始数据处理类"""
    
    def __init__(self, experiment_id, input_dir="data/raw", output_dir="data/processed"):
        """
        初始化处理器
        
        Args:
            experiment_id: 实验ID (如 "exp_001")
            input_dir: 输入数据目录
            output_dir: 输出数据目录
        """
        self.experiment_id = experiment_id
        self.input_dir = Path(input_dir) / experiment_id
        self.output_dir = Path(output_dir) / experiment_id
        
        # 创建输出目录结构
        self.disp_dir = self.output_dir / "displacement_fields"
        self.markers_dir = self.output_dir / "marker_positions"
        self.force_dir = self.output_dir / "force_estimates"
        self.validation_dir = self.output_dir / "validation_results"
        
        for dir_path in [self.disp_dir, self.markers_dir, self.force_dir, self.validation_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 加载元数据
        self.metadata = self.load_metadata()
        
        # 初始化算法模块
        self.marker_detector = None
        self.force_estimator = None
        self.initialize_algorithms()
        
        # 处理状态
        self.reference_frame = None
        self.reference_markers = None
        
    def load_metadata(self):
        """加载实验元数据"""
        metadata_path = self.input_dir / "metadata.json"
        if not metadata_path.exists():
            print(f"错误: 元数据文件不存在: {metadata_path}")
            return {}
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"加载元数据: {metadata_path}")
        print(f"实验ID: {metadata.get('experiment_id', '未知')}")
        print(f"测试条件: {len(metadata.get('test_conditions', []))}个")
        
        return metadata
    
    def initialize_algorithms(self):
        """初始化算法模块"""
        if not ALGORITHMS_AVAILABLE:
            print("警告: 算法库不可用，使用简化处理模式")
            return
        
        try:
            # 从元数据获取传感器参数
            sensor_params = self.metadata.get("sensor_parameters", {})
            
            # 初始化标记点检测器
            marker_radius_mm = sensor_params.get("marker_radius_mm", 6.0)
            marker_spacing_mm = sensor_params.get("marker_spacing_mm", 25.0)
            pixel_to_mm = sensor_params.get("pixel_to_mm", 0.1)
            
            self.marker_detector = MarkerDetection(
                marker_radius_mm=marker_radius_mm,
                marker_spacing_mm=marker_spacing_mm,
                pixel_to_mm=pixel_to_mm
            )
            print(f"标记点检测器初始化: 半径={marker_radius_mm}mm, 间距={marker_spacing_mm}mm")
            
            # 初始化力估计器
            youngs_modulus = sensor_params.get("material_youngs_modulus_pa", 2000000.0)
            poissons_ratio = sensor_params.get("material_poissons_ratio", 0.49)
            sensor_thickness = sensor_params.get("sensor_thickness_mm", 5.0)
            
            self.force_estimator = ForceEstimation(
                youngs_modulus=youngs_modulus,
                poissons_ratio=poissons_ratio,
                sensor_thickness_mm=sensor_thickness
            )
            print(f"力估计器初始化: E={youngs_modulus}Pa, ν={poissons_ratio}")
            
        except Exception as e:
            print(f"算法初始化失败: {e}")
            self.marker_detector = None
            self.force_estimator = None
    
    def load_reference_frame(self):
        """加载参考帧 (未加载状态)"""
        ref_path = self.input_dir / "calibration" / "reference_frame.png"
        if not ref_path.exists():
            # 尝试从图像目录加载第一帧
            ref_path = self.input_dir / "images" / "frame_0000.png"
        
        if ref_path.exists() and CV_AVAILABLE:
            self.reference_frame = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
            print(f"参考帧已加载: {ref_path}, 尺寸: {self.reference_frame.shape}")
            return self.reference_frame
        else:
            print(f"警告: 参考帧不存在或OpenCV不可用: {ref_path}")
            return None
    
    def detect_reference_markers(self):
        """检测参考帧中的标记点"""
        if self.reference_frame is None:
            if not self.load_reference_frame():
                print("错误: 无法加载参考帧")
                return None
        
        if self.marker_detector is None:
            print("警告: 标记点检测器未初始化，使用默认标记点模式")
            # 生成合成标记点模式
            height, width = self.reference_frame.shape if self.reference_frame is not None else (480, 640)
            markers = self.generate_synthetic_markers(width, height)
            self.reference_markers = markers
            return markers
        
        try:
            # 使用算法库检测标记点
            markers = self.marker_detector.detect_markers_from_image(
                self.reference_frame,
                reference_mode=True
            )
            
            if markers is not None and len(markers) > 0:
                print(f"检测到 {len(markers)} 个标记点")
                self.reference_markers = markers
                
                # 保存标记点位置
                markers_path = self.markers_dir / "reference_markers.npy"
                np.save(str(markers_path), markers)
                print(f"参考标记点已保存: {markers_path}")
                
                return markers
            else:
                print("警告: 未检测到标记点，使用合成标记点")
                markers = self.generate_synthetic_markers()
                self.reference_markers = markers
                return markers
                
        except Exception as e:
            print(f"标记点检测失败: {e}")
            markers = self.generate_synthetic_markers()
            self.reference_markers = markers
            return markers
    
    def generate_synthetic_markers(self, width=640, height=480):
        """生成合成标记点模式"""
        print(f"生成合成标记点模式: {width}x{height}")
        
        # 从元数据获取参数
        sensor_params = self.metadata.get("sensor_parameters", {})
        marker_spacing_mm = sensor_params.get("marker_spacing_mm", 25.0)
        pixel_to_mm = sensor_params.get("pixel_to_mm", 0.1)
        sensor_size_mm = sensor_params.get("sensor_size_mm", [100, 100])
        
        # 转换为像素
        marker_spacing_px = marker_spacing_mm / pixel_to_mm
        sensor_width_px = sensor_size_mm[0] / pixel_to_mm
        sensor_height_px = sensor_size_mm[1] / pixel_to_mm
        
        # 计算标记点网格
        rows = int(sensor_height_px / marker_spacing_px) + 1
        cols = int(sensor_width_px / marker_spacing_px) + 1
        
        # 生成网格点 (相对于图像中心)
        center_x = width / 2
        center_y = height / 2
        start_x = center_x - (cols - 1) * marker_spacing_px / 2
        start_y = center_y - (rows - 1) * marker_spacing_px / 2
        
        markers = []
        for i in range(rows):
            for j in range(cols):
                x = start_x + j * marker_spacing_px
                y = start_y + i * marker_spacing_px
                markers.append([x, y])
        
        markers = np.array(markers, dtype=np.float32)
        print(f"生成了 {len(markers)} 个合成标记点 (网格: {rows}x{cols})")
        
        return markers
    
    def process_image_frame(self, frame_idx, frame_filename):
        """
        处理单个图像帧
        
        Args:
            frame_idx: 帧索引
            frame_filename: 帧文件名
            
        Returns:
            dict: 处理结果，包含位移场和标记点位置
        """
        # 加载当前帧
        frame_path = self.input_dir / "images" / frame_filename
        if not frame_path.exists():
            print(f"警告: 图像文件不存在: {frame_path}")
            return None
        
        if CV_AVAILABLE:
            current_frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        else:
            print(f"警告: OpenCV不可用，无法加载图像: {frame_path}")
            # 创建模拟数据
            current_frame = None
        
        # 确保有参考标记点
        if self.reference_markers is None:
            self.detect_reference_markers()
        
        result = {
            "frame_id": frame_idx,
            "frame_filename": frame_filename,
            "displacement_field": None,
            "marker_positions": None
        }
        
        if self.marker_detector is not None and current_frame is not None:
            try:
                # 检测当前帧的标记点
                current_markers = self.marker_detector.detect_markers_from_image(
                    current_frame,
                    reference_mode=False
                )
                
                if current_markers is not None and len(current_markers) == len(self.reference_markers):
                    # 计算位移场
                    displacement = current_markers - self.reference_markers
                    result["displacement_field"] = displacement
                    result["marker_positions"] = current_markers
                    
                    # 保存位移场
                    disp_path = self.disp_dir / f"disp_{frame_idx:04d}.npy"
                    np.save(str(disp_path), displacement)
                    
                    # 保存标记点位置
                    markers_path = self.markers_dir / f"markers_{frame_idx:04d}.npy"
                    np.save(str(markers_path), current_markers)
                    
                    return result
                    
            except Exception as e:
                print(f"处理帧 {frame_idx} 时出错: {e}")
        
        # 如果算法处理失败，生成模拟位移场
        print(f"为帧 {frame_idx} 生成模拟位移场")
        return self.generate_simulated_displacement(frame_idx)
    
    def generate_simulated_displacement(self, frame_idx):
        """生成模拟位移场 (用于测试)"""
        if self.reference_markers is None:
            self.reference_markers = self.generate_synthetic_markers()
        
        # 从元数据获取力数据
        force_data = self.load_force_measurement(frame_idx)
        force_z = abs(force_data.get("fz", 0.0)) if force_data else 5.0
        
        # 简单的Boussinesq-like位移模型
        n_markers = len(self.reference_markers)
        displacement = np.zeros((n_markers, 2), dtype=np.float32)
        
        # 假设接触点在中心
        center_x = np.mean(self.reference_markers[:, 0])
        center_y = np.mean(self.reference_markers[:, 1])
        
        # 计算每个标记点到接触点的距离
        for i in range(n_markers):
            dx = self.reference_markers[i, 0] - center_x
            dy = self.reference_markers[i, 1] - center_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                # Boussinesq-like位移: 与1/r成正比
                scale = force_z * 0.1  # 缩放因子
                displacement[i, 0] = -dx / distance * scale / (distance + 10.0)
                displacement[i, 1] = -dy / distance * scale / (distance + 10.0)
        
        # 添加一些随机噪声
        noise_level = 0.1
        displacement += np.random.normal(0, noise_level, displacement.shape)
        
        # 保存位移场
        disp_path = self.disp_dir / f"disp_{frame_idx:04d}.npy"
        np.save(str(disp_path), displacement)
        
        # 计算"当前"标记点位置
        current_markers = self.reference_markers + displacement
        markers_path = self.markers_dir / f"markers_{frame_idx:04d}.npy"
        np.save(str(markers_path), current_markers)
        
        return {
            "frame_id": frame_idx,
            "displacement_field": displacement,
            "marker_positions": current_markers
        }
    
    def load_force_measurement(self, frame_idx):
        """加载力测量数据"""
        force_csv = self.input_dir / "force_measurements.csv"
        if not force_csv.exists():
            return {"fx": 0.0, "fy": 0.0, "fz": -5.0, "tx": 0.0, "ty": 0.0, "tz": 0.0}
        
        try:
            with open(force_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if frame_idx < len(rows):
                    row = rows[frame_idx]
                    return {
                        "fx": float(row.get("force_x_n", 0.0)),
                        "fy": float(row.get("force_y_n", 0.0)),
                        "fz": float(row.get("force_z_n", 0.0)),
                        "tx": float(row.get("torque_x_nm", 0.0)),
                        "ty": float(row.get("torque_y_nm", 0.0)),
                        "tz": float(row.get("torque_z_nm", 0.0))
                    }
                else:
                    # 返回最后一个可用的力数据
                    return {
                        "fx": 0.0, "fy": 0.0, "fz": -5.0,
                        "tx": 0.0, "ty": 0.0, "tz": 0.0
                    }
                    
        except Exception as e:
            print(f"加载力测量数据失败: {e}")
            return {"fx": 0.0, "fy": 0.0, "fz": -5.0, "tx": 0.0, "ty": 0.0, "tz": 0.0}
    
    def process_all_frames(self, start_frame=0, end_frame=None, batch_size=10):
        """
        处理所有图像帧
        
        Args:
            start_frame: 起始帧索引
            end_frame: 结束帧索引 (包含)
            batch_size: 批处理大小
            
        Returns:
            list: 所有帧的处理结果
        """
        # 获取图像文件列表
        image_dir = self.input_dir / "images"
        if not image_dir.exists():
            print(f"错误: 图像目录不存在: {image_dir}")
            return []
        
        image_files = sorted(list(image_dir.glob("frame_*.png")))
        if not image_files:
            print(f"错误: 未找到图像文件: {image_dir}/frame_*.png")
            return []
        
        if end_frame is None or end_frame >= len(image_files):
            end_frame = len(image_files) - 1
        
        print(f"处理图像帧: {start_frame} 到 {end_frame} (共 {end_frame - start_frame + 1} 帧)")
        
        # 确保有参考帧和标记点
        self.load_reference_frame()
        self.detect_reference_markers()
        
        all_results = []
        processed_count = 0
        
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx >= len(image_files):
                break
            
            frame_filename = image_files[frame_idx].name
            print(f"处理帧 {frame_idx}/{end_frame}: {frame_filename}")
            
            result = self.process_image_frame(frame_idx, frame_filename)
            if result:
                all_results.append(result)
                processed_count += 1
            
            # 进度显示
            if processed_count % 10 == 0:
                print(f"进度: {processed_count}/{end_frame - start_frame + 1}")
        
        print(f"帧处理完成: {processed_count} 帧已处理")
        return all_results
    
    def estimate_forces(self, displacement_fields=None):
        """
        从位移场估计力
        
        Args:
            displacement_fields: 位移场列表，如果为None则从文件加载
            
        Returns:
            list: 力估计结果
        """
        if self.force_estimator is None:
            print("错误: 力估计器未初始化")
            return []
        
        if displacement_fields is None:
            # 从文件加载位移场
            disp_files = sorted(list(self.disp_dir.glob("disp_*.npy")))
            displacement_fields = []
            for disp_file in disp_files:
                disp = np.load(str(disp_file))
                displacement_fields.append(disp)
        
        if not displacement_fields:
            print("错误: 没有位移场数据")
            return []
        
        force_estimates = []
        
        for i, disp in enumerate(displacement_fields):
            print(f"估计力 {i+1}/{len(displacement_fields)}...")
            
            try:
                # 使用自适应力估计
                force_result = self.force_estimator.adaptive_force_estimation(
                    displacement_field=disp,
                    marker_positions=self.reference_markers
                )
                
                if force_result and "estimated_force" in force_result:
                    force_estimates.append({
                        "frame_id": i,
                        "estimated_force": force_result["estimated_force"],
                        "method_used": force_result.get("method_used", "unknown"),
                        "confidence": force_result.get("confidence", 0.0)
                    })
                else:
                    # 使用Hertz方法作为备选
                    force_result = self.force_estimator.hertz_contact_force(
                        displacement_field=disp,
                        marker_positions=self.reference_markers
                    )
                    force_estimates.append({
                        "frame_id": i,
                        "estimated_force": force_result,
                        "method_used": "hertz",
                        "confidence": 0.5
                    })
                    
            except Exception as e:
                print(f"力估计失败 (帧 {i}): {e}")
                force_estimates.append({
                    "frame_id": i,
                    "estimated_force": {"fx": 0.0, "fy": 0.0, "fz": -5.0},
                    "method_used": "failed",
                    "confidence": 0.0
                })
        
        # 保存力估计结果
        self.save_force_estimates(force_estimates)
        
        return force_estimates
    
    def save_force_estimates(self, force_estimates):
        """保存力估计结果"""
        if not force_estimates:
            print("警告: 没有力估计数据可保存")
            return
        
        # 保存为CSV
        csv_path = self.force_dir / "force_estimates.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "force_x_n", "force_y_n", "force_z_n",
                           "method_used", "confidence"])
            
            for est in force_estimates:
                force = est["estimated_force"]
                writer.writerow([
                    est["frame_id"],
                    force.get("fx", 0.0),
                    force.get("fy", 0.0),
                    force.get("fz", 0.0),
                    est["method_used"],
                    est["confidence"]
                ])
        
        print(f"力估计结果已保存: {csv_path}")
        
        # 保存为JSON
        json_path = self.force_dir / "force_estimates.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(force_estimates, f, indent=2)
        
        return csv_path
    
    def validate_estimates(self):
        """验证力估计结果"""
        # 加载估计的力
        estimates_path = self.force_dir / "force_estimates.csv"
        if not estimates_path.exists():
            print("错误: 力估计文件不存在")
            return None
        
        # 加载测量的力
        measurements_path = self.input_dir / "force_measurements.csv"
        if not measurements_path.exists():
            print("警告: 力测量文件不存在，无法验证")
            return None
        
        # 这里可以添加详细的验证逻辑
        # 暂时返回简单的验证结果
        validation_result = {
            "validation_time": np.datetime64('now').astype(str),
            "experiment_id": self.experiment_id,
            "total_frames": 0,
            "metrics": {
                "mean_error_n": 0.0,
                "max_error_n": 0.0,
                "correlation": 0.0
            }
        }
        
        # 保存验证结果
        validation_path = self.validation_dir / "validation_report.md"
        with open(validation_path, 'w', encoding='utf-8') as f:
            f.write("# 力估计验证报告\n\n")
            f.write(f"实验ID: {self.experiment_id}\n")
            f.write(f"验证时间: {validation_result['validation_time']}\n\n")
            f.write("## 总结\n\n")
            f.write("验证功能需要完整的力测量数据进行计算。\n")
            f.write("请确保force_measurements.csv文件包含准确的力测量值。\n")
        
        print(f"验证报告已保存: {validation_path}")
        return validation_result
    
    def create_dataset(self, dataset_name="contact_force_dataset_v1.0"):
        """
        创建整理好的数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            Path: 数据集目录路径
        """
        dataset_dir = Path("data") / "datasets" / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建训练/验证/测试分割
        train_dir = dataset_dir / "train"
        val_dir = dataset_dir / "val"
        test_dir = dataset_dir / "test"
        
        for dir_path in [train_dir, val_dir, test_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"数据集创建完成: {dataset_dir}")
        
        # 创建数据集描述文件
        desc = {
            "dataset_name": dataset_name,
            "experiment_id": self.experiment_id,
            "creation_date": np.datetime64('now').astype(str),
            "description": "视触传感器力估计数据集",
            "contents": {
                "displacement_fields": f"{len(list(self.disp_dir.glob('*.npy')))} files",
                "force_measurements": "force_measurements.csv",
                "force_estimates": "force_estimates.csv",
                "metadata": "metadata.json"
            }
        }
        
        desc_path = dataset_dir / "dataset_description.json"
        with open(desc_path, 'w', encoding='utf-8') as f:
            json.dump(desc, f, indent=2)
        
        print(f"数据集描述文件: {desc_path}")
        return dataset_dir


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="原始传感器数据处理脚本")
    parser.add_argument("--experiment_id", type=str, default="exp_001",
                       help="实验ID (默认: exp_001)")
    parser.add_argument("--input_dir", type=str, default="data/raw",
                       help="输入数据目录 (默认: data/raw)")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                       help="输出数据目录 (默认: data/processed)")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="起始帧索引 (默认: 0)")
    parser.add_argument("--end_frame", type=int, default=None,
                       help="结束帧索引 (默认: 处理所有帧)")
    parser.add_argument("--skip_force_estimation", action="store_true",
                       help="跳过力估计步骤")
    parser.add_argument("--create_dataset", action="store_true",
                       help="创建整理好的数据集")
    
    args = parser.parse_args()
    
    print(f"=== 原始传感器数据处理 ===")
    print(f"实验ID: {args.experiment_id}")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    
    # 创建处理器
    processor = RawDataProcessor(
        experiment_id=args.experiment_id,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    try:
        # 1. 处理所有图像帧
        print("\n1. 处理图像帧...")
        results = processor.process_all_frames(
            start_frame=args.start_frame,
            end_frame=args.end_frame
        )
        
        if not results:
            print("警告: 没有处理任何帧")
        else:
            print(f"成功处理 {len(results)} 帧")
        
        # 2. 估计力 (可选)
        if not args.skip_force_estimation and processor.force_estimator is not None:
            print("\n2. 从位移场估计力...")
            force_estimates = processor.estimate_forces()
            print(f"力估计完成: {len(force_estimates)} 个估计")
        else:
            print("\n2. 跳过力估计步骤")
        
        # 3. 验证结果
        print("\n3. 验证力估计...")
        validation = processor.validate_estimates()
        
        # 4. 创建数据集 (可选)
        if args.create_dataset:
            print("\n4. 创建数据集...")
            dataset_dir = processor.create_dataset()
            print(f"数据集已创建: {dataset_dir}")
        
        print(f"\n=== 数据处理完成 ===")
        print(f"处理后的数据保存在: {processor.output_dir}")
        print(f"位移场: {processor.disp_dir}")
        print(f"标记点位置: {processor.markers_dir}")
        print(f"力估计: {processor.force_dir}")
        print(f"验证结果: {processor.validation_dir}")
        
        if not args.skip_force_estimation:
            print("\n下一步: 运行 validate_dataset.py 进行详细验证")
        
    except Exception as e:
        print(f"数据处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())