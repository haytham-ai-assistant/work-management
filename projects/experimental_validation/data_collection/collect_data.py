#!/usr/bin/env python3
"""
视触传感器数据采集脚本模板

本脚本提供了数据采集的基本框架，用户需要根据实际硬件接口进行适配。
支持从相机和力传感器同步采集数据，并保存为标准化格式。

使用方法:
    python collect_data.py --experiment_id exp_001 --duration 5.0 --force_range 40.0

注意: 需要根据实际硬件修改以下部分:
    1. 相机接口 (OpenCV, FLIR, Basler等)
    2. 力传感器接口 (ATI, Serial, USB等)
    3. 同步机制 (硬件触发或软件同步)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

# 尝试导入可能需要的硬件库
try:
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    print("警告: OpenCV不可用，图像采集功能将受限")

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("警告: pyserial不可用，串口通信功能将受限")


class SensorDataCollector:
    """传感器数据采集器基类"""
    
    def __init__(self, experiment_id, output_dir="data/raw"):
        """
        初始化采集器
        
        Args:
            experiment_id: 实验ID (如 "exp_001")
            output_dir: 输出目录
        """
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir) / experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.image_dir = self.output_dir / "images"
        self.image_dir.mkdir(exist_ok=True)
        self.calibration_dir = self.output_dir / "calibration"
        self.calibration_dir.mkdir(exist_ok=True)
        
        # 初始化硬件状态
        self.camera_initialized = False
        self.force_sensor_initialized = False
        self.sync_initialized = False
        
        # 数据缓冲区
        self.image_buffer = []
        self.force_buffer = []
        self.timestamps = []
        
        # 元数据
        self.metadata = {
            "experiment_id": experiment_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "start_time": datetime.now().isoformat(),
            "data_acquisition": {
                "software_version": "1.0",
                "collection_script": "collect_data.py"
            }
        }
    
    def initialize_camera(self, **camera_params):
        """
        初始化相机
        
        Args:
            camera_params: 相机参数，如:
                - camera_id: 相机ID (默认0)
                - resolution: 分辨率 (默认[640, 480])
                - fps: 帧率 (默认30)
                - exposure: 曝光时间 (默认自动)
        
        Returns:
            bool: 初始化是否成功
        """
        # 这里需要根据实际相机SDK实现
        print(f"初始化相机: {camera_params}")
        
        if CV_AVAILABLE:
            # 示例: 使用OpenCV打开相机
            camera_id = camera_params.get("camera_id", 0)
            try:
                self.cap = cv2.VideoCapture(camera_id)
                if not self.cap.isOpened():
                    print(f"错误: 无法打开相机 {camera_id}")
                    return False
                
                # 设置相机参数
                width = camera_params.get("resolution", [640, 480])[0]
                height = camera_params.get("resolution", [640, 480])[1]
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                fps = camera_params.get("fps", 30)
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                
                self.camera_initialized = True
                self.metadata["data_acquisition"]["camera_params"] = camera_params
                print(f"相机初始化成功: {width}x{height} @ {fps}fps")
                return True
                
            except Exception as e:
                print(f"相机初始化失败: {e}")
                return False
        else:
            print("警告: OpenCV不可用，使用模拟相机模式")
            self.camera_initialized = True  # 模拟模式
            self.metadata["data_acquisition"]["camera_params"] = {
                **camera_params,
                "simulation_mode": True
            }
            return True
    
    def initialize_force_sensor(self, **force_params):
        """
        初始化力传感器
        
        Args:
            force_params: 力传感器参数，如:
                - port: 串口端口 (如 "COM3" 或 "/dev/ttyUSB0")
                - baudrate: 波特率 (默认115200)
                - sensor_type: 传感器类型 ("ATI_Nano17", "custom")
        
        Returns:
            bool: 初始化是否成功
        """
        print(f"初始化力传感器: {force_params}")
        
        sensor_type = force_params.get("sensor_type", "simulated")
        
        if sensor_type == "simulated":
            # 模拟力传感器 (用于测试)
            self.force_sensor_initialized = True
            self.force_sensor_type = "simulated"
            print("力传感器: 模拟模式")
            
        elif sensor_type == "ATI_Nano17" and SERIAL_AVAILABLE:
            # 这里需要根据ATI Nano17的实际协议实现
            port = force_params.get("port", "/dev/ttyUSB0")
            baudrate = force_params.get("baudrate", 115200)
            
            try:
                self.ser = serial.Serial(port, baudrate, timeout=1)
                self.force_sensor_initialized = True
                self.force_sensor_type = "ATI_Nano17"
                print(f"ATI Nano17初始化成功: {port} @ {baudrate}bps")
                
            except Exception as e:
                print(f"力传感器初始化失败: {e}")
                self.force_sensor_initialized = False
                
        else:
            print(f"警告: 不支持力传感器类型 '{sensor_type}' 或缺少依赖库")
            self.force_sensor_initialized = False
        
        if self.force_sensor_initialized:
            self.metadata["data_acquisition"]["force_sensor_params"] = force_params
            self.metadata["experiment_setup"] = {
                "force_sensor_type": sensor_type,
                "force_sensor_range_n": force_params.get("range", [0, 50])
            }
        
        return self.force_sensor_initialized
    
    def initialize_synchronization(self, sync_method="software"):
        """
        初始化同步机制
        
        Args:
            sync_method: 同步方法 ("hardware", "software", "none")
        
        Returns:
            bool: 初始化是否成功
        """
        print(f"初始化同步: {sync_method}")
        self.sync_method = sync_method
        self.sync_initialized = True
        self.metadata["data_acquisition"]["synchronization_method"] = sync_method
        return True
    
    def capture_reference_frame(self):
        """采集参考帧 (未加载状态)"""
        if not self.camera_initialized:
            print("错误: 相机未初始化")
            return None
        
        print("采集参考帧...")
        
        if CV_AVAILABLE and hasattr(self, 'cap'):
            ret, frame = self.cap.read()
            if ret:
                # 转换为灰度图
                if len(frame.shape) == 3:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame_gray = frame
                
                # 保存参考帧
                ref_path = self.calibration_dir / "reference_frame.png"
                cv2.imwrite(str(ref_path), frame_gray)
                print(f"参考帧已保存: {ref_path}")
                
                # 也保存到图像目录作为frame_0000
                frame_path = self.image_dir / "frame_0000.png"
                cv2.imwrite(str(frame_path), frame_gray)
                
                return frame_gray
        else:
            # 模拟模式: 生成合成图像
            print("模拟模式: 生成合成参考帧")
            # 这里可以调用算法库的合成图像生成函数
            ref_path = self.calibration_dir / "reference_frame.png"
            # 创建简单的测试图像
            img = np.ones((480, 640), dtype=np.uint8) * 128
            cv2.imwrite(str(ref_path), img) if CV_AVAILABLE else None
            
            frame_path = self.image_dir / "frame_0000.png"
            cv2.imwrite(str(frame_path), img) if CV_AVAILABLE else None
            
            return img
        
        return None
    
    def read_force_sensor(self):
        """
        读取力传感器数据
        
        Returns:
            dict: 力/力矩数据，或模拟数据
        """
        if not self.force_sensor_initialized:
            return {"fx": 0.0, "fy": 0.0, "fz": 0.0, 
                    "tx": 0.0, "ty": 0.0, "tz": 0.0}
        
        if self.force_sensor_type == "simulated":
            # 模拟力数据 (正弦波)
            t = time.time()
            import math
            force_z = 5.0 + 3.0 * math.sin(t * 2 * math.pi * 0.5)  # 0.5Hz正弦波
            return {
                "fx": 0.1 * math.sin(t * 2 * math.pi * 1.0),
                "fy": 0.1 * math.cos(t * 2 * math.pi * 1.0),
                "fz": -force_z,  # 负值表示压向传感器
                "tx": 0.001, "ty": 0.002, "tz": 0.003
            }
        
        elif self.force_sensor_type == "ATI_Nano17":
            # 这里需要根据ATI的实际协议实现
            if SERIAL_AVAILABLE and hasattr(self, 'ser'):
                try:
                    # 示例: 读取一行数据并解析
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        # 假设格式: "Fx,Fy,Fz,Tx,Ty,Tz"
                        values = line.split(',')
                        if len(values) >= 6:
                            return {
                                "fx": float(values[0]), "fy": float(values[1]), 
                                "fz": float(values[2]),
                                "tx": float(values[3]), "ty": float(values[4]), 
                                "tz": float(values[5])
                            }
                except Exception as e:
                    print(f"读取力传感器数据失败: {e}")
        
        return {"fx": 0.0, "fy": 0.0, "fz": 0.0, 
                "tx": 0.0, "ty": 0.0, "tz": 0.0}
    
    def collect_data(self, duration_seconds=5.0, condition_id="cond_001", 
                     description="默认测试条件"):
        """
        采集数据
        
        Args:
            duration_seconds: 采集持续时间(秒)
            condition_id: 测试条件ID
            description: 测试条件描述
        
        Returns:
            bool: 采集是否成功
        """
        if not (self.camera_initialized and self.force_sensor_initialized):
            print("错误: 传感器未完全初始化")
            return False
        
        print(f"开始数据采集: {duration_seconds}秒, 条件: {condition_id}")
        
        # 重置缓冲区
        self.image_buffer = []
        self.force_buffer = []
        self.timestamps = []
        
        # 添加测试条件到元数据
        if "test_conditions" not in self.metadata:
            self.metadata["test_conditions"] = []
        
        self.metadata["test_conditions"].append({
            "condition_id": condition_id,
            "description": description,
            "duration_s": duration_seconds,
            "start_time": datetime.now().isoformat()
        })
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                frame_time = time.time()
                
                # 1. 采集图像
                if CV_AVAILABLE and hasattr(self, 'cap'):
                    ret, frame = self.cap.read()
                    if ret:
                        if len(frame.shape) == 3:
                            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        else:
                            frame_gray = frame
                        
                        # 保存图像
                        frame_filename = f"frame_{frame_count:04d}.png"
                        frame_path = self.image_dir / frame_filename
                        cv2.imwrite(str(frame_path), frame_gray)
                        
                        self.image_buffer.append(frame_filename)
                    else:
                        print(f"警告: 第{frame_count}帧采集失败")
                        frame_gray = None
                else:
                    # 模拟模式
                    frame_gray = None
                    frame_filename = f"frame_{frame_count:04d}.png"
                    self.image_buffer.append(frame_filename)
                
                # 2. 读取力传感器
                force_data = self.read_force_sensor()
                
                # 3. 记录时间戳
                timestamp_data = {
                    "frame_id": frame_count,
                    "image_timestamp_s": frame_time - start_time,
                    "force_timestamp_s": frame_time - start_time,
                    "sync_offset_s": 0.0  # 假设完美同步
                }
                
                # 4. 保存到缓冲区
                self.force_buffer.append(force_data)
                self.timestamps.append(timestamp_data)
                
                frame_count += 1
                
                # 控制采集频率
                time.sleep(0.033)  # 约30fps
            
            print(f"数据采集完成: {frame_count}帧")
            return True
            
        except KeyboardInterrupt:
            print("\n数据采集被用户中断")
            return False
        except Exception as e:
            print(f"数据采集出错: {e}")
            return False
    
    def save_metadata(self):
        """保存元数据文件"""
        # 更新结束时间
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["total_frames"] = len(self.image_buffer)
        
        # 保存元数据
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        print(f"元数据已保存: {metadata_path}")
        return metadata_path
    
    def save_force_data(self):
        """保存力数据为CSV格式"""
        if not self.force_buffer:
            print("警告: 力数据为空")
            return None
        
        import csv
        
        csv_path = self.output_dir / "force_measurements.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入标题
            writer.writerow(["frame_id", "timestamp_s", "force_x_n", "force_y_n", 
                           "force_z_n", "torque_x_nm", "torque_y_nm", "torque_z_nm", 
                           "condition_id"])
            
            # 写入数据
            for i, force_data in enumerate(self.force_buffer):
                # 使用最后一个条件ID
                condition_id = "cond_001"
                if self.metadata.get("test_conditions"):
                    condition_id = self.metadata["test_conditions"][-1]["condition_id"]
                
                timestamp = self.timestamps[i]["image_timestamp_s"] if i < len(self.timestamps) else i * 0.033
                
                writer.writerow([
                    i, timestamp,
                    force_data.get("fx", 0.0), force_data.get("fy", 0.0), 
                    force_data.get("fz", 0.0),
                    force_data.get("tx", 0.0), force_data.get("ty", 0.0), 
                    force_data.get("tz", 0.0),
                    condition_id
                ])
        
        print(f"力数据已保存: {csv_path}")
        return csv_path
    
    def save_timestamps(self):
        """保存时间戳数据"""
        if not self.timestamps:
            print("警告: 时间戳数据为空")
            return None
        
        import csv
        
        csv_path = self.output_dir / "timestamps.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入标题
            writer.writerow(["frame_id", "image_timestamp_s", "force_timestamp_s", 
                           "sync_offset_s"])
            
            # 写入数据
            for ts in self.timestamps:
                writer.writerow([
                    ts["frame_id"],
                    ts["image_timestamp_s"],
                    ts["force_timestamp_s"],
                    ts["sync_offset_s"]
                ])
        
        print(f"时间戳已保存: {csv_path}")
        return csv_path
    
    def cleanup(self):
        """清理资源"""
        if CV_AVAILABLE and hasattr(self, 'cap'):
            self.cap.release()
            print("相机资源已释放")
        
        if SERIAL_AVAILABLE and hasattr(self, 'ser'):
            self.ser.close()
            print("串口资源已释放")
        
        print("数据采集器清理完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="视触传感器数据采集脚本")
    parser.add_argument("--experiment_id", type=str, default="exp_001",
                       help="实验ID (默认: exp_001)")
    parser.add_argument("--duration", type=float, default=5.0,
                       help="采集持续时间(秒) (默认: 5.0)")
    parser.add_argument("--force_range", type=float, default=40.0,
                       help="模拟力传感器范围(N) (默认: 40.0)")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                       help="输出目录 (默认: data/raw)")
    parser.add_argument("--camera_id", type=int, default=0,
                       help="相机ID (默认: 0)")
    parser.add_argument("--resolution", type=str, default="640,480",
                       help="相机分辨率 (默认: 640,480)")
    parser.add_argument("--fps", type=int, default=30,
                       help="帧率 (默认: 30)")
    
    args = parser.parse_args()
    
    # 解析分辨率
    resolution = [int(x) for x in args.resolution.split(',')]
    
    print(f"=== 视触传感器数据采集 ===")
    print(f"实验ID: {args.experiment_id}")
    print(f"持续时间: {args.duration}秒")
    print(f"输出目录: {args.output_dir}")
    print(f"相机分辨率: {resolution[0]}x{resolution[1]} @ {args.fps}fps")
    
    # 创建采集器
    collector = SensorDataCollector(args.experiment_id, args.output_dir)
    
    try:
        # 初始化相机
        camera_params = {
            "camera_id": args.camera_id,
            "resolution": resolution,
            "fps": args.fps
        }
        if not collector.initialize_camera(**camera_params):
            print("警告: 相机初始化失败，使用模拟模式")
        
        # 初始化力传感器 (模拟模式)
        force_params = {
            "sensor_type": "simulated",
            "range": [0, args.force_range]
        }
        if not collector.initialize_force_sensor(**force_params):
            print("警告: 力传感器初始化失败")
        
        # 初始化同步
        collector.initialize_synchronization(sync_method="software")
        
        # 采集参考帧
        collector.capture_reference_frame()
        
        # 采集数据
        success = collector.collect_data(
            duration_seconds=args.duration,
            condition_id="cond_001",
            description=f"{args.duration}秒数据采集测试"
        )
        
        if success:
            # 保存所有数据
            collector.save_metadata()
            collector.save_force_data()
            collector.save_timestamps()
            
            print(f"\n=== 数据采集完成 ===")
            print(f"实验数据保存在: {collector.output_dir}")
            print(f"帧数: {len(collector.image_buffer)}")
            print(f"下一步: 运行 process_raw_data.py 处理原始数据")
        else:
            print("数据采集失败")
        
    except KeyboardInterrupt:
        print("\n数据采集被用户中断")
    except Exception as e:
        print(f"数据采集过程中发生错误: {e}")
    finally:
        collector.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())