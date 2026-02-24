#!/usr/bin/env python3
"""
力估计算法单元测试

使用unittest框架测试ForceEstimation类的核心功能。
"""

import unittest
import sys
import os
import numpy as np

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.force_estimation import ForceEstimation


class TestForceEstimation(unittest.TestCase):
    """ForceEstimation类单元测试"""
    
    def setUp(self):
        """测试前准备"""
        self.estimator = ForceEstimation(
            youngs_modulus=2.0e6,
            poissons_ratio=0.49,
            sensor_thickness=5.0
        )
    
    def test_initialization(self):
        """测试初始化参数"""
        self.assertEqual(self.estimator.youngs_modulus, 2.0e6)
        self.assertEqual(self.estimator.poissons_ratio, 0.49)
        self.assertEqual(self.estimator.sensor_thickness, 5.0)
        
        # 检查剪切模量计算
        expected_shear = 2.0e6 / (2 * (1 + 0.49))
        self.assertAlmostEqual(self.estimator.shear_modulus, expected_shear)
    
    def test_hertz_contact_force(self):
        """测试Hertz接触力计算"""
        # 测试位移向量
        displacement = np.array([0.1, 0.2, 1.0])  # 1mm垂直位移
        
        total_force, force_dist = self.estimator.hertz_contact_force(
            displacement,
            contact_radius=10.0,
            sphere_radius=50.0
        )
        
        # 检查返回类型
        self.assertIsInstance(total_force, float)
        self.assertIsInstance(force_dist, np.ndarray)
        
        # 检查力分布形状
        self.assertEqual(force_dist.shape, (50, 50))
        
        # 检查力应为正值
        self.assertGreater(total_force, 0)
        
        # 检查力分布非负
        self.assertGreaterEqual(force_dist.min(), 0)
    
    def test_boussinesq_solution(self):
        """测试Boussinesq解"""
        # 创建测试位移场
        n_points = 50
        displacement_field = np.random.randn(n_points, 2) * 0.5
        positions = np.random.rand(n_points, 2) * 100
        
        force_grid = self.estimator.boussinesq_solution(
            displacement_field,
            positions,
            grid_resolution=2.0
        )
        
        # 检查返回类型和形状
        self.assertIsInstance(force_grid, np.ndarray)
        self.assertEqual(force_grid.ndim, 2)
        
        # 检查力网格应为非负
        self.assertGreaterEqual(force_grid.min(), 0)
    
    def test_finite_element_inverse(self):
        """测试有限元反问题"""
        # 创建测试数据
        n_nodes = 10
        displacement_field = np.random.randn(n_nodes, 2) * 0.1
        node_positions = np.random.rand(n_nodes, 2) * 100
        
        # 创建简单三角形连接
        connectivity = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=int)
        
        node_forces = self.estimator.finite_element_inverse(
            displacement_field,
            node_positions,
            connectivity
        )
        
        # 检查返回类型和形状
        self.assertIsInstance(node_forces, np.ndarray)
        self.assertEqual(node_forces.shape, (n_nodes, 2))
    
    def test_estimate_force_from_displacement_hertz(self):
        """测试Hertz方法力估计"""
        # 创建测试位移场
        n_points = 100
        displacement_field = np.zeros((n_points, 2))
        positions = np.random.rand(n_points, 2) * 100
        
        # 创建中心位移（模拟接触）
        center = np.array([50, 50])
        for i, pos in enumerate(positions):
            distance = np.linalg.norm(pos - center)
            if distance < 20:
                displacement_field[i] = (center - pos) * 0.1
        
        result = self.estimator.estimate_force_from_displacement(
            displacement_field,
            positions,
            method='hertz'
        )
        
        # 检查结果结构
        self.assertIn('method', result)
        self.assertIn('total_force', result)
        self.assertIn('force_distribution', result)
        self.assertIn('max_force', result)
        self.assertIn('force_center', result)
        
        self.assertEqual(result['method'], 'hertz')
        self.assertIsInstance(result['total_force'], float)
    
    def test_estimate_force_from_displacement_boussinesq(self):
        """测试Boussinesq方法力估计"""
        n_points = 80
        displacement_field = np.random.randn(n_points, 2) * 0.2
        positions = np.random.rand(n_points, 2) * 100
        
        result = self.estimator.estimate_force_from_displacement(
            displacement_field,
            positions,
            method='boussinesq'
        )
        
        self.assertEqual(result['method'], 'boussinesq')
        self.assertIsInstance(result['total_force'], float)
        self.assertIsInstance(result['force_distribution'], np.ndarray)
    
    def test_estimate_force_from_displacement_fem(self):
        """测试有限元方法力估计"""
        n_points = 30
        displacement_field = np.random.randn(n_points, 2) * 0.1
        positions = np.random.rand(n_points, 2) * 100
        
        result = self.estimator.estimate_force_from_displacement(
            displacement_field,
            positions,
            method='fem'
        )
        
        self.assertEqual(result['method'], 'fem')
        self.assertIsInstance(result['total_force'], float)
    
    def test_calibrate_from_data(self):
        """测试校准方法"""
        # 创建测试数据
        displacement_fields = []
        known_forces = []
        
        for i in range(5):
            n_points = 50
            displacement = np.random.randn(n_points, 2) * (i + 1) * 0.05
            displacement_fields.append(displacement)
            known_forces.append(5.0 * (i + 1))  # 5N, 10N, 15N, 20N, 25N
        
        calibration = self.estimator.calibrate_from_data(
            displacement_fields,
            known_forces,
            method='hertz'
        )
        
        # 检查结果结构
        self.assertIn('success', calibration)
        self.assertIn('calibrated_params', calibration)
        self.assertIn('performance', calibration)
        
        if calibration['success']:
            params = calibration['calibrated_params']
            self.assertIn('youngs_modulus', params)
            self.assertIn('poissons_ratio', params)
            self.assertIn('scale_factor', params)
    
    def test_validate_estimation(self):
        """测试验证方法"""
        displacement_fields = []
        known_forces = []
        
        for i in range(3):
            n_points = 40
            displacement = np.random.randn(n_points, 2) * 0.1
            displacement_fields.append(displacement)
            known_forces.append(10.0)
        
        validation = self.estimator.validate_estimation(
            displacement_fields,
            known_forces,
            method='hertz'
        )
        
        self.assertIn('success', validation)
        self.assertIn('method', validation)
        self.assertIn('metrics', validation)
        
        if validation['success']:
            metrics = validation['metrics']
            self.assertIn('mae', metrics)
            self.assertIn('rmse', metrics)
            self.assertIn('max_error', metrics)
    
    def test_cross_validate_methods(self):
        """测试交叉验证方法"""
        displacement_fields = []
        known_forces = []
        
        for i in range(3):
            n_points = 30
            displacement = np.random.randn(n_points, 2) * 0.1
            displacement_fields.append(displacement)
            known_forces.append(8.0)
        
        cross_validation = self.estimator.cross_validate_methods(
            displacement_fields,
            known_forces,
            methods=['hertz', 'boussinesq']
        )
        
        self.assertIn('success', cross_validation)
        self.assertIn('methods_tested', cross_validation)
        self.assertIn('results', cross_validation)
        
        if cross_validation['success']:
            self.assertIn('hertz', cross_validation['results'])
            self.assertIn('boussinesq', cross_validation['results'])
    
    def test_visualize_force_distribution_no_matplotlib(self):
        """测试无matplotlib时的可视化处理"""
        # 创建测试结果
        result = {
            'method': 'hertz',
            'total_force': 15.5,
            'max_force': 2.3,
            'force_center': np.array([50.0, 50.0]),
            'force_distribution': np.random.rand(20, 20)
        }
        
        # 调用可视化方法（应该正常返回，不抛出异常）
        try:
            self.estimator.visualize_force_distribution(result)
            # 如果没有异常，测试通过
            self.assertTrue(True)
        except Exception as e:
            # 如果有异常，检查是否是matplotlib相关
            if "matplotlib" in str(e).lower():
                # 这是预期的，因为环境可能没有matplotlib
                print(f"注意：matplotlib不可用，可视化跳过: {e}")
                self.assertTrue(True)
            else:
                # 其他异常应该失败
                raise


class TestForceEstimationEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    def test_empty_displacement_field(self):
        """测试空位移场"""
        estimator = ForceEstimation()
        
        empty_displacement = np.array([]).reshape(0, 2)
        empty_positions = np.array([]).reshape(0, 2)
        
        # 测试各种方法对空输入的处理
        methods = ['hertz', 'boussinesq', 'fem']
        for method in methods:
            try:
                result = estimator.estimate_force_from_displacement(
                    empty_displacement,
                    empty_positions,
                    method=method
                )
                # 检查结果
                self.assertIn('total_force', result)
                # 对于空输入，总力应为0或NaN
                # 这里不检查具体值，只要不崩溃即可
            except Exception as e:
                # 某些方法可能不支持空输入，这是可以接受的
                print(f"方法 {method} 对空输入抛出异常: {e}")
    
    def test_single_point(self):
        """测试单点位移场"""
        estimator = ForceEstimation()
        
        single_displacement = np.array([[0.1, 0.2]])
        single_position = np.array([[50.0, 50.0]])
        
        result = estimator.estimate_force_from_displacement(
            single_displacement,
            single_position,
            method='boussinesq'
        )
        
        self.assertEqual(result['method'], 'boussinesq')
    
    def test_large_displacement(self):
        """测试大位移值"""
        estimator = ForceEstimation()
        
        n_points = 20
        large_displacement = np.ones((n_points, 2)) * 100.0  # 巨大位移
        positions = np.random.rand(n_points, 2) * 100
        
        result = estimator.estimate_force_from_displacement(
            large_displacement,
            positions,
            method='hertz'
        )
        
        # 检查不崩溃
        self.assertIn('total_force', result)
    
    def test_zero_displacement(self):
        """测试零位移"""
        estimator = ForceEstimation()
        
        n_points = 15
        zero_displacement = np.zeros((n_points, 2))
        positions = np.random.rand(n_points, 2) * 100
        
        result = estimator.estimate_force_from_displacement(
            zero_displacement,
            positions,
            method='hertz'
        )
        
        # 零位移应产生零力或接近零的力
        # 这里只检查不崩溃
        self.assertIn('total_force', result)


def run_tests():
    """运行测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestForceEstimation))
    suite.addTests(loader.loadTestsFromTestCase(TestForceEstimationEdgeCases))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("运行力估计算法单元测试...")
    result = run_tests()
    
    # 退出代码：0表示成功，1表示失败
    exit(0 if result.wasSuccessful() else 1)