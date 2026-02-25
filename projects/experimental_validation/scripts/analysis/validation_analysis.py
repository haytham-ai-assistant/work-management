#!/usr/bin/env python3
"""
Validation results analysis script.
Analyzes 64 test case validation results to identify patterns and optimization opportunities.
No pandas dependency - uses basic Python data structures.
"""

import json
import csv
import math
import statistics
from pathlib import Path
from collections import defaultdict

def load_validation_data():
    """Load validation results from CSV and JSON files."""
    results_dir = Path(__file__).parent.parent.parent / "results" / "force_estimation_validation"
    
    # Load CSV data
    csv_path = results_dir / "validation_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            numeric_fields = ['force_magnitude', 'youngs_modulus', 'noise_level', 
                             'ground_truth_force', 'estimated_force', 'absolute_error',
                             'relative_error', 'estimation_time']
            for field in numeric_fields:
                row[field] = float(row[field])
            data.append(row)
    
    # Load JSON data for additional metadata
    json_path = results_dir / "validation_results.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    else:
        json_data = None
    
    return data, json_data

def group_by(data, key_func):
    """Group data by a key function."""
    groups = defaultdict(list)
    for item in data:
        key = key_func(item)
        groups[key].append(item)
    return groups

def analyze_error_patterns(data):
    """Analyze error patterns across different conditions."""
    print("=" * 80)
    print("VALIDATION RESULTS ANALYSIS")
    print("=" * 80)
    
    # Group by method
    methods_group = group_by(data, lambda x: x['method'])
    
    # 1. Overall performance by method
    print("\n1. OVERALL PERFORMANCE BY METHOD:")
    print("-" * 40)
    
    for method in ['hertz', 'boussinesq', 'fem']:
        if method in methods_group:
            group_data = methods_group[method]
            rel_errors = [d['relative_error'] for d in group_data]
            abs_errors = [d['absolute_error'] for d in group_data]
            times = [d['estimation_time'] for d in group_data]
            
            mean_rel = statistics.mean(rel_errors) * 100
            std_rel = statistics.stdev(rel_errors) * 100 if len(rel_errors) > 1 else 0
            mean_abs = statistics.mean(abs_errors)
            mean_time = statistics.mean(times) * 1000
            
            print(f"\n{method.upper():12s} ({len(group_data)} cases):")
            print(f"  Relative error: {mean_rel:6.1f}% ± {std_rel:5.1f}%")
            print(f"  Absolute error: {mean_abs:6.2f} N")
            print(f"  Computation time: {mean_time:6.1f} ms")
    
    # 2. Performance by force magnitude
    print("\n2. PERFORMANCE BY FORCE MAGNITUDE:")
    print("-" * 40)
    
    force_group = group_by(data, lambda x: x['force_magnitude'])
    
    for force in sorted(force_group.keys()):
        group_data = force_group[force]
        print(f"\nForce: {force} N ({len(group_data)//3} test cases)")
        
        # Further group by method
        force_method_group = group_by(group_data, lambda x: x['method'])
        
        for method in ['hertz', 'boussinesq', 'fem']:
            if method in force_method_group:
                method_data = force_method_group[method]
                rel_errors = [d['relative_error'] for d in method_data]
                mean_rel = statistics.mean(rel_errors) * 100
                std_rel = statistics.stdev(rel_errors) * 100 if len(rel_errors) > 1 else 0
                mean_time = statistics.mean([d['estimation_time'] for d in method_data]) * 1000
                
                print(f"  {method:12s}: {mean_rel:6.1f}% ± {std_rel:5.1f}% error, {mean_time:6.1f} ms")
    
    # 3. Performance by noise level
    print("\n3. PERFORMANCE BY NOISE LEVEL:")
    print("-" * 40)
    
    noise_group = group_by(data, lambda x: x['noise_level'])
    
    for noise in sorted(noise_group.keys()):
        group_data = noise_group[noise]
        print(f"\nNoise level: {noise} ({len(group_data)//3} test cases)")
        
        # Further group by method
        noise_method_group = group_by(group_data, lambda x: x['method'])
        
        for method in ['hertz', 'boussinesq', 'fem']:
            if method in noise_method_group:
                method_data = noise_method_group[method]
                rel_errors = [d['relative_error'] for d in method_data]
                mean_rel = statistics.mean(rel_errors) * 100
                std_rel = statistics.stdev(rel_errors) * 100 if len(rel_errors) > 1 else 0
                
                print(f"  {method:12s}: {mean_rel:6.1f}% ± {std_rel:5.1f}% error")
    
    # 4. Performance by Young's modulus
    print("\n4. PERFORMANCE BY YOUNG'S MODULUS:")
    print("-" * 40)
    
    youngs_group = group_by(data, lambda x: x['youngs_modulus'])
    
    for youngs in sorted(youngs_group.keys()):
        group_data = youngs_group[youngs]
        youngs_gpa = youngs / 1e9
        print(f"\nYoung's modulus: {youngs_gpa:.1f} GPa ({len(group_data)//3} test cases)")
        
        # Further group by method
        youngs_method_group = group_by(group_data, lambda x: x['method'])
        
        for method in ['hertz', 'boussinesq', 'fem']:
            if method in youngs_method_group:
                method_data = youngs_method_group[method]
                rel_errors = [d['relative_error'] for d in method_data]
                mean_rel = statistics.mean(rel_errors) * 100
                std_rel = statistics.stdev(rel_errors) * 100 if len(rel_errors) > 1 else 0
                
                print(f"  {method:12s}: {mean_rel:6.1f}% ± {std_rel:5.1f}% error")
    
    # 5. Identify best-performing conditions for each method
    print("\n5. BEST-PERFORMING CONDITIONS FOR EACH METHOD:")
    print("-" * 40)
    
    for method in ['hertz', 'boussinesq', 'fem']:
        if method in methods_group:
            method_data = methods_group[method]
            
            # Find best and worst cases by relative error
            best_case = min(method_data, key=lambda x: x['relative_error'])
            worst_case = max(method_data, key=lambda x: x['relative_error'])
            
            print(f"\n{method.upper()} method:")
            print(f"  Best case: Force={best_case['force_magnitude']}N, "
                  f"E={best_case['youngs_modulus']/1e9:.1f}GPa, "
                  f"Noise={best_case['noise_level']}, "
                  f"Error={best_case['relative_error']*100:.1f}%")
            print(f"  Worst case: Force={worst_case['force_magnitude']}N, "
                  f"E={worst_case['youngs_modulus']/1e9:.1f}GPa, "
                  f"Noise={worst_case['noise_level']}, "
                  f"Error={worst_case['relative_error']*100:.1f}%")
    
    return data

def generate_optimization_recommendations(data):
    """Generate specific optimization recommendations based on analysis."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    # Group by method for analysis
    methods_group = group_by(data, lambda x: x['method'])
    
    # Hertz method analysis
    if 'hertz' in methods_group:
        hertz_data = methods_group['hertz']
        underestimation_count = sum(1 for d in hertz_data if d['estimated_force'] < d['ground_truth_force'])
        underestimation_pct = (underestimation_count / len(hertz_data)) * 100
        
        recommendations.append({
            'method': 'hertz',
            'issue': f"Underestimates force in {underestimation_pct:.1f}% of cases",
            'recommendation': 'Adjust contact radius estimation or material parameter scaling',
            'priority': 'medium'
        })
    
    # Boussinesq method analysis
    if 'boussinesq' in methods_group:
        boussinesq_data = methods_group['boussinesq']
        times = [d['estimation_time'] for d in boussinesq_data]
        mean_time = statistics.mean(times)
        slow_cases = sum(1 for t in times if t > 1.0)
        
        recommendations.append({
            'method': 'boussinesq',
            'issue': f"Very slow computation ({mean_time:.2f}s average, {slow_cases} cases >1s)",
            'recommendation': 'Optimize grid resolution and implement convergence acceleration',
            'priority': 'high'
        })
    
    # FEM method analysis
    if 'fem' in methods_group:
        fem_data = methods_group['fem']
        overestimation_count = sum(1 for d in fem_data if d['estimated_force'] > d['ground_truth_force'])
        overestimation_pct = (overestimation_count / len(fem_data)) * 100
        large_error_count = sum(1 for d in fem_data if d['relative_error'] > 5.0)  # >500% error
        
        recommendations.append({
            'method': 'fem',
            'issue': f"Severe overestimation ({overestimation_pct:.1f}% of cases, {large_error_count} cases >500% error)",
            'recommendation': 'Implement Tikhonov regularization and condition number monitoring',
            'priority': 'high'
        })
    
    # Noise sensitivity analysis
    noise_sensitivity = {}
    for method in ['hertz', 'boussinesq', 'fem']:
        if method in methods_group:
            method_data = methods_group[method]
            # Simple correlation calculation
            rel_errors = [d['relative_error'] for d in method_data]
            noise_levels = [d['noise_level'] for d in method_data]
            
            if len(method_data) > 1:
                # Calculate Pearson correlation coefficient
                mean_error = statistics.mean(rel_errors)
                mean_noise = statistics.mean(noise_levels)
                
                numerator = sum((e - mean_error) * (n - mean_noise) for e, n in zip(rel_errors, noise_levels))
                denom1 = math.sqrt(sum((e - mean_error)**2 for e in rel_errors))
                denom2 = math.sqrt(sum((n - mean_noise)**2 for n in noise_levels))
                
                if denom1 > 0 and denom2 > 0:
                    correlation = numerator / (denom1 * denom2)
                    noise_sensitivity[method] = abs(correlation)
    
    if noise_sensitivity:
        most_sensitive = max(noise_sensitivity, key=noise_sensitivity.get)
        recommendations.append({
            'method': 'all',
            'issue': f"Noise sensitivity highest for {most_sensitive} method (corr={noise_sensitivity[most_sensitive]:.3f})",
            'recommendation': 'Add displacement field smoothing/preprocessing for noise robustness',
            'priority': 'medium'
        })
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['method'].upper()} METHOD:")
        print(f"   Issue: {rec['issue']}")
        print(f"   Recommendation: {rec['recommendation']}")
        print(f"   Priority: {rec['priority']}")
    
    return recommendations

def save_analysis_report(data, recommendations, output_dir=None):
    """Save analysis report to file."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "results" / "force_estimation_validation"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "validation_analysis_report.md"
    
    # Calculate basic statistics
    force_values = list(set(d['force_magnitude'] for d in data))
    youngs_values = list(set(d['youngs_modulus'] for d in data))
    noise_values = list(set(d['noise_level'] for d in data))
    
    with open(report_path, 'w') as f:
        f.write("# Validation Results Analysis Report\n\n")
        from datetime import datetime
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write(f"- Total data points: {len(data)}\n")
        f.write(f"- Test cases: {len(data)//3}\n")
        f.write(f"- Force range: {min(force_values)} - {max(force_values)} N\n")
        f.write(f"- Young's modulus range: {min(youngs_values)/1e9:.1f} - {max(youngs_values)/1e9:.1f} GPa\n")
        f.write(f"- Noise levels: {', '.join(map(str, sorted(noise_values)))}\n\n")
        
        # Method performance
        f.write("## Method Performance\n\n")
        
        methods_group = group_by(data, lambda x: x['method'])
        for method in ['hertz', 'boussinesq', 'fem']:
            if method in methods_group:
                method_data = methods_group[method]
                rel_errors = [d['relative_error'] for d in method_data]
                times = [d['estimation_time'] for d in method_data]
                
                mean_error = statistics.mean(rel_errors) * 100
                std_error = statistics.stdev(rel_errors) * 100 if len(rel_errors) > 1 else 0
                mean_time = statistics.mean(times) * 1000
                
                f.write(f"### {method.capitalize()} Method\n")
                f.write(f"- Mean relative error: {mean_error:.1f}% ± {std_error:.1f}%\n")
                f.write(f"- Mean computation time: {mean_time:.1f} ms\n\n")
        
        # Recommendations
        f.write("## Optimization Recommendations\n\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. **{rec['method'].upper()}**: {rec['issue']}\n")
            f.write(f"   - **Recommendation**: {rec['recommendation']}\n")
            f.write(f"   - **Priority**: {rec['priority']}\n\n")
        
        # Detailed analysis tables
        f.write("## Detailed Analysis\n\n")
        
        # Force magnitude analysis
        f.write("### Performance by Force Magnitude\n\n")
        f.write("| Force (N) | Hertz Error (%) | Boussinesq Error (%) | FEM Error (%) |\n")
        f.write("|-----------|----------------|----------------------|---------------|\n")
        
        force_group = group_by(data, lambda x: x['force_magnitude'])
        for force in sorted(force_group.keys()):
            group_data = force_group[force]
            force_method_group = group_by(group_data, lambda x: x['method'])
            
            row = [f"{force}"]
            for method in ['hertz', 'boussinesq', 'fem']:
                if method in force_method_group:
                    method_data = force_method_group[method]
                    rel_errors = [d['relative_error'] for d in method_data]
                    mean_error = statistics.mean(rel_errors) * 100
                    row.append(f"{mean_error:.1f}")
                else:
                    row.append("N/A")
            
            f.write("| " + " | ".join(row) + " |\n")
        
        # Noise analysis
        f.write("\n### Performance by Noise Level\n\n")
        f.write("| Noise Level | Hertz Error (%) | Boussinesq Error (%) | FEM Error (%) |\n")
        f.write("|-------------|----------------|----------------------|---------------|\n")
        
        noise_group = group_by(data, lambda x: x['noise_level'])
        for noise in sorted(noise_group.keys()):
            group_data = noise_group[noise]
            noise_method_group = group_by(group_data, lambda x: x['method'])
            
            row = [f"{noise}"]
            for method in ['hertz', 'boussinesq', 'fem']:
                if method in noise_method_group:
                    method_data = noise_method_group[method]
                    rel_errors = [d['relative_error'] for d in method_data]
                    mean_error = statistics.mean(rel_errors) * 100
                    row.append(f"{mean_error:.1f}")
                else:
                    row.append("N/A")
            
            f.write("| " + " | ".join(row) + " |\n")
    
    print(f"\nAnalysis report saved to: {report_path}")
    return report_path

def main():
    """Main analysis function."""
    try:
        print("Loading validation data...")
        data, json_data = load_validation_data()
        
        print(f"Loaded {len(data)} data points ({len(data)//3} test cases)")
        
        # Perform analysis
        data = analyze_error_patterns(data)
        
        # Generate recommendations
        recommendations = generate_optimization_recommendations(data)
        
        # Save report
        report_path = save_analysis_report(data, recommendations)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        # Calculate overall statistics
        methods_group = group_by(data, lambda x: x['method'])
        hertz_error = statistics.mean([d['relative_error'] for d in methods_group.get('hertz', [])]) * 100
        boussinesq_error = statistics.mean([d['relative_error'] for d in methods_group.get('boussinesq', [])]) * 100
        fem_error = statistics.mean([d['relative_error'] for d in methods_group.get('fem', [])]) * 100
        
        print(f"\nKey findings:")
        print(f"1. Hertz method: {hertz_error:.1f}% average error")
        print(f"2. Boussinesq method: {boussinesq_error:.1f}% average error")
        print(f"3. FEM method: {fem_error:.1f}% average error")
        print(f"\nDetailed report: {report_path}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()