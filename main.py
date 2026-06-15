import pandas as pd
import numpy as np
import json
import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.data_cleaning import DataCleaner
from src.metrics_calculator import MetricsCalculator
from src.analysis import BottleneckAnalyzer


def load_data(data_dir='sample_data'):
    required_files = {
        'gps': 'gps_data.csv',
        'stops': 'stop_data.csv',
        'swipe': 'swipe_data.csv',
        'schedule': 'schedule_data.csv',
        'congestion': 'congestion_data.csv',
        'complaint': 'complaint_data.csv',
        'weather': 'weather_data.csv'
    }
    data = {}
    for key, fname in required_files.items():
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            for col in ['timestamp', 'arrival_time', 'departure_time',
                        'scheduled_arrival', 'scheduled_departure',
                        'swipe_time', 'complaint_time', 'service_date', 'date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            data[key] = df
            print(f"  已加载 {fname}: {len(df)} 条记录")
        else:
            print(f"  警告: {fname} 不存在，使用空数据")
            data[key] = pd.DataFrame()
    return data


def export_results(results, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'analysis_report_{timestamp}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        def convert(o):
            if isinstance(o, (pd.Timestamp, datetime)):
                return o.isoformat()
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, pd.DataFrame):
                return o.to_dict(orient='records')
            if isinstance(o, pd.Series):
                return o.to_dict()
            return str(o)
        json.dump(results, f, ensure_ascii=False, indent=2, default=convert)
    print(f"\n完整分析报告已保存到: {report_path}")
    return report_path


def print_summary(results):
    print("\n" + "="*70)
    print("                   公交线路运行数据分析报告")
    print("="*70)
    print("\n【数据清洗报告】")
    cleaning = results.get('cleaning_report', {})
    for k, v in cleaning.items():
        print(f"  - {k}: {v}")
    print("\n" + "-"*70)
    for route_id, route_data in results.get('route_analysis', {}).items():
        print(f"\n{'='*70}")
        print(f"  线路: {route_id}")
        print(f"{'='*70}")
        print("\n【瓶颈站点 Top 5】")
        bottlenecks = route_data.get('bottleneck_stops', [])
        for i, bn in enumerate(bottlenecks[:5], 1):
            metrics_str = ', '.join([f"{k}={v}" for k, v in bn.get('metrics', {}).items()])
            print(f"  {i}. 站点 {bn.get('stop_id', '?')} (序号{bn.get('stop_sequence', '?')}) "
                  f"- 类型: {bn.get('bottleneck_type', '?')}, "
                  f"严重度: {round(bn.get('severity_score', 0), 1)}")
            print(f"     指标: {metrics_str}")
        print("\n【晚点原因拆分 (%)】")
        breakdown = route_data.get('delay_breakdown', {})
        pct = breakdown.get('percentage_breakdown', {})
        for cause, pct_val in sorted(pct.items(), key=lambda x: -x[1]):
            if pct_val > 0:
                print(f"  - {cause}: {pct_val}%")
        events = breakdown.get('event_counts', {})
        if events:
            print(f"  事件统计: {', '.join([f'{k}={v}' for k, v in events.items()])}")
        print("\n【调度建议】")
        suggestions = route_data.get('dispatch_suggestions', [])
        for i, sug in enumerate(suggestions, 1):
            priority_flag = '!!!' if sug.get('priority') == 'high' else ('!' if sug.get('priority') == 'medium' else '')
            print(f"  {priority_flag}[{sug.get('priority', '').upper()}] {sug.get('suggestion_type', '')}:")
            print(f"     {sug.get('action', '')}")
            details = sug.get('details', {})
            if details:
                detail_str = ', '.join([f"{k}={v}" for k, v in details.items()])
                print(f"     详情: {detail_str}")

        period_analysis = route_data.get('period_segment_comparison', {})
        if period_analysis and period_analysis.get('segments'):
            print("\n【时段分段对比】")
            period_names = period_analysis.get('period_names', {})
            ot_segments = period_analysis.get('segments', {}).get('on_time_rate', {})
            if ot_segments:
                print("  准点率:")
                for seg_key, seg_data in ot_segments.items():
                    name = period_names.get(seg_key, seg_key)
                    print(f"    {name}: {round(seg_data['on_time_rate']*100, 1)}% "
                          f"(平均偏差 {seg_data['avg_deviation_minutes']} 分钟)")
            load_segments = period_analysis.get('segments', {}).get('load_factor', {})
            if load_segments:
                print("  平均小时客流:")
                for seg_key, seg_data in load_segments.items():
                    name = period_names.get(seg_key, seg_key)
                    print(f"    {name}: {seg_data['avg_hourly_passengers']} 人次")
            key_findings = period_analysis.get('key_findings', [])
            if key_findings:
                print("  关键发现:")
                for finding in key_findings:
                    print(f"    ! {finding}")

        rainy_analysis = route_data.get('rainy_day_analysis', {})
        if rainy_analysis and rainy_analysis.get('has_data'):
            print("\n【雨天影响分析】")
            ist_comp = rainy_analysis.get('inter_stop_time_comparison', {})
            if ist_comp:
                print(f"  站间耗时: 晴天 {ist_comp.get('sunny_avg_sec', 0)}s → "
                      f"雨天 {ist_comp.get('rainy_avg_sec', 0)}s "
                      f"(+{ist_comp.get('avg_time_increase_pct', 0)}%)")
            load_comp = rainy_analysis.get('load_factor_comparison', {})
            if load_comp:
                print(f"  满载率: 晴天 {load_comp.get('sunny_avg_hourly', 0)} → "
                      f"雨天 {load_comp.get('rainy_avg_hourly', 0)} "
                      f"(+{load_comp.get('avg_load_increase_pct', 0)}%)")
            comp_comp = rainy_analysis.get('complaint_comparison', {})
            if comp_comp:
                print(f"  投诉量: 晴天 {comp_comp.get('sunny_avg_daily', 0)}/天 → "
                      f"雨天 {comp_comp.get('rainy_avg_daily', 0)}/天 "
                      f"(+{comp_comp.get('complaint_increase_pct', 0)}%)")
            top_rainy_complaints = rainy_analysis.get('top_rainy_complaint_types', [])
            if top_rainy_complaints:
                print(f"  雨天高发投诉类型: {', '.join(top_rainy_complaints)}")
            recommendations = rainy_analysis.get('recommendations', [])
            if recommendations:
                print("  建议:")
                for rec in recommendations:
                    print(f"    → {rec}")
            severity = rainy_analysis.get('summary', {}).get('severity', 'unknown')
            needs_extra = rainy_analysis.get('needs_rainy_extra_service', False)
            if needs_extra:
                print(f"  ⚠ 建议雨天临时加密班次 (影响程度: {severity})")
    if 'route_metrics' in results:
        rm = results['route_metrics']
        print("\n" + "-"*70)
        print("【整体运营指标】")
        if 'on_time_rate' in rm and 'route_summary' in rm['on_time_rate']:
            print("\n  线路准点率:")
            for _, row in rm['on_time_rate']['route_summary'].iterrows():
                print(f"    {row['route_id']} ({'上行' if row['direction']==0 else '下行'}): "
                      f"{round(row['on_time_rate']*100, 1)}% "
                      f"(平均偏差 {round(row['avg_deviation']/60, 1)} 分钟)")
        if 'headway' in rm and 'summary' in rm['headway']:
            hw = rm['headway']['summary']
            print(f"\n  发车间隔不规则度(CV>0.3的时段数): {len(hw[hw['headway_irregularity']>0.3])}")
        if 'peak_periods' in rm and not rm['peak_periods'].empty:
            pp = rm['peak_periods']
            print(f"\n  客流高峰时段数: {len(pp)}")
            for rid in pp['route_id'].unique():
                rid_pp = pp[pp['route_id'] == rid]
                wd = sorted(rid_pp[rid_pp['is_weekday'] == True]['hour'].unique().tolist())
                we = sorted(rid_pp[rid_pp['is_weekday'] == False]['hour'].unique().tolist())
                print(f"    {rid} - 工作日: {wd}, 周末: {we}")
        if 'congested_segments' in rm and not rm['congested_segments'].empty:
            cs = rm['congested_segments']
            congested = cs[cs['is_congested'] == True]
            print(f"\n  常发拥堵路段数: {congested['road_segment_id'].nunique()}")


def run_analysis(data_dir='sample_data', output_dir='output'):
    print("公交线路运行数据分析系统")
    print("="*50)
    print("\n[1/4] 加载数据...")
    raw_data = load_data(data_dir)
    print("\n[2/4] 数据清洗...")
    cleaner = DataCleaner()
    holidays = ['2025-10-01', '2025-10-02', '2025-10-03']
    cleaned = cleaner.clean_all(
        gps_data=raw_data.get('gps', pd.DataFrame()),
        stop_data=raw_data.get('stops', pd.DataFrame()),
        swipe_data=raw_data.get('swipe', pd.DataFrame()),
        schedule_data=raw_data.get('schedule', pd.DataFrame()),
        congestion_data=raw_data.get('congestion', pd.DataFrame()),
        complaint_data=raw_data.get('complaint', pd.DataFrame()),
        weather_data=raw_data.get('weather', pd.DataFrame()),
        holidays=holidays
    )
    print("\n[3/4] 计算指标...")
    calculator = MetricsCalculator(on_time_threshold_minutes=2)
    metrics = calculator.calculate_all(cleaned)
    print("\n[4/4] 瓶颈分析与调度建议...")
    analyzer = BottleneckAnalyzer()
    route_analysis = analyzer.analyze_all(metrics, cleaned)
    results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'cleaning_report': cleaned.get('report', {}),
        'route_metrics': {k: v for k, v in metrics.items() if k != 'delay_analysis'},
        'route_analysis': route_analysis
    }
    print_summary(results)
    export_results(results, output_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description='公交线路运行数据分析系统')
    parser.add_argument('--data-dir', type=str, default='sample_data',
                        help='输入数据目录 (默认: sample_data)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='输出结果目录 (默认: output)')
    parser.add_argument('--generate-sample', action='store_true',
                        help='先运行示例数据生成')
    args = parser.parse_args()
    if args.generate_sample:
        print("生成示例数据...")
        from generate_sample_data import generate_sample_data
        generate_sample_data(args.data_dir)
    run_analysis(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
