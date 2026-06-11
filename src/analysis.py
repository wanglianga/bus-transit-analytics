import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class BottleneckAnalyzer:
    def __init__(self, on_time_rate_threshold=0.85, headway_cv_threshold=0.3):
        self.on_time_rate_threshold = on_time_rate_threshold
        self.headway_cv_threshold = headway_cv_threshold

    def analyze_all(self, metrics, cleaned_data):
        results = {}
        routes = self._get_all_routes(metrics)
        for route_id in routes:
            route_result = {}
            route_result['bottleneck_stops'] = self.identify_bottleneck_stops(
                metrics, route_id
            )
            route_result['delay_breakdown'] = self.breakdown_delay_causes(
                metrics, cleaned_data, route_id
            )
            route_result['dispatch_suggestions'] = self.generate_dispatch_suggestions(
                metrics, route_id
            )
            results[route_id] = route_result
        return results

    def _get_all_routes(self, metrics):
        routes = set()
        if 'on_time_rate' in metrics and 'route_summary' in metrics['on_time_rate']:
            routes.update(metrics['on_time_rate']['route_summary']['route_id'].unique())
        if 'headway' in metrics and 'summary' in metrics['headway']:
            routes.update(metrics['headway']['summary']['route_id'].unique())
        if 'inter_stop_time' in metrics and 'summary' in metrics['inter_stop_time']:
            routes.update(metrics['inter_stop_time']['summary']['route_id'].unique())
        return sorted(routes)

    def identify_bottleneck_stops(self, metrics, route_id):
        bottlenecks = []
        if 'on_time_rate' in metrics and 'stop_summary' in metrics['on_time_rate']:
            on_time_stops = metrics['on_time_rate']['stop_summary'][
                metrics['on_time_rate']['stop_summary']['route_id'] == route_id
            ].copy()
            low_ot = on_time_stops[
                on_time_stops['on_time_rate'] < self.on_time_rate_threshold
            ]
            for _, row in low_ot.iterrows():
                bottlenecks.append({
                    'stop_id': row['stop_id'],
                    'stop_sequence': int(row['stop_sequence']),
                    'bottleneck_type': 'low_on_time_rate',
                    'severity_score': (1 - row['on_time_rate']) * 100,
                    'metrics': {
                        'on_time_rate': round(row['on_time_rate'], 3),
                        'avg_delay_minutes': round(row['avg_deviation'] / 60, 1),
                        'late_count': int(row['late_count'])
                    }
                })
        if 'delay_analysis' in metrics and 'by_stop' in metrics['delay_analysis']:
            delay_stops = metrics['delay_analysis']['by_stop'][
                metrics['delay_analysis']['by_stop']['route_id'] == route_id
            ].copy()
            if not delay_stops.empty and delay_stops['total_delay'].sum() > 0:
                delay_stops['delay_share'] = (
                    delay_stops['total_delay'] / delay_stops['total_delay'].sum()
                )
                top_delays = delay_stops[delay_stops['delay_share'] >= 0.1].head(5)
                for _, row in top_delays.iterrows():
                    bottlenecks.append({
                        'stop_id': row['stop_id'],
                        'stop_sequence': int(row['stop_sequence']),
                        'bottleneck_type': 'high_delay_accumulation',
                        'severity_score': row['delay_share'] * 100,
                        'metrics': {
                            'delay_count': int(row['delay_count']),
                            'avg_delay_minutes': round(row['avg_delay'], 1),
                            'total_delay_minutes': round(row['total_delay'], 1),
                            'delay_share_pct': round(row['delay_share'] * 100, 1)
                        }
                    })
        if 'load_factor' in metrics and 'summary' in metrics['load_factor']:
            load_stops = metrics['load_factor']['summary'][
                metrics['load_factor']['summary']['route_id'] == route_id
            ].copy()
            overloaded = load_stops[load_stops['load_level'].isin(['high', 'overloaded'])]
            for _, row in overloaded.iterrows():
                bottlenecks.append({
                    'stop_id': row['stop_id'],
                    'stop_sequence': int(row['stop_sequence']),
                    'bottleneck_type': 'high_passenger_load',
                    'severity_score': min(row['avg_passengers'] / 2, 100),
                    'metrics': {
                        'hour': int(row['hour']),
                        'avg_passengers': round(row['avg_passengers'], 1),
                        'max_passengers': int(row['max_passengers']),
                        'load_level': row['load_level']
                    }
                })
        if 'inter_stop_time' in metrics and 'summary' in metrics['inter_stop_time']:
            ist = metrics['inter_stop_time']['summary'][
                metrics['inter_stop_time']['summary']['route_id'] == route_id
            ].copy()
            if not ist.empty and 'std_run_time' in ist.columns:
                ist['cv'] = ist['std_run_time'] / ist['avg_run_time'].replace(0, np.nan)
                high_cv = ist[ist['cv'] > 1.0].head(5)
                for _, row in high_cv.iterrows():
                    bottlenecks.append({
                        'stop_id': row['stop_id'],
                        'stop_sequence': int(row['stop_sequence']),
                        'segment': row['segment'],
                        'bottleneck_type': 'high_run_time_variability',
                        'severity_score': min(row['cv'] * 50, 100),
                        'metrics': {
                            'avg_run_time_sec': round(row['avg_run_time'], 1),
                            'p95_run_time_sec': round(row['p95_run_time'], 1),
                            'coefficient_of_variation': round(row['cv'], 2)
                        }
                    })
        return sorted(bottlenecks, key=lambda x: x['severity_score'], reverse=True)

    def breakdown_delay_causes(self, metrics, cleaned_data, route_id):
        breakdown = defaultdict(float)
        detail_counts = defaultdict(int)
        if 'delay_analysis' in metrics:
            da = metrics['delay_analysis']
            if 'by_hour' in da:
                hour_data = da['by_hour'][da['by_hour']['route_id'] == route_id]
                if not hour_data.empty:
                    total = hour_data['total_delay'].sum()
                    if total > 0:
                        morning = hour_data[
                            hour_data['hour'].between(7, 9)
                        ]['total_delay'].sum()
                        evening = hour_data[
                            hour_data['hour'].between(17, 19)
                        ]['total_delay'].sum()
                        breakdown['morning_peak_congestion'] = morning
                        breakdown['evening_peak_congestion'] = evening
                        breakdown['other_periods'] = total - morning - evening
                        detail_counts['total_delay_events'] = int(hour_data['delay_count'].sum())
            if 'detour_delays' in da:
                detour_data = da['detour_delays'][
                    da['detour_delays']['route_id'] == route_id
                ]
                if not detour_data.empty:
                    breakdown['detour_and_route_deviation'] = \
                        detour_data['detour_total_delay'].sum()
                    detail_counts['detour_events'] = int(
                        detour_data['detour_delay_count'].sum()
                    )
            if 'skipped_summary' in da:
                skip_data = da['skipped_summary'][
                    da['skipped_summary']['route_id'] == route_id
                ]
                if not skip_data.empty:
                    detail_counts['skipped_stop_events'] = int(skip_data['skip_count'].sum())
        if 'congested_segments' in metrics and not metrics['congested_segments'].empty:
            cs = metrics['congested_segments']
            congested = cs[cs['is_congested'] == True]
            if not congested.empty:
                breakdown['road_congestion'] = congested['avg_congestion'].sum() * 10
        if 'complaint_hotspots' in metrics and isinstance(metrics['complaint_hotspots'], dict):
            ch = metrics['complaint_hotspots']
            if 'by_route_type' in ch:
                complaints = ch['by_route_type'][
                    ch['by_route_type']['route_id'] == route_id
                ]
                if not complaints.empty:
                    detail_counts['complaints_total'] = int(complaints['complaint_count'].sum())
                    complaint_types = complaints['complaint_type'].unique()
                    if len(complaint_types) > 0:
                        breakdown['passenger_complaints'] = complaints['complaint_count'].sum() * 5
        if 'on_time_rate' in metrics and 'stop_summary' in metrics['on_time_rate']:
            os_df = metrics['on_time_rate']['stop_summary'][
                metrics['on_time_rate']['stop_summary']['route_id'] == route_id
            ]
            if not os_df.empty:
                high_late = os_df[os_df['avg_deviation'] > 120]
                if not high_late.empty:
                    breakdown['dwell_time_and_boarding'] = high_late['avg_deviation'].sum() / 60
        total = sum(breakdown.values())
        if total > 0:
            pct_breakdown = {k: round(v / total * 100, 1) for k, v in breakdown.items()}
        else:
            pct_breakdown = {k: 0.0 for k in breakdown.keys()}
        return {
            'percentage_breakdown': pct_breakdown,
            'absolute_values': {k: round(v, 1) for k, v in breakdown.items()},
            'event_counts': dict(detail_counts)
        }

    def generate_dispatch_suggestions(self, metrics, route_id):
        suggestions = []
        on_time_rate = None
        if 'on_time_rate' in metrics and 'route_summary' in metrics['on_time_rate']:
            ot_routes = metrics['on_time_rate']['route_summary'][
                metrics['on_time_rate']['route_summary']['route_id'] == route_id
            ]
            if not ot_routes.empty:
                on_time_rate = ot_routes.iloc[0]['on_time_rate']
        if 'headway' in metrics and 'summary' in metrics['headway']:
            hw = metrics['headway']['summary'][
                metrics['headway']['summary']['route_id'] == route_id
            ].copy()
            if not hw.empty:
                irregular = hw[hw['headway_irregularity'] > self.headway_cv_threshold]
                if not irregular.empty:
                    peak_hours = irregular[irregular['is_weekday'] == True]['hour'].unique()
                    suggestions.append({
                        'suggestion_type': 'headway_regularization',
                        'priority': 'high' if len(peak_hours) >= 2 else 'medium',
                        'action': '加强高峰时段发车频率控制，使用智能调度系统实时调整发车间隔',
                        'details': {
                            'affected_hours': sorted([int(h) for h in peak_hours])[:5],
                            'max_irregularity': round(
                                irregular['headway_irregularity'].max(), 2
                            ),
                            'avg_irregularity': round(
                                irregular['headway_irregularity'].mean(), 2
                            )
                        }
                    })
        if 'load_factor' in metrics and 'summary' in metrics['load_factor']:
            lf = metrics['load_factor']['summary'][
                metrics['load_factor']['summary']['route_id'] == route_id
            ].copy()
            if not lf.empty:
                overloaded = lf[lf['load_level'].isin(['high', 'overloaded'])]
                if not overloaded.empty:
                    peak_hours = sorted(overloaded['hour'].unique())
                    suggestions.append({
                        'suggestion_type': 'add_vehicles',
                        'priority': 'high',
                        'action': f"在客流高峰时段（{', '.join([f'{h}:00' for h in peak_hours[:3]])}）增加运力，建议加车 {max(1, len(peak_hours) // 2)} 辆",
                        'details': {
                            'peak_hours': [int(h) for h in peak_hours],
                            'max_avg_passengers': round(overloaded['avg_passengers'].max(), 1),
                            'affected_stops_count': overloaded['stop_id'].nunique()
                        }
                    })
        if 'delay_analysis' in metrics and 'detour_delays' in metrics['delay_analysis']:
            dd = metrics['delay_analysis']['detour_delays']
            if isinstance(dd, pd.DataFrame) and not dd.empty:
                detour_route = dd[dd['route_id'] == route_id]
                if not detour_route.empty and detour_route.iloc[0]['detour_delay_count'] > 5:
                    suggestions.append({
                        'suggestion_type': 'route_optimization',
                        'priority': 'medium',
                        'action': '分析绕行频发路段，评估临时绕行对准点率的影响，考虑常态化优化线路走向',
                        'details': {
                            'detour_event_count': int(detour_route.iloc[0]['detour_delay_count']),
                            'avg_detour_delay_min': round(
                                detour_route.iloc[0]['detour_avg_delay'], 1
                            )
                        }
                    })
        if on_time_rate is not None and on_time_rate < self.on_time_rate_threshold:
            suggestions.append({
                'suggestion_type': 'schedule_adjustment',
                'priority': 'high',
                'action': '重新评估班次计划运行时间，当前计划可能偏紧，建议增加站点间运行缓冲时间',
                'details': {
                    'current_on_time_rate': round(on_time_rate, 3),
                    'target_rate': self.on_time_rate_threshold
                }
            })
        if 'complaint_hotspots' in metrics and isinstance(metrics['complaint_hotspots'], dict):
            ch = metrics['complaint_hotspots']
            if 'by_route_type' in ch:
                cr = ch['by_route_type'][ch['by_route_type']['route_id'] == route_id]
                if not cr.empty and cr['complaint_count'].sum() >= 10:
                    top_complaint = cr.sort_values('complaint_count', ascending=False).iloc[0]
                    suggestions.append({
                        'suggestion_type': 'stop_optimization',
                        'priority': 'medium',
                        'action': f"针对'{top_complaint['complaint_type']}'类高发投诉，优化站点停靠策略或开展司机专项培训",
                        'details': {
                            'top_complaint_type': top_complaint['complaint_type'],
                            'top_complaint_count': int(top_complaint['complaint_count']),
                            'total_complaints': int(cr['complaint_count'].sum())
                        }
                    })
        if not suggestions:
            suggestions.append({
                'suggestion_type': 'maintain_current',
                'priority': 'low',
                'action': '当前线路运行状况良好，保持现有调度策略，持续监控关键指标',
                'details': {}
            })
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        return sorted(suggestions, key=lambda s: priority_order.get(s['priority'], 3))
