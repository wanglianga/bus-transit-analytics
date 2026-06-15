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
            route_result['period_segment_comparison'] = self.analyze_period_segment_comparison(
                metrics, route_id
            )
            route_result['rainy_day_analysis'] = self.analyze_rainy_day_impact(
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
            ls = metrics['load_factor']['summary']
            if 'route_id' in ls.columns:
                load_stops = ls[ls['route_id'] == route_id].copy()
            else:
                load_stops = pd.DataFrame()
            if not load_stops.empty:
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
                    detour_delay = float(detour_data['detour_total_delay'].sum())
                    detour_trips = int(detour_data.get('detour_trip_count', 0).sum())
                    detour_events = int(detour_data['detour_delay_count'].sum())
                    breakdown['detour_and_route_deviation'] = detour_delay + detour_trips * 10
                    detail_counts['detour_events'] = detour_events
                    detail_counts['detour_trip_count'] = detour_trips
            if 'skip_route_summary' in da:
                sr = da['skip_route_summary'][
                    da['skip_route_summary']['route_id'] == route_id
                ]
                if not sr.empty:
                    row = sr.iloc[0]
                    skip_delay = float(row.get('skip_related_total_delay_min', 0))
                    skip_events = int(row.get('skipped_stop_event_count', 0))
                    affected_trips = int(row.get('affected_trip_count', 0))
                    skip_weighted = skip_delay + skip_events * 15
                    breakdown['skipped_stops'] = skip_weighted
                    detail_counts['skipped_stop_events'] = skip_events
                    detail_counts['skipped_affected_trips'] = affected_trips
                    detail_counts['skip_related_delay_min'] = skip_delay
            if 'skipped_summary' in da and 'skipped_stop_events' not in detail_counts:
                skip_data = da['skipped_summary'][
                    da['skipped_summary']['route_id'] == route_id
                ]
                if not skip_data.empty:
                    cnt = int(skip_data['skip_count'].sum())
                    breakdown['skipped_stops'] = cnt * 15
                    detail_counts['skipped_stop_events'] = cnt
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
            ].copy() if 'route_id' in metrics['load_factor']['summary'].columns else pd.DataFrame()
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
                if not detour_route.empty:
                    r0 = detour_route.iloc[0]
                    trip_count = int(r0.get('detour_trip_count', 0))
                    delay_count = int(r0['detour_delay_count'])
                    if trip_count >= 1 or delay_count >= 1:
                        suggestions.append({
                            'suggestion_type': 'route_optimization',
                            'priority': 'high' if trip_count >= 3 else 'medium',
                            'action': '分析绕行频发路段，评估临时绕行对准点率的影响，考虑常态化优化线路走向',
                            'details': {
                                'detour_trip_count': trip_count,
                                'detour_delay_event_count': delay_count,
                                'avg_detour_delay_min': round(float(r0['detour_avg_delay']), 1)
                            }
                        })
        if 'delay_analysis' in metrics and 'skip_route_summary' in metrics['delay_analysis']:
            srs = metrics['delay_analysis']['skip_route_summary']
            if isinstance(srs, pd.DataFrame) and not srs.empty:
                skip_route = srs[srs['route_id'] == route_id]
                if not skip_route.empty:
                    r0 = skip_route.iloc[0]
                    skip_events = int(r0.get('skipped_stop_event_count', 0))
                    affected = int(r0.get('affected_trip_count', 0))
                    if skip_events >= 1:
                        suggestions.append({
                            'suggestion_type': 'skip_stop_regulation',
                            'priority': 'high' if skip_events >= 3 else 'medium',
                            'action': '跳站事件频发，需开展司机规范停靠培训并安装站点到离站自动上报系统',
                            'details': {
                                'skipped_stop_event_count': skip_events,
                                'affected_trip_count': affected,
                                'skip_related_delay_min': round(
                                    float(r0.get('skip_related_total_delay_min', 0)), 1
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
        if 'rainy_day_comparison' in metrics and metrics['rainy_day_comparison'].get('has_data', False):
            rainy = metrics['rainy_day_comparison']
            if 'inter_stop_time_diff' in rainy and not rainy['inter_stop_time_diff'].empty:
                ist_diff = rainy['inter_stop_time_diff'][
                    rainy['inter_stop_time_diff']['route_id'] == route_id
                ]
                if not ist_diff.empty:
                    max_time_increase = ist_diff['avg_time_increase_pct'].max()
                    if max_time_increase >= 15:
                        suggestions.append({
                            'suggestion_type': 'rainy_day_extra_service',
                            'priority': 'high' if max_time_increase >= 25 else 'medium',
                            'action': (
                                f"雨天站间耗时增加{round(max_time_increase, 1)}%，"
                                f"建议在雨天临时加密班次，缓解延误和拥挤"
                            ),
                            'details': {
                                'avg_time_increase_pct': round(max_time_increase, 1),
                                'recommendation': 'add_trips_on_rainy_days'
                            }
                        })
            if 'complaint_diff' in rainy and not rainy['complaint_diff'].empty:
                comp_diff = rainy['complaint_diff'][
                    rainy['complaint_diff']['route_id'] == route_id
                ]
                if not comp_diff.empty:
                    comp_increase = comp_diff.iloc[0].get('complaint_increase_pct', 0)
                    if comp_increase >= 30:
                        suggestions.append({
                            'suggestion_type': 'rainy_day_complaint_mitigation',
                            'priority': 'medium',
                            'action': (
                                f"雨天投诉量增加{round(comp_increase, 1)}%，"
                                f"建议加强雨天服务质量管控，增设雨天乘车指引"
                            ),
                            'details': {
                                'complaint_increase_pct': round(comp_increase, 1)
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

    def analyze_period_segment_comparison(self, metrics, route_id):
        result = {
            'segments': {},
            'key_findings': []
        }
        if 'period_segment_analysis' not in metrics:
            return result

        psa = metrics['period_segment_analysis']
        period_names = {
            'morning_peak': '早高峰(7-9点)',
            'evening_peak': '晚高峰(17-19点)',
            'off_peak': '平峰时段',
            'weekend': '周末'
        }

        if 'on_time_by_period' in psa and not psa['on_time_by_period'].empty:
            ot = psa['on_time_by_period'][
                psa['on_time_by_period']['route_id'] == route_id
            ]
            if not ot.empty:
                on_time_data = {}
                for _, row in ot.iterrows():
                    seg = row['period_segment']
                    on_time_data[seg] = {
                        'on_time_rate': round(row['on_time_rate'], 4),
                        'avg_deviation_minutes': round(row['avg_deviation_seconds'] / 60, 2),
                        'late_count': int(row['late_count']),
                        'total_count': int(row['total_count'])
                    }
                result['segments']['on_time_rate'] = on_time_data
                if 'morning_peak' in on_time_data and 'off_peak' in on_time_data:
                    morning_rate = on_time_data['morning_peak']['on_time_rate']
                    offpeak_rate = on_time_data['off_peak']['on_time_rate']
                    diff_pct = (offpeak_rate - morning_rate) * 100
                    if diff_pct > 5:
                        result['key_findings'].append(
                            f"早高峰准点率比平峰低{round(diff_pct, 1)}个百分点，通勤时段拥堵影响显著"
                        )
                if 'evening_peak' in on_time_data and 'off_peak' in on_time_data:
                    evening_rate = on_time_data['evening_peak']['on_time_rate']
                    offpeak_rate = on_time_data['off_peak']['on_time_rate']
                    diff_pct = (offpeak_rate - evening_rate) * 100
                    if diff_pct > 5:
                        result['key_findings'].append(
                            f"晚高峰准点率比平峰低{round(diff_pct, 1)}个百分点，晚高峰延误问题突出"
                        )

        if 'inter_stop_time_by_period' in psa and not psa['inter_stop_time_by_period'].empty:
            ist = psa['inter_stop_time_by_period'][
                psa['inter_stop_time_by_period']['route_id'] == route_id
            ]
            if not ist.empty:
                ist_data = {}
                for _, row in ist.iterrows():
                    seg = row['period_segment']
                    ist_data[seg] = {
                        'avg_run_time_sec': round(row['avg_run_time_sec'], 1),
                        'p95_run_time_sec': round(row['p95_run_time_sec'], 1),
                        'sample_count': int(row['sample_count'])
                    }
                result['segments']['inter_stop_time'] = ist_data

        if 'load_factor_by_period' in psa and not psa['load_factor_by_period'].empty:
            lf = psa['load_factor_by_period'][
                psa['load_factor_by_period']['route_id'] == route_id
            ]
            if not lf.empty:
                load_data = {}
                for _, row in lf.iterrows():
                    seg = row['period_segment']
                    load_data[seg] = {
                        'avg_hourly_passengers': round(row['avg_hourly_passengers'], 1),
                        'max_hourly_passengers': int(row['max_hourly_passengers']),
                        'total_passengers': int(row['total_passengers'])
                    }
                result['segments']['load_factor'] = load_data
                if 'morning_peak' in load_data and 'off_peak' in load_data:
                    morning_load = load_data['morning_peak']['avg_hourly_passengers']
                    offpeak_load = load_data['off_peak']['avg_hourly_passengers']
                    if offpeak_load > 0:
                        ratio = morning_load / offpeak_load
                        if ratio >= 1.5:
                            result['key_findings'].append(
                                f"早高峰客流是平峰的{round(ratio, 1)}倍，运力需求差异明显"
                            )

        if 'headway_by_period' in psa and not psa['headway_by_period'].empty:
            hw = psa['headway_by_period'][
                psa['headway_by_period']['route_id'] == route_id
            ]
            if not hw.empty:
                headway_data = {}
                for _, row in hw.iterrows():
                    seg = row['period_segment']
                    headway_data[seg] = {
                        'avg_headway_minutes': round(row['avg_headway_sec'] / 60, 2),
                        'headway_cv': round(row['headway_cv'], 3),
                        'sample_count': int(row['sample_count'])
                    }
                result['segments']['headway'] = headway_data

        result['period_names'] = period_names
        return result

    def analyze_rainy_day_impact(self, metrics, route_id):
        result = {
            'has_data': False,
            'inter_stop_time_comparison': {},
            'load_factor_comparison': {},
            'complaint_comparison': {},
            'congestion_comparison': {},
            'summary': {},
            'needs_rainy_extra_service': False,
            'recommendations': []
        }
        if 'rainy_day_comparison' not in metrics:
            return result
        rainy = metrics['rainy_day_comparison']
        if not rainy.get('has_data', False):
            return result

        result['has_data'] = True

        if 'inter_stop_time_diff' in rainy and not rainy['inter_stop_time_diff'].empty:
            ist_diff = rainy['inter_stop_time_diff'][
                rainy['inter_stop_time_diff']['route_id'] == route_id
            ]
            if not ist_diff.empty:
                row = ist_diff.iloc[0]
                result['inter_stop_time_comparison'] = {
                    'sunny_avg_sec': round(row['avg_run_time_sec_sunny'], 1),
                    'rainy_avg_sec': round(row['avg_run_time_sec_rainy'], 1),
                    'avg_time_increase_pct': round(row['avg_time_increase_pct'], 1),
                    'p95_time_increase_pct': round(row['p95_time_increase_pct'], 1)
                }
                time_increase = row['avg_time_increase_pct']
                if time_increase >= 20:
                    result['recommendations'].append(
                        '雨天运行时间显著增加，建议雨天加密班次20-30%'
                    )
                    result['needs_rainy_extra_service'] = True
                elif time_increase >= 10:
                    result['recommendations'].append(
                        '雨天运行时间有所增加，建议重点时段临时增派车辆'
                    )
                if time_increase >= 15:
                    result['summary']['severity'] = 'high'
                elif time_increase >= 8:
                    result['summary']['severity'] = 'medium'
                else:
                    result['summary']['severity'] = 'low'

        if 'load_factor_diff' in rainy and not rainy['load_factor_diff'].empty:
            load_diff = rainy['load_factor_diff'][
                rainy['load_factor_diff']['route_id'] == route_id
            ]
            if not load_diff.empty:
                row = load_diff.iloc[0]
                result['load_factor_comparison'] = {
                    'sunny_avg_hourly': round(row['avg_hourly_passengers_sunny'], 1),
                    'rainy_avg_hourly': round(row['avg_hourly_passengers_rainy'], 1),
                    'avg_load_increase_pct': round(row['avg_load_increase_pct'], 1)
                }
                load_increase = row['avg_load_increase_pct']
                if load_increase >= 15:
                    result['recommendations'].append(
                        '雨天客流明显增加，需增加运力应对满载率上升'
                    )
                    if load_increase >= 20:
                        result['needs_rainy_extra_service'] = True

        if 'complaint_diff' in rainy and not rainy['complaint_diff'].empty:
            comp_diff = rainy['complaint_diff'][
                rainy['complaint_diff']['route_id'] == route_id
            ]
            if not comp_diff.empty:
                row = comp_diff.iloc[0]
                result['complaint_comparison'] = {
                    'sunny_avg_daily': round(row['avg_daily_complaints_sunny'], 2),
                    'rainy_avg_daily': round(row['avg_daily_complaints_rainy'], 2),
                    'complaint_increase_pct': round(row['complaint_increase_pct'], 1)
                }
                comp_increase = row['complaint_increase_pct']
                if comp_increase >= 30:
                    result['recommendations'].append(
                        '雨天投诉显著增加，需加强服务质量和安全提醒'
                    )

        if 'complaint_type_by_weather' in rainy and not rainy['complaint_type_by_weather'].empty:
            ct = rainy['complaint_type_by_weather'][
                rainy['complaint_type_by_weather']['route_id'] == route_id
            ]
            if not ct.empty:
                rainy_types = ct[ct['is_rainy'] == True].sort_values(
                    'complaint_count', ascending=False
                )
                if not rainy_types.empty:
                    top_rainy_complaints = rainy_types.head(3)['complaint_type'].tolist()
                    result['top_rainy_complaint_types'] = top_rainy_complaints

        if 'congestion_by_weather' in rainy and not rainy['congestion_by_weather'].empty:
            cw = rainy['congestion_by_weather']
            sunny_cong = cw[cw['is_rainy'] == False]
            rainy_cong = cw[cw['is_rainy'] == True]
            if not sunny_cong.empty and not rainy_cong.empty:
                result['congestion_comparison'] = {
                    'sunny_avg_index': round(sunny_cong.iloc[0]['avg_congestion_index'], 2),
                    'rainy_avg_index': round(rainy_cong.iloc[0]['avg_congestion_index'], 2),
                    'sunny_severe_ratio': round(sunny_cong.iloc[0]['severe_ratio'] * 100, 1),
                    'rainy_severe_ratio': round(rainy_cong.iloc[0]['severe_ratio'] * 100, 1)
                }

        if not result['recommendations']:
            result['recommendations'].append(
                '雨天对运营影响较小，维持现有调度策略，持续关注天气变化'
            )

        return result
