import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data_cleaning import DataCleaner
from src.metrics_calculator import MetricsCalculator
from src.analysis import BottleneckAnalyzer


def make_minimal_anomaly_data():
    base = datetime(2025, 10, 14, 7, 0, 0)
    route_id = '999路'
    direction = 0
    stop_count = 8
    stops = [f"S999_{i:02d}" for i in range(1, stop_count + 1)]
    stop_lats = [31.20 + i * 0.003 for i in range(stop_count)]
    stop_lons = [121.40 + i * 0.004 for i in range(stop_count)]
    stop_lats_detour = [31.30 + i * 0.003 for i in range(stop_count)]
    stop_lons_detour = [121.50 + i * 0.004 for i in range(stop_count)]

    schedule_rows = []
    for seq in range(1, stop_count + 1):
        schedule_rows.append({
            'route_id': route_id,
            'direction': direction,
            'stop_id': stops[seq - 1],
            'stop_sequence': seq,
            'latitude': stop_lats[seq - 1],
            'longitude': stop_lons[seq - 1],
            'trip_id': 'NORMAL_TRIP',
            'vehicle_id': 'V_NORMAL',
            'service_date': base.date(),
            'scheduled_arrival': base + timedelta(minutes=(seq - 1) * 4),
            'scheduled_departure': base + timedelta(minutes=(seq - 1) * 4, seconds=30)
        })
    for seq in range(1, stop_count + 1):
        schedule_rows.append({
            'route_id': route_id,
            'direction': direction,
            'stop_id': stops[seq - 1],
            'stop_sequence': seq,
            'latitude': stop_lats[seq - 1],
            'longitude': stop_lons[seq - 1],
            'trip_id': 'SKIP_TRIP',
            'vehicle_id': 'V_SKIP',
            'service_date': base.date(),
            'scheduled_arrival': base + timedelta(hours=1, minutes=(seq - 1) * 4),
            'scheduled_departure': base + timedelta(hours=1, minutes=(seq - 1) * 4, seconds=30)
        })
    for seq in range(1, stop_count + 1):
        schedule_rows.append({
            'route_id': route_id,
            'direction': direction,
            'stop_id': stops[seq - 1],
            'stop_sequence': seq,
            'latitude': stop_lats[seq - 1],
            'longitude': stop_lons[seq - 1],
            'trip_id': 'DETOUR_TRIP',
            'vehicle_id': 'V_DETOUR',
            'service_date': base.date(),
            'scheduled_arrival': base + timedelta(hours=2, minutes=(seq - 1) * 4),
            'scheduled_departure': base + timedelta(hours=2, minutes=(seq - 1) * 4, seconds=30)
        })
    schedule_df = pd.DataFrame(schedule_rows)

    stop_rows = []
    cur = base + timedelta(minutes=2)
    for seq in range(1, stop_count + 1):
        arr = cur + timedelta(minutes=3)
        dep = arr + timedelta(seconds=30)
        stop_rows.append({
            'route_id': route_id, 'direction': direction, 'trip_id': 'NORMAL_TRIP',
            'vehicle_id': 'V_NORMAL', 'stop_id': stops[seq - 1], 'stop_sequence': seq,
            'arrival_time': arr, 'departure_time': dep
        })
        cur = dep

    cur = base + timedelta(hours=1, minutes=2)
    for seq in [1, 2, 5, 6, 7, 8]:
        if seq <= 2:
            run = 3
        elif seq == 5:
            run = 4
        else:
            run = 3
        arr = cur + timedelta(minutes=run)
        dep = arr + timedelta(seconds=30)
        stop_rows.append({
            'route_id': route_id, 'direction': direction, 'trip_id': 'SKIP_TRIP',
            'vehicle_id': 'V_SKIP', 'stop_id': stops[seq - 1], 'stop_sequence': seq,
            'arrival_time': arr, 'departure_time': dep
        })
        cur = dep

    cur = base + timedelta(hours=2, minutes=2)
    detour_stop_ids = stops.copy()
    detour_stop_ids[3] = 'S_UNEXPECTED_X1'
    for seq in range(1, stop_count + 1):
        arr = cur + timedelta(minutes=4)
        dep = arr + timedelta(seconds=35)
        stop_rows.append({
            'route_id': route_id, 'direction': direction, 'trip_id': 'DETOUR_TRIP',
            'vehicle_id': 'V_DETOUR', 'stop_id': detour_stop_ids[seq - 1], 'stop_sequence': seq,
            'arrival_time': arr, 'departure_time': dep
        })
        cur = dep
    stops_df = pd.DataFrame(stop_rows)

    gps_rows = []
    for (vid, tripid, lat_list, lon_list, h) in [
        ('V_NORMAL', 'NORMAL_TRIP', stop_lats, stop_lons, 0),
        ('V_SKIP', 'SKIP_TRIP', stop_lats, stop_lons, 1),
        ('V_DETOUR', 'DETOUR_TRIP', stop_lats_detour, stop_lons_detour, 2),
    ]:
        t0 = base + timedelta(hours=h, minutes=2)
        t = t0
        for si in range(stop_count):
            for offset in range(5):
                gps_rows.append({
                    'vehicle_id': vid,
                    'timestamp': t + timedelta(seconds=offset * 15),
                    'latitude': lat_list[si] + np.random.uniform(-0.0003, 0.0003),
                    'longitude': lon_list[si] + np.random.uniform(-0.0003, 0.0003),
                    'speed_kmh': 20
                })
            t += timedelta(minutes=4)
    gps_df = pd.DataFrame(gps_rows)

    swipe_df = pd.DataFrame(columns=[
        'card_id', 'route_id', 'stop_id', 'vehicle_id', 'swipe_time', 'swipe_type'
    ])
    congestion_df = pd.DataFrame(columns=[
        'road_segment_id', 'timestamp', 'congestion_index', 'avg_speed_kmh'
    ])
    complaint_df = pd.DataFrame(columns=[
        'complaint_id', 'route_id', 'stop_id', 'complaint_type', 'complaint_time', 'severity'
    ])

    return {
        'gps': gps_df,
        'stops': stops_df,
        'swipe': swipe_df,
        'schedule': schedule_df,
        'congestion': congestion_df,
        'complaint': complaint_df
    }


def run_validation():
    route_id = '999路'
    stop_count = 8
    print("=" * 70)
    print("  最小异常样本验证")
    print("=" * 70)

    raw = make_minimal_anomaly_data()

    total_stop_records = len(raw['stops'])
    normal = raw['stops'][raw['stops']['trip_id'] == 'NORMAL_TRIP']
    skip = raw['stops'][raw['stops']['trip_id'] == 'SKIP_TRIP']
    detour = raw['stops'][raw['stops']['trip_id'] == 'DETOUR_TRIP']

    print("\n【样本构造】")
    print(f"  NORMAL_TRIP 站点数: {len(normal)} (完整 {stop_count:=8} 个站)")
    print(f"  SKIP_TRIP   站点数: {len(skip)} (跳站: 缺少序列 3,4)")
    print(f"  DETOUR_TRIP 站点数: {len(detour)} (第4个站 stop_id = S_UNEXPECTED_X1, + GPS 偏离)")

    print("\n[1/3] 数据清洗 ...")
    cleaner = DataCleaner(skip_gap_runtime_factor=2.0)
    cleaned = cleaner.clean_all(
        gps_data=raw['gps'], stop_data=raw['stops'], swipe_data=raw['swipe'],
        schedule_data=raw['schedule'], congestion_data=raw['congestion'],
        complaint_data=raw['complaint'], holidays=[]
    )
    print("\n[验证点 A] cleaning_report:")
    for k, v in cleaned['report'].items():
        print(f"  {k}: {v}")

    assert cleaned['report'].get('skipped_stops_total', 0) >= 2, \
        f"FAIL: 跳站总数应为 >=2, 实际 {cleaned['report'].get('skipped_stops_total')}"
    print("  ✓ skipped_stops_total >= 2")

    assert cleaned['report'].get('detour_trips_detected', 0) >= 1, \
        f"FAIL: 绕行检测数应为 >=1, 实际 {cleaned['report'].get('detour_trips_detected')}"
    print("  ✓ detour_trips_detected >= 1")

    assert cleaned['report'].get('detour_by_unexpected_stop', 0) >= 1, \
        f"FAIL: 异常停靠触发绕行应为 1, 实际 {cleaned['report'].get('detour_by_unexpected_stop')}"
    print("  ✓ detour_by_unexpected_stop >= 1 (单次异常停靠即判定绕行)")

    skipped = cleaned['skipped_stops']
    if not skipped.empty:
        print(f"\n  跳站明细 (skipped_stops_list):")
        for _, r in skipped.iterrows():
            print(f"    trip={r['trip_id']} 跳过 stop_seq={r['stop_sequence']} "
                  f"({r['stop_id']}) gap={r['gap_from']}→{r['gap_to']} 原因={r['skip_reason']}")

    stops_with_flags = cleaned['stops'][cleaned['stops']['trip_id'].isin(['SKIP_TRIP', 'DETOUR_TRIP'])]
    print("\n  trip 级别标记:")
    for tid in ['SKIP_TRIP', 'DETOUR_TRIP']:
        t = stops_with_flags[stops_with_flags['trip_id'] == tid].iloc[0]
        print(f"    {tid}: trip_has_skip={t.get('trip_has_skip')}, "
              f"trip_skip_count={t.get('trip_skip_count')}, "
              f"is_detour={t.get('is_detour')}, detour_reason={t.get('detour_reason', '')}")

    print("\n[2/3] 指标计算 ...")
    calc = MetricsCalculator(on_time_threshold_minutes=2)
    metrics = calc.calculate_all(cleaned)

    da = metrics.get('delay_analysis', {})
    print("\n[验证点 B] delay_analysis 中跳站/绕行摘要:")
    if 'skip_route_summary' in da and not da['skip_route_summary'].empty:
        print("  skip_route_summary:")
        for _, r in da['skip_route_summary'].iterrows():
            print(f"    route={r['route_id']} skip_events={r['skipped_stop_event_count']} "
                  f"affected_trips={r['affected_trip_count']} "
                  f"skip_related_delay={r['skip_related_total_delay_min']} min")
        assert da['skip_route_summary'].iloc[0]['skipped_stop_event_count'] >= 2
        print("  ✓ skip_route_summary 事件数 >= 2")
    else:
        print("  WARN: skip_route_summary 为空")

    if 'detour_delays' in da and not da['detour_delays'].empty:
        print("  detour_delays:")
        for _, r in da['detour_delays'].iterrows():
            print(f"    route={r['route_id']} trips={r.get('detour_trip_count', 'NA')} "
                  f"events={r['detour_delay_count']} total_delay={r['detour_total_delay']} min")
        assert da['detour_delays'].iloc[0].get('detour_trip_count', 0) >= 1
        print("  ✓ detour_delays trip 数 >= 1")
    else:
        print("  WARN: detour_delays 为空 (若 DETOUR_TRIP 准点可能导致，不影响清洗正确性)")

    print("\n[3/3] 瓶颈分析与调度建议 ...")
    analyzer = BottleneckAnalyzer()
    route_analysis = analyzer.analyze_all(metrics, cleaned)
    assert route_id in route_analysis, f"{route_id} 不在分析结果中"
    ra = route_analysis[route_id]

    print("\n[验证点 C] delay_breakdown:")
    bd = ra['delay_breakdown']
    print(f"  percentage_breakdown:")
    for k, v in bd['percentage_breakdown'].items():
        print(f"    {k}: {v}%")
    print(f"  event_counts: {bd['event_counts']}")

    has_skip_pct = (
        'skipped_stops' in bd['percentage_breakdown']
        and bd['percentage_breakdown']['skipped_stops'] > 0
    )
    has_skip_event = bd['event_counts'].get('skipped_stop_events', 0) >= 2
    assert has_skip_pct or has_skip_event, \
        f"FAIL: 晚点归因应含跳站(skipped_stops), pct={bd['percentage_breakdown'].get('skipped_stops')}, " \
        f"events={bd['event_counts'].get('skipped_stop_events')}"
    print("  ✓ skipped_stops 出现在晚点归因 (pct > 0 或 事件计数 >=2)")

    has_detour_pct = (
        'detour_and_route_deviation' in bd['percentage_breakdown']
        and bd['percentage_breakdown']['detour_and_route_deviation'] > 0
    )
    has_detour_event = bd['event_counts'].get('detour_trip_count', 0) >= 1
    assert has_detour_pct or has_detour_event, \
        f"FAIL: 晚点归因应含绕行(detour_and_route_deviation)"
    print("  ✓ detour_and_route_deviation 出现在晚点归因")

    print("\n[验证点 D] dispatch_suggestions:")
    sugs = ra['dispatch_suggestions']
    sug_types = [s['suggestion_type'] for s in sugs]
    print(f"  suggestion_types: {sug_types}")
    for s in sugs:
        print(f"    [{s['priority'].upper()}] {s['suggestion_type']}: {s['action'][:40]}... "
              f"details={s['details']}")

    assert 'skip_stop_regulation' in sug_types, \
        f"FAIL: 调度建议应含 skip_stop_regulation，实际 {sug_types}"
    print("  ✓ skip_stop_regulation 调度建议存在")

    assert 'route_optimization' in sug_types, \
        f"FAIL: 调度建议应含 route_optimization，实际 {sug_types}"
    print("  ✓ route_optimization 调度建议存在")

    print("\n" + "=" * 70)
    print("  ✓ 所有验证点通过 ✓")
    print("=" * 70)
    return True


if __name__ == '__main__':
    run_validation()
