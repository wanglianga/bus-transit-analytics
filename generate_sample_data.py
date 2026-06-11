import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)


def generate_sample_data(output_dir='sample_data'):
    os.makedirs(output_dir, exist_ok=True)
    routes = ['101路', '202路']
    directions = [0, 1]
    route_config = {
        '101路': {
            'stop_count': 15,
            'base_lat': 31.23,
            'base_lon': 121.47,
            'vehicles': ['V10101', 'V10102', 'V10103', 'V10104']
        },
        '202路': {
            'stop_count': 12,
            'base_lat': 31.22,
            'base_lon': 121.50,
            'vehicles': ['V20201', 'V20202', 'V20203']
        }
    }
    stops_info = {}
    for route_id in routes:
        cfg = route_config[route_id]
        for direction in directions:
            key = (route_id, direction)
            stop_list = []
            for seq in range(1, cfg['stop_count'] + 1):
                factor = seq if direction == 0 else (cfg['stop_count'] - seq + 1)
                stop_list.append({
                    'route_id': route_id,
                    'direction': direction,
                    'stop_id': f"S{route_id[:3]}{direction}{seq:02d}",
                    'stop_sequence': seq,
                    'stop_name': f"{route_id}{'上行' if direction==0 else '下行'}第{seq}站",
                    'latitude': cfg['base_lat'] + factor * 0.003 + random.uniform(-0.0005, 0.0005),
                    'longitude': cfg['base_lon'] + factor * 0.004 + random.uniform(-0.0005, 0.0005)
                })
            stops_info[key] = pd.DataFrame(stop_list)
    gps_records = []
    stop_records = []
    swipe_records = []
    schedule_records = []
    congestion_records = []
    complaint_records = []
    base_date = datetime(2025, 10, 13)
    service_days = 7
    complaint_id = 1
    for day in range(service_days):
        current_date = base_date + timedelta(days=day)
        is_weekend = current_date.weekday() >= 5
        for route_id in routes:
            cfg = route_config[route_id]
            for direction in directions:
                stops_df = stops_info[(route_id, direction)]
                if is_weekend:
                    start_hours = [7, 8, 9, 10, 12, 14, 16, 18, 20]
                else:
                    start_hours = [6, 7, 7, 8, 8, 8, 9, 10, 12, 14, 16, 17, 17, 18, 18, 19, 20]
                for trip_idx, start_hour in enumerate(start_hours):
                    vehicle_id = random.choice(cfg['vehicles'])
                    trip_id = f"T{route_id[:3]}{direction}{day}{trip_idx:03d}"
                    start_time = current_date.replace(
                        hour=start_hour,
                        minute=random.randint(0, 59),
                        second=0
                    )
                    schedule_records.append({
                        'route_id': route_id,
                        'direction': direction,
                        'trip_id': trip_id,
                        'vehicle_id': vehicle_id,
                        'service_date': current_date.date(),
                        'schedule_type': 'holiday' if is_weekend else 'weekday'
                    })
                    current_time = start_time
                    gps_time = start_time
                    for _, stop_row in stops_df.iterrows():
                        run_seconds = random.randint(90, 240)
                        if 7 <= start_hour <= 9 or 17 <= start_hour <= 19:
                            run_seconds += random.randint(60, 180)
                        dwell_seconds = random.randint(15, 60)
                        if random.random() < 0.05:
                            dwell_seconds += random.randint(60, 180)
                        arrival_time = current_time + timedelta(seconds=run_seconds)
                        departure_time = arrival_time + timedelta(seconds=dwell_seconds)
                        if random.random() < 0.03 and _ > 0:
                            current_time = departure_time
                            continue
                        if random.random() < 0.97:
                            stop_records.append({
                                'route_id': route_id,
                                'direction': direction,
                                'trip_id': trip_id,
                                'vehicle_id': vehicle_id,
                                'stop_id': stop_row['stop_id'],
                                'stop_sequence': stop_row['stop_sequence'],
                                'arrival_time': arrival_time,
                                'departure_time': departure_time,
                                'is_interpolated': False
                            })
                            boarding = 0
                            if 7 <= start_hour <= 9 or 17 <= start_hour <= 19:
                                boarding = random.randint(10, 45)
                            else:
                                boarding = random.randint(2, 20)
                            for p in range(boarding):
                                swipe_time = arrival_time + timedelta(
                                    seconds=random.randint(5, dwell_seconds - 5)
                                )
                                swipe_records.append({
                                    'card_id': f"C{random.randint(10000, 99999)}",
                                    'route_id': route_id,
                                    'stop_id': stop_row['stop_id'],
                                    'vehicle_id': vehicle_id,
                                    'swipe_time': swipe_time,
                                    'swipe_type': 'boarding'
                                })
                            scheduled_arrival = arrival_time + timedelta(
                                seconds=random.randint(-120, 180)
                            )
                            schedule_records[-1].update({
                                f'stop_{stop_row["stop_sequence"]}_id': stop_row['stop_id'],
                                f'stop_{stop_row["stop_sequence"]}_arrival': scheduled_arrival
                            })
                            if 'scheduled_arrival' not in schedule_records[-1]:
                                schedule_records[-1]['scheduled_arrival'] = scheduled_arrival
                                schedule_records[-1]['stop_sequence'] = stop_row['stop_sequence']
                            schedule_records.append({
                                'route_id': route_id,
                                'direction': direction,
                                'trip_id': trip_id,
                                'vehicle_id': vehicle_id,
                                'service_date': current_date.date(),
                                'schedule_type': 'holiday' if is_weekend else 'weekday',
                                'stop_id': stop_row['stop_id'],
                                'stop_sequence': stop_row['stop_sequence'],
                                'scheduled_arrival': scheduled_arrival,
                                'scheduled_departure': scheduled_arrival + timedelta(seconds=dwell_seconds)
                            })
                        seg_start = current_time
                        while gps_time < departure_time:
                            lat = stop_row['latitude'] + random.uniform(-0.001, 0.001)
                            lon = stop_row['longitude'] + random.uniform(-0.001, 0.001)
                            if random.random() < 0.02:
                                lat += random.uniform(-0.01, 0.01)
                                lon += random.uniform(-0.01, 0.01)
                            gps_records.append({
                                'vehicle_id': vehicle_id,
                                'route_id': route_id,
                                'timestamp': gps_time,
                                'latitude': lat,
                                'longitude': lon,
                                'speed_kmh': random.uniform(5, 45)
                            })
                            gps_time += timedelta(seconds=30)
                        current_time = departure_time
    for day in range(service_days):
        current_date = base_date + timedelta(days=day)
        for hour in range(6, 22):
            for seg in range(1, 21):
                congestion = np.random.exponential(3)
                if hour in [7, 8, 9, 17, 18, 19]:
                    congestion += random.uniform(2, 5)
                congestion = min(congestion, 10)
                congestion_records.append({
                    'road_segment_id': f"SEG{seg:03d}",
                    'timestamp': current_date.replace(hour=hour, minute=random.randint(0, 59)),
                    'congestion_index': round(congestion, 1),
                    'avg_speed_kmh': max(5, 60 - congestion * 5)
                })
    complaint_types = [
        '晚点严重', '车辆拥挤', '越站不停', '司机服务态度',
        '车厢卫生', '空调故障', '停靠不规范', '绕行未告知'
    ]
    for _ in range(60):
        route_id = random.choice(routes)
        c_type = random.choice(complaint_types)
        day_offset = random.randint(0, service_days - 1)
        c_date = base_date + timedelta(days=day_offset)
        complaint_records.append({
            'complaint_id': f"CP{complaint_id:05d}",
            'route_id': route_id,
            'stop_id': random.choice(list(stops_info.keys())),
            'complaint_type': c_type,
            'complaint_time': c_date.replace(
                hour=random.choice([7, 8, 9, 17, 18, 19, 12]),
                minute=random.randint(0, 59)
            ),
            'description': f"乘客反映{c_type}",
            'severity': random.choice(['low', 'medium', 'high'])
        })
        complaint_id += 1
    for cr in complaint_records:
        if isinstance(cr['stop_id'], tuple):
            cr['stop_id'] = stops_info[cr['stop_id']]['stop_id'].iloc[0]
    gps_df = pd.DataFrame(gps_records)
    stops_df = pd.DataFrame(stop_records)
    swipe_df = pd.DataFrame(swipe_records)
    schedule_df = pd.DataFrame([r for r in schedule_records if 'stop_id' in r])
    congestion_df = pd.DataFrame(congestion_records)
    complaint_df = pd.DataFrame(complaint_records)
    gps_df.to_csv(f'{output_dir}/gps_data.csv', index=False)
    stops_df.to_csv(f'{output_dir}/stop_data.csv', index=False)
    swipe_df.to_csv(f'{output_dir}/swipe_data.csv', index=False)
    schedule_df.to_csv(f'{output_dir}/schedule_data.csv', index=False)
    congestion_df.to_csv(f'{output_dir}/congestion_data.csv', index=False)
    complaint_df.to_csv(f'{output_dir}/complaint_data.csv', index=False)
    print(f"数据生成完成，保存到 {output_dir}/ 目录")
    print(f"  GPS 轨迹: {len(gps_df)} 条")
    print(f"  站点记录: {len(stops_df)} 条")
    print(f"  刷卡记录: {len(swipe_df)} 条")
    print(f"  班次计划: {len(schedule_df)} 条")
    print(f"  拥堵指数: {len(congestion_df)} 条")
    print(f"  投诉记录: {len(complaint_df)} 条")
    return {
        'gps': gps_df,
        'stops': stops_df,
        'swipe': swipe_df,
        'schedule': schedule_df,
        'congestion': congestion_df,
        'complaint': complaint_df
    }


if __name__ == '__main__':
    generate_sample_data()
