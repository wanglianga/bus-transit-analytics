import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import math
warnings.filterwarnings('ignore')

try:
    from geopy.distance import geodesic
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _distance_km(p1, p2):
    if HAS_GEOPY:
        return geodesic(p1, p2).km
    return _haversine_km(p1[0], p1[1], p2[0], p2[1])


def _distance_meters(p1, p2):
    return _distance_km(p1, p2) * 1000


class DataCleaner:
    def __init__(self, gps_radius_threshold=50, speed_threshold=80):
        self.gps_radius_threshold = gps_radius_threshold
        self.speed_threshold = speed_threshold
        self.cleaning_report = {}

    def clean_all(self, gps_data, stop_data, swipe_data, schedule_data,
                  congestion_data, complaint_data, holidays=None):
        self.cleaning_report = {}
        gps_cleaned = self.clean_gps_drift(gps_data)
        stop_cleaned = self.clean_stop_records(stop_data, gps_cleaned)
        stop_fixed = self.fix_missing_stops(stop_cleaned, schedule_data)
        stop_fixed = self.detect_skipped_stops(stop_fixed, schedule_data)
        stop_fixed = self.detect_detour(stop_fixed, gps_cleaned, schedule_data)
        schedule_adjusted = self.adjust_holiday_schedule(schedule_data, holidays)
        swipe_cleaned = self.clean_swipe_data(swipe_data)
        congestion_cleaned = self.clean_congestion_data(congestion_data)
        complaint_cleaned = self.clean_complaint_data(complaint_data)
        return {
            'gps': gps_cleaned,
            'stops': stop_fixed,
            'swipe': swipe_cleaned,
            'schedule': schedule_adjusted,
            'congestion': congestion_cleaned,
            'complaint': complaint_cleaned,
            'report': self.cleaning_report
        }

    def clean_gps_drift(self, gps_data):
        df = gps_data.copy()
        original_count = len(df)
        df = df.sort_values(['vehicle_id', 'timestamp']).reset_index(drop=True)
        df['prev_lat'] = df.groupby('vehicle_id')['latitude'].shift(1)
        df['prev_lon'] = df.groupby('vehicle_id')['longitude'].shift(1)
        df['prev_ts'] = df.groupby('vehicle_id')['timestamp'].shift(1)
        def calc_speed(row):
            if pd.isna(row['prev_lat']) or pd.isna(row['prev_ts']):
                return 0
            dist = _distance_km(
                (row['prev_lat'], row['prev_lon']),
                (row['latitude'], row['longitude'])
            )
            time_diff = (row['timestamp'] - row['prev_ts']).total_seconds() / 3600
            return dist / time_diff if time_diff > 0 else 999
        df['speed_kmh'] = df.apply(calc_speed, axis=1)
        df['distance_from_prev'] = df.apply(
            lambda r: _distance_meters(
                (r['prev_lat'], r['prev_lon']),
                (r['latitude'], r['longitude'])
            ) if not pd.isna(r['prev_lat']) else 0, axis=1
        )
        drift_mask = (df['speed_kmh'] > self.speed_threshold) | \
                     (df['distance_from_prev'] > self.gps_radius_threshold * 10)
        df.loc[drift_mask, ['latitude', 'longitude']] = np.nan
        df['latitude'] = df.groupby('vehicle_id')['latitude'].transform(
            lambda s: s.interpolate(method='linear')
        )
        df['longitude'] = df.groupby('vehicle_id')['longitude'].transform(
            lambda s: s.interpolate(method='linear')
        )
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df.drop(columns=['prev_lat', 'prev_lon', 'prev_ts', 'speed_kmh', 'distance_from_prev'])
        self.cleaning_report['gps_drift_removed'] = original_count - len(df)
        return df.reset_index(drop=True)

    def clean_stop_records(self, stop_data, gps_data):
        df = stop_data.copy()
        original_count = len(df)
        df = df.dropna(subset=['stop_id', 'vehicle_id'])
        df['arrival_time'] = pd.to_datetime(df['arrival_time'], errors='coerce')
        df['departure_time'] = pd.to_datetime(df['departure_time'], errors='coerce')
        time_invalid = df['arrival_time'].isna() | df['departure_time'].isna()
        df = df[~time_invalid]
        dwell_time = (df['departure_time'] - df['arrival_time']).dt.total_seconds()
        invalid_dwell = (dwell_time < 0) | (dwell_time > 3600)
        df = df[~invalid_dwell]
        self.cleaning_report['invalid_stop_records_removed'] = original_count - len(df)
        return df.reset_index(drop=True)

    def fix_missing_stops(self, stop_data, schedule_data):
        df = stop_data.copy()
        missing_count = 0
        for (route_id, direction), grp in df.groupby(['route_id', 'direction']):
            route_stops = schedule_data[
                (schedule_data['route_id'] == route_id) &
                (schedule_data['direction'] == direction)
            ]['stop_sequence'].unique()
            route_stops = sorted(route_stops)
            for trip_id, trip in grp.groupby('trip_id'):
                actual_stops = sorted(trip['stop_sequence'].unique())
                expected = [s for s in route_stops
                            if s >= min(actual_stops) and s <= max(actual_stops)]
                missing = [s for s in expected if s not in actual_stops]
                missing_count += len(missing)
                for stop_seq in missing:
                    prev_stop = trip[trip['stop_sequence'] < stop_seq].tail(1)
                    next_stop = trip[trip['stop_sequence'] > stop_seq].head(1)
                    if len(prev_stop) > 0 and len(next_stop) > 0:
                        prev_row = prev_stop.iloc[0]
                        next_row = next_stop.iloc[0]
                        mid_arrival = prev_row['departure_time'] + \
                            (next_row['arrival_time'] - prev_row['departure_time']) / 2
                        mid_departure = mid_arrival + timedelta(seconds=30)
                        new_row = prev_row.copy()
                        new_row['stop_sequence'] = stop_seq
                        new_row['stop_id'] = schedule_data[
                            (schedule_data['route_id'] == route_id) &
                            (schedule_data['direction'] == direction) &
                            (schedule_data['stop_sequence'] == stop_seq)
                        ]['stop_id'].values[0]
                        new_row['arrival_time'] = mid_arrival
                        new_row['departure_time'] = mid_departure
                        new_row['is_interpolated'] = True
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df['is_interpolated'] = df.get('is_interpolated', False)
        self.cleaning_report['missing_stops_filled'] = missing_count
        return df.sort_values(['route_id', 'trip_id', 'stop_sequence']).reset_index(drop=True)

    def detect_skipped_stops(self, stop_data, schedule_data):
        df = stop_data.copy()
        skip_records = []
        for (route_id, direction), grp in df.groupby(['route_id', 'direction']):
            route_stops = schedule_data[
                (schedule_data['route_id'] == route_id) &
                (schedule_data['direction'] == direction)
            ]['stop_sequence'].unique()
            route_stops = sorted(route_stops)
            for trip_id, trip in grp.groupby('trip_id'):
                actual_stops = sorted(trip['stop_sequence'].unique())
                skipped = []
                for i in range(len(actual_stops) - 1):
                    gap = actual_stops[i + 1] - actual_stops[i]
                    if gap > 1:
                        for s in range(actual_stops[i] + 1, actual_stops[i + 1]):
                            if s in route_stops:
                                skipped.append(s)
                for s in skipped:
                    skip_records.append({
                        'route_id': route_id,
                        'trip_id': trip_id,
                        'stop_sequence': s,
                        'direction': direction,
                        'is_skipped': True
                    })
        if skip_records:
            skip_df = pd.DataFrame(skip_records)
            df = df.merge(
                skip_df[['route_id', 'trip_id', 'stop_sequence', 'direction', 'is_skipped']],
                on=['route_id', 'trip_id', 'stop_sequence', 'direction'],
                how='left'
            )
        else:
            df['is_skipped'] = False
        df['is_skipped'] = df['is_skipped'].fillna(False)
        self.cleaning_report['skipped_stops_detected'] = int(df['is_skipped'].sum())
        return df

    def detect_detour(self, stop_data, gps_data, schedule_data):
        df = stop_data.copy()
        df['is_detour'] = False
        detour_count = 0
        route_stop_info = {}
        for (route_id, direction), grp in schedule_data.groupby(['route_id', 'direction']):
            route_stop_info[(route_id, direction)] = dict(zip(grp['stop_sequence'], grp['stop_id']))
        for (route_id, direction, trip_id), trip_stops in df.groupby(
                ['route_id', 'direction', 'trip_id']):
            key = (route_id, direction)
            if key not in route_stop_info:
                continue
            expected_ids = set(route_stop_info[key].values())
            actual_ids = set(trip_stops['stop_id'].unique())
            unexpected = actual_ids - expected_ids
            if len(unexpected) > 2:
                df.loc[
                    (df['route_id'] == route_id) &
                    (df['trip_id'] == trip_id) &
                    (df['direction'] == direction),
                    'is_detour'
                ] = True
                detour_count += 1
        self.cleaning_report['detour_trips_detected'] = detour_count
        return df

    def adjust_holiday_schedule(self, schedule_data, holidays=None):
        df = schedule_data.copy()
        if holidays is None:
            holidays = []
        df['service_date'] = pd.to_datetime(df.get('service_date', pd.Timestamp.now().date()))
        df['is_holiday'] = df['service_date'].dt.date.astype(str).isin(
            [str(h) for h in holidays]
        ) | (df['service_date'].dt.weekday >= 5)
        df['schedule_type'] = np.where(df['is_holiday'], 'holiday', 'weekday')
        return df

    def clean_swipe_data(self, swipe_data):
        df = swipe_data.copy()
        original_count = len(df)
        df = df.dropna(subset=['card_id', 'stop_id', 'swipe_time'])
        df['swipe_time'] = pd.to_datetime(df['swipe_time'], errors='coerce')
        df = df.dropna(subset=['swipe_time'])
        df = df.drop_duplicates(subset=['card_id', 'swipe_time'])
        self.cleaning_report['invalid_swipe_records_removed'] = original_count - len(df)
        return df.reset_index(drop=True)

    def clean_congestion_data(self, congestion_data):
        df = congestion_data.copy()
        original_count = len(df)
        df = df.dropna(subset=['road_segment_id', 'timestamp', 'congestion_index'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df['congestion_index'] = df['congestion_index'].clip(0, 10)
        self.cleaning_report['invalid_congestion_records_removed'] = original_count - len(df)
        return df.reset_index(drop=True)

    def clean_complaint_data(self, complaint_data):
        df = complaint_data.copy()
        original_count = len(df)
        df = df.dropna(subset=['complaint_time', 'complaint_type'])
        df['complaint_time'] = pd.to_datetime(df['complaint_time'], errors='coerce')
        df = df.dropna(subset=['complaint_time'])
        self.cleaning_report['invalid_complaint_records_removed'] = original_count - len(df)
        return df.reset_index(drop=True)
