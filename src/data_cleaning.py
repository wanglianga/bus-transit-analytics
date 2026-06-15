import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import math
from collections import defaultdict
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
    def __init__(self, gps_radius_threshold=50, speed_threshold=80,
                 skip_gap_runtime_factor=1.3, gps_deviation_meters=300,
                 detour_unexpected_threshold=1):
        self.gps_radius_threshold = gps_radius_threshold
        self.speed_threshold = speed_threshold
        self.skip_gap_runtime_factor = skip_gap_runtime_factor
        self.gps_deviation_meters = gps_deviation_meters
        self.detour_unexpected_threshold = detour_unexpected_threshold
        self.cleaning_report = {}
        self.skipped_stops_list = pd.DataFrame()

    def clean_all(self, gps_data, stop_data, swipe_data, schedule_data,
                  congestion_data, complaint_data, weather_data=None, holidays=None):
        self.cleaning_report = {}
        self.skipped_stops_list = pd.DataFrame()

        gps_cleaned = self.clean_gps_drift(gps_data)
        stop_cleaned = self.clean_stop_records(stop_data, gps_cleaned)
        weather_cleaned = self.clean_weather_data(weather_data)

        analysis = self._analyze_gaps(stop_cleaned, schedule_data)
        skip_df = analysis['skipped_stops']
        miss_df = analysis['missing_stops']

        self.skipped_stops_list = skip_df
        self.cleaning_report['skipped_stops_total'] = len(skip_df)
        self.cleaning_report['skipped_trips_with_skip'] = (
            skip_df[['route_id', 'trip_id']].drop_duplicates().shape[0]
            if len(skip_df) else 0
        )
        self.cleaning_report['missing_stops_to_interpolate'] = len(miss_df)

        stop_fixed = self._interpolate_missing_only(stop_cleaned, schedule_data, miss_df)
        stop_fixed = self._mark_trip_flags(stop_fixed, skip_df)

        stop_fixed = self.detect_detour(stop_fixed, gps_cleaned, schedule_data)
        stop_fixed = self._merge_weather_to_stops(stop_fixed, weather_cleaned)
        schedule_adjusted = self.adjust_holiday_schedule(schedule_data, holidays)
        swipe_cleaned = self.clean_swipe_data(swipe_data)
        swipe_cleaned = self._merge_weather_to_swipe(swipe_cleaned, weather_cleaned)
        congestion_cleaned = self.clean_congestion_data(congestion_data)
        congestion_cleaned = self._merge_weather_to_congestion(congestion_cleaned, weather_cleaned)
        complaint_cleaned = self.clean_complaint_data(complaint_data)
        complaint_cleaned = self._merge_weather_to_complaint(complaint_cleaned, weather_cleaned)

        return {
            'gps': gps_cleaned,
            'stops': stop_fixed,
            'swipe': swipe_cleaned,
            'schedule': schedule_adjusted,
            'congestion': congestion_cleaned,
            'complaint': complaint_cleaned,
            'weather': weather_cleaned,
            'skipped_stops': skip_df,
            'report': self.cleaning_report
        }

    def _analyze_gaps(self, stop_data, schedule_data):
        skipped_rows = []
        missing_rows = []

        route_ref = {}
        for (route_id, direction), grp in schedule_data.groupby(['route_id', 'direction']):
            sorted_seq = sorted(grp['stop_sequence'].unique())
            stop_map = dict(zip(grp['stop_sequence'], grp['stop_id']))
            route_ref[(route_id, direction)] = {
                'sequences': sorted_seq,
                'stop_map': stop_map
            }

        baseline = self._compute_segment_baseline(stop_data)

        for (route_id, direction), grp in stop_data.groupby(['route_id', 'direction']):
            key = (route_id, direction)
            if key not in route_ref:
                continue
            ref = route_ref[key]

            for trip_id, trip in grp.groupby('trip_id'):
                actual = sorted(trip['stop_sequence'].unique().tolist())
                if len(actual) < 2:
                    continue

                trip_by_seq = dict(zip(trip['stop_sequence'], trip.to_dict('records')))
                expected_range = [s for s in ref['sequences']
                                  if s >= actual[0] and s <= actual[-1]]

                for i in range(len(actual) - 1):
                    prev_seq, curr_seq = actual[i], actual[i + 1]
                    gap_seqs = [s for s in range(prev_seq + 1, curr_seq)
                                if s in expected_range]
                    if not gap_seqs:
                        continue

                    prev_row = trip_by_seq[prev_seq]
                    curr_row = trip_by_seq[curr_seq]
                    actual_gap_secs = (
                        pd.Timestamp(curr_row['arrival_time']) -
                        pd.Timestamp(prev_row['departure_time'])
                    ).total_seconds()

                    normal_secs = baseline.get(
                        (route_id, direction, prev_seq, curr_seq),
                        180 * (curr_seq - prev_seq)
                    )

                    if actual_gap_secs < normal_secs * self.skip_gap_runtime_factor:
                        for gs in gap_seqs:
                            skipped_rows.append({
                                'route_id': route_id,
                                'direction': direction,
                                'trip_id': trip_id,
                                'stop_sequence': gs,
                                'stop_id': ref['stop_map'].get(gs, f"UNKNOWN_{gs}"),
                                'vehicle_id': prev_row.get('vehicle_id'),
                                'gap_from': prev_seq,
                                'gap_to': curr_seq,
                                'actual_gap_secs': actual_gap_secs,
                                'baseline_secs': normal_secs,
                                'skip_reason': 'runtime_too_short_for_gap'
                            })
                    else:
                        for gs in gap_seqs:
                            missing_rows.append({
                                'route_id': route_id,
                                'direction': direction,
                                'trip_id': trip_id,
                                'stop_sequence': gs,
                                'stop_id': ref['stop_map'].get(gs, f"UNKNOWN_{gs}"),
                                'gap_from': prev_seq,
                                'gap_to': curr_seq,
                                'actual_gap_secs': actual_gap_secs,
                                'baseline_secs': normal_secs
                            })

        return {
            'skipped_stops': pd.DataFrame(skipped_rows),
            'missing_stops': pd.DataFrame(missing_rows)
        }

    def _compute_segment_baseline(self, stop_data):
        baseline = {}
        df = stop_data.sort_values(['route_id', 'direction', 'trip_id', 'stop_sequence']).copy()
        df['next_seq'] = df.groupby(['route_id', 'direction', 'trip_id'])['stop_sequence'].shift(-1)
        df['next_arrival'] = df.groupby(['route_id', 'direction', 'trip_id'])['arrival_time'].shift(-1)
        df['departure_time'] = pd.to_datetime(df['departure_time'])
        df['next_arrival'] = pd.to_datetime(df['next_arrival'])
        df['seg_secs'] = (df['next_arrival'] - df['departure_time']).dt.total_seconds()
        valid = df.dropna(subset=['next_seq', 'seg_secs'])
        valid = valid[valid['seg_secs'] > 0]
        valid['next_seq'] = valid['next_seq'].astype(int)
        grouped = valid.groupby(['route_id', 'direction', 'stop_sequence', 'next_seq'])
        for (rid, dr, ss, ns), grp in grouped:
            if len(grp) >= 2:
                baseline[(rid, dr, ss, ns)] = grp['seg_secs'].median()
            else:
                baseline[(rid, dr, ss, ns)] = grp['seg_secs'].iloc[0]
        return baseline

    def _interpolate_missing_only(self, stop_data, schedule_data, missing_df):
        df = stop_data.copy()
        df['is_interpolated'] = False

        if missing_df.empty:
            self.cleaning_report['missing_stops_filled'] = 0
            return df

        filled = 0
        for _, mrow in missing_df.iterrows():
            trip_mask = (
                (df['route_id'] == mrow['route_id']) &
                (df['direction'] == mrow['direction']) &
                (df['trip_id'] == mrow['trip_id'])
            )
            trip = df[trip_mask]
            prev = trip[trip['stop_sequence'] < mrow['stop_sequence']].tail(1)
            nxt = trip[trip['stop_sequence'] > mrow['stop_sequence']].head(1)
            if prev.empty or nxt.empty:
                continue

            prev_row = prev.iloc[0]
            nxt_row = nxt.iloc[0]

            mid_arrival = pd.Timestamp(prev_row['departure_time']) + \
                (pd.Timestamp(nxt_row['arrival_time']) - pd.Timestamp(prev_row['departure_time'])) / 2
            mid_departure = mid_arrival + timedelta(seconds=30)

            new_row = prev_row.copy()
            new_row['stop_sequence'] = int(mrow['stop_sequence'])
            new_row['stop_id'] = mrow['stop_id']
            new_row['arrival_time'] = mid_arrival
            new_row['departure_time'] = mid_departure
            new_row['is_interpolated'] = True

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            filled += 1

        self.cleaning_report['missing_stops_filled'] = filled
        return df.sort_values(['route_id', 'trip_id', 'stop_sequence']).reset_index(drop=True)

    def _mark_trip_flags(self, stop_data, skipped_df):
        df = stop_data.copy()
        df['trip_skip_count'] = 0
        df['trip_has_skip'] = False

        if skipped_df.empty:
            return df

        trip_counts = skipped_df.groupby(
            ['route_id', 'trip_id']
        ).size().reset_index(name='_count')

        for _, tc in trip_counts.iterrows():
            mask = (
                (df['route_id'] == tc['route_id']) &
                (df['trip_id'] == tc['trip_id'])
            )
            df.loc[mask, 'trip_skip_count'] = tc['_count']
            df.loc[mask, 'trip_has_skip'] = True
        return df

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
            time_diff = (pd.Timestamp(row['timestamp']) - pd.Timestamp(row['prev_ts'])).total_seconds() / 3600
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

    def detect_detour(self, stop_data, gps_data, schedule_data):
        df = stop_data.copy()
        df['is_detour'] = False
        df['detour_reason'] = ''

        route_stop_info = {}
        for (route_id, direction), grp in schedule_data.groupby(['route_id', 'direction']):
            if 'latitude' in grp.columns and 'longitude' in grp.columns:
                coords = grp[['latitude', 'longitude']].values.tolist()
            else:
                coords = []
            route_stop_info[(route_id, direction)] = {
                'stop_ids': set(grp['stop_id'].unique()),
                'stop_coords': coords
            }

        vehicle_trip_gps = defaultdict(list)
        if not gps_data.empty and 'vehicle_id' in gps_data.columns:
            gps_sorted = gps_data.sort_values(['vehicle_id', 'timestamp'])
            for _, g in gps_sorted.iterrows():
                vehicle_trip_gps[g['vehicle_id']].append(
                    (float(g['latitude']), float(g['longitude']), pd.Timestamp(g['timestamp']))
                )

        detour_trips = 0
        detour_by_unexpected_stop = 0
        detour_by_gps_deviation = 0

        for (route_id, direction, trip_id), trip_stops in df.groupby(
                ['route_id', 'direction', 'trip_id']):
            key = (route_id, direction)
            if key not in route_stop_info:
                continue

            expected_ids = route_stop_info[key]['stop_ids']
            actual_ids = set(trip_stops['stop_id'].dropna().unique())
            unexpected = actual_ids - expected_ids
            has_unexpected = len(unexpected) >= self.detour_unexpected_threshold

            has_gps_deviation = False
            ref_coords = route_stop_info[key]['stop_coords']
            if ref_coords and len(trip_stops) > 0:
                vid = trip_stops.iloc[0].get('vehicle_id')
                t_start = pd.Timestamp(trip_stops.iloc[0]['arrival_time'])
                t_end = pd.Timestamp(trip_stops.iloc[-1]['departure_time'])
                trip_gps = [
                    (lat, lon, ts) for (lat, lon, ts) in vehicle_trip_gps.get(vid, [])
                    if t_start <= ts <= t_end
                ]
                if trip_gps:
                    deviations = []
                    for lat, lon, _ in trip_gps:
                        min_dist = min(
                            (_distance_meters((lat, lon), (c[0], c[1])) for c in ref_coords),
                            default=self.gps_deviation_meters * 2
                        )
                        deviations.append(min_dist)
                    if deviations:
                        ratio = sum(
                            1 for d in deviations if d > self.gps_deviation_meters
                        ) / len(deviations)
                        if ratio >= 0.3:
                            has_gps_deviation = True

            is_trip_detour = has_unexpected or has_gps_deviation
            if is_trip_detour:
                mask = (
                    (df['route_id'] == route_id) &
                    (df['trip_id'] == trip_id) &
                    (df['direction'] == direction)
                )
                df.loc[mask, 'is_detour'] = True
                reasons = []
                if has_unexpected:
                    reasons.append(f"unexpected_stops[{len(unexpected)}]")
                    detour_by_unexpected_stop += 1
                if has_gps_deviation:
                    reasons.append(f"gps_deviation[ratio>30%]")
                    detour_by_gps_deviation += 1
                df.loc[mask, 'detour_reason'] = ','.join(reasons)
                detour_trips += 1

        self.cleaning_report['detour_trips_detected'] = detour_trips
        self.cleaning_report['detour_by_unexpected_stop'] = detour_by_unexpected_stop
        self.cleaning_report['detour_by_gps_deviation'] = detour_by_gps_deviation
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

    def clean_weather_data(self, weather_data):
        if weather_data is None or weather_data.empty:
            self.cleaning_report['weather_records'] = 0
            return pd.DataFrame()
        df = weather_data.copy()
        original_count = len(df)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        if 'weather' not in df.columns:
            df['weather'] = 'unknown'
        if 'rain_intensity' not in df.columns:
            df['rain_intensity'] = 'none'
        df['is_rainy'] = df['weather'].str.lower().isin(['rain', 'rainy', '小雨', '中雨', '大雨', '暴雨'])
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        self.cleaning_report['weather_records'] = len(df)
        self.cleaning_report['rainy_days'] = int(df['is_rainy'].sum())
        return df.reset_index(drop=True)

    def _merge_weather_to_stops(self, stops_df, weather_df):
        if stops_df.empty or weather_df.empty:
            if 'is_rainy' not in stops_df.columns:
                stops_df['is_rainy'] = False
                stops_df['weather'] = 'unknown'
                stops_df['rain_intensity'] = 'none'
            return stops_df
        df = stops_df.copy()
        df['date_str'] = df['arrival_time'].dt.strftime('%Y-%m-%d')
        weather_map = weather_df.set_index('date_str')
        df['is_rainy'] = df['date_str'].map(weather_map['is_rainy']).fillna(False)
        df['weather'] = df['date_str'].map(weather_map['weather']).fillna('unknown')
        df['rain_intensity'] = df['date_str'].map(weather_map['rain_intensity']).fillna('none')
        df = df.drop(columns=['date_str'])
        return df

    def _merge_weather_to_swipe(self, swipe_df, weather_df):
        if swipe_df.empty or weather_df.empty:
            if 'is_rainy' not in swipe_df.columns:
                swipe_df['is_rainy'] = False
                swipe_df['weather'] = 'unknown'
            return swipe_df
        df = swipe_df.copy()
        df['date_str'] = df['swipe_time'].dt.strftime('%Y-%m-%d')
        weather_map = weather_df.set_index('date_str')
        df['is_rainy'] = df['date_str'].map(weather_map['is_rainy']).fillna(False)
        df['weather'] = df['date_str'].map(weather_map['weather']).fillna('unknown')
        df['rain_intensity'] = df['date_str'].map(weather_map['rain_intensity']).fillna('none')
        df = df.drop(columns=['date_str'])
        return df

    def _merge_weather_to_congestion(self, congestion_df, weather_df):
        if congestion_df.empty or weather_df.empty:
            if 'is_rainy' not in congestion_df.columns:
                congestion_df['is_rainy'] = False
                congestion_df['weather'] = 'unknown'
            return congestion_df
        df = congestion_df.copy()
        df['date_str'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        weather_map = weather_df.set_index('date_str')
        df['is_rainy'] = df['date_str'].map(weather_map['is_rainy']).fillna(False)
        df['weather'] = df['date_str'].map(weather_map['weather']).fillna('unknown')
        df['rain_intensity'] = df['date_str'].map(weather_map['rain_intensity']).fillna('none')
        df = df.drop(columns=['date_str'])
        return df

    def _merge_weather_to_complaint(self, complaint_df, weather_df):
        if complaint_df.empty or weather_df.empty:
            if 'is_rainy' not in complaint_df.columns:
                complaint_df['is_rainy'] = False
                complaint_df['weather'] = 'unknown'
            return complaint_df
        df = complaint_df.copy()
        df['date_str'] = df['complaint_time'].dt.strftime('%Y-%m-%d')
        weather_map = weather_df.set_index('date_str')
        df['is_rainy'] = df['date_str'].map(weather_map['is_rainy']).fillna(False)
        df['weather'] = df['date_str'].map(weather_map['weather']).fillna('unknown')
        df['rain_intensity'] = df['date_str'].map(weather_map['rain_intensity']).fillna('none')
        df = df.drop(columns=['date_str'])
        return df
