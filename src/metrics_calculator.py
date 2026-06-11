import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    def __init__(self, on_time_threshold_minutes=2):
        self.on_time_threshold = timedelta(minutes=on_time_threshold_minutes)

    def calculate_all(self, cleaned_data):
        stops_df = cleaned_data['stops']
        schedule_df = cleaned_data['schedule']
        swipe_df = cleaned_data['swipe']
        congestion_df = cleaned_data['congestion']
        complaint_df = cleaned_data['complaint']
        skipped_stops_df = cleaned_data.get('skipped_stops', pd.DataFrame())

        metrics = {}
        metrics['inter_stop_time'] = self.calculate_inter_stop_time(stops_df)
        metrics['headway'] = self.calculate_headway(stops_df, schedule_df)
        metrics['on_time_rate'] = self.calculate_on_time_rate(stops_df, schedule_df)
        metrics['load_factor'] = self.calculate_load_factor(stops_df, swipe_df)
        metrics['peak_periods'] = self.identify_peak_periods(swipe_df)
        metrics['congested_segments'] = self.identify_congested_segments(
            congestion_df, stops_df
        )
        metrics['complaint_hotspots'] = self.identify_complaint_hotspots(complaint_df)
        metrics['delay_analysis'] = self.analyze_delays(
            stops_df, schedule_df, congestion_df, skipped_stops_df
        )
        return metrics

    def calculate_inter_stop_time(self, stops_df):
        df = stops_df.copy()
        df = df.sort_values(['route_id', 'direction', 'trip_id', 'stop_sequence'])
        df['next_arrival'] = df.groupby(
            ['route_id', 'direction', 'trip_id']
        )['arrival_time'].shift(-1)
        df['next_stop_seq'] = df.groupby(
            ['route_id', 'direction', 'trip_id']
        )['stop_sequence'].shift(-1)
        df['run_time_seconds'] = (
            df['next_arrival'] - df['departure_time']
        ).dt.total_seconds()
        df['segment'] = df.apply(
            lambda r: f"{int(r['stop_sequence'])}->{int(r['next_stop_seq'])}"
            if not pd.isna(r['next_stop_seq']) else None, axis=1
        )
        result = df.dropna(subset=['run_time_seconds', 'segment'])
        summary = result.groupby(
            ['route_id', 'direction', 'stop_id', 'stop_sequence', 'segment']
        ).agg(
            avg_run_time=('run_time_seconds', 'mean'),
            median_run_time=('run_time_seconds', 'median'),
            std_run_time=('run_time_seconds', 'std'),
            p95_run_time=('run_time_seconds', lambda x: np.percentile(x, 95)),
            sample_count=('run_time_seconds', 'count')
        ).reset_index()
        return {'detail': result, 'summary': summary}

    def calculate_headway(self, stops_df, schedule_df):
        df = stops_df.copy()
        df = df.sort_values(['route_id', 'direction', 'stop_id', 'arrival_time'])
        df['prev_arrival'] = df.groupby(
            ['route_id', 'direction', 'stop_id']
        )['arrival_time'].shift(1)
        df['headway_seconds'] = (
            df['arrival_time'] - df['prev_arrival']
        ).dt.total_seconds()
        df['hour'] = df['arrival_time'].dt.hour
        df['is_weekday'] = df['arrival_time'].dt.weekday < 5
        result = df.dropna(subset=['headway_seconds'])
        summary = result.groupby(
            ['route_id', 'direction', 'hour', 'is_weekday']
        ).agg(
            avg_headway=('headway_seconds', 'mean'),
            median_headway=('headway_seconds', 'median'),
            std_headway=('headway_seconds', 'std'),
            headway_irregularity=('headway_seconds', lambda x: np.std(x) / np.mean(x) if np.mean(x) > 0 else 0),
            sample_count=('headway_seconds', 'count')
        ).reset_index()
        return {'detail': result, 'summary': summary}

    def calculate_on_time_rate(self, stops_df, schedule_df):
        merged = stops_df.merge(
            schedule_df[
                ['route_id', 'direction', 'trip_id', 'stop_sequence',
                 'scheduled_arrival', 'scheduled_departure']
            ],
            on=['route_id', 'direction', 'trip_id', 'stop_sequence'],
            how='left'
        )
        merged['scheduled_arrival'] = pd.to_datetime(
            merged['scheduled_arrival'], errors='coerce'
        )
        merged['arrival_deviation'] = (
            merged['arrival_time'] - merged['scheduled_arrival']
        ).dt.total_seconds()
        merged['is_early'] = merged['arrival_deviation'] < -self.on_time_threshold.total_seconds()
        merged['is_late'] = merged['arrival_deviation'] > self.on_time_threshold.total_seconds()
        merged['is_on_time'] = ~merged['is_early'] & ~merged['is_late']
        route_summary = merged.groupby(['route_id', 'direction']).agg(
            on_time_count=('is_on_time', 'sum'),
            late_count=('is_late', 'sum'),
            early_count=('is_early', 'sum'),
            total_trips=('is_on_time', 'count'),
            avg_deviation=('arrival_deviation', 'mean')
        ).reset_index()
        route_summary['on_time_rate'] = route_summary['on_time_count'] / route_summary['total_trips']
        stop_summary = merged.groupby(
            ['route_id', 'direction', 'stop_id', 'stop_sequence']
        ).agg(
            on_time_count=('is_on_time', 'sum'),
            late_count=('is_late', 'sum'),
            total_count=('is_on_time', 'count'),
            avg_deviation=('arrival_deviation', 'mean')
        ).reset_index()
        stop_summary['on_time_rate'] = stop_summary['on_time_count'] / stop_summary['total_count']
        return {
            'detail': merged,
            'route_summary': route_summary,
            'stop_summary': stop_summary
        }

    def calculate_load_factor(self, stops_df, swipe_df):
        if swipe_df.empty:
            return {'detail': pd.DataFrame(), 'summary': pd.DataFrame()}
        swipe = swipe_df.copy()
        swipe['hour'] = swipe['swipe_time'].dt.hour
        swipe['date'] = swipe['swipe_time'].dt.date
        stop_boarding = swipe.groupby(
            ['route_id', 'stop_id', 'date', 'hour']
        ).agg(passenger_count=('card_id', 'nunique')).reset_index()
        merged = stops_df.copy()
        merged['hour'] = merged['arrival_time'].dt.hour
        merged['date'] = merged['arrival_time'].dt.date
        merged = merged.merge(
            stop_boarding,
            on=['route_id', 'stop_id', 'date', 'hour'],
            how='left'
        )
        merged['passenger_count'] = merged['passenger_count'].fillna(0)
        summary = merged.groupby(
            ['route_id', 'direction', 'stop_id', 'stop_sequence', 'hour']
        ).agg(
            avg_passengers=('passenger_count', 'mean'),
            max_passengers=('passenger_count', 'max'),
            median_passengers=('passenger_count', 'median'),
            total_passengers=('passenger_count', 'sum')
        ).reset_index()
        summary['load_level'] = pd.cut(
            summary['avg_passengers'],
            bins=[-np.inf, 10, 30, 50, np.inf],
            labels=['low', 'medium', 'high', 'overloaded']
        )
        return {'detail': merged, 'summary': summary}

    def identify_peak_periods(self, swipe_df):
        if swipe_df.empty:
            return pd.DataFrame()
        df = swipe_df.copy()
        df['hour'] = df['swipe_time'].dt.hour
        df['is_weekday'] = df['swipe_time'].dt.weekday < 5
        df['date'] = df['swipe_time'].dt.date
        hourly = df.groupby(
            ['route_id', 'is_weekday', 'date', 'hour']
        ).agg(passenger_count=('card_id', 'nunique')).reset_index()
        summary = hourly.groupby(
            ['route_id', 'is_weekday', 'hour']
        ).agg(
            avg_passengers=('passenger_count', 'mean'),
            median_passengers=('passenger_count', 'median')
        ).reset_index()
        result_list = []
        for (route_id, is_weekday), grp in summary.groupby(['route_id', 'is_weekday']):
            if grp['avg_passengers'].sum() == 0:
                continue
            threshold = grp['avg_passengers'].quantile(0.7)
            peaks = grp[grp['avg_passengers'] >= threshold].copy()
            peaks['is_peak'] = True
            result_list.append(peaks)
        if result_list:
            return pd.concat(result_list, ignore_index=True)
        return pd.DataFrame(columns=summary.columns.tolist() + ['is_peak'])

    def identify_congested_segments(self, congestion_df, stops_df):
        if congestion_df.empty:
            return pd.DataFrame()
        df = congestion_df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['is_weekday'] = df['timestamp'].dt.weekday < 5
        df['congestion_level'] = pd.cut(
            df['congestion_index'],
            bins=[-np.inf, 2, 5, 8, np.inf],
            labels=['smooth', 'light', 'moderate', 'severe']
        )
        summary = df.groupby(
            ['road_segment_id', 'hour', 'is_weekday']
        ).agg(
            avg_congestion=('congestion_index', 'mean'),
            max_congestion=('congestion_index', 'max'),
            severe_count=('congestion_index', lambda x: (x >= 8).sum()),
            total_count=('congestion_index', 'count')
        ).reset_index()
        summary['severe_ratio'] = summary['severe_count'] / summary['total_count']
        summary['is_congested'] = (summary['avg_congestion'] >= 5) | (summary['severe_ratio'] >= 0.3)
        return summary

    def identify_complaint_hotspots(self, complaint_df):
        if complaint_df.empty:
            return pd.DataFrame()
        df = complaint_df.copy()
        df['date'] = df['complaint_time'].dt.date
        df['hour'] = df['complaint_time'].dt.hour
        by_route_type = df.groupby(
            ['route_id', 'complaint_type']
        ).agg(
            complaint_count=('complaint_id', 'count'),
            unique_dates=('date', 'nunique')
        ).reset_index()
        by_route_stop = df.groupby(
            ['route_id', 'stop_id', 'complaint_type']
        ).agg(complaint_count=('complaint_id', 'count')).reset_index()
        by_time = df.groupby(
            ['route_id', 'hour', 'complaint_type']
        ).agg(complaint_count=('complaint_id', 'count')).reset_index()
        return {
            'by_route_type': by_route_type,
            'by_route_stop': by_route_stop,
            'by_time': by_time
        }

    def analyze_delays(self, stops_df, schedule_df, congestion_df, skipped_stops_df=None):
        if skipped_stops_df is None:
            skipped_stops_df = pd.DataFrame()

        on_time = self.calculate_on_time_rate(stops_df, schedule_df)
        detail = on_time['detail'].copy()
        detail['date'] = detail['arrival_time'].dt.date
        detail['hour'] = detail['arrival_time'].dt.hour
        delay_detail = detail[detail['is_late']].copy()
        delay_detail['delay_minutes'] = delay_detail['arrival_deviation'] / 60

        trip_extra = stops_df.groupby(
            ['route_id', 'trip_id']
        ).agg({
            'is_detour': 'max',
            'trip_has_skip': 'max',
            'trip_skip_count': 'max',
            'detour_reason': 'first'
        }).reset_index()
        detail = detail.merge(
            trip_extra, on=['route_id', 'trip_id'], how='left', suffixes=('', '_trip')
        )
        if 'is_detour_trip' in detail.columns:
            detail['is_detour'] = detail['is_detour_trip'].fillna(detail.get('is_detour', False))
        if 'trip_has_skip_trip' in detail.columns:
            detail['trip_has_skip'] = detail['trip_has_skip_trip'].fillna(
                detail.get('trip_has_skip', False)
            )
        if 'trip_skip_count_trip' in detail.columns:
            detail['trip_skip_count'] = detail['trip_skip_count_trip'].fillna(
                detail.get('trip_skip_count', 0)
            )

        delay_detail = detail[detail['is_late']].copy()
        delay_detail['delay_minutes'] = delay_detail['arrival_deviation'] / 60

        by_cause = {}
        by_cause['by_stop'] = delay_detail.groupby(
            ['route_id', 'stop_id', 'stop_sequence']
        ).agg(
            delay_count=('delay_minutes', 'count'),
            avg_delay=('delay_minutes', 'mean'),
            total_delay=('delay_minutes', 'sum')
        ).reset_index().sort_values(
            ['route_id', 'total_delay'], ascending=[True, False]
        )
        by_cause['by_hour'] = delay_detail.groupby(
            ['route_id', 'hour']
        ).agg(
            delay_count=('delay_minutes', 'count'),
            avg_delay=('delay_minutes', 'mean'),
            total_delay=('delay_minutes', 'sum')
        ).reset_index()

        detour_mask = delay_detail.get('is_detour', False) == True
        if detour_mask.any():
            dd_grp = delay_detail[detour_mask].groupby('route_id').agg(
                detour_delay_count=('delay_minutes', 'count'),
                detour_avg_delay=('delay_minutes', 'mean'),
                detour_total_delay=('delay_minutes', 'sum')
            ).reset_index()
            trip_detour_counts = trip_extra[trip_extra['is_detour'] == True].groupby(
                'route_id'
            ).size().reset_index(name='detour_trip_count')
            dd_grp = dd_grp.merge(trip_detour_counts, on='route_id', how='left')
            by_cause['detour_delays'] = dd_grp

        if not skipped_stops_df.empty:
            skip_summary = skipped_stops_df.groupby(
                ['route_id', 'stop_id', 'stop_sequence']
            ).agg(skip_count=('trip_id', 'count')).reset_index()
            by_cause['skipped_summary'] = skip_summary

            skip_by_route = skipped_stops_df.groupby('route_id').agg(
                skipped_stop_event_count=('trip_id', 'count'),
                affected_trip_count=('trip_id', 'nunique')
            ).reset_index()
            skip_delay_routes = []
            for _, sr in skip_by_route.iterrows():
                rd = delay_detail[delay_detail['route_id'] == sr['route_id']]
                skip_trips = detail[
                    (detail['route_id'] == sr['route_id']) &
                    (detail.get('trip_has_skip', False) == True)
                ]['trip_id'].unique()
                skip_late = delay_detail[
                    (delay_detail['route_id'] == sr['route_id']) &
                    (delay_detail['trip_id'].isin(skip_trips))
                ]
                skip_delay_routes.append({
                    'route_id': sr['route_id'],
                    'skipped_stop_event_count': int(sr['skipped_stop_event_count']),
                    'affected_trip_count': int(sr['affected_trip_count']),
                    'skip_related_delay_count': int(len(skip_late)),
                    'skip_related_total_delay_min': round(
                        skip_late['delay_minutes'].sum(), 1
                    ) if len(skip_late) else 0,
                    'skip_related_avg_delay_min': round(
                        skip_late['delay_minutes'].mean(), 1
                    ) if len(skip_late) else 0
                })
            by_cause['skip_route_summary'] = pd.DataFrame(skip_delay_routes)

        return by_cause
