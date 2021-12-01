import os
import numpy as np
import pandas as pd

def get_df(path, measure):
    '''
    Lee los datos de las distintas fuentes de datos y devuelve un df
    '''

    # Get poits
    points_df = pd.read_excel(path + 'PointsMapping.ods', engine='odf', index_col=0)

    # Get smartphone sensor data
    sensor_sphone_df = pd.read_csv(path + measure + '_smartphone_sens.csv')
    sensor_sphone_df.columns = sensor_sphone_df.columns.str.strip()

    # Get timestamp
    timestamp_df = pd.read_csv(path + measure + '_timestamp_id.csv',
                              header=None, names=['arrival_ts', 'departure_ts', 'place_id'])

    # Join sensor and timestamp
    sensor_sphone_df['place_id'] = sensor_sphone_df['timestamp'].apply(
        lambda x: return_place_id(timestamp_df, x))
    # Drop nulls
    sensor_sphone_df.dropna(axis=0, inplace=True)
    # Convert to int
    sensor_sphone_df['place_id'] = sensor_sphone_df['place_id'].apply(lambda x: int(x))

    # Get WIFI data
    col = [f'WAP{str(num).zfill(3)}' for num in range(1, 128)]
    wifi_df = pd.read_csv(path + measure + '_smartphone_wifi.csv',
                              header=None, names=col)
    wifi_df.index.name = 'id'
    wifi_df.index = range(1,326)

    # Join WIFI signal with sensor signals             
    wifi_df = wifi_df.reset_index()
    wifi_df.rename(columns={'index': 'place_id'}, inplace=True)    
    signals = wifi_df.copy()
    signals = signals.merge(sensor_sphone_df[['MagneticFieldX', 'MagneticFieldY', 'MagneticFieldZ', 'place_id']])

    # Add coordinates
    signals['x'] = [points_df.loc[id, 'X'] for id in signals.place_id]
    signals['y'] = [points_df.loc[id, 'Y'] for id in signals.place_id]

    # Drop duplicates
    print("Length original", len(signals))
    print("Length duplicates:", signals[signals.duplicated()].shape[0])
    signals = signals.drop_duplicates(keep = 'last')
    print("Length without duplicates:", len(signals), '\n')

    return signals


def return_place_id(timestamp_df, timestamp):
    '''
    Devuelte el id del lugar en el que se encuentra en un momento dado.
    En caso de no encontrar un cajón devuelve None
    '''
    selected = timestamp_df[(timestamp_df.arrival_ts <= timestamp) &
             (timestamp_df.departure_ts >= timestamp)]
    if selected.shape[0] == 0:
        return None
    else:
        return timestamp_df[(timestamp_df.arrival_ts <= timestamp) &
             (timestamp_df.departure_ts >= timestamp)].iloc[0]['place_id']
