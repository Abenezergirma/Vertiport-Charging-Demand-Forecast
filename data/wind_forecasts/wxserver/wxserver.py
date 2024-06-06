#!/usr/bin/env python
"""Query the weather server and return the list of weather conditions"""


import numpy as np
import pyproj
import matplotlib.pyplot as plt
from time import perf_counter
from typing import List
import requests
import numpy as np
import json
from tqdm import tqdm
import time

CERT = "./gwu_client.crt"
KEY = "./gwu_client.key"

TIMESTAMPS=['2021-04-20T01:40:00+00:00']
ROUTE = np.array([[-97.40959,32.58769],[-96.46249, 33.3]])
OUTPATH = 'wind_data_at_5km.json'
SEP_DIST = 5000# In m
ENSEMBLE = False
# timestamps=['2021-08-10T05:30:00+00:00']


min_lat = 32.58769
max_lat = 33.3#33.18819

max_lon = -96.46249
min_lon = -97.40959

box = np.clip(ROUTE,[min_lon,min_lat],[max_lon,max_lat])


def server_request(lats: List[float], lons: List[float], alts: List[float], timestamps: List[str], certificate: str = '', key: str = '', ensemble: bool = False):
    """Handles the requestinng of the weather data from the wxserver"""

    # Assert that valid data is passed
    assert len(lats) == len(lons) == len(alts) == len(timestamps)

    no_points = len(lats)

    if not certificate:
        certificate = CERT

    if not key:
        key = KEY

    session = requests.Session()
    session.cert = (certificate, key)

    request_data = {
        'num_elements': no_points,
        'coordinates': {
            'latitudes': lats,
            'longitudes': lons,
            'altitudes_m': alts,
            'timestamps': timestamps,
        }
    }

    print(
        f'[MESSAGE]:: Requesting {no_points} points of data from the weather server')
    start = perf_counter()

    if not ensemble:
        result = session.request(
            'REPORT',
            'https://microwx.wx.ll.mit.edu/get_weather_data',
            json=request_data)
    else:
        result = session.request(
            'REPORT',
            'https://microwx.wx.ll.mit.edu/get_ensemble_weather_data',
            json=request_data)

    run_time = perf_counter() - start

    print(
        f'[MESSAGE]:: Response ({result.status_code}) from server in {run_time:.3f} seconds')

    if int(result.status_code) != 200:
        print(f'Invalid server response status_code {result.status_code}')
        return None

    return result.json()


def prepare_area(box, d = 100, alts = [400,450], plot=False):
    # Define the WGS 84 (World Geodetic System 1984) and the target projection (in meters)
    print("Perparing Area...")
    wgs84 = pyproj.Proj(init="epsg:4326")  # WGS 84 (latitude, longitude)
    target_proj = pyproj.Proj(init="epsg:3857")  # Spherical Mercator (meters)
    print("Converting to Cartesian...")
    cartesian_coordinates = np.array([pyproj.transform(wgs84, target_proj, lon, lat) for lon, lat in box])

    normalised_cartesian = cartesian_coordinates-cartesian_coordinates.min(axis=0)

    # Calculate the minimum and maximum coordinates for the bounding box
    min_x, min_y = np.min(normalised_cartesian, axis=0)
    max_x, max_y = np.max(normalised_cartesian, axis=0)

    print("Generating x/y vals...")
    x_vals = np.linspace(min_x+(0*d),max_x-(0*d),int((max_x-min_x)/d), endpoint=True)
    y_vals = np.linspace(min_y+(0*d),max_y-(0*d),int((max_y-min_y)/d), endpoint=True)


    points_array = np.array([[x,y] for x in x_vals for y in y_vals])

    if plot:
        x, y = normalised_cartesian[:, 0], normalised_cartesian[:, 1]
        plt.plot(x, y, 'bo-', label='Origional Area')  # 'bo-' means blue circles connected by lines

        plt.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], 'r--', label='Bounding Box')

        plt.scatter(points_array[:,0], points_array[:,1], s=20, c='g', label='Sampling Points')

        plt.xlabel("Lon (meters)")
        plt.ylabel("Lat (meters)")

        for xi, yi in zip(x, y):
            plt.annotate(f"({xi:.2f}, {yi:.2f})", (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.title("Box and Bounding Box in Cartesian Coordinates")
        plt.grid(True)
        plt.legend()
        plt.show()

    print(f"Converting {len(points_array)} points from Cartesian...")
    converted_points = np.clip(np.array([pyproj.transform(target_proj, wgs84, x + min_x, y + min_y) for x, y in points_array+cartesian_coordinates.min(axis=0)]),(min_lon,min_lat),(max_lon,max_lat))
    
    points_with_alt = []
    print("Generating alts...")
    for alt in tqdm(alts):
        points_with_alt.extend(
            [point[0].item(), point[1].item(), alt]
            for point in tqdm(converted_points)
        )
        
    print("Sample points generated!")
    return points_with_alt


def multi_point_query(box, d=500, alts=[400], timestamps=['2021-08-10T05:30:00+00:00'], ensemble=True, file_name='no_ensemble.json', sleep_every = 10, sleep_time=60):
    sample_points = prepare_area(box, d=d, alts=alts, plot=True)
    print(sample_points)

    print(f'\n\nSampling {len(sample_points)} Points')

    data = []
    
    if ensemble:
        for i, (lons, lats, alts) in tqdm(enumerate(sample_points)):
            data.append(server_request(
                [lats], [lons], [alts], [timestamps[0]], ensemble=ensemble
            ))
            if i%sleep_every == 0:
                print(f'Sleeping for {sleep_time} seconds to avoid overloading the server')
                time.sleep(sleep_time)
    else:
        print("Requesting points...")
        print(sample_points)
        data = server_request(
            [lats for _, lats, _ in sample_points],
            [lons for lons, _, _ in sample_points],
            [alts for _, _, alts in sample_points],
            [timestamps[0]]*len(sample_points),
            ensemble=False
        )
    print('Request done!')
    
    if ensemble:
        out = {'order': sample_points, 'results':{}}
        
        merged_data = {}

        for elements in data:
            for item in elements:
                if item is None:
                    continue
                
                name = item['name']
                reports = item['reports']

                if name not in merged_data:
                    merged_data[name] = {'name': name, 'reports': {'wind_speeds': [], 'wind_directions': []}}

                for report in reports:
                    merged_data[name]['reports']['wind_speeds'].append(report['wind_speed'])
                    merged_data[name]['reports']['wind_directions'].append(report['wind_direction'])

        result = list(merged_data.values())
        
        out['results'] = result
    
    else:
        print('Preparing final structure...')
        out = {'order': sample_points, 'results':{'wind_speeds': [], 'wind_directions': []}}
        
        for point in tqdm(data):
                if point is None:
                    continue
                out['results']['wind_speeds'].append(point['wind_speed'])
                out['results']['wind_directions'].append(point['wind_direction'])
    
    print(f'Saving file (dump JASON to {file_name})...')
    with open(file_name, "w") as outfile: 
        json.dump(out, outfile)
    print('Done!')

    
    
    



multi_point_query(box, timestamps=TIMESTAMPS,d=SEP_DIST, ensemble=ENSEMBLE, file_name=OUTPATH)


