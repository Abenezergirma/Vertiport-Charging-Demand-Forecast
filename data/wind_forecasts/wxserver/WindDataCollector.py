import numpy as np
import pyproj
import matplotlib.pyplot as plt
from time import perf_counter
from typing import List
import requests
import json
from tqdm import tqdm
import time
import os

CERT = "./gwu_client.crt"
KEY = "./gwu_client.key"

TIMESTAMPS=['2021-04-20T07:40:00+00:00']
ROUTE = np.array([[-96.935,32.5876],[-96.6058,32.802]])
OUTPATH = 'wind_data_at_2_5km.json'
SEP_DIST = 1000# In m
ENSEMBLE = False


class WindDataCollector:
    def __init__(self, cert_path: str, key_path: str, out_dir='Wind-Forecasts'):
        self.certificate = cert_path
        self.key = key_path
        self.out_dir = os.path.join(os.getcwd(), out_dir)  # Create full path to the output directory
        os.makedirs(self.out_dir, exist_ok=True)  # Create the directory if it does not exist
        self.session = requests.Session()
        self.session.cert = (self.certificate, self.key)
        self.timestamps = ['2021-04-20T06:40:00+00:00']

    def clean_line(self, line: str):
        route = line.split(",")[-1]
        points = route.split(" ")
        points_converted = np.array([p.split("/") for p in points])
        # print(points_converted)
        lats = points_converted[:, 0].astype(float).tolist()
        lons = points_converted[:, 1].astype(float).tolist()
        alts = points_converted[:, 2].astype(float).tolist()
        timestamps = [self.timestamps[0] for _ in range(len(lats))]
        return lats, lons, alts, timestamps

    def server_request(self,lats: List[float], lons: List[float], alts: List[float], timestamps: List[str], ensemble: bool = False):
        """Handles the requestinng of the weather data from the wxserver"""

        # Assert that valid data is passed
        assert len(lats) == len(lons) == len(alts) == len(timestamps)

        no_points = len(lats)

        if not self.certificate:
            self.certificate = CERT

        if not self.key:
            self.key = KEY

        session = requests.Session()
        session.cert = (self.certificate, self.key)

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

    def process_trajectory_file(self, file_path: str, initial_index: int, final_index: int):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data_collection = {'data': []}
        for line in lines[initial_index:final_index]:
            lats, lons, alts, timestamps = self.clean_line(line)
            # print(timestamps)
            
            # Ensure altitudes are at least 100 meters
            adjusted_alts = [max(float(alt), 100.0) for alt in alts]

            data = self.server_request(lats, lons, adjusted_alts, timestamps)
            if not data:
                continue

            wind_data = {'order': [lats, lons, adjusted_alts, timestamps], 'results': {'wind_speeds': [], 'wind_directions': []}}
            for point in data:
                wind_data['results']['wind_speeds'].append(point['wind_speed'])
                wind_data['results']['wind_directions'].append(point['wind_direction'])
            data_collection['data'].append(wind_data)

        with open(self.out_path, 'w') as outfile:
            json.dump(data_collection, outfile)
            print(f'Data saved to {self.out_path}')
    
    def process_trajectory_file_in_mass(self, file_path: str, initial_index: int, final_index: int):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        chunk_size = 20  # Define the size of each chunk
        for start_idx in range(initial_index, final_index, chunk_size):
            end_idx = min(start_idx + chunk_size, final_index)
            data_collection = {'data': []}

            for line in lines[start_idx:end_idx]:
                lats, lons, alts, timestamps = self.clean_line(line)
                adjusted_alts = [max(float(alt), 100.0) for alt in alts]
                data = self.server_request(lats, lons, adjusted_alts, timestamps)
                if not data:
                    continue

                wind_data = {'order': [lats, lons, adjusted_alts, timestamps], 'results': {'wind_speeds': [], 'wind_directions': []}}
                for point in data:
                    wind_data['results']['wind_speeds'].append(point['wind_speed'])
                    wind_data['results']['wind_directions'].append(point['wind_direction'])
 
                data_collection['data'].append(wind_data)

            out_filename = f'{self.out_dir}/wind_data_{start_idx}_{end_idx-1}.json'
            with open(out_filename, 'w') as outfile:
                json.dump(data_collection, outfile)
                print(f'Data saved to {out_filename}')
                
        
    def project_coordinates(self, min_lat, max_lat, min_lon, max_lon):
        transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        min_x, min_y = transformer.transform(min_lon, min_lat)
        max_x, max_y = transformer.transform(max_lon, max_lat)
        return min_x, max_x, min_y, max_y

    def divide_and_sample(self, min_x, max_x, min_y, max_y, distance_between_points=100, alts=[400]):
        transformer_cartesian_to_geo = pyproj.Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
        small_boxes = []

        width = max_x - min_x
        height = max_y - min_y

        # Calculate the number of divisions needed to get close to 3200 meters for each small box
        points_per_side = 32  # This should give us 32x32 points = 1024 points per box
        box_width = points_per_side * distance_between_points
        box_height = points_per_side * distance_between_points

        # Number of boxes along each dimension
        boxes_x = int(width / box_width)
        boxes_y = int(height / box_height)

        for i in range(boxes_x):
            for j in range(boxes_y):
                box_min_x = min_x + i * box_width
                box_max_x = box_min_x + box_width
                box_min_y = min_y + j * box_height
                box_max_y = box_min_y + box_height

                # Convert box corners to geographic coordinates
                geo_min_lon, geo_min_lat = transformer_cartesian_to_geo.transform(box_min_x, box_min_y)
                geo_max_lon, geo_max_lat = transformer_cartesian_to_geo.transform(box_max_x, box_max_y)

                # Create grid points
                grid_points = np.mgrid[box_min_x:box_max_x:distance_between_points, box_min_y:box_max_y:distance_between_points].reshape(2, -1).T
                converted_points = [transformer_cartesian_to_geo.transform(x, y) for x, y in grid_points]
                points_with_altitudes = [(lon, lat, alt) for lon, lat in converted_points for alt in alts]

                small_boxes.append({
                    'box': (geo_min_lon, geo_max_lon, geo_min_lat, geo_max_lat),
                    'points': points_with_altitudes
                })

        return small_boxes


    def plot_boxes(self, large_box, small_boxes, plot_points=False):
        fig, ax = plt.subplots()
        # Plot large box
        min_x, max_x, min_y, max_y = large_box
        ax.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], 'r-', label='Large Box')

        # Plot small boxes and optionally points
        for box in small_boxes:
            box_min_x, box_max_x, box_min_y, box_max_y = box['box']
            ax.plot([box_min_x, box_max_x, box_max_x, box_min_x, box_min_x], [box_min_y, box_min_y, box_max_y, box_max_y, box_min_y], 'b--', label='Small Box')
            if plot_points:
                points = [point[:2] for point in box['points']]  # Extract only x and y for plotting
                ax.scatter(*zip(*points), s=10, c='g', label='Sampled Points')

        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title("Box Division and Sampling")
        # ax.legend()
        ax.grid(True)
        plt.show()
    
    def plot_small_box(self, box, file_index):
        fig, ax = plt.subplots()
        geo_min_lon, geo_max_lon, geo_min_lat, geo_max_lat = box['box']
        points = box['points']

        # Draw small box boundaries
        ax.plot([geo_min_lon, geo_max_lon, geo_max_lon, geo_min_lon, geo_min_lon],
                [geo_min_lat, geo_min_lat, geo_max_lat, geo_max_lat, geo_min_lat], 'b-', label='Small Box')
        # Plot points
        ax.scatter([p[0] for p in points], [p[1] for p in points], c='r', label='Sample Points')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Small Box {file_index} Visualization')
        ax.legend()
        plt.show()

    def multi_point_query(self, min_lat, max_lat, min_lon, max_lon, d=500, alts=[400], timestamps=['2021-04-20T07:40:00+00:00'], ensemble=False, dir_name='data_storage', sleep_every=10, sleep_time=60):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        # Project and sample points
        min_x, max_x, min_y, max_y = self.project_coordinates(min_lat, max_lat, min_lon, max_lon)
        small_boxes = self.divide_and_sample(min_x, max_x, min_y, max_y, distance_between_points=d, alts=alts)

        print(f'Sampling Points Across Small Boxes')

        # Process each small box individually
        for file_index, box in enumerate(small_boxes):
            print(f'\nProcessing Small Box {file_index + 1}/{len(small_boxes)}')
            print(len(box['points']))
            # self.plot_small_box(box, file_index)
            
            data = []
            if ensemble:
                for point in box['points']:
                    response = self.server_request([point[1]], [point[0]], [point[2]], [timestamps[0]], ensemble=ensemble)
                    data.append(response)
                    if len(data) % sleep_every == 0:
                        print(f'Sleeping for {sleep_time} seconds to avoid overloading the server')
                        time.sleep(sleep_time)
            else:
                lats, lons,  alts = zip(*[(point[1], point[0], point[2]) for point in box['points']])
                sample_points =  [list(t) for t in zip(list(lons), list(lats), list(alts))]
                # print(sample_points)
                print("Requesting points...")
                data = self.server_request(
                    [lats for _, lats, _ in sample_points],
                    [lons for lons, _, _ in sample_points],
                    [alts for _, _, alts in sample_points],
                    [timestamps[0]]*len(sample_points),
                    ensemble=False
                )
            print('Request done!')
            print('Preparing final structure...')
            out = {'order': sample_points, 'results':{'wind_speeds': [], 'wind_directions': []}}
        
            for point in tqdm(data):
                if point is None:
                    continue
                out['results']['wind_speeds'].append(point['wind_speed'])
                out['results']['wind_directions'].append(point['wind_direction'])
                
            file_path = os.path.join(dir_name, f'box_{file_index}.json')
            print(f'Saving data for Small Box {file_index + 1}')
            with open(file_path, "w") as outfile:
                json.dump(out, outfile)

            print(f'Data saved for Small Box {file_index + 1} in {file_path}')

    

# Usage
collector = WindDataCollector(cert_path='./gwu_client.crt', key_path='./gwu_client.key')
# collector.process_trajectory_file_in_mass('../DFW_SCENARIOS/DFW_CMU_APT_CGO.txt', initial_index = 101, final_index=105)
min_lat = 32.58769
max_lat = 33.18819
min_lon = -97.40959
max_lon = -96.46249
collector.multi_point_query(32.58769, 33.18819, -97.40959, -96.46249)

# min_x, max_x, min_y, max_y = collector.project_coordinates(min_lat, max_lat, min_lon, max_lon)
# small_boxes = collector.divide_and_sample(min_x, max_x, min_y, max_y)
# collector.plot_boxes((min_x, max_x, min_y, max_y), small_boxes, plot_points=False)



 
