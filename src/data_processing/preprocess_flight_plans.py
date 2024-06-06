import os
import pandas as pd
import geopy.distance
import matplotlib.pyplot as plt
import json
from shapely.geometry import Point, LineString, mapping
import fiona 
from fiona.crs import from_epsg 
from collections import defaultdict
plt.rcParams['text.usetex'] = True

DATAFOLDER = '/home/abenezertaye/Desktop/Research/Codes/NASA/Vertiport-Charging-Demand-Forecast/data'

class PreprocessFlightPlans:
    def __init__(self, flight_plan_file, wind_model_file):
        self.flight_plan_file = os.path.join(DATAFOLDER, flight_plan_file)
        self.wind_model_file = os.path.join(DATAFOLDER, wind_model_file)
        self.filtered_flight_plans = {}
        self.filtered_wind_models = {}
        self.unique_points = set()
        self.features_points = []
        self.features_lines = []
        self.vertiport_flights = defaultdict(list)  # Maps vertiports to flight plans
        self.vertiport_energy = {}  # Maps vertiports to lists of energy consumptions
        self.flight_plan_data = {}
        self.unique_vertiport_flights = None
        self.new_flight_plans_path = os.path.join(DATAFOLDER, 'flight_plans/new_filtered_flight_plans.json')

    def load_json(self, file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    def filter_and_rename_flight_plans(self):
        flight_plans = self.load_json(self.flight_plan_file)
        wind_models = self.load_json(self.wind_model_file)
        
        new_key_index = 1
        for key in sorted(flight_plans.keys(), key=lambda x: int(x.split('_')[-1])):
            coordinates = flight_plans[key]
            if len(coordinates) != 2:
                print(f"Skipping {key} as it does not have two coordinate pairs")
                continue

            initial_pos = tuple(coordinates[0])  # Using as is
            final_pos = tuple(coordinates[1])
            
            # Calculate distance
            try:
                distance = geopy.distance.geodesic(initial_pos, final_pos).km
            except ValueError as e:
                print(f"Error calculating distance for {key}: {e}")
                continue

            if distance >= 40:
                new_key = f"Flight_plan_{new_key_index}"
                self.filtered_flight_plans[new_key] = coordinates
                print(f"{key} calculated distance: {distance} km")  # Debugging distance calculation

                if key in wind_models:
                    self.filtered_wind_models[new_key] = wind_models[key]
                
                new_key_index += 1

        self.save_filtered_data(os.path.join(DATAFOLDER, 'flight_plans/new_filtered_flight_plans.json'), self.filtered_flight_plans)
        self.save_filtered_data(os.path.join(DATAFOLDER, 'wind_forecasts/new_charging_wind_models.json'), self.filtered_wind_models)

    def split_and_save_json_files(self, batch_size=5):
        # Load the processed and filtered flight plans and wind models
        flight_plans = self.load_json(os.path.join(DATAFOLDER, 'flight_plans/new_filtered_flight_plans.json'))
        wind_models = self.load_json(os.path.join(DATAFOLDER, 'wind_forecasts/new_charging_wind_models.json'))
        
        # Calculate the number of batches
        num_batches = (len(flight_plans) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size
            batch_flight_plans = {k: flight_plans[k] for j, k in enumerate(flight_plans) if start_index <= j < end_index}
            batch_wind_models = {k: wind_models[k] for j, k in enumerate(wind_models) if start_index <= j < end_index}
            print(f'new_filtered_flight_plans_{start_index+1}_{end_index}')
            print(f'new_charging_wind_models_{start_index+1}_{end_index}')
            # Save the batches to new files
            self.save_filtered_data(f'flight_plans/new_filtered_flight_plans_{start_index+1}_{end_index}.json', batch_flight_plans)
            self.save_filtered_data(f'wind_forecasts/new_charging_wind_models_{start_index+1}_{end_index}.json', batch_wind_models)


    def save_filtered_data(self, file_name, data):
        with open(os.path.join(DATAFOLDER, file_name), 'w') as file:
            json.dump(data, file, indent=4)
            
    def print_distance(self, file_name):
        flight_plans = self.load_json(os.path.join(DATAFOLDER, file_name))
        
        for key in sorted(flight_plans.keys(), key=lambda x: int(x.split('_')[-1])):
            coordinates = flight_plans[key]
            if len(coordinates) != 2:
                print(f"Skipping {key} as it does not have two coordinate pairs")
                continue

            initial_pos = tuple(coordinates[0])  # Using as is
            final_pos = tuple(coordinates[1])
            points = [initial_pos, final_pos]
            # Extracting latitude and longitude
            latitudes, longitudes = zip(*points)
            
            # Calculate distance
            try:
                distance = geopy.distance.geodesic(initial_pos, final_pos).km
                print(f"{key} calculated distance: {distance} km")  # Debugging distance calculation
            except ValueError as e:
                print(f"Error calculating distance for {key}: {e}")
                continue
         
    def parse_excel_files(self, directory_path):
        data = []

        # Loop through all .xlsx files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith(".xlsx"):
                file_path = os.path.join(directory_path, filename)
                self.process_excel_file(file_path, data)

        if data:
            print("Data loaded correctly")
        else:
            print("No data to plot.")

    def process_excel_file(self, file_path, data):
        # Load the Excel file
        df = pd.read_excel(file_path, header=None)

        # Loop through the DataFrame row by row
        current_flight_plan = None
        for index, row in df.iterrows():
            # Assuming the Flight Plan Name is in the first cell of some rows
            if pd.notna(row[0]) and 'Flight Plan Name:' in str(row[0]):
                current_flight_plan = str(row[0]).split(':')[-1].strip()
            # Check for the row that contains 'Speed' directly above the speed value
            if pd.notna(row[0]) and row[0] == 'Speed':
                if current_flight_plan:
                    speed = df.iloc[index + 1, 0]  # Speed value is in the next row, same column
                    total_optimal_cost = df.iloc[index + 1, 1]  # Cost is in the next row, next column
                    data.append({
                        'Flight Plan Name': current_flight_plan,
                        'Speed': speed,
                        'Total Optimal Cost': total_optimal_cost
                    })
                    # Update dictionary for global access if needed
                    self.flight_plan_data[current_flight_plan] = {
                        'Speed': speed,
                        'Optimal Cost': total_optimal_cost
                    }

    def plot_energy_consumption_data(self, data):
        # Convert list of data into DataFrame for easier plotting
        plot_df = pd.DataFrame(data)
        
        # Extract flight plan numbers
        plot_df['Flight Plan Number'] = plot_df['Flight Plan Name'].apply(lambda x: int(x.split('_')[-1]))

        # First plot: Total Optimal Cost vs. Flight Plan Number
        fig1, ax1 = plt.subplots(figsize=(4, 4))  # Half A4 size in height

        plot_df.groupby('Flight Plan Number')['Total Optimal Cost'].mean().plot(
            kind='bar', color='deepskyblue', ax=ax1
        )
        ax1.set_title('Energy Consumption of Each Flight Plan', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Flight Plan', fontsize=12)
        ax1.set_ylabel('Energy Consumption (MJ)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='both', labelsize=10)
        # Customize the x-axis
        N =112
        ax1.set_xticks(range(1, N+1, 10))  # Set ticks at intervals, for example, every 10th item
        ax1.set_xticklabels(range(1, N+1, 10))

        plt.tight_layout(pad=2.0)
        plt.savefig(os.path.join('data', 'total_optimal_cost_by_flight_plan.png'), dpi=300, bbox_inches='tight')

        
    def process_data(self,file_name):
        flight_plans = self.load_json(file_name)
        for flight_plan_key, coordinates in self.filtered_flight_plans.items():
            initial_pos = tuple(coordinates[0][::-1])  # First coordinate pair in the list
            # print(initial_pos)
            final_pos = tuple(coordinates[1][::-1])    # Second coordinate pair in the list
            
            
            # Map initial and final positions to the flight plan
            self.map_vertiport_to_flight_plan(initial_pos, flight_plan_key)
            self.map_vertiport_to_flight_plan(final_pos, flight_plan_key)


            # Handle unique points
            if initial_pos not in self.unique_points:
                self.unique_points.add(initial_pos)
                point_feature = {
                    'type': 'Feature',
                    'properties': {},
                    'geometry': mapping(Point(initial_pos))
                }
                self.features_points.append(point_feature)
                # print('here_______________________',self.features_points)


            if final_pos not in self.unique_points:
                self.unique_points.add(final_pos)
                point_feature = {
                    'type': 'Feature',
                    'properties': {},
                    'geometry': mapping(Point(final_pos))
                }
                self.features_points.append(point_feature)
                        # Update unique vertiports set
            self.unique_points.update([initial_pos, final_pos])

            # Handle line feature
            line = LineString([initial_pos, final_pos])
            line_feature = {
                'type': 'Feature',
                'properties': {},
                'geometry': mapping(line)
            }
            self.features_lines.append(line_feature)

    def save_geojson(self, output_file_points, output_file_lines):
        schema_points = {'geometry': 'Point', 'properties': {}}
        schema_lines = {'geometry': 'LineString', 'properties': {}}

        with fiona.open(output_file_points, 'w', driver='GeoJSON', crs=from_epsg(4326), schema=schema_points) as points:
            # print('here _________________________',self.features_points)
            points.writerecords(self.features_points)
        
        with fiona.open(output_file_lines, 'w', driver='GeoJSON', crs=from_epsg(4326), schema=schema_lines) as lines:
            lines.writerecords(self.features_lines)

        print(f"Files saved: {output_file_points} and {output_file_lines}")

    def get_vertiport_coordinates(self, vertiport):

        return vertiport

    def map_vertiport_to_flight_plan(self, vertiport, flight_plan_key):
        # Adds the flight plan to the list of plans associated with the vertiport
        if vertiport not in self.vertiport_flights:
            self.vertiport_flights[vertiport] = []
        self.vertiport_flights[vertiport].append(flight_plan_key)
        # print(self.vertiport_flights)

    def map_vertiports_to_flight_plans(self):
        # print("here")
        if not self.vertiport_flights:
            self.load_json(self.new_flight_plans_path)
            self.process_data(self.new_flight_plans_path)

        unique_vertiport_flights = defaultdict(list)
        # flight_plans = self.load_json(self.flight_plan_file)

        for flight_plan_key, coordinates in self.filtered_flight_plans.items():
            departure_coords = tuple(coordinates[0][::-1])  # First set of coordinates
            for vertiport in self.vertiport_flights:
                if self.get_vertiport_coordinates(vertiport) == departure_coords:
                    unique_vertiport_flights[vertiport].append(flight_plan_key)
                    break
            else:
                print(f"No matching vertiport found for departure coordinates {departure_coords} of flight plan {flight_plan_key}")

        # Print the final mappings
        for vertiport, flights in unique_vertiport_flights.items():
            print(f"Vertiport {vertiport} is uniquely associated with flight plans: {flights}")
        print(len(unique_vertiport_flights))
        self.unique_vertiport_flights = unique_vertiport_flights
        
        return unique_vertiport_flights
    
if __name__ == '__main__':
    flight_plans = 'flight_plans/filtered_flight_plans.json'
    wind_models = 'wind_forecasts/Charging_wind_models.json'
    preprocess_FP = PreprocessFlightPlans(flight_plans, wind_models)
    preprocess_FP.filter_and_rename_flight_plans()
    preprocess_FP.split_and_save_json_files()
    preprocess_FP.print_distance("flight_plans/new_filtered_flight_plans_1_5.json")

    energy_consumption_files = os.path.join(DATAFOLDER, 'energy_consumptions/Total-Energy-Profiles')
    preprocess_FP.parse_excel_files(energy_consumption_files)

    preprocess_FP.process_data(os.path.join(DATAFOLDER,"flight_plans/new_filtered_flight_plans.json"))

    vertiport_geojson = os.path.join(DATAFOLDER,'flight_plans/vertiports_new.geojson')
    flights_geojson = os.path.join(DATAFOLDER,'flight_plans/flight_paths.geojson')

    preprocess_FP.save_geojson(vertiport_geojson, flights_geojson)
    preprocess_FP.map_vertiports_to_flight_plans()
    # preprocess_FP.integrate_and_visualize_data(energy_consumption_files) vertiport_geojson