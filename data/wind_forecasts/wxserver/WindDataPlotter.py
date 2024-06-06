import json
import matplotlib.pyplot as plt
import numpy as np
import folium 
from scipy.interpolate import griddata
import os
# Enable LaTeX in Matplotlib
plt.rcParams['text.usetex'] = True

class WindDataPlotter:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.load_data()

    def load_data(self):
        """Load data from a JSON file."""
        with open(self.filename, 'r') as file:
            data = json.load(file)
        return data

    def plot_trajectories(self, amplify=False, amplification_factor=1):
        """Plot each trajectory with wind speeds and directions, using a unified color scale."""
        plt.figure(figsize=(14, 10))  # Adjust the figure size as needed

        all_wind_speeds = []  # List to store all valid wind speeds across all trajectories
        for item in self.data['data']:
            wind_speeds = [float(ws) for ws in item['results']['wind_speeds'] if ws is not None]
            lats = [float(lat) for lat in item['order'][0] if lat is not None]
            lons = [float(lon) for lon in item['order'][1] if lon is not None]
            valid = np.isfinite(wind_speeds) & np.isfinite(lats) & np.isfinite(lons)
            valid_wind_speeds = np.array(wind_speeds)[valid]  # Filter only valid wind speeds
            if amplify:
                valid_wind_speeds *= amplification_factor
            all_wind_speeds.extend(valid_wind_speeds)

        vmin = np.min(all_wind_speeds) if all_wind_speeds else 0
        vmax = np.max(all_wind_speeds) if all_wind_speeds else 1

        cmap = plt.cm.coolwarm  # Diverging colormap

        for index, item in enumerate(self.data['data']):
            lats = np.array([float(lat) for lat in item['order'][0] if lat is not None], dtype=np.float64)
            lons = np.array([float(lon) for lon in item['order'][1] if lon is not None], dtype=np.float64)
            wind_speeds = np.array([float(ws) for ws in item['results']['wind_speeds'] if ws is not None], dtype=np.float64)
            wind_directions = np.array([float(wd) for wd in item['results']['wind_directions'] if wd is not None], dtype=np.float64)
            valid = ~np.isnan(wind_speeds) & ~np.isnan(wind_directions) & ~np.isnan(lats) & ~np.isnan(lons)
            lats, lons, wind_speeds, wind_directions = lats[valid], lons[valid], wind_speeds[valid], wind_directions[valid]

            # Normalize the wind direction vectors
            wind_dirs_radians = np.deg2rad(wind_directions)
            U = np.cos(wind_dirs_radians)
            V = np.sin(wind_dirs_radians)
            magnitude = np.sqrt(U**2 + V**2)
            U_norm = U / magnitude
            V_norm = V / magnitude
            arrow_spacing = 10
            scatter = plt.scatter(lons, lats, c=wind_speeds, cmap=cmap, vmin=vmin, vmax=vmax, label=f'Trajectory {index + 1}')
            plt.quiver(lons[::arrow_spacing], lats[::arrow_spacing], U_norm[::arrow_spacing], V_norm[::arrow_spacing], color='red', scale=90)

        plt.colorbar(scatter, label='Wind Speed (m/s)')
        plt.title('Trajectories with Wind Data')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def plot_individual_trajectories(self, arrow_spacing=10):
        """Plot wind data for each trajectory on separate plots."""
        # Initialize variables to find the global min and max wind speed
        global_min_speed = float('inf')
        global_max_speed = float('-inf')

        # First pass to determine the global min and max wind speeds
        for item in self.data['data']:
            wind_speeds = [float(ws) for ws in item['results']['wind_speeds'] if ws is not None]
            if wind_speeds:  # Ensure the list is not empty
                local_min = min(wind_speeds)
                local_max = max(wind_speeds)
                global_min_speed = min(global_min_speed, local_min)
                global_max_speed = max(global_max_speed, local_max)

        # Plotting each trajectory individually
        for index, item in enumerate(self.data['data']):
            plt.figure(figsize=(10, 8))
            lats = np.array([float(lat) for lat in item['order'][0] if lat is not None], dtype=np.float64)
            lons = np.array([float(lon) for lon in item['order'][1] if lon is not None], dtype=np.float64)
            wind_speeds = np.array([float(ws) for ws in item['results']['wind_speeds'] if ws is not None], dtype=np.float64)
            wind_directions = np.array([float(wd) for wd in item['results']['wind_directions'] if wd is not None], dtype=np.float64)
            
            # Ensure all data points are valid
            valid = ~np.isnan(wind_speeds) & ~np.isnan(wind_directions) & ~np.isnan(lats) & ~np.isnan(lons)
            valid_lats = lats[valid]
            valid_lons = lons[valid]
            valid_wind_speeds = wind_speeds[valid]
            valid_wind_directions = wind_directions[valid]

            # Determine color scale limits from valid data
            vmin = np.min(valid_wind_speeds) if len(valid_wind_speeds) > 0 else 0
            vmax = np.max(valid_wind_speeds) if len(valid_wind_speeds) > 0 else 1


            # valid = ~np.isnan(wind_speeds) & ~np.isnan(wind_directions) & ~np.isnan(lats) & ~np.isnan(lons)
            # lats, lons, wind_speeds, wind_directions = lats[valid], lons[valid], wind_speeds[valid], wind_directions[valid]

            wind_dirs_radians = np.deg2rad(valid_wind_directions)
            U = np.cos(wind_dirs_radians)
            V = np.sin(wind_dirs_radians)
            
            plt.plot(valid_lons, valid_lats)

            scatter = plt.scatter(valid_lons, valid_lats, c=valid_wind_speeds, cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar(scatter, label='Wind Speed (m/s)')
            plt.quiver(valid_lons[::arrow_spacing], valid_lats[::arrow_spacing], U[::arrow_spacing], V[::arrow_spacing], color='red', scale=50)

            plt.title(f'Trajectory {index + 1} with Wind Data')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True)
            plt.show()
    

    def plot_wind_data(self, file_path, subsample_rate=2, vector_scale=60):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                order_data = data.get('order')
                wind_speeds = data.get('results', {}).get('wind_speeds')
                wind_directions = data.get('results', {}).get('wind_directions')

                if order_data and wind_speeds and wind_directions and \
                len(order_data) == len(wind_speeds) == len(wind_directions):
                    # Subsample the data
                    subsampled_indices = np.arange(0, len(order_data), subsample_rate)
                    longitudes = np.array([point[0] for point in order_data])[subsampled_indices]
                    latitudes = np.array([point[1] for point in order_data])[subsampled_indices]
                    U = np.array([speed * np.cos(np.deg2rad(direction)) for speed, direction in zip(wind_speeds, wind_directions)])[subsampled_indices]
                    V = np.array([speed * np.sin(np.deg2rad(direction)) for speed, direction in zip(wind_speeds, wind_directions)])[subsampled_indices]

                    # Normalize the vectors
                    plt.figure(figsize=(15, 10))
                    vector_magnitudes = np.sqrt(U**2 + V**2)
                    U_normalized = U / vector_magnitudes
                    V_normalized = V / vector_magnitudes
                    # Create a grid for the contour plot
                    grid_x, grid_y = np.mgrid[min(longitudes):max(longitudes):100j, min(latitudes):max(latitudes):100j]
                    grid_speed = griddata((longitudes, latitudes), vector_magnitudes, (grid_x, grid_y), method='linear')
                    plt.contourf(grid_x, grid_y, grid_speed, cmap=plt.cm.jet)
                    plt.colorbar(label=r'Wind Speed ($\frac{m}{s}$)')
                    
                    indices = np.arange(0, len(longitudes), subsample_rate)
                    plt.quiver(longitudes[indices], latitudes[indices], U_normalized[indices], V_normalized[indices], color='black', scale=vector_scale,width=0.002)
                    # quiver = plt.quiver(longitudes[indices], latitudes[indices], U[indices], V[indices], color='black', scale=vector_scale,width=0.002)                    
                    # quiver = plt.quiver(longitudes, latitudes, U_normalized, V_normalized, vector_magnitudes, angles='xy', scale_units='xy', scale=vector_scale, cmap=plt.cm.jet, width=0.002)

                    plt.title(r'\textbf{Wind Field: Direction and Magnitude}')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    plt.grid(True)
                    plt.show()
                else:
                    print("Data is missing or not in the expected format.")
        except FileNotFoundError:
            print("File not found.")
        except JSONDecodeError:
            print("Error decoding JSON.")
        except Exception as e:
            print(f"An error occurred: {e}")
        
    def plot_trajectories_on_map(self):
        """Plot valid trajectories on a folium map."""
        # Initialize the map centered around an average location
        mean_lat = np.mean([float(lat) for item in self.data['data'] for lat in item['order'][0] if lat is not None])
        mean_lon = np.mean([float(lon) for item in self.data['data'] for lon in item['order'][1] if lon is not None])
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=10)

        # Loop through each trajectory
        for index, item in enumerate(self.data['data']):
            # Extract the latitude and longitude lists
            lats = np.array([float(lat) for lat in item['order'][0] if lat is not None], dtype=np.float64)
            lons = np.array([float(lon) for lon in item['order'][1] if lon is not None], dtype=np.float64)
            wind_speeds = np.array([float(ws) for ws in item['results']['wind_speeds'] if ws is not None], dtype=np.float64)
            wind_directions = np.array([float(wd) for wd in item['results']['wind_directions'] if wd is not None], dtype=np.float64)
            
            # Ensure all data points are valid
            valid = ~np.isnan(wind_speeds) & ~np.isnan(wind_directions) & ~np.isnan(lats) & ~np.isnan(lons)
            valid_lats = lats[valid]
            valid_lons = lons[valid]
     
   
            # Create a line for the trajectory
            trajectory = list(zip(valid_lats, valid_lons))
            # folium.PolyLine(locations=trajectory, color='blue', weight=5, tooltip=f'Trajectory {index+1}').add_to(m)
                        # Plot each point with a marker
            for lat, lon in zip(valid_lats, valid_lons):
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=3,  # Size of the marker
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    tooltip=f'Trajectory {index + 1}'
                ).add_to(m)

        # Display the map
        return m

    def plot_aggregated_wind_data(self, directory_path, subsample_rate=10, vector_scale=60):
        longitudes = []
        latitudes = []
        wind_speeds = []
        wind_directions = []
        
        # Load all JSON files in the specified directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.json'):
                file_path = os.path.join(directory_path, filename)
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        order_data = data.get('order')
                        results = data.get('results')
                        if order_data and results:
                            wind_speeds.extend(results.get('wind_speeds', []))
                            wind_directions.extend(results.get('wind_directions', []))
                            for point in order_data:
                                longitudes.append(point[0])
                                latitudes.append(point[1])
                except Exception as e:
                    print(f"An error occurred while processing {filename}: {e}")

        # Prepare data for plotting
        if longitudes and latitudes and wind_speeds and wind_directions:
            longitudes = np.array(longitudes)
            latitudes = np.array(latitudes)
            wind_speeds = np.array(wind_speeds)
            wind_directions = np.array(wind_directions)

            # Calculate wind vector components
            U = wind_speeds * np.cos(np.deg2rad(wind_directions))
            V = wind_speeds * np.sin(np.deg2rad(wind_directions))

            # Normalize the vectors
            plt.figure(figsize=(15, 10))
            vector_magnitudes = np.sqrt(U**2 + V**2)
            U_normalized = U / vector_magnitudes
            V_normalized = V / vector_magnitudes

            # Create a grid for the contour plot
            grid_x, grid_y = np.mgrid[min(longitudes):max(longitudes):100j, min(latitudes):max(latitudes):100j]
            grid_speed = griddata((longitudes, latitudes), vector_magnitudes, (grid_x, grid_y), method='linear')
            
            plt.contourf(grid_x, grid_y, grid_speed, cmap=plt.cm.jet)
            plt.colorbar(label='Wind Speed (m/s)')
            
            indices = np.arange(0, len(longitudes), subsample_rate)
            plt.quiver(longitudes[indices], latitudes[indices], U_normalized[indices], V_normalized[indices], color='black', scale=vector_scale,width=0.002)
                   

            # plt.quiver(longitudes, latitudes, U_normalized, V_normalized, color='black', scale=vector_scale, width=0.002)
            plt.title('Wind Field: Direction and Magnitude')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True)
            plt.show()
        else:
            print("Data is missing or not in the expected format.")
    

# Usage
# plotter = WindDataPlotter('wind_data_at_2_5km.json')
# # plotter.plot_trajectories(amplify=True, amplification_factor=1)  # Amplify to visualize minor differences
# plotter.plot_wind_data('wind_data_at_2_5km.json')
# plotter.plot_aggregated_wind_data('data_storage')

# plotter.plot_individual_trajectories(arrow_spacing=10)
# map_plot = plotter.plot_trajectories_on_map()
# map_plot.save('trajectories_map.html')

