import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import math 
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from data_processing.preprocess_flight_plans import PreprocessFlightPlans  # Import the PreprocessFlightPlans class
from charging_model import ChargingModel  # Import the ChargingModel class
import random
DATAFOLDER = '/home/abenezertaye/Desktop/Research/Codes/NASA/Vertiport-Charging-Demand-Forecast/data'


class ChargingDemandForecasting(PreprocessFlightPlans):
    def __init__(self, flight_plans: str, wind_models: str):
        super().__init__(flight_plans, wind_models)
        M = 320
        S = 3.75
        self.charging_model = ChargingModel(M, S)  # Instantiate the charging class
        self.charging_related_results = {}
        self.cruise_speeds = {}
        self.aircraft_data = None

    def map_vertiports_to_flight_plans(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Map vertiports to their associated departing and arriving flight plans.
        """
        self.filter_and_rename_flight_plans()

        if not self.vertiport_flights:
            self.load_json(self.flight_plan_file)
            self.process_data(self.flight_plan_file)

        vertiport_flight_mapping = defaultdict(lambda: {'departing': [], 'arriving': []})

        for flight_plan_key, coordinates in self.filtered_flight_plans.items():
            departure_coords = tuple(coordinates[0][::-1])  # First set of coordinates for departure
            arrival_coords = tuple(coordinates[-1][::-1])   # Last set of coordinates for arrival

            for vertiport in self.vertiport_flights:
                vertiport_coords = self.get_vertiport_coordinates(vertiport)

                if vertiport_coords == departure_coords:
                    vertiport_flight_mapping[vertiport]['departing'].append(flight_plan_key)
                if vertiport_coords == arrival_coords:
                    vertiport_flight_mapping[vertiport]['arriving'].append(flight_plan_key)

        # Print the final mappings
        for vertiport, flights in vertiport_flight_mapping.items():
            print(f"Vertiport {vertiport} is associated with departing flight plans: {flights['departing']} and arriving flight plans: {flights['arriving']}")
        print(len(vertiport_flight_mapping))
        self.unique_vertiport_flights = vertiport_flight_mapping

        return vertiport_flight_mapping
    
    def categorize_and_print_vertiports(self, vertiport_flight_mapping: Dict[str, Dict[str, List[str]]], threshold: float = 500) -> None:
        """
        Categorize vertiports and print the ones that only have departing flight plans, only arriving flight plans,
        and those with both, along with their flight plans. Vertiports with coordinates within a certain distance
        threshold are merged.
        
        :param vertiport_flight_mapping: Dictionary mapping vertiports to their departing and arriving flight plans.
        :param threshold: Distance threshold in meters to merge vertiports.
        """
        vertiports = {vertiport: self.get_vertiport_coordinates(vertiport) for vertiport in vertiport_flight_mapping.keys()}
        merged_vertiports = self.merge_close_vertiports(vertiports, threshold)

        # Remap flight plans to merged vertiports
        merged_vertiport_flight_mapping = defaultdict(lambda: {'departing': [], 'arriving': []})
        for vertiport, flights in vertiport_flight_mapping.items():
            merged_vertiport = next(mv for mv, coords in merged_vertiports.items() if self.haversine(coords, self.get_vertiport_coordinates(vertiport)) < threshold)
            merged_vertiport_flight_mapping[merged_vertiport]['departing'].extend(flights['departing'])
            merged_vertiport_flight_mapping[merged_vertiport]['arriving'].extend(flights['arriving'])

        # Remove duplicates in flight plans
        for vertiport in merged_vertiport_flight_mapping:
            merged_vertiport_flight_mapping[vertiport]['departing'] = list(set(merged_vertiport_flight_mapping[vertiport]['departing']))
            merged_vertiport_flight_mapping[vertiport]['arriving'] = list(set(merged_vertiport_flight_mapping[vertiport]['arriving']))

        # Categorize vertiports
        departing_only = {}
        arriving_only = {}
        both_departing_and_arriving = {}

        for vertiport, flights in merged_vertiport_flight_mapping.items():
            num_departing = len(flights['departing'])
            num_arriving = len(flights['arriving'])

            if num_departing > 0 and num_arriving == 0:
                departing_only[vertiport] = flights
            elif num_arriving > 0 and num_departing == 0:
                arriving_only[vertiport] = flights
            elif num_departing > 0 and num_arriving > 0:
                both_departing_and_arriving[vertiport] = flights

        def print_vertiport_info(vertiports, category):
            print(f"\n{category}:")
            vertiport_counter = 1
            for vertiport, flights in vertiports.items():
                num_departing = len(flights['departing'])
                num_arriving = len(flights['arriving'])
                print(f"Vertiport {vertiport_counter}: {vertiport}")
                print(f"  Departing flight plans ({num_departing}): {flights['departing']}")
                print(f"  Arriving flight plans ({num_arriving}): {flights['arriving']}")
                vertiport_counter += 1

        print_vertiport_info(departing_only, "Vertiports with Only Departing Flight Plans")
        print_vertiport_info(arriving_only, "Vertiports with Only Arriving Flight Plans")
        print_vertiport_info(both_departing_and_arriving, "Vertiports with Both Departing and Arriving Flight Plans")
        
        return departing_only, arriving_only, both_departing_and_arriving
            
    def merge_close_vertiports(self, vertiports: Dict[str, Tuple[float, float]], threshold: float = 1000) -> Dict[str, Tuple[float, float]]:
        """
        Merge vertiports that are within a certain distance threshold.
        :param vertiports: Dictionary of vertiports with their coordinates.
        :param threshold: Distance threshold in meters to merge vertiports.
        :return: Merged vertiports dictionary.
        """
        merged_vertiports = {}
        used = set()

        vertiport_items = list(vertiports.items())
        for i, (vertiport1, coords1) in enumerate(vertiport_items):
            if vertiport1 in used:
                continue
            merged_vertiports[vertiport1] = coords1
            for j, (vertiport2, coords2) in enumerate(vertiport_items):
                if i != j and vertiport2 not in used:
                    if self.haversine(coords1, coords2) < threshold:
                        used.add(vertiport2)
                        # Average the coordinates
                        merged_vertiports[vertiport1] = (
                            (merged_vertiports[vertiport1][0] + coords2[0]) / 2,
                            (merged_vertiports[vertiport1][1] + coords2[1]) / 2
                        )

        return merged_vertiports
    
    
    def haversine(self, coord1, coord2):
        """
        Calculate the great-circle distance between two points on the Earth.
        :param coord1: Tuple (longitude, latitude)
        :param coord2: Tuple (longitude, latitude)
        :return: Distance in meters
        """
        R = 6371000  # Radius of the Earth in meters
        lon1, lat1 = math.radians(coord1[0]), math.radians(coord1[1])
        lon2, lat2 = math.radians(coord2[0]), math.radians(coord2[1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    def are_coordinates_close(self, coord1, coord2, threshold):
        return self.haversine(coord1, coord2) < threshold
    
    
    def assign_aircraft_to_flight_plans(self, vertiport_flight_mapping) -> Dict[int, List[str]]:
        departing_only, arriving_only, both_departing_and_arriving = self.categorize_and_print_vertiports(vertiport_flight_mapping)
        aircraft_assignment = {}
        aircraft_id = 1

        assigned_flight_plans = set()

        # Assign aircraft to flight plans not requiring recharging
        for vertiport, flights in departing_only.items():
            flight_plans = flights['departing']
            for flight_plan in flight_plans:
                if flight_plan not in assigned_flight_plans:
                    aircraft_assignment[aircraft_id] = [flight_plan]
                    assigned_flight_plans.add(flight_plan)
                    aircraft_id += 1

        for vertiport, flights in arriving_only.items():
            flight_plans = flights['arriving']
            for flight_plan in flight_plans:
                if flight_plan not in assigned_flight_plans:
                    aircraft_assignment[aircraft_id] = [flight_plan]
                    assigned_flight_plans.add(flight_plan)
                    aircraft_id += 1

        # Assign aircraft to arriving flight plans in both_departing_and_arriving
        arriving_aircraft_ids = {}
        for vertiport, flights in both_departing_and_arriving.items():
            arriving_flight_plans = flights['arriving']
            departing_flight_plans = flights['departing']

            # Assign arriving flight plans
            for flight_plan in arriving_flight_plans:
                if flight_plan not in assigned_flight_plans:
                    aircraft_assignment[aircraft_id] = [flight_plan]
                    arriving_aircraft_ids[flight_plan] = aircraft_id
                    assigned_flight_plans.add(flight_plan)
                    aircraft_id += 1

            # Assign departing flight plans to the same aircraft as arriving, if possible
            for flight_plan in departing_flight_plans:
                if flight_plan in assigned_flight_plans:
                    continue
                assigned_aircraft_id = None
                for arriving_flight_plan in arriving_flight_plans:
                    if arriving_flight_plan in arriving_aircraft_ids:
                        assigned_aircraft_id = arriving_aircraft_ids[arriving_flight_plan]
                        break
                if assigned_aircraft_id is None:
                    assigned_aircraft_id = aircraft_id
                    aircraft_assignment[aircraft_id] = []
                    aircraft_id += 1
                aircraft_assignment[assigned_aircraft_id].append(flight_plan)
                assigned_flight_plans.add(flight_plan)

        # Print the assignment in an easy-to-track format
        for aircraft, flight_plans in aircraft_assignment.items():
            print(f"Aircraft {aircraft}: {', '.join(flight_plans)}")

        return aircraft_assignment

    def integrate_energy_consumption(self, cost_files, aircraft_assignment: Dict[int, List[str]]):
        total_battery_energy = 676.8 * 10**6  # Total battery energy in Joules
        self.parse_excel_files(cost_files)
        results = {}

        for aircraft_id, flight_plans in aircraft_assignment.items():
            aircraft_data = []
            initial_soc = random.uniform(0.2, 0.3)  # Initial SoC between 20% and 30%
            
            for i, flight_plan in enumerate(flight_plans):
                # print(self.flight_plan_data)
                if i == 0:
                    energy_consumed = (1 - initial_soc) * total_battery_energy
                    soc_after_flight = 1 - self.flight_plan_data[flight_plan]['Optimal Cost'] * 10**6 / total_battery_energy
                    
                else:
                    previous_energy_consumption = self.flight_plan_data[flight_plans[i - 1]]['Optimal Cost'] * 10**6
                    soc_after_flight = 1 - previous_energy_consumption / total_battery_energy
                    energy_consumed = total_battery_energy - previous_energy_consumption
                    
                time_to_charge = self.charging_model.calculate_charging_time(soc_after_flight * 100, total_battery_energy) * 60

                aircraft_data.append({
                    'flight_plan': flight_plan,
                    'energy_consumed': energy_consumed,
                    'soc_after_flight': soc_after_flight,
                    'time_to_charge': time_to_charge
                })

            results[aircraft_id] = aircraft_data
            
        self.aircraft_data = results

        for aircraft_id, data in results.items():
            print(f"Aircraft {aircraft_id}:")
            for entry in data:
                print(f"  Flight Plan: {entry['flight_plan']}, Energy Consumed: {entry['energy_consumed']} J, SoC After Flight: {entry['soc_after_flight'] if entry['soc_after_flight'] is not None else 'N/A'}, Time to Charge: {entry['time_to_charge']} mins")

    def plot_aircraft_vs_flight_plans(self, aircraft_assignment):
        # Count the number of flight plans assigned to each aircraft
        aircraft_flight_counts = {aircraft_id: len(flight_plans) for aircraft_id, flight_plans in aircraft_assignment.items()}

        # Separate the aircraft with only one flight plan
        others_count = 1#sum(1 for count in aircraft_flight_counts.values() if count == 1)
        filtered_aircraft_flight_counts = {aircraft_id: count for aircraft_id, count in aircraft_flight_counts.items() if count > 1}

        # Prepare data for plotting
        x_labels = list(map(str, filtered_aircraft_flight_counts.keys())) + ['Others']
        y_values = list(filtered_aircraft_flight_counts.values()) + [others_count]

        # Create the bar plot
        plt.figure(figsize=(3.15, 2.36))
        plt.bar(x_labels, y_values, color='skyblue')
        # plt.title('Aircraft ID vs Number of Flight Plans',fontsize=10)
        plt.xlabel('Aircraft ID',fontsize=8)
        plt.ylabel('Number of Flight Plans',fontsize=8)
        plt.xticks(rotation=45,fontsize=8)
        plt.grid(True, axis='y', linestyle='--', linewidth=0.7)
        plt.tight_layout()

        # Save the figure
        plt.savefig('aircraft_vs_flight_plans.pdf')

        # Show the plot
        plt.show()
        
        # Filter out aircraft with more than one flight plan and prepare SoC data
        soc_after_first_flight = {}
        energy_needed_to_fully_charge = {}
        total_battery_capacity = 676.8 * 10**6  # in Joules

        for aircraft_id, data in self.aircraft_data.items():
            if len(data) > 1:
                first_flight_data = data[0]
                soc_after_first_flight[aircraft_id] = first_flight_data['soc_after_flight']
                energy_needed_to_fully_charge[aircraft_id] = (1 - soc_after_first_flight[aircraft_id]) * total_battery_capacity/10**6

        # Prepare data for plotting the SoC values and energy needed
        x_labels_soc = list(map(str, soc_after_first_flight.keys()))
        y_values_soc = list(soc_after_first_flight.values())
        y_values_energy = list(energy_needed_to_fully_charge.values())

        # Create the bar plot for SoC values and energy needed
        fig, ax1 = plt.subplots(figsize=(3.15, 2.36))

        color = 'tab:green'
        ax1.set_xlabel('Aircraft ID',fontsize=8)
        ax1.set_ylabel('SoC After First Flight Plan', color=color,fontsize=8)
        ax1.bar(x_labels_soc, y_values_soc, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, axis='y', linestyle='--', linewidth=0.7)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Energy Needed to Fully Charge (MJ)', color=color,fontsize=8)  # we already handled the x-label with ax1
        ax2.plot(x_labels_soc, y_values_energy, color=color, marker='o', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(True, axis='y', linestyle='--', linewidth=0.7)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.title('SoC and Energy Needed to Fully Charge',fontsize=10)
        plt.savefig('aircraft_vs_soc_and_energy.pdf')
        plt.show()    
    def compute_vertiport_energy_demand(self, vertiport_flight_mapping):
        # Retrieve the vertiport data
        departing_only, arriving_only, both_departing_and_arriving = self.categorize_and_print_vertiports(vertiport_flight_mapping)

        # Create a dictionary to store the total energy demand and number of flight plans for each vertiport
        vertiport_energy_demand = {}
        vertiport_flight_counts = {}

        # Helper function to find the departing vertiport for a flight plan
        def find_departing_vertiport(flight_plan):
            for location, flights in departing_only.items():
                if flight_plan in flights['departing']:
                    return location
            for location, flights in both_departing_and_arriving.items():
                if flight_plan in flights['departing']:
                    return location
            return None

        # Iterate over the aircraft data to aggregate energy consumption by vertiport
        for aircraft_id, flights_data in self.aircraft_data.items():
            for flight_data in flights_data:
                flight_plan = flight_data['flight_plan']
                energy_consumed = flight_data['energy_consumed'] / 10**6  # Convert to MJ
                
                # Find the departing vertiport for the flight plan
                departing_vertiport = find_departing_vertiport(flight_plan)
                if departing_vertiport:
                    if departing_vertiport not in vertiport_energy_demand:
                        vertiport_energy_demand[departing_vertiport] = 0
                        vertiport_flight_counts[departing_vertiport] = 0
                    vertiport_energy_demand[departing_vertiport] += energy_consumed
                    vertiport_flight_counts[departing_vertiport] += 1

        # Print the total energy demand for each vertiport
        for vertiport, total_energy in vertiport_energy_demand.items():
            print(f"Vertiport {vertiport}: Total Energy Demand = {total_energy} MJ")

        # Prepare data for plotting
        labels = [f'{i+1}' for i in range(len(vertiport_energy_demand))]
        energies = list(vertiport_energy_demand.values())
        num_flights = list(vertiport_flight_counts.values())

        # Creating the bar plot
        fig, ax1 = plt.subplots(figsize=(6.3, 3.36))  # Half A4 size in height

        bars = ax1.bar(labels, energies, color='mediumblue')
        ax1.set_xlabel('Vertiport', fontsize=10)
        ax1.set_ylabel('Total Energy Consumption (MJ)', fontsize=10)
        # ax1.set_title('Total Energy Consumption by Vertiport', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Create a secondary y-axis to plot the number of flight plans
        ax2 = ax1.twinx()
        ax2.plot(labels, num_flights, color='tomato', marker='o', linestyle='--', linewidth=2, markersize=8)
        ax2.set_ylabel('Number of Flight Plans', fontsize=10, color='tomato')
        ax2.tick_params(axis='y', labelcolor='tomato', labelsize=8)
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()

        # Add data labels for the number of flight plans
        for i, (bar, num) in enumerate(zip(bars, num_flights)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, num + 0.01 * max(num_flights), f'{num}', color='tomato', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout(pad=2.0)
        plt.savefig('total_energy_consumption_by_vertiport.pdf', dpi=300, bbox_inches='tight')
        plt.show()

        return vertiport_energy_demand
    

    # def integrate_energy_consumption(self, cost_files, aircraft_assignment: Dict[int, List[str]]):
    #     total_battery_energy = 676.8 * 10**6  # in Kilo Joules
    #     self.parse_excel_files(cost_files)
    #     results = {}

    #     for aircraft_id, flight_plans in aircraft_assignment.items():
    #         aircraft_data = []
    #         initial_soc = random.uniform(0.2, 0.3)  # Initial SoC between 20% and 30%

    #         for i, flight_plan in enumerate(flight_plans):
    #             if i == 0:
    #                 soc_after_flight = 1 - self.flight_plan_data[flight_plan]['Optimal Cost'] * 10**6 / total_battery_energy
    #             else:
    #                 soc_after_flight = aircraft_data[-1]['soc_after_flight'] - self.flight_plan_data[flight_plan]['Optimal Cost'] * 10**6 / total_battery_energy

    #             energy_consumed = (1 - soc_after_flight) * total_battery_energy
    #             time_to_charge = self.charging_model.calculate_charging_time(soc_after_flight * 100, total_battery_energy) * 60

    #             aircraft_data.append({
    #                 'flight_plan': flight_plan,
    #                 'energy_consumed': energy_consumed,
    #                 'soc_after_flight': soc_after_flight,
    #                 'time_to_charge': time_to_charge
    #             })

    #         results[aircraft_id] = aircraft_data

    #     self.aircraft_data = results

    #     for aircraft_id, data in results.items():
    #         print(f"Aircraft {aircraft_id}:")
    #         for entry in data:
    #             print(f"  Flight Plan: {entry['flight_plan']}, Energy Consumed: {entry['energy_consumed']} KJ, SoC After Flight: {entry['soc_after_flight'] if entry['soc_after_flight'] is not None else 'N/A'}, Time to Charge: {entry['time_to_charge']} mins")

    def plot_charging_power_profile(self):
        # Vertiport IDs to be plotted
        vertiports_to_plot = ['1', '2', '8']
        total_battery_energy = 676.8 * 10**3  # in Kilo Joules
        time_step = 60  # Time step in seconds (1 minute)

        for vertiport_id in vertiports_to_plot:
            # Get the flight plans for the vertiport
            flight_plans = []
            for aircraft_data in self.aircraft_data.values():
                for flight in aircraft_data:
                    if vertiport_id in flight['flight_plan']:
                        flight_plans.append(flight)

            # Disperse the arrival times equally within a 3-hour time horizon
            num_flights = len(flight_plans)
            arrival_times = np.linspace(0, 3 * 3600, num_flights)

            print(f"Charging start times for Vertiport {vertiport_id}:")
            for arrival_time in arrival_times:
                print(f"Flight Plan Arrival Time: {arrival_time / 3600:.2f} hours")

            # Initialize time and power arrays
            end_time = 3 * 3600 + max(flight['time_to_charge'] * 60 for flight in flight_plans)
            times = np.arange(0, end_time + time_step, time_step)
            total_power = np.zeros_like(times)

            end_times = []

            for i, flight in enumerate(flight_plans):
                initial_soc = 1 - (flight['energy_consumed'] / total_battery_energy)
                final_soc = flight['soc_after_flight']
                arrival_time = arrival_times[i]

                # Calculate the charging power over time
                current_time = arrival_time
                current_soc = final_soc * 100

                while current_soc < 100:
                    power = self.charging_model.charging_curve(current_soc)
                    time_index = int(current_time // time_step)
                    if time_index >= len(total_power):
                        break  # Ensure we don't exceed the array bounds
                    total_power[time_index] += power
                    current_time += time_step  # Increment time by time_step
                    current_soc += (power * time_step) / total_battery_energy * 100

                end_times.append(current_time)

            print(f"Charging end times for Vertiport {vertiport_id}:")
            for end_time in end_times:
                print(f"Flight Plan End Time: {end_time / 3600:.2f} hours")

            # Plot the power profile
            plt.figure(figsize=(3.15, 2.36))  

            # Plot total power profile
            plt.plot(times / 3600, total_power, label=f'Vertiport {vertiport_id}', color='blue', linestyle='-', linewidth=1.6)

            # Indicate charging start and end times for the first and last flight plans
            first_last_indices = [0, -1] if len(arrival_times) > 1 else [0]
            for idx in first_last_indices:
                arrival_time = arrival_times[idx]
                end_time = end_times[idx]
                plt.axvline(x=arrival_time / 3600, color='red', linestyle='--', linewidth=1)
                plt.axvline(x=end_time / 3600, color='green', linestyle='--', linewidth=1)
                plt.fill_betweenx(np.arange(0, max(total_power) * 1.1), arrival_time / 3600, end_time / 3600, color='gray', alpha=0.3)
                flight_number = flight_plans[idx]['flight_plan'].split('_')[-1]
                plt.text(arrival_time / 3600, max(total_power) * 1.05, flight_number, rotation=0, verticalalignment='center', color='black', fontsize=8)
                plt.text(end_time / 3600, max(total_power) * 1.05, flight_number, rotation=0, verticalalignment='center', color='black', fontsize=8)

            plt.xlabel('Time (hours)', fontsize=8)
            plt.ylabel('Charging Power (kW)', fontsize=8)
            # plt.title(f'Charging Power Profile for Vertiport {vertiport_id}', fontsize=12)
            # plt.legend()
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(f'charging_power_profile_vertiport_{vertiport_id}.pdf', dpi=300, bbox_inches='tight')
            plt.show()

if __name__ == '__main__':
    flight_plans = 'flight_plans/filtered_flight_plans.json'
    wind_models = 'wind_forecasts/Charging_wind_models.json'
    charging_demand = ChargingDemandForecasting(flight_plans, wind_models)
    vertiport_flight_mapping = charging_demand.map_vertiports_to_flight_plans()
    departing_only, arriving_only, both_departing_and_arriving = charging_demand.categorize_and_print_vertiports(vertiport_flight_mapping)
    aircraft_assignment = charging_demand.assign_aircraft_to_flight_plans(vertiport_flight_mapping)
    energy_consumption_files = os.path.join(DATAFOLDER, 'energy_consumptions/Total-Energy-Profiles')
    charging_demand.integrate_energy_consumption(energy_consumption_files, aircraft_assignment)
    charging_demand.plot_aircraft_vs_flight_plans(aircraft_assignment)
    charging_demand.compute_vertiport_energy_demand(vertiport_flight_mapping)
    charging_demand.plot_charging_power_profile()






