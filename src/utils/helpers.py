import math 
import matplotlib.pyplot as plt
import random

def assign_cruise_speeds(flight_plans, cruise_speeds):
    grouped_flight_plans = [flight_plans[i:i + 5] for i in range(0, len(flight_plans), 5)]
    for i, group in enumerate(grouped_flight_plans):
        cruise_speed = random.uniform(50, 60)
        for flight_plan in group:
            cruise_speeds[flight_plan] = cruise_speed
            print(f"Assigned cruise speed {cruise_speed:.2f} m/s to {flight_plan}")  
    return cruise_speeds


def generate_cruise_speed_plot():
    cruise_speeds = []
    flight_plans = [f'Flight_plan_{i+1}' for i in range(112)]
    cruise_speeds = assign_cruise_speeds(flight_plans, cruise_speeds)
    grouped_flight_plans = [flight_plans[i:i + 5] for i in range(0, len(flight_plans), 5)]
    x_labels = [f'{i*5+1}-{(i+1)*5}' for i in range(len(grouped_flight_plans))]
    cruise_speeds = [cruise_speeds[group[0]] for group in grouped_flight_plans]
    print(cruise_speeds)
            
    plt.figure(figsize=(5, 4))
    plt.bar(x_labels, cruise_speeds, color='b')
    plt.title('Cruise Speed Values for Flight Plans')
    plt.xlabel('Flight Plan')
    plt.ylabel('Cruise Speed (m/s)')
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cruise_speed_assignments.pdf', dpi=300, bbox_inches='tight')
    plt.show()