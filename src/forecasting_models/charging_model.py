import matplotlib.pyplot as plt
from typing import List
import math
from scipy.integrate import quad
plt.rcParams['text.usetex'] = True

class ChargingModel:
    def __init__(self, M: float, S: float):
        self.M = M
        self.S = S

    def charging_curve(self, x: float) -> float:
        """
        Piecewise function:
        charging_curve(x) = M if x <= 20% SoC
        charging_curve(x) = M - S * (x - 20) if x > 20% SoC
        """
        if x <= 20:
            return self.M
        else:
            return self.M - self.S * (x - 20)

    def plot_charging_curve(self) -> None:
        """
        Plot the piecewise function for a range of x values
        """
        x_values = list(range(0, 101))  # SoC from 0% to 100%
        y_values = [self.charging_curve(x) for x in x_values]

        plt.figure(figsize=(3.15, 2.36))
        plt.plot(x_values, y_values, label='Charging Power', color='blue', linewidth=2)
        plt.xlabel('State of Charge $(\%)$', fontsize=8)
        plt.ylabel('Charge Power (kW)', fontsize=8)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig('Charging_Curve.pdf', dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_charging_time(self, initial_soc: float, total_capacity_joules: float) -> float:
        """
        Calculates the time required to fully charge the battery from the initial SoC.

        :param initial_soc: Initial state of charge (percentage, 0-100)
        :param total_capacity_joules: Total battery capacity in joules
        :return: Time required to fully charge the battery in hours
        """
        def integrand(soc):
            charging_power_watts = self.charging_curve(soc) * 10**3
            soc_increment_per_second = (charging_power_watts / total_capacity_joules) * 100
            return 1 / (soc_increment_per_second + 0.0000001)

        time_required_seconds, _ = quad(integrand, initial_soc, 100)
        time_required_hours = time_required_seconds / 3600  # Convert seconds to hours
        return time_required_hours

    def plot_charging_time(self, total_capacity_joules: float):
        soc_values = range(0, 101)  # SoC values from 0% to 100%
        time_required = [self.calculate_charging_time(soc, total_capacity_joules) for soc in soc_values]

        plt.figure(figsize=(3.15, 2.36))
        plt.plot(soc_values, time_required, label='Charging Time', color='blue', linewidth=2)
        plt.xlabel('State of Charge $(\%)$', fontsize=8)
        plt.ylabel('Time Required to Fully Charge (hrs)', fontsize=8)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig('Charging_time.pdf', dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_cumulative_energy(self, initial_soc: float) -> float:
        """
        Calculates the cumulative energy required to charge the battery from the initial SoC to 100%.

        :param initial_soc: Initial state of charge (percentage, 0-100)
        :return: Cumulative energy required in joules
        """
        def integrand(soc):
            return self.charging_curve(soc)

        cumulative_energy, _ = quad(integrand, initial_soc, 100)
        return cumulative_energy

if __name__ == '__main__':
    M = 320  #  M
    S = 3.75  # S
    charging_model = ChargingModel(M, S)
    battery_capacity =  676.8 * 10**6
    charging_model.plot_charging_time(battery_capacity)
    charging_model.plot_charging_curve()
