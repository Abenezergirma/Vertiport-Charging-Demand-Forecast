# Vertiport Charging Demand Forecasting

This repository contains the code for forecasting the charging demand at vertiports in an urban air mobility (UAM) setting. The project focuses on predicting the energy consumption and managing the charging infrastructure required for electric vertical takeoff and landing (eVTOL) aircraft at various vertiports.

## Project Overview

The project is designed to handle various tasks, including:
- Loading and processing flight plan data.
- Mapping vertiports to their associated flight plans.
- Computing the energy demand at each vertiport.
- Plotting charging power profiles for selected vertiports.
- Visualizing the locations of vertiports and their energy demands on a map.

## Directory Structure
```bash
Vertiport-Charging-Demand-Forecast/
│
├── data/
│ ├── energy_consumptions/
│ ├── flight_plans/
│ ├── wind_forecasts/
│ └── vertiports.geojson
│
├── data_processing/
│ ├── init.py
│ └── preprocess_flight_plans.py
│
├── forecasting_models/
│ ├── init.py
│ ├── charging_demand_forecasting.py
│ └── charging_model.py
│
├── outputs/
│ ├── charging_power_profile_vertiport_1.pdf
│ ├── charging_power_profile_vertiport_2.pdf
│ ├── charging_power_profile_vertiport_8.pdf
│ └── vertiport_energy_demand_map.png
│
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Vertiport-Charging-Demand-Forecast.git
   cd Vertiport-Charging-Demand-Forecast
   
2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   
3. Install the required packages: 
   ```bash 
   pip install -r requirements.txt

## Usage
### Preprocess Flight Plans
The PreprocessFlightPlans class in the data_processing/preprocess_flight_plans.py module is responsible for loading and processing flight plan data, mapping vertiports to flight plans, and computing distances between waypoints.

### Charging Model
The ChargingModel class in the forecasting_models/charging_model.py module contains methods related to the charging curve and calculating charging times.

### Charging Demand Forecasting
The ChargingDemandForecasting class in the forecasting_models/charging_demand_forecasting.py module integrates the preprocessing and charging models to forecast the charging demand at vertiports, plot charging power profiles, and visualize energy demands on a map.
