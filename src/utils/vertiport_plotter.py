import folium
import pandas as pd
from folium.plugins import HeatMap

# Vertiport data
data = {
    "vertiport_id": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    ],
    "latitude": [
        32.6756856631, 32.9934469, 32.7344259197, 32.6451586239, 32.8747416374,
        33.0207255857, 33.0471788798, 32.9061183861, 32.8443092, 32.9919516082,
        32.9902699455, 32.8335464415, 32.9934624553, 32.7488714744, 33.0247283882,
        32.9020118731, 32.7788115917, 32.7081265686, 33.0270319201, 32.9803994265,
        32.64043387, 32.9467685372, 32.7508677176, 33.0337819238, 32.801954546
    ],
    "longitude": [
        -97.4095502941, -97.3133984, -97.3390828244, -96.900192401, -97.2980396294,
        -96.6121279792, -97.0530247807, -97.0404952877, -96.8614056, -97.1866838062,
        -96.6611381227, -96.7539956129, -96.6371079753, -96.8231793045, -96.6964477248,
        -97.2774446221, -96.8039822771, -97.3586033542, -97.1039357917, -96.7164350473,
        -96.9514139944, -96.6315451532, -97.0838872824, -97.0705762169, -96.6062218646
    ],
    "energy_demand": [
        14319.167102938682, 11673.161318971237, 512.1155356052186, 961.979, 527.4629969670299,
        1517.2907918902297, 501.774, 7388.223530479694, 5647.204158335296, 1511.9952883252322,
        458.41, 506.8204704384315, 533.9646991818387, 528.7425034591936, 517.9302526197779,
        872.15, 483.2772846999459, 995.2959999999999, 993.8175198988916, 533.8001729438588,
        510.268, 989.9886553184151, 524.025, 529.8401104346376, 3025.2681674057662
    ]
}

# Create a DataFrame
vertiports_df = pd.DataFrame(data)

# Create a Folium map centered around the mean latitude and longitude
map_center = [vertiports_df['latitude'].mean(), vertiports_df['longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=10)

# Create a MarkerCluster object to manage the markers
marker_cluster = folium.FeatureGroup(name="Vertiports").add_to(m)

# Add markers to the map
for _, row in vertiports_df.iterrows():
    energy_demand = row['energy_demand']
    vertiport_id = row['vertiport_id']
    
    # Set the color based on energy demand
    color = 'red' if energy_demand > 1000 else 'blue'
    
    # Create a circle marker for the vertiport
    # folium.CircleMarker(
    #     location=[row['latitude'], row['longitude']],
    #     radius=10 + energy_demand / 500,  # Adjust the radius based on energy demand
    #     color=None,
    #     fill=True,
    #     fill_color=color,
    #     fill_opacity=0.6,
    #     popup=f"Vertiport ID: {vertiport_id}<br>Energy Demand: {energy_demand:.2f} MJ"
    # ).add_to(marker_cluster)
    
    # Add a label for the vertiport
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black;">{int(vertiport_id)}</div>')
    ).add_to(marker_cluster)

# Add a HeatMap
heat_data = [[row['latitude'], row['longitude'], row['energy_demand']] for index, row in vertiports_df.iterrows()]
HeatMap(heat_data).add_to(m)

# Add a legend to the map
legend_html = '''
<div style="position: fixed;
     bottom: 50px; left: 50px; width: 200px; height: 90px;
     background-color: white; z-index: 1000; font-size: 14px;">
     <b>&nbsp;Legend</b><br>
     &nbsp;<i class="fa fa-circle fa-1x" style="color:red"></i>&nbsp;High Energy Demand<br>
     &nbsp;<i class="fa fa-circle fa-1x" style="color:blue"></i>&nbsp;Low Energy Demand<br>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Save the map as an HTML file
m.save('vertiport_charging_stations_map.html')

print("Map has been created and saved as vertiport_charging_stations_map.html")
