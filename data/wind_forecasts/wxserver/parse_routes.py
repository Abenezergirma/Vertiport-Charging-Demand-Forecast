# Server error 500 code has something to do with the timestamps. That needs to be figured out later.

import numpy
import matplotlib.pyplot as plt
from wxserver import server_request
import json
from tqdm import tqdm

out_path = 'weather.json' #This is where the data is dumped

traj_data = open('../DFW_SCENARIOS/DFW_CMU_APT_CGO.txt', 'r')

TIMESTAMPS=['2021-04-20T06:40:00+00:00'] #This timestamp is used in all queries, since the server returns NaN for other timestamps

num_routes = 9 #Controls the number of routes

def clean_line(line):
    #Takes a line of data from the trajectory file and returns the lats, lons, alts, and timestamps from that file

    route = line.split(",")[-1]
    points = route.split(" ")
        
    points_converted = []
    for p in points:
            points_converted.append(p.split("/"))
    points_converted = numpy.array(points_converted)

    #Formatting: make the first three columns integers, remove the /n from the timestamp of the last entry.
    lats = points_converted[:,0].astype(float)
    lons = points_converted[:,1].astype(float)
    alts = points_converted[:,2].astype(float)
    timestamps = [i for i in points_converted[:,3]]
    timestamps[-1] = timestamps[-1][:-1]

    sample_points = [
        [lat for lat in lats],
        [lon for lon in lons],
        [a for a in alts],
        ["2021-04-20T"+"07:40:00"+"+00:00" for t in timestamps]
    ]
    
    return sample_points


lines = traj_data.readlines()

out = {'data': []}

for line in lines[1:num_routes+1]:
    sample_points = clean_line(line)

    data = server_request(
        sample_points[0],
        sample_points[1],
        sample_points[2],
        sample_points[3],
        ensemble=False) 
    print('Preparing final structure...')
    wind_data = {'order': sample_points, 'results':{'wind_speeds': [], 'wind_directions': []}}

    for point in tqdm(data):
        if point is None:
                continue
        wind_data['results']['wind_speeds'].append(point['wind_speed'])
        wind_data['results']['wind_directions'].append(point['wind_direction'])

    out['data'].append(wind_data)



print(f'Saving file (dump JASON to {out_path})...')
with open(out_path, "w") as outfile: 
    json.dump(out, outfile)
print('Done!')
