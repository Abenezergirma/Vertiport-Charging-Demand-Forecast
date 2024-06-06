# Chapter-1

## Overview of the Repository:

This repository contains the code and results from the first chapter of my dissertation on **Urban Air Mobility (UAM)** trajectory planning. The goal is to provide a real-time planning framework that's both safe and scalable.

### Key Features:

1. **Decentralized Free-Flight Concept**: 
   - Every aircraft independently handles conflict resolution and ensures safety by considering the predicted positions of nearby planes.
   
2. **Two-Part Framework**:
   - **Reachability Analysis Tool**: Predicts potential aircraft movements using data from simulated trajectories.
   - **Decision Maker**: Uses a 6-degree freedom model for fixed-wing aircraft to plan collision-free paths.

3. **Safety Enhancements**: 
   - Incorporates techniques like reward shaping and action shielding to boost safety.
   
4. **Performance Testing**: 
   - Simulated tests with up to 32 aircraft showed the effectiveness of this framework, gauged by reduced Near Mid Air Collisions (NMAC) and computational efficiency.

# Illustrations

# Images
<table>
  <tr>
    <td><img src="img/state_x.png" alt="Image 1"></td>
    <td><img src="img/xy_proj.png" alt="Image 2"></td>
  </tr>
  <tr>
    <td><img src="img/state_y.png" alt="Image 3"></td>
    <td><img src="img/xz_proj.png" alt="Image 4"></td>
  </tr>
  <tr>
    <td><img src="img/state_z.png" alt="Image 5"></td>
    <td><img src="img/yz_proj.png" alt="Image 6"></td>
  </tr>
</table>


# Videos

- [Video 1](img/video1.avi)
- [Video 2](img/video2.avi)
- [Video 3](img/video3.avi)
... (add as many videos as you have)




### Code Overview:

To ensure a seamless reproduction of the results discussed in the chapter, follow the guide below:

**Repository Structure**:
- The repository is organized into four distinct branches:
  1. `main`: Contains all the implemented techniques.
  2. `baseline`: Implements the baseline technique discussed.
  3. `reward_shaping`: Focuses on the reward shaping technique.
  4. `action_shielding`: Highlights the action shielding technique.

**Steps to Run the Code**:
1. Navigate to the `main.m` script.
2. Configure the desired number of agents you wish to simulate.
3. Execute the script.
4. Once completed, gather key flight statistics, including NMAC and Computation time.
5. Save your environment as a `.mat` file.
6. To visualize the results, import the saved `.mat` file into the `animate.m` script and run it.
