# Panda5gsim
Panda5gSim is a game engine-based 3D mobility and ray-tracing simulator for 5G/6G networks in 3D synthetic urban environments.
It supports air-to-ground (A2G), ground-to-Air (G2A), air-to-air (A2A) communication channels.

## Panda5gSim Features:
* Generate synthetic urban layout based on the built-up parameters of the ITU-R P.1410-5.
* Simulate 3D mobility for UAVs (unmanned aerial vehicles) and ground users.
* Collects transformations of the UAVs and ground users in the 3D space.
* Collects ray-tracing line-of-sight (LoS) status for directional and omnidirectional antennas on the UAVs.
* Implements the 3GPP 3D channel model and other popular models especially the LoS probability models.
* Use PandAI for mobility and pathfinding of the UAVs and ground users.
    Can model communication networks in urban environment and simulate the performance of the networks.
* We used Panda5gSim in sojourn time modeling for UAVs and ground users.

## Installation
1. Install anaconda or miniconda from [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)
2. Create a new conda environment
```bash
conda create -n panda5gsim --file requirements.txt
```
3. Activate the conda environment
```bash
conda activate panda5gsim
```
4. Clone the repository
```bash
git clone https://github.com/yemenlinux/panda5gsim.git
```
5. Change directory to the cloned repository
```bash
cd panda5gsim
```
6. Run a file from examples folder or create your own code. For example, to run the directional line-of-sight with mobility, run the following command:
```bash
python examples/directional_los_with_mobility.py
```




