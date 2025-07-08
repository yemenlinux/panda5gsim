# Panda5gsim
Panda5gSim is a game-based 3D mobility and ray-tracing simulator for 5G/6G networks in 3D synthetic urban environments. It simulates the mobility of unmanned aerial vehicles (UAVs) and ground users in any urban environment that can be described using ITU-R P.1410 parameters. The Panda5gSim provides a cost-effective alternative to trace-based mobility. It generates accurate ray-tracing line-of-sight (LoS) status for directional and omnidirectional antennas on the UAVs as well as tracing positions and orientations of mobile objects in the 3D space. This approach is the first of its kind.

See a video demonstration of simulating 67 urban environments that are generated using ITU-R P.1410 parameters (PPP distribution).

[![Video Title](https://img.youtube.com/vi/wOamYmyLu3I/0.jpg)](https://www.youtube.com/watch?v=wOamYmyLu3I)

Simulating mobility in dense urban environment without buildings.
[![Video Title](https://img.youtube.com/vi/UG95zqqhYUI/0.jpg)](https://www.youtube.com/watch?v=UG95zqqhYUI)

Simulating mobility in dense urban environment with buildings.
[![Video Title](https://img.youtube.com/vi/eZwWh9pH6e8/0.jpg)](https://www.youtube.com/watch?v=eZwWh9pH6e8)


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

## Citation
If you use Panda5gSim in your research, please cite the following paper:

```bibtex
@ARTICLE{11050367,
  author={Raddwan, Basheer Ameen and Al-Baltah, Ibrahim Ahmed},
  journal={IEEE Access}, 
  title={Mobility-Aware Bivariate Line-of-Sight Probability for Air-to-Ground Communications Using Millimeter and Terahertz Waves}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Air to ground communication;Atmospheric modeling;Layout;Geometry;Line-of-sight propagation;Directional antennas;Ray tracing;Communication channels;Buildings;ITU;Line-of-Sight probability;mobility;air-to-ground communication;multi-Access edge Computing;ray-tracing;unmanned aerial vehicles;service time;sojourn time;urban;simulation},
  doi={10.1109/ACCESS.2025.3582890}}
```
The Panda5gSim components are described in the following paper:

```bibtex
@INPROCEEDINGS{10777167,
  author={Raddwan, Basheer Ameen and Ahmed Al-Baltah, Ibrahim and Ghaleb, Mukhtar},
  booktitle={2024 1st International Conference on Emerging Technologies for Dependable Internet of Things (ICETI)}, 
  title={Environment-Aware 3D Mobility Simulation for the 5G and 6G Wireless Networks}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  keywords={Three-dimensional displays;Mobility models;Biological system modeling;Wireless networks;Urban areas;Interference;Ray tracing;Throughput;3GPP;Signal to noise ratio;urban;5G;6G;mobility;ray-tracing;simulation;open-source;handover rate;framework;multi-access edge computing;unmanned aerial vehicle;3D mobility;3D environment},
  doi={10.1109/ICETI63946.2024.10777167}}
```

## License
Panda5gSim is licensed under the MIT License for academic use only. For commercial use, please contact the authors.