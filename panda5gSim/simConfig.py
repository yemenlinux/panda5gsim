#
SCENARIOS = {
    # Parameters follow the work in 
    # Mohammed, I., Gopalam, S., Collings, I. B., & Hanly, S. v. (2023). 
    # Closed Form Approximations for UAV Line-of-Sight Probability in 
    # Urban Environments. IEEE Access, 11, 40162–40174. 
    # https://doi.org/10.1109/ACCESS.2023.3267808
    'Suburban': {
        'Scenario': '3D-UMa',
        'Alpha': 0.1, 
        'Beta': 750, 
        'Gamma': 8, 
    },
    'Urban': {
        'Scenario': '3D-UMa',
        'Alpha': 0.3, 
        'Beta': 500, 
        'Gamma': 15, 
    },
    'Dense Urban': {
        'Scenario': '3D-UMi',
        'Alpha': 0.5, # Built-up area percentage of the total area
        'Beta': 300, # Number of buildings per square km
        'Gamma': 20, # Building height distribution parameter
    },
    'High-rise Urban': {
        'Scenario': '3D-UMi',
        'Alpha': 0.5, 
        'Beta': 300, 
        'Gamma': 50, 
    },
}
# Simulation configuration data structure
SimulationData = {
    'Scenarios': SCENARIOS,
    'Seed value': 1, 
    'Iterations': 20,
    'Network load': 0.5, # 50 %
    'Confidence intervals': [0.5, 0.5, 0.95],
    #
    'Frequency (GHz)': 28.0, # 28 GHz
    
    # simulation area
    'Simulation area per square km': 2, 
    'Simulation area boundaries per m': (-600.0, -600.0, 600.0, 600.0), 
    
    # Mobility parameters
    'Ground nav mesh': 'denise_urban_ground_nav_mesh.csv',
    
    # Ground Base Stations parameters
    'Number of sites': 19, 
    'Number of sectors': 3,
    
    # Air Base Stations parameters
    'Number of air eNB': 3, 
    'Allowed flight range': [5, 50, 100, 150, 200, 500],
    # Users parameters
    
    
    
    
    # Available Simulation Scenarios are: 
    'RAN Generation': '5G', 
    'Scenario': 'denise_urban', 
    'Cell layout': 'hexagonal_grid', 
    
    'ISD': 200, 
    'Cell radius': 173.2, 
    # 'Frequency (GHz)': 2, 
    'Bandwidth (MHz)': 1.0, 
    'ISD': 200, 
    'Hexagon area': 77937.71372646748, 
    
    'Number of air users': 5, 
    
    
    
    'Mobility': {
        'Outdoor percentage': 0.2,
        'Mobility modes': ['PoI', 'Random'],
        'Mobility modes probability': [0.2, 0.8],
        'Place of Interest': ['home', 'work'], 
        'Place of interest probability': [0.5, 0.5],
        'Mobility medium': ['walk', 'car', 'bus', 'train', 'airplane'],
        'Mobility medium probability': [0.2, 0.2, 0.2, 0.2, 0.2],
        'Walking speed (m/s)': 0.83,
        'Car speed (m/s)': 19.4,
        'Bullet train Speed (m/s)': 83.3,
    },
    
    'Building Defaults': {
        'Max. building height': 24.0, 
        'Max. number of building floors': 8, 
        'Average building height': 12.0, 
        'Floor height': 3.0, 
        'Above roof': 0.0, 
        'Average street width': 20.0, 
        'Street width range': (5.0, 50.0), 
        'Alpha': 0.1, 
        'Beta': 100, 
        'Gamma': 20, 
        # 'Scenario': 'denise_urban', 
        # 'Number of buildings per square km': 100, 
        'Width': 20, 
        'Depth': 20,
        }, 
    
    'BS defaults': {
        # 'Scenario': 'denise_urban', 
        'MIMO': (1, 1), 
        'Antenna down tilt': 0, 
        'BS antenna height': 10.0, 
        'BS Tx. power': 30.0, 
        'Tx. power': 30.0, 
        'Antenna gain': 16.0, 
        'Antenna losses': 0, 
        'Mobility': False, 
        'Mobility speed': 0.0, 
        'Frequency (GHz)': 2, 
        'Bandwidth (MHz)': 1.0, 
        'Graphical model': None, 
        'Power consumption': 0, 
        'Operation state': 'on', 
        'Antenna height': 10.0,
        },
    # users
    'Population density per sq. km': 650, 
    'Telephony density per population': 0.8,
    'Number of ground users': 20,
    'UT defaults': {
        # 'scenario': 'denise_urban', 
        'Antenna height': 1.5, 
        'Antenna gain': 0, 
        'Antenna losses': 0, 
        'Misc losses': 0, 
        'Tx. power': 0, 
        'Rx. sensitivity': -130, 
        'Supported frequencies (GHz)': (0.7, 1.8, 2.0, 2.4), 
        'Indoor ratio': 0.8, 
        'MIMO': (1, 1), 
        'Is actor': True, 
        'Mobility': True, 
        'Mobility speed': 3.0, 
        'Min BS-UT distance': 10, 
        'Distribution': 'uniform', 
        'Frequency (GHz)': 2, 
        'Bandwidth (MHz)': 1.0, 
        'Graphical model': None, 
        'Power consumption': 0, 
        'Operation state': 'idle', 
        'Run state': 'on',
        },
    
    'Number of air BSs': 3,
    'UAV user defaults': {
        # 'scenario': 'denise_urban', 
        # 'ID': '0_0', 
        # 'position': (0, 0, 0), 
        'Flight height': 20, 
        'Air nav_mesh': 'denise_urban_air_nav_mesh_20.csv', 
        'Frequency (GHz)': 2, 
        'Bandwidth (MHz)': 1.0,
        },
    
    'UAV BS defaults': {
        # 'scenario': 'denise_urban', 
        # 'ID': '0_0', 
        # 'position': (0, 0, 0), 
        'Antenna height': 0.0, 
        'Flight height': 50, 
        'Air nav_mesh': 'denise_urban_air_nav_mesh_50.csv', 
        'Frequency (GHz)': 2, 
        'Bandwidth (MHz)': 1.0,
        }, 
    'Cloud defaults': {
        # 'scenario': '3D-UMi', 
        # 'ID': '0_0', 
        # 'position': (0, 0, 0),
        }, 
    'MEC defaults': {
        # 'scenario': '3D-UMi', 
        # 'ID': '0_0', 
        # 'position': (0, 0, 0)
    },
    
    'Dynamic ranges': {
        # 'scenario': '3D-UMi', 
        'Fixed cell radius': False, 
        'Cell radius range': range(40, 540, 80), 
        'Antenna types': ['omni', 'directional'], 
        'MIMO range': [(1, 1), (2, 2), (4, 4)], 
        'Fixed antenna height': False, 
        'BS antenna height range': (10.0, 25.0), 
        'UT antenna height range': (1.5, 22.5), 
        'Frequency range (GHz)': [0.7, 0.9, 1.8, 2.6, 3.5, 6.0, 28.0], 
        'Bandwidth range (MHz)': [1, 10, 20], 
        'BS Tx. power for bw': [30.0, 41.0, 44.0],
    },
    
    
}

