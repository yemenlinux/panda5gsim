# Radio link budgets and metrics

import numpy as np
import pandas as pd

from panda5gSim.metrics.pathloss import FSPL
from panda5gSim.core.helpers import pairwise

frequency_ghz = 28

MODULATION_AND_CODING_LUT =[
    # ETSI. 2018. ‘5G; NR; Physical Layer Procedures for Data
    # (3GPP TS 38.214 Version 15.3.0 Release 15)’. Valbonne, France: ETSI.
    # Generation MIMO CQI Index	Modulation	Coding rate
    # Spectral efficiency (bps/Hz) SINR estimate (dB)
    # (Generation (4G/5G), MIMO, CQI (Channel Quality Indicator), modulation, coding rate, spectral efficiency (bps/Hz), SINR estimate (dB))
    # (0,   1,   2,   3,     4,   5,       6) index of tuple
    ('4G', '1x1', 1, 'QPSK', 78, 0.1523, -6.7),
    ('4G', '1x1', 2, 'QPSK', 120, 0.2344, -4.7),
    ('4G', '1x1', 3, 'QPSK', 193, 0.377, -2.3),
    ('4G', '1x1', 4, 'QPSK', 308, 0.6016, 0.2),
    ('4G', '1x1', 5, 'QPSK', 449, 0.877, 2.4),
    ('4G', '1x1', 6, 'QPSK', 602, 1.1758, 4.3),
    ('4G', '1x1', 7, '16QAM', 378, 1.4766, 5.9),
    ('4G', '1x1', 8, '16QAM', 490, 1.9141, 8.1),
    ('4G', '1x1', 9, '16QAM', 616, 2.4063, 10.3),
    ('4G', '1x1', 10, '64QAM', 466, 2.7305, 11.7),
    ('4G', '1x1', 11, '64QAM', 567, 3.3223, 14.1),
    ('4G', '1x1', 12, '64QAM', 666, 3.9023, 16.3),
    ('4G', '1x1', 13, '64QAM', 772, 4.5234, 18.7),
    ('4G', '1x1', 14, '64QAM', 973, 5.1152, 21),
    ('4G', '1x1', 15, '64QAM', 948, 5.5547, 22.7),
    ('5G', '8x8', 1, 'QPSK', 78, 0.30, -6.7),
    ('5G', '8x8', 2, 'QPSK', 193, 2.05, -4.7),
    ('5G', '8x8', 3, 'QPSK', 449, 4.42, -2.3),
    ('5G', '8x8', 4, '16QAM', 378, 6.40, 0.2),
    ('5G', '8x8', 5, '16QAM', 490, 8.00, 2.4),
    ('5G', '8x8', 6, '16QAM', 616, 10.82, 4.3),
    ('5G', '8x8', 7, '64QAM', 466, 12.40, 5.9),
    ('5G', '8x8', 8, '64QAM', 567, 16.00, 8.1),
    ('5G', '8x8', 9, '64QAM', 666, 19.00, 10.3),
    ('5G', '8x8', 10, '64QAM', 772, 22.00, 11.7),
    ('5G', '8x8', 11, '64QAM', 873, 28.00, 14.1),
    ('5G', '8x8', 12, '256QAM', 711, 32.00, 16.3),
    ('5G', '8x8', 13, '256QAM', 797, 38.00, 18.7),
    ('5G', '8x8', 14, '256QAM', 885, 44.00, 21),
    ('5G', '8x8', 15, '256QAM', 948, 50.00, 22.7),
]

def path_loss(row, Frequency_ghz = 28, TxAntennaGain = 16, RxAntennaGain = 1):
    if hasattr(row, 'Gain'):
        TxAntennaGain = row['Gain']
    if hasattr(row, 'd3D'):
        d3d = row['d3D']
    else:
        cos =np.cos(np.deg2rad(row['theta']))
        if cos == 0:
            d3d = row['h']
        else:
            d3d = row['d2D'] / cos
    return FSPL(d3d, Frequency_ghz, TxAntennaGain, RxAntennaGain)

def get_alpha_beta_gamma(env):
    standard = {'High-rise Urban': (0.5, 300, 50), 'Dense Urban': (0.5, 300, 20), 'Urban': (0.3, 500, 15), 'Suburban': (0.1, 750, 8)}
    if env in standard.keys():
        return standard[env]
    else:
        _, alpha, beta, gamma = env.split('_')
        alpha = eval(alpha)
        beta = eval(beta)
        gamma = eval(gamma)
        return alpha, beta, gamma

def received_power1(row, 
                   Frequency_ghz = 28,
                   TxPower = 46, # dBm
                   RxAntennaGain = 1, 
                   TxAntennaGain = 16,
                   building_penetration_loss = 25, # dB
                   ):
    if hasattr(row, 'nBuildings'):
        n = row['nBuildings']
    else:
        # _, alpha, beta, gamma = row['Environment'].split('_')
        # alpha = eval(alpha)
        # beta = eval(beta)
        # gamma = eval(gamma)
        alpha, beta, gamma = get_alpha_beta_gamma(row['Environment'])
        n = np.floor((row['d2D']/1000 * np.sqrt(alpha * beta)))
    los = row['RayLoS']
    #
    if hasattr(row, 'Gain'):
        TxAntennaGain = row['Gain']
    else:
        TxAntennaGain = 1
    # 
    if hasattr(row, 'd3D'):
        d3d = row['d3D']
    else:
        cos =np.cos(np.deg2rad(row['theta']))
        if cos == 0:
            d3d = row['h']
        else:
            d3d = row['d2D'] / cos
    
    
    # path loss
    if hasattr(row, 'PL'):
        PL = row['PL']
    else:
        PL = FSPL(d3d, Frequency_ghz, TxAntennaGain, RxAntennaGain)
    if los == 1:
        P_rx = TxPower - PL + RxAntennaGain + TxAntennaGain 
    else:
        P_rx = TxPower - PL - n * building_penetration_loss + RxAntennaGain + TxAntennaGain
    return 10**(P_rx/10)

def estimate_noise_watt(bandwidth_MHz = 10):
    k = 1.38e-23
    T = 290
    noise_figure = 1.5
    # convert MHz to Hz
    BW_Hz = bandwidth_MHz * 1e6
    # noise in dB
    return 10**((10 * np.log10(k * T * 1e3) 
            + noise_figure + 10 * np.log10(BW_Hz))/10)

def get_max_info(row):
    # Get max value and its index (position)
    max_index = row.idxmax()
    max_value = row.max()
    interference = row.sum() - max_value
    # Return a Series with desired values
    cols = ['sTxNode', 'RxPower', 'Interference', ]
    return pd.Series([max_index, max_value, interference], index=cols)

def connected_BS_power(data):
    cols = ['Time', 'RxNode', 'sTxNode', 'RayLoS', 'P_rx', 'Interference', 'd2D',
            'Gain', 'PL',
           'phi', 'theta', 'v_rx', 'v_tx', 'Environment']
    new_df = pd.DataFrame(columns=cols)
    #
    for t in data.Time.unique():
        cdata = data[data['Time'] == t]
        table = cdata.pivot(index=['RxNode'], columns='TxNode', values='P_rx')
        table2 = table.apply(get_max_info, axis=1)
        #
        table2['Time'] = t
        table2 = table2.reset_index()
        #
        for idx, old in table2.iterrows():
            new = {}
            rowc = cdata.loc[(cdata['RxNode'] == table2.iloc[idx]['RxNode']) & (cdata['TxNode'] == table2.iloc[idx]['sTxNode'])].squeeze()
            new.update(old)
            new.update(rowc.to_dict())
            
            new_df.loc[len(new_df)] = new
    #
    return new_df

def get_max_SINR(row, bandwidth_MHz = 10 ):
    # Get max value and its index (position)
    noise = estimate_noise_watt(bandwidth_MHz)
    interference = row.sum() 
    I = interference + noise
    # SINR
    SINR =  row / (I - row)
    #
    max_index = SINR.idxmax()
    max_SINR = SINR.max()
    # 
    max_power = row[max_index]
    # Return a Series with desired values
    cols = ['sTxNode', 'RxPower', 'Interference', 'SINR' ]
    return pd.Series([max_index, max_power, I, max_SINR], index=cols)

# def connected_BS_SINR1(data, bandwidth_MHz = 10):
#     # bw = bandwidth_MHz
#     cols = ['Time', 'RxNode', 'sTxNode', 'RayLoS', 'd2D',
#             'Gain', 'PL', 'P_rx', 'Interference', 'SINR',
#            'phi', 'theta', 'v_rx', 'v_tx', 'Environment',  ]
#     new_df = pd.DataFrame(columns=cols)
#     #
#     for t in data.Time.unique():
#         cdata = data[data['Time'] == t]
#         table = cdata.pivot(index=['RxNode'], columns='TxNode', values='P_rx')
#         table2 = table.apply(lambda x: get_max_SINR(x, bandwidth_MHz), axis=1)
#         #
#         table2['Time'] = t
#         table2 = table2.reset_index()
#         #
#         for idx, old in table2.iterrows():
#             new = {}
#             rowc = cdata.loc[(cdata['RxNode'] == table2.iloc[idx]['RxNode']) & (cdata['TxNode'] == table2.iloc[idx]['sTxNode'])].squeeze()
#             new.update(old)
#             new.update(rowc.to_dict())
#             # append to new_df
#             new_df.loc[len(new_df)] = new
#     # 
#     return new_df

def received_power(row, 
                   Frequency_ghz = 28,
                   TxPower = 43, # dBm ~ 20 W
                   RxAntennaGain = 1, 
                   TxAntennaGain = 1,
                   building_penetration_loss = 40, # dB
                   antenna_type = 'directional'
                   ):
    # if hasattr(row, 'nBuildings'):
    #     n = row['nBuildings']
    # else:
    #     # _, alpha, beta, gamma = row['Environment'].split('_')
    #     # alpha = eval(alpha)
    #     # beta = eval(beta)
    #     # gamma = eval(gamma)
    #     alpha, beta, gamma = get_alpha_beta_gamma(row['Environment'])
    #     n = np.floor((row['d2D']/1000 * np.sqrt(alpha * beta)))
    if antenna_type == 'omni':
        los = row['RayLoS']
    if antenna_type == 'directional':
        los = row['dRayLoS'] 
    #
    if antenna_type == 'omni':
        pass
    else:
        if hasattr(row, 'Gain'):
            TxAntennaGain = row['Gain']
    # 
    if hasattr(row, 'd3D'):
        d3d = row['d3D']
    else:
        cos =np.cos(np.deg2rad(row['theta']))
        if cos == 0:
            d3d = row['h']
        else:
            d3d = row['d2D'] / cos
    # path loss
    if hasattr(row, 'PL'):
        PL = row['PL']
    else:
        PL = FSPL(d3d, Frequency_ghz, TxAntennaGain, RxAntennaGain)
    if los == 1:
        return 10**((TxPower - PL + RxAntennaGain + TxAntennaGain)/10)
    else:
        # return 1e-20
        P_rx = (TxPower - PL 
                - building_penetration_loss 
                + RxAntennaGain + TxAntennaGain)
    return 10**(P_rx/10)
    
def connected_BS_SINR(data, bandwidth_MHz = 10):
    # bw = bandwidth_MHz
    cols = ['Time', 'RxNode', 
            # 'TxNode', 'RayLoS', 
            'd2D_o', 'h_o', 'd2D_d', 'h_d',
            # 'Gain', 'PL', 
            'sTxNode_d', #'RxPower_d', 'IpN_d', 
            'SINR_d',
            'sTxNode_o', #'RxPower_o', 'IpN_o', 
            'SINR_o',
           'phi_o', 'theta_o', 'phi_d', 'theta_d',
            # 'v_rx', 'v_tx', 'Environment',  
           ]
    new_df = pd.DataFrame(columns=cols)
    #
    data = data.sort_values('Time', inplace=False)
    for t in data.Time.unique():
        cdata = data[data['Time'] == t]
        table1 = cdata.pivot(index=['RxNode'], columns='TxNode', values='P_rx_o')
        table1 = table1.apply(lambda x: get_max_SINR(x, bandwidth_MHz), axis=1)
        #
        # table1['Time'] = t
        table1 = table1.reset_index()
        #
        table2 = cdata.pivot(index=['RxNode'], columns='TxNode', values='P_rx_d')
        table2 = table2.apply(lambda x: get_max_SINR(x, bandwidth_MHz), axis=1)
        table2 = table2.reset_index()
        #
        for idx, old in table1.iterrows():
            new = {}
            old.index = ['RxNode', 'sTxNode_o', 'RxPower_o', 'IpN_o', 'SINR_o' ]
            
            row1 = cdata.loc[(cdata['RxNode'] == table1.iloc[idx]['RxNode']) & (cdata['TxNode'] == table1.iloc[idx]['sTxNode'])].squeeze()
            row1 = row1.to_dict()
            row1 = {'Time': row1['Time'], 
                    'd2D_o': row1['d2D'], 
                    'h_o': row1['h'], 
                    'phi_o': row1['phi'], 
                    'theta_o': row1['theta']
                   }
            #
            
            # print(row2)
            # row2 = table2.iloc[idx]
            row2 = table2.loc[(table2['RxNode'] == table1.iloc[idx]['RxNode']) ].squeeze()
            # row2.index = ['index', 'RxNode', 'sTxNode_o', 'RxPower_o', 'IpN_o', 'SINR_o' ]
            row2.index = ['RxNode', 'sTxNode_d', 'RxPower_d', 'IpN_d', 'SINR_d' ]
            row2 = row2.to_dict()
            # row2.pop('RxNode')
            new.update(old)
            new.update(row1)
            new.update(row2)
            #
            row3 = cdata.loc[(cdata['RxNode'] == row2['RxNode']) & (cdata['TxNode'] == row2['sTxNode_d']) ].squeeze()
            row3 = row3.to_dict()
            row3 = {'d2D_d': row3['d2D'],
                    'h_d': row3['h'],
                    'phi_d': row3['phi'],
                    'theta_d': row3['theta']
                   }
            new.update(row3)
            # append to new_df
            new_df.loc[len(new_df)] = new
    # 
    return new_df


def directional_rayLoS(row, phi_3dB = 65, theta_3dB = 65):
    raylos = row['RayLoS']
    phi = row['phi']
    theta = row['theta']
    if (abs(phi) <= phi_3dB/2 
        and abs(theta) <= theta_3dB/2
        and raylos == 1):
        return 1
    else:
        return 0

def buildingWidth(row):
    alpha = row['alpha']
    beta = row['beta']
    gamma = row['gamma']
    return 1000 * np.sqrt(alpha/beta)
    
def streetWidth(row):
    alpha = row['alpha']
    beta = row['beta']
    gamma = row['gamma']
    w = 1000 * np.sqrt(alpha/beta)
    return 1000/np.sqrt(beta) - w
    
def extract_alpha(row):
    _, alpha, beta, gamma = row['Environment'].split('_')
    return eval(alpha)

def extract_beta(row):
    _, alpha, beta, gamma = row['Environment'].split('_')
    return eval(beta)

def extract_gamma(row):
    _, alpha, beta, gamma = row['Environment'].split('_')
    return eval(gamma)

def extract_polar_env_angle(row):
    _, alpha, beta, gamma = row['Environment'].split('_')
    alpha = eval(alpha)
    beta = eval(beta)
    gamma = eval(gamma)
    w_b = 1000 * np.sqrt(alpha/beta)
    w_s = 1000/np.sqrt(beta) - w_b
    # complex
    ex = (w_s - w_b) + 1j * (w_s - gamma) # good
    return np.arctan2(ex.imag, ex.real)

def spectral_efficiency(row, sinr = 'SINR_o', bandwidth_MHz = 10, generation = '5G', MCLut = None):
    sinr = row[sinr]
    return bandwidth_MHz * np.log(1 + sinr)

def capacity(row, se = 'SE_o', bandwidth_MHz = 10):
    se = row[se]
    return se * bandwidth_MHz

