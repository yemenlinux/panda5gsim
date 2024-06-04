
import numpy as np

np.random.seed(42)

# Line of sight probability for simulation scenarios and distances
def probability_of_los(scenario, distance2d, distance3d, indoor):
    """
    """
    if scenario == 'RMa':
        return probability_of_los_rma(distance2d)
        


def probability_of_los_rma(distance2d):
    """
    """
    if distance2d <= 10:
        return 1
    else:
        return np.exp(-((distance2d - 10)/1000))
    
def c_apostrophe(h_ut):
    """
    """
    if h_ut <= 13:
        return 0
    elif 13 < h_ut and h_ut <= 23:
        return ((h_ut - 13)/10)**1.5
    
def probability_of_los_uma(distance2d, h_ut):
    """
    """
    if distance2d <= 18:
        return 1
    else:
        return ((18/distance2d) + np.exp(-(distance2d/63)) * (1 - (18/distance2d)) 
        * (1 + c_apostrophe(h_ut) *(5/4) * ((distance2d/100)**3)*np.exp(-(distance2d/150)))
        )
        
def probability_of_los_umi(distance2d):
    """
    """
    if distance2d <= 18:
        return 1
    else:
        return ((18/distance2d) + np.exp(-(distance2d/36)) * (1 - (18/distance2d)))
    
def probability_of_los_inh_mixed(distance2d):
    """
    """
    if distance2d <= 1.2:
        return 1
    elif 1.2 < distance2d and distance2d < 6:
        return np.exp(-(distance2d - 1.2)/4.7) 
    else:
        return np.exp(-(distance2d - 6.5)/32.6) * 0.32
    
def probability_of_los_inh_open(distance2d):
    """
    """
    if distance2d <= 5:
        return 1
    elif 5 < distance2d and distance2d <= 49:
        return np.exp(-(distance2d - 5)/70.8) 
    else:
        return np.exp(-(distance2d - 49)/32.6) * 0.54
    
def probability_of_los_inf(scenario, distance2d, d_clutter, hbs, hut, r, hc):
    """ LOS probability for indoor hotspot scenarios
        for parameters see 3GPP TR 38.901 V16.0.0 (2020-01) Table 7.2-4: Evaluation parameters for InF
    """
    k_subsce = calculate_k_subsce(scenario, d_clutter, hbs, hut, r, hc)
    return np.exp(-(distance2d / k_subsce))
    
def calculate_k_subsce(scenario, d_clutter, hbs, hut, r, hc):
    """ Calculate k_subsce for indoor hotspot scenarios
        for parameters see 3GPP TR 38.901 V16.0.0 (2020-01) Table 7.2-4: Evaluation parameters for InF
    """
    if scenario == 'InF-SL' or scenario == 'InF-DL':
        return -(d_clutter / np.ln(1 - r))
    if scenario == 'InF-SH' or scenario == 'InF-DH':
        return -(d_clutter / np.ln(1 - r)) * ((hbs - hut) / (hc - hbs))