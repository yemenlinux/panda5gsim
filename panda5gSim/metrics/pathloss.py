"""
Path loss calculator

Author: Edward Oughton
Date: April 2019

An implementation of a path loss calculator utilising ETSI TR 138 901 (0.5 - 100 GHz).

"""
import numpy as np
from math import pi, sqrt

from scipy import constants as scipy_constants
c = scipy_constants.speed_of_light

def FSPL(d3d, frequency_Ghz, Gtx = 0.0, Grx = 0.0):
    # FSPLn is the free space path loss for a certain receiver, obtained from Friis transmission equation 
    # A. Saakian, Radio Wave Propagation Fundamentals. Artech House, 2011.
    f_MHz = frequency_Ghz *1000
    # refernce FSPL at 1 m, 1 MHz.
    if d3d == 0:
        d3d = 1
    fspl0 = 20 * np.log10(1) + 20 * np.log10(1e6) + 20 * np.log10(4 * np.pi  / scipy_constants.speed_of_light) - Gtx - Grx
    return 20 * np.log10(d3d) + 20 * np.log10(f_MHz) + fspl0

def test_FSPL():
    return round(FSPL(10, 28, Gtx = 0.0, Grx = 0.0),2) == 81.39

#
def get_indoor_distance(distance2d, street_width):
    """ estimate indoor distance with mean value of 10m
    """
    if distance2d < street_width:
        indoor_distance = 10
    if distance2d < 10 and distance2d > 2.5:
        #todo: check this value 
        indoor_distance = 2.5
    else:
        indoor_distance = 0
    return distance2d - indoor_distance, indoor_distance
    
def get_breakpoint_distance(scenario, frequency, cell_height, ue_height):
    """Breakpoint distance 
    see 3GPP TR 38.901 R16.0.0 (2020-11)
    Args:
        BS (base station): base station, eNodeB, gNodeB, gNB, or AP

    Returns:
        _type_: _description_
    """
    # the effective environment height hE
    if scenario == "UMi" or scenario == "UMa":
        # note 1: the effective environment height hE is defined with 
        # probability equal to 1/(1+C(d2D, hUT)) and chosen from a discrete 
        # uniform distribution uniform(12,15,…,(hUT-1.5)) otherwise. 
        # With C(d2D, hUT) given by
        #TODO: implement note 1 of ETSI TR 138 901 Table 7.4.1-1: Pathloss models ETSI (page 29)
        hE = 1
    # indoor office or rural macrocell
    elif scenario == "InH" or scenario == "RMa":
        hE = 0
    else:
        hE = 0
    # the effective antenna height at the BS
    hBS = cell_height - hE
    # the effective antenna height at the UE
    hUT = ue_height - hE
    # for UMi and UMa
    h_apost_bs = cell_height - ue_height
    h_apost_ut = ue_height - hE
    # the frequency in Hz
    fc = (frequency * 1e9)/3e8 
    # the propagation velocity in free space
    # the breakpoint distance
    if scenario == "UMi" or scenario == "UMa":
        dBP = 4 * h_apost_bs * h_apost_ut * fc
        return dBP, h_apost_bs, h_apost_ut
    elif scenario == "InH":
        dBP = 4 * hBS * hUT *fc
        return dBP, hBS, hUT
    elif scenario == "RMa":
        dBP = 2 * np.pi * hBS * hUT * fc
        return dBP, hBS, hUT
    else:
        dBP = 2 * np.pi * hBS * hUT * fc
        return dBP, hBS, hUT
    #return dBP

def path_loss_rma_los(
                    fc, 
                    distance2d,
                    distance3d,
                    building_height,
                    dbp,
                    iterations,
                    seed_value):
    """
    """
    PL_rma_los = None
    # if 10 <= distance2d <= dbp:
    if distance2d <= dbp:
        PL_rma_los = round(
            20*np.log10(40 * np.pi * distance3d * fc/3) 
            + min(0.03 * building_height**1.72, 10) 
            * np.log10(distance3d) - min(0.044 * building_height**1.72, 14.77)
            + 0.002 * np.log10(building_height) * distance3d 
            + generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value)
        )
    if dbp <= distance2d <= 10000:
        PL_rma_los = round(
            20*np.log10(40*np.pi*dbp*fc/3) + min(0.03*building_height**1.72,10) *
            np.log10(dbp) - min(0.044*building_height**1.72,14.77) +
            0.002*np.log10(building_height)*dbp +
            generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value) +
            40*np.log10(distance3d / dbp) +
            generate_log_normal_dist_value(fc, 1, 6, iterations, seed_value)
        )
    #
    if PL_rma_los is None:
        print(f'f:{fc}, d:{distance2d}, d3d:{distance3d}, h:{building_height}, seed_value:{seed_value}, iterations:{iterations}')
        print(f'dbp:{dbp}')
        raise ValueError(
            "PL_rma_los is None"
        )
    return PL_rma_los
    

def path_loss_rma_nlos(fc, distance2d, distance3d, building_height, street_width, hbs, hut, dbp, iterations, seed_value):
    """
    """
    PL_rma_los = 0
    PL_apostrophe_rma_nlos = round(
                    161.04 - 7.1 * np.log10(street_width)+7.5*np.log10(building_height) -
                    (24.37 - 3.7 * (building_height/hbs)**2)*np.log10(hbs) +
                    (43.42 - 3.1*np.log10(hbs))*(np.log10(distance3d)-3) +
                    20*np.log10(fc) - (3.2 * (np.log10(11.75*hut))**2 - 4.97) +
                    generate_log_normal_dist_value(fc, 1, 8, iterations, seed_value)
                )
    if 10 <= distance2d <= 5000:
        PL_rma_los = path_loss_rma_los(fc, distance2d, distance3d, building_height, dbp, iterations, seed_value)
        
    return max(PL_rma_los, PL_apostrophe_rma_nlos)
        
def path_loss_uma_los(fc, distance2d, distance3d, building_height, street_width, hbs, hut, dbp, iterations, seed_value):
    """
    """
    #if 10.0 <= distance2d and distance2d < dbp:
    if distance2d < dbp:
        PL_uma_los = round(28.0 + 22.0*np.log10(distance3d) + 20.0*np.log10(fc)
                   + generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value)
        )
    elif dbp <= distance2d and distance2d < 5000.0:
        PL_uma_los = round(28.0 + 40.0*np.log10(distance3d) + 20.0*np.log10(fc) - 9.0*np.log10(pow(dbp,2)+pow((hbs-hut),2))
                           + generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value)
        )
    else:
        PL_uma_los = round(28.0 + 40.0*np.log10(distance3d) + 20.0*np.log10(fc) 
                           - 9.0*np.log10(dbp**2 + (hbs-hut)**2) 
                           - 26.0*np.log10(distance3d/1000.0)
                           + generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value)
        )
    return PL_uma_los

def path_loss_uma_nlos(fc, distance2d, distance3d, building_height, street_width, hbs, hut, dbp, iterations, seed_value):
    """
    """
    # if 10 <= distance2d and  distance2d <= 5000:
    if 10 <= distance2d and  distance2d <= 5000:
        PL_uma_los = path_loss_uma_los(fc, distance2d, distance3d, building_height, street_width, hbs, hut, dbp, iterations, seed_value)
    else:
        PL_uma_los = 0
    PL_uma_nlos = round(13.54 + 39.08*np.log10(distance3d) + 20*np.log10(fc) - 0.6*(hut-1.5)
                        + generate_log_normal_dist_value(fc, 1, 6, iterations, seed_value)
    )
    PL_uma_nlos = max(PL_uma_los, PL_uma_nlos)
    return PL_uma_nlos

def path_loss_umi_los(fc, distance2d, distance3d, hbs, hut, dbp, iterations, seed_value):
    """
    """
    if 10 <= distance2d and distance2d <= dbp:
        PL_umi_los = round(32.4 + 21*np.log10(distance3d) + 20*np.log10(fc)
                           + generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value)
        )
    else:
        PL_umi_los = round(32.4 + 40*np.log10(distance3d) + 20*np.log10(fc) - 9.5*np.log10(pow(dbp,2)+pow((hbs-hut),2))
                           + generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value)
        )
    return PL_umi_los

def path_loss_umi_nlos(fc, distance2d, distance3d, hbs, hut, dbp, iterations, seed_value):
    """
    """
    if 10 <= distance2d and distance2d <= 5000:
        PL_umi_los = path_loss_umi_los(fc, distance2d, distance3d, hbs, hut, dbp, iterations, seed_value)
    else:
        PL_umi_los = 0
    PL_umi_nlos = round(22.4 + 35.3*np.log10(distance3d) + 21.3*np.log10(fc) - 0.3*(hut-1.5)
                        + generate_log_normal_dist_value(fc, 1, 7.82, iterations, seed_value)
    )
    return max(PL_umi_los, PL_umi_nlos)

def path_loss_inh_los(fc, distance3d, iterations, seed_value):
    """
    """
    PL_inh_los = round(32.4 + 17.3*np.log10(distance3d) + 20*np.log10(fc)
                           + generate_log_normal_dist_value(fc, 1, 3, iterations, seed_value)
    )
    return PL_inh_los

def path_loss_inh_nlos(fc, distance3d, iterations, seed_value):
    """
    """
    PL_inh_los = path_loss_inh_los(fc, distance3d, iterations, seed_value)
    PL_inh_nlos = round(17.3 + 38.3*np.log10(distance3d) + 24.9*np.log10(fc) 
                        + generate_log_normal_dist_value(fc, 1, 8.03, iterations, seed_value))
    return max(PL_inh_los, PL_inh_nlos)

def path_loss_inf_los(fc, distance3d, iterations, seed_value):
    """
    """
    return round(31.84 + 22.0*np.log10(distance3d) + 20*np.log10(fc)
                       + generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value)
    )
    
def path_loss_inf_nlos(scenario, fc, distance3d, iterations, seed_value):
    """
    """
    PL_inf_los = path_loss_inf_los(fc, distance3d, iterations, seed_value)
    # InF-SL
    if scenario == 'InF-SL':
        PL_inf_sl_nlos = round(33 + 25.5*np.log10(distance3d) + 20*np.log10(fc)
                            + generate_log_normal_dist_value(fc, 1, 5.7, iterations, seed_value))
        return max(PL_inf_los, PL_inf_sl_nlos)
    #  InF-DL
    if scenario == 'InF-DL':
        PL_inf_dl_nlos = round(18.6 + 35.7*np.log10(distance3d) + 20*np.log10(fc)
                            + generate_log_normal_dist_value(fc, 1, 7.2, iterations, seed_value))
        return max(PL_inf_los, PL_inf_dl_nlos)
    # Inf-SH
    if scenario == 'InF-SH':
        PL_inf_sh_nlos = round(32.4 + 23.0*np.log10(distance3d) + 20*np.log10(fc)
                            + generate_log_normal_dist_value(fc, 1, 5.9, iterations, seed_value))
        return max(PL_inf_los, PL_inf_sh_nlos)
    # INF-DH
    if scenario == 'InF-DH':
        PL_inf_dh_nlos = round(33.63 + 21.9*np.log10(distance3d) + 24.9*np.log10(fc)
                            + generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value))
        return max(PL_inf_los, PL_inf_dh_nlos)
    

# modified from pysim5G
def path_loss_etsi_tr_138_901(scenario,
                            frequency, 
                            distance2d, 
                            distance3d, 
                            cell_height,
                            ue_height,
                            building_height,
                            type_of_sight,
                            indoor,
                            street_width,
                            seed_value,
                            iterations
                            ):
    """

    Model requires:
        - Frequency in gigahertz
        - Distance in meters

    c = speed of light
    he = effective environment height
    hbs = effective antenna height
    hut = effective user terminal height

    """
    fc = frequency # GHz
    #c = 3e8
    #
    dbp, hbs, hut = get_breakpoint_distance(scenario, frequency, cell_height, ue_height)
    # if indoor: estimate indoor distance
    if indoor:
        d2d_out, d2d_in = get_indoor_distance(distance2d, street_width)
    else:
        d2d_out = distance2d
    #
    check_3gpp_applicability(building_height, street_width, cell_height, ue_height)
    #
    
    if scenario == 'RMa':
        # path loss for suburban macrocell and rural macrocell
        if type_of_sight == 1:
            PL_rma_los = None
            PL_rma_los = path_loss_rma_los(fc, distance2d, distance3d, building_height, dbp, iterations, seed_value)
            return PL_rma_los
        if type_of_sight == 0:
            PL_rma_nlos = path_loss_rma_nlos(fc, distance2d, distance3d, building_height, street_width, hbs, hut, dbp, iterations, seed_value)
            return PL_rma_nlos
    elif scenario == 'UMa':
        if type_of_sight == 1:
            PL_uma_los = path_loss_uma_los(fc, distance2d, distance3d, building_height, street_width, hbs, hut, dbp, iterations, seed_value)
            return PL_uma_los
        if type_of_sight == 0:
            PL_uma_nlos = path_loss_uma_nlos(fc, distance2d, distance3d, building_height, street_width, hbs, hut, dbp, iterations, seed_value)
            return PL_uma_nlos
    elif scenario == 'UMi':
        if type_of_sight == 1:
            PL_umi_los = path_loss_umi_los(fc, distance2d, distance3d, hbs, hut, dbp, iterations, seed_value)
            return PL_umi_los
        if type_of_sight == 0:
            PL_umi_nlos = path_loss_umi_nlos(fc, distance2d, distance3d, hbs, hut, dbp, iterations, seed_value)
            return PL_umi_nlos
    elif scenario == '3D-UMi':
        #todo: implement 3D-UMi as in 3GPP TR 36.873
        if type_of_sight == 1:
            PL_umi_los = path_loss_umi_los(fc, distance2d, distance3d, hbs, hut, dbp, iterations, seed_value)
            return PL_umi_los
        if type_of_sight == 0:
            PL_umi_nlos = path_loss_umi_nlos(fc, distance2d, distance3d, hbs, hut, dbp, iterations, seed_value)
            return PL_umi_nlos
    elif scenario == 'InH':
        if type_of_sight == 1:
            PL_inh_los = path_loss_inh_los(fc, distance3d, iterations, seed_value)
            return PL_inh_los
        if type_of_sight == 0:
            PL_inh_nlos = path_loss_inh_nlos(fc, distance3d, iterations, seed_value)
            return PL_inh_nlos
    elif scenario == 'InF' or scenario == 'InF-SL' or scenario == 'InF-DL' or scenario == 'InF-SH' or scenario == 'InF-DH':
        if type_of_sight == 1:
            PL_inf_los = path_loss_inf_los(fc, distance3d, iterations, seed_value)
            return PL_inf_los
        if type_of_sight == 0:
            PL_inf_nlos = path_loss_inf_nlos(scenario, fc, distance3d, iterations, seed_value)
            return PL_inf_nlos
    else:
        raise ValueError(f'Unknown Scenario: {scenario}, the scenario must be one of the following: RMa, UMa, UMi, InH, InF, InF-SL, InF-DL, InF-SH, InF-DH')

            

def calculate_path_loss(scenario,
                        frequency, 
                        distance2d, 
                        distance3d, 
                        cell_height,
                        ue_height,
                        building_height,
                        type_of_sight,
                        indoor,
                        street_width,
                        seed_value,
                        iterations
                        ):
    """
    Calculate the correct path loss given a range of critera.

    Parameters
    ----------
    frequency : float
        Frequency band given in GHz.
    distance : float
        Distance between the transmitter and receiver in km.
    ant_height:
        Height of the antenna.
    ant_type : string
        Indicates the type of site antenna (hotspot, micro, macro).
    building_height : int
        Height of surrounding buildings in meters (m).
    street_width : float
        Width of street in meters (m).
    settlement_type : string
        Gives the type of settlement (urban, suburban or rural).
    type_of_sight : string
        Indicates whether the path is (Non) Line of Sight (LOS or NLOS).
    ue_height : float
        Height of the User Equipment.
    above_roof : int
        Indicates if the propagation line is above or below building roofs.
        Above = 1, below = 0.
    indoor : binary
        Indicates if the user is indoor (True) or outdoor (False).
    seed_value : int
        Dictates repeatable random number generation.
    iterations : int
        Specifies how many iterations a specific calculation should be run for.

    Returns
    -------
    path_loss : float
        Path loss in decibels (dB)
    model : string
        Type of model used for path loss estimation.

    """
    # distance = distance3d
    # ant_type = cell_type
    # ant_height = cell_height
    # building_height
    # street_width
    # settlement_type
    # type_of_sight = los
    # ue_height = ue_height
    # above_roof 
    # indoor
    # seed_value
    # iterations
    
    if 0.05 < frequency <= 100:
        path_loss = path_loss_etsi_tr_138_901(scenario,
                                            frequency, 
                                            distance2d, 
                                            distance3d, 
                                            cell_height,
                                            ue_height,
                                            building_height,
                                            type_of_sight,
                                            indoor,
                                            street_width,
                                            seed_value,
                                            iterations
                                            )
        #print(f'path_loss = {path_loss}, {outdoor_to_indoor_path_loss(frequency, indoor, seed_value)}')
        path_loss = path_loss + outdoor_to_indoor_path_loss(
            frequency, indoor, seed_value
            )
        model = 'etsi_tr_138_901'
    else:
        raise ValueError (
            "frequency of {} is NOT within correct range".format(frequency)
        )
    return round(path_loss), model


##
# ----------- Code from pysim5G  ------------------
def path_loss_calculator(frequency, distance, ant_height, ant_type,
    building_height, street_width, settlement_type, type_of_sight,
    ue_height, above_roof, indoor, seed_value, iterations):
    """
    Calculate the correct path loss given a range of critera.

    Parameters
    ----------
    frequency : float
        Frequency band given in GHz.
    distance : float
        Distance between the transmitter and receiver in km.
    ant_height:
        Height of the antenna.
    ant_type : string
        Indicates the type of site antenna (hotspot, micro, macro).
    building_height : int
        Height of surrounding buildings in meters (m).
    street_width : float
        Width of street in meters (m).
    settlement_type : string
        Gives the type of settlement (urban, suburban or rural).
    type_of_sight : string
        Indicates whether the path is (Non) Line of Sight (LOS or NLOS).
    ue_height : float
        Height of the User Equipment.
    above_roof : int
        Indicates if the propagation line is above or below building roofs.
        Above = 1, below = 0.
    indoor : binary
        Indicates if the user is indoor (True) or outdoor (False).
    seed_value : int
        Dictates repeatable random number generation.
    iterations : int
        Specifies how many iterations a specific calculation should be run for.

    Returns
    -------
    path_loss : float
        Path loss in decibels (dB)
    model : string
        Type of model used for path loss estimation.

    """
    if 0.05 < frequency <= 100:

        path_loss = etsi_tr_138_901(frequency, distance, ant_height, ant_type,
            building_height, street_width, settlement_type, type_of_sight,
            ue_height, above_roof, indoor, seed_value, iterations
        )

        #print(f'path_loss = {path_loss}, {outdoor_to_indoor_path_loss(frequency, indoor, seed_value)}')
        path_loss = path_loss + outdoor_to_indoor_path_loss(
            frequency, indoor, seed_value
        )

        model = 'etsi_tr_138_901'

    else:

        raise ValueError (
            "frequency of {} is NOT within correct range".format(frequency)
        )

    return round(path_loss), model


def etsi_tr_138_901(frequency, distance, ant_height, ant_type,
    building_height, street_width, settlement_type, type_of_sight,
    ue_height, above_roof, indoor, seed_value, iterations):
    """

    Model requires:
        - Frequency in gigahertz
        - Distance in meters

    c = speed of light
    he = effective environment height
    hbs = effective antenna height
    hut = effective user terminal height

    """
    fc = frequency
    c = 3e8

    he = 1 #enviroment_height
    hbs = ant_height
    hut = ue_height
    h_apost_bs = ant_height - ue_height
    h_apost_ut = ue_height - he
    w = street_width # mean street width is 20m
    h = building_height # mean building height

    dbp = 2 * pi * hbs * hut * (fc * 1e9) / c
    d_apost_bp = 4 * h_apost_bs * h_apost_ut * (fc*1e9) / c
    d2d_in = 10 #mean d2d_in value
    d2d_out = distance - d2d_in
    d2d = d2d_out + d2d_in
    d3d = sqrt((d2d_out + d2d_in)**2 + (hbs - hut)**2)

    check_3gpp_applicability(building_height, street_width, ant_height, ue_height)

    if ant_type == 'macro':
        if settlement_type == 'suburban' or settlement_type == 'rural':
            pl1 = round(
                20*np.log10(40*pi*d3d*fc/3) + min(0.03*h**1.72,10) *
                np.log10(d3d) - min(0.044*h**1.72,14.77) +
                0.002*np.log10(h)*d3d +
                generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value)
            )

            if 10 <= d2d <= dbp:
                if type_of_sight == 1:
                    return pl1
            pl_rma_los = pl1

            pl2 = round(
                20*np.log10(40*pi*dbp*fc/3) + min(0.03*h**1.72,10) *
                np.log10(dbp) - min(0.044*h**1.72,14.77) +
                0.002*np.log10(h)*dbp +
                generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value) +
                40*np.log10(d3d / dbp) +
                generate_log_normal_dist_value(fc, 1, 6, iterations, seed_value)
            )

            if dbp <= d2d <= 10000:
                if type_of_sight == 1:
                    return pl2
            pl_rma_los = pl2

            if type_of_sight == 0:

                pl_apostrophe_rma_nlos = round(
                    161.04 - 7.1 * np.log10(w)+7.5*np.log10(h) -
                    (24.37 - 3.7 * (h/hbs)**2)*np.log10(hbs) +
                    (43.42 - 3.1*np.log10(hbs))*(np.log10(d3d)-3) +
                    20*np.log10(fc) - (3.2 * (np.log10(11.75*hut))**2 - 4.97) +
                    generate_log_normal_dist_value(fc, 1, 8, iterations, seed_value)
                )

                # # currently does not cap at 5km, which this should
                pl_rma_nlos = max(pl_apostrophe_rma_nlos, pl_rma_los)

                return pl_rma_nlos

            if d2d > 10000:
                return uma_nlos_optional(frequency, distance, ant_height, ue_height,
                    seed_value, iterations)

        elif settlement_type == 'urban':
            #print(f'd2d: {d2d}, d_apost_bp: {d_apost_bp}')
            pl1 = round(
                28 + 22 * np.log10(d3d) + 20 * np.log10(fc) +
                generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value)
            )

            if 10 <= d2d <= d_apost_bp:
                if type_of_sight == 1:
                    return pl1
            pl_uma_los = pl1

            pl2 = round(
                28 + 40*np.log10(d3d) + 20 * np.log10(fc) -
                9*np.log10((d_apost_bp)**2 + (hbs-hut)**2) +
                generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value)
            )

            if d_apost_bp <= d2d <= 5000:
                if type_of_sight == 1:
                    return pl2
            pl_uma_los = pl2

            if type_of_sight == 0:

                if d2d <= 5000:
                    pl_apostrophe_uma_nlos = round(
                        13.54 + 39.08 * np.log10(d3d) + 20 *
                        np.log10(fc) - 0.6 * (hut - 1.5) +
                        generate_log_normal_dist_value(fc, 1, 6, iterations, seed_value)
                    )

                if d2d > 5000:
                    pl_apostrophe_uma_nlos = uma_nlos_optional(frequency, distance, ant_height,
                        ue_height, seed_value, iterations)

                pl_uma_nlos = max(pl_apostrophe_uma_nlos, pl_uma_los)

                return pl_uma_nlos

        else:
            # return uma_nlos_optional(frequency, distance, ant_height, ue_height,
            #     seed_value, iterations)
            raise ValueError('Did not recognise settlement_type')

    elif ant_type == 'micro':

            pl1 = round(
                32.4 + 21 * np.log10(d3d) + 20 * np.log10(fc) +
                generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value)
            )

            if 10 <= d2d <= d_apost_bp:
                if type_of_sight == 1:
                    return pl1
                pl_umi_los = pl1

            pl2 = round(
                32.4 + 40*np.log10(d3d) + 20 * np.log10(fc) -
                9.5*np.log10((d_apost_bp)**2 + (hbs-hut)**2) +
                generate_log_normal_dist_value(fc, 1, 4, iterations, seed_value)
            )

            if d_apost_bp <= d2d <= 5000:
                if type_of_sight == 1:
                    return pl2
            pl_umi_los = pl2

            if type_of_sight == 0:

                if d2d <= 5000:
                    pl_apostrophe_umi_nlos = round(
                        35.3 * np.log10(d3d) + 22.4 +
                        21.3 * np.log10(fc) - 0.3 * (hut - 1.5) +
                        generate_log_normal_dist_value(fc, 1, 7.82, iterations, seed_value)
                    )

            pl_uma_nlos = max(pl_apostrophe_umi_nlos, pl_umi_los)

            return pl_uma_nlos

    else:
        raise ValueError('Did not recognise ant_type')
    #
    print(f'frequency: {frequency} , distance:{distance}, ant_height:{ant_height}, ant_type:{ant_type}, building_height:{building_height}, street_width:{street_width}, settlement_type:{settlement_type}, type_of_sight:{type_of_sight}, ue_height:{ue_height}, above_roof:{above_roof}, indoor:{indoor}, seed_value:{seed_value}, iterations:{iterations}')
    return 'complete'


def uma_nlos_optional(frequency, distance, ant_height, ue_height,
    seed_value, iterations):
    """

    UMa NLOS / Optional from ETSI TR 138.901 / 3GPP TR 38.901

    Parameters
    ----------
    frequency : int
        Carrier band (f) required in GHz.
    distance : int
        Distance (d) between transmitter and receiver (km).
    ant_height : int
        Transmitter antenna height (h1) (m, above ground).
    ue_height : int
        Receiver antenna height (h2) (m, above ground).
    sigma : int
        Variation in path loss (dB) which is 2.5dB for free space.
    seed_value : int
        Dictates repeatable random number generation.
    iterations : int
        Specifies iterations for a specific calculation.

    Returns
    -------
    path_loss : float
        Path loss in decibels (dB)

    """
    fc = frequency
    d3d = sqrt((distance)**2 + (ant_height - ue_height)**2)

    path_loss = 32.4 + 20*np.log10(fc) + 30*np.log10(d3d)

    random_variation = generate_log_normal_dist_value(
        frequency, 1, 7.8, iterations, seed_value
    )

    return round(path_loss + random_variation)


def check_3gpp_applicability(building_height, street_width, ant_height, ue_height):

    if 5 <= building_height < 50 :
        building_height_compliant = True
    else:
        building_height_compliant = False
        print('building_height not compliant')

    if 5 <= street_width < 50:
        street_width_compliant = True
    else:
        street_width_compliant = False
        print('Street_width not compliant')

    if 10 <= ant_height < 150:
        ant_height_compliant = True
    else:
        ant_height_compliant = False
        print(f'ant_height: {ant_height} not compliant')

    #if 1 <= ue_height < 10:
    if 0.07 <= ue_height < 100:
        ue_height_compliant = True
    else:
        ue_height_compliant = False
        print(f'ue_height not compliant {ue_height}')

    if (building_height_compliant + street_width_compliant +
        ant_height_compliant + ue_height_compliant) == 4:
        overall_compliant = True
    else:
        overall_compliant = False

    return overall_compliant


def generate_log_normal_dist_value(frequency, mu, sigma, draws, seed_value):
    """
    Generates random values using a lognormal distribution,
    given a specific mean (mu) and standard deviation (sigma).

    https://stackoverflow.com/questions/51609299/python-np-lognormal-gives-infinite-
    results-for-big-average-and-st-dev

    The parameters mu and sigma in np.random.lognormal are not the mean
    and STD of the lognormal distribution. They are the mean and STD
    of the underlying normal distribution.

    Parameters
    ----------
    mu : int
        Mean of the desired distribution.
    sigma : int
        Standard deviation of the desired distribution.
    draws : int
        Number of required values.

    Returns
    -------
    random_variation : float
        Mean of the random variation over the specified itations.

    """
    if seed_value == None:
        pass
    else:
        frequency_seed_value = seed_value * frequency * 100

        np.random.seed(int(str(frequency_seed_value)[:2]))

    normal_std = np.sqrt(np.log10(1 + (sigma/mu)**2))
    normal_mean = np.log10(mu) - normal_std**2 / 2

    hs = np.random.lognormal(normal_mean, normal_std, draws)

    return round(np.mean(hs),2)


def outdoor_to_indoor_path_loss(frequency, indoor, seed_value):
    """

    ITU-R M.1225 suggests building penetration loss for shadow fading can be modelled
    as a log-normal distribution with a mean and  standard deviation of 12 dB and
    8 dB respectively.

    frequency : int
        Carrier band (f) required in MHz.
    indoor : binary
        Indicates if the user is indoor (True) or outdoor (False).
    seed_value : int
        Dictates repeatable random number generation.

    Returns
    -------
    path_loss : float
        Outdoor to indoor path loss in decibels (dB)

    """
    if indoor:
        outdoor_to_indoor_path_loss = generate_log_normal_dist_value(frequency, 12, 8, 1, seed_value)
    else:
        outdoor_to_indoor_path_loss = 0
    return outdoor_to_indoor_path_loss
