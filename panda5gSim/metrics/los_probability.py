
import numpy as np
import math
from scipy.special import erf

# np.random.seed(42)
from panda5gSim.core.transformations import TransformProcessor

# LoS probability
# Rec. ITU-R P.1410-5
# calculate number of buildings from the transmitter to the receiver
def num_buildings_between_TxRX(d2D, alpha, beta):
    """ calculate number of buildings from the transmitter to the receiver
    based on the alpha, beta and ground distance
    d2D (float): distance in meters
    """
    d2D = d2D/1000 # distance in km
    return np.floor((d2D * np.sqrt(alpha * beta)))

# calculate the probability of LoS
def P_LoS_Rec_ITU_R_P_1410_5(d2d, h_tx, h_rx, alpha, beta, gamma):
    """ calculate the probability of LoS
        based on the alpha, beta and ground distance
        implementation of Rec. ITU-R P.1410-5 2.1.5
    """
    # calculate the number of buildings
    N = num_buildings_between_TxRX(d2d, alpha, beta)
    if N == 0:
        return 1.0
    # calculate delta
    delta = d2d / N
    # calculate di = (i+1/2)*delta for i in {0, 1, 2, ..., (N-1)}
    di = np.array([(i + 0.5) * delta for i in range(int(N))])
    # the height of the ray joining the transmitter and receiver 
    # at the point where the ray crosses the building.
    # h_{los} = h_{tx} - \frac{r_{los}(h_{tx} - h_{rx})}{r_{rx}}
    hi = [(h_tx - (d_i * (h_tx - h_rx) / d2d)) for d_i in di]
    # The probability Pi that a building is smaller than height hi is given by
    # Pi(hi) = 1 - e^{-\frac{h_{i}^{2}}{2 \gamma^2}} for i in hi
    Pi = [(1 - np.exp(-((h_i**2) / (2 * gamma**2)))) for h_i in hi]
    # The probability PLoS,i that there is a LoS ray at 
    # position di is given by: 
    # P_LoS = \prod_{j=0}^{i} Pi(h_{j})
    P_LoS = np.prod(Pi)
    return P_LoS

# implementation of the paper
# Kang, H., Joung, J., Ahn, J., & Kang, J. (2019). 
# Secrecy-Aware Altitude Optimization for Quasi-Static UAV 
# Base Station Without Eavesdropper Location Information. 
# IEEE Communications Letters, 23(5), 851–854. 
# https://doi.org/10.1109/LCOMM.2019.2909880
#
def P_LoS_Kang2019Secrecy(d2d, h_tx, h_rx, alpha, beta, gamma):
    """ calculate the LoS probability
        implementation of Kang2019
        P_{LoS} (d,h) = e^{- \sqrt{\frac{\mu \pi}{2}} \frac{d \gamma}{h} 
        erf(\frac{h}{\sqrt{2} \gamma})}
        
        parameters:
        d2d (float): distance in meter
    """
    d2d = d2d / 1000 # distance in km
    _erf = erf((h_tx - h_rx)/(np.sqrt(2)* gamma))
    mu = alpha * beta
    P_LoS = np.exp(- np.sqrt(mu * np.pi / 2) * (d2d * gamma / h_tx) * _erf)
    return P_LoS
    

# implementation of 
# Mohammed, I., Gopalam, S., Collings, I. B., & Hanly, S. v. (2023). 
# Closed Form Approximations for UAV Line-of-Sight Probability in 
# Urban Environments. IEEE Access, 11, 40162–40174. 
# https://doi.org/10.1109/ACCESS.2023.3267808
#
def P_LoS_Mohammed2023Closed(d2d, h_tx, h_rx, alpha, beta, gamma):
    """ calculate the LoS probability
        implementation of Mohammed2023
        P_{LoS} (d,h) = e^{- \sqrt{\frac{\mu \pi}{2}} \frac{d \gamma}{h} 
        erf(\frac{h}{\sqrt{2} \gamma})}
    """
    if d2d == 0:
        # return 1
        d2d = d2d + 0.0001
    # w =  1000 \sqrt{\frac{\alpha}{\beta}}
    w = 1000 * np.sqrt(alpha / beta)
    # 
    # s = \frac{1000}{\sqrt{\beta}}-w
    s = (1000 / np.sqrt(beta)) - w
    # s = (1 / np.sqrt(beta)) - w
    # \rho = \frac{h_{tx}^2}{2 \gamma^2 d^2}
    rho = ((h_tx-h_rx)**2) / (2 * gamma**2 * d2d**2)
    # \Lambda_1 = \frac{1}{s} \(\frac{613}{753}-\frac{901}{2116}\frac{s}{w} 
    # +\frac{1258}{8477}\(frac{s}{w})^2 - \frac{239}{10712}(\frac{s}{w})^3
    # \)
    Lambda_1 = ((1 / s) * ((613 / 753) - (901 / 2116) * (s / w) 
                          + (1258 / 8477) * (s / w)**2 
                          - (239 / 10712) * (s / w)**3))
    # if (s/w) <0.25 or (s/w)>2.5:
    #     print(f"s/w = {s/w}, lambda_1 = {Lambda_1}")
    # num = \sqrt{\pi} e^{\frac{\Lambda_1^2}{4 \rho}}
    x = (Lambda_1**2) / (4 * rho)
    #Note: in the source there is a typo in exponentiation function
    # the exponential exp(x) --> typo
    # the correct is exp(- x)
    num = np.sqrt(np.pi) * np.exp(-x)
    # P_LoS = F(r,h|\alpha, \beta, \gamma) =
    # 1 - \Lambda_1 \fraq{num}{2 \sqrt{\rho}} (erf(\frac{2\rho r 
    # + \Lambda_1}{2 \sqrt{\rho}}) 
    # - erf(\frac{\Lambda_1}{2 \sqrt{\rho}})
    erf1 = math.erf((2 * rho * d2d + Lambda_1) / (2 * np.sqrt(rho)))
    erf2 = math.erf(Lambda_1 / (2 * np.sqrt(rho)))
    P_LoS = (Lambda_1 * (num / (2 * np.sqrt(rho)))
          * (erf1 - erf2))
    return 1 - P_LoS

def P_LoS_Pang2023Geometry(d2D, h_tx, h_rx, alpha, beta, gamma, wl):
    """_summary_

    Args:
        d2D (float): ground distance
        h_tx (float): transmitter height
        h_rx (float): receiver height
        alpha (float): ratio of build-up area to land area
        beta (float): number of buildings in km^2
        gamma (float): average building height
        wl (float): signal wavelength

    Returns:
        P_LoS : line-of-sight probability, can be calculated by
        P_{LoS}(d_{2D}, h_{tx}, h_{rx}| \alpha, \beta, \gamma) = 
        \prod_{i=1}^{N_b} \left[
        1 - exp\left( - \frac{\left[h_{tx}- \frac{d_i (h_{tx} - h_{rx})}{d_{2D}} - 
            \frac{\sqrt{\lambda d_{2D}} \sqrt{d_{2D}^2 + (h_{tx} - h_{rx})^2}}{d_{2D}^2} min(d_i, d_{2D} - d_i)
            \right]^2}{2 \gamma^2}
        \right)
        \right]
        where $\lambda$, $h_{tx}$, $h_{rx}$, $d_{2D}$ are the wavelength, 
        transmitter height, receiver height, and ground distance between 
        transmitter and receiver. Number of buildings along the LoS path 
        between transmitter and receiver can be calculated by 
        $N = floor(d_{2D} \sqrt{\aleph \beta}/1000)$.
    """
    N = num_buildings_between_TxRX(d2D, alpha, beta)
    # bilding width
    w_b = np.sqrt(alpha/beta)
    # p_i
    p_i = []
    for i in range(int(N)):
        # d_i = \frac{(i+1/2)d_{2D}}{d_{2D} \sqrt{\alpha \beta}/1000} + \frac{w_b}{2}
        di = ((((i + 0.5) * d2D) 
                / np.floor(d2D * np.sqrt(alpha * beta)/1000))
                + (w_b / 2))
        # Ri
        if di <= d2D / 2:
            Ri = (di * (np.sqrt(wl * d2D) / d2D))
        else:
            Ri = ((d2D - di) * (np.sqrt(wl * d2D) / d2D))
        # hi = h_tx - \frac{d_i (h_tx - h_rx)}{d_{2D}} 
        #     - \frac{R_i}{cos(\theta)}
        # cos(\theta) = \frac{d_{2D}}{\sqrt{d_{2D}^2 + (h_tx - h_rx)^2}}
        cos_theta = d2D / np.sqrt(d2D**2 + (h_tx - h_rx)**2)
        hi = (h_tx - (di * (h_tx - h_rx) / d2D)
                - (Ri / cos_theta))
        # p_i
        p_i.append((1-np.exp(-(hi**2 / (2 * gamma**2)))))
    
    # p_i
    # p_i = np.array([(1-np.exp(-((h_tx - (di(i) * (h_tx - h_rx) / d2D) 
    #             - ((np.sqrt(wl * d2D) * np.sqrt(d2D**2 + (h_tx - h_rx)**2)) 
    #             / d2D**2) * min(di(i), d2D - di(i)))**2 
    #                 / (2 * gamma**2))) )
    #             for i in range(int(N))])
    # P_LoS
    P_LoS = np.prod(p_i)
    return P_LoS

# implementation for 
# A. Al-Hourani, “On the Probability of Line-of-Sight in Urban Envi-
# ronments,” IEEE Wireless Communications Letters, vol. 9, no. 8, pp.
# 1178–1181, aug 2020.
#
def log_normal_mu(mean_h_b, std_h_b):
    """ convert from Rayleigh
    mean of log normal distribution parameters
    
    Parameters:
        mean_h_b: mean of building heights, 
        std_h_b: std of building heights
    returns:
        mean of log-normal distribution 
        $\mu_0 = \ln (\frac{m_0}{\sqrt{1 + \frac{v_0}{m_0^2} } })$ 
    """
    return np.log(mean_h_b/np.sqrt(1 + (std_h_b/mean_h_b**2)))

def log_normal_std(mean_h_b, std_h_b):
    """ convert from Rayleigh 
    the standard deviation of log normal distribution parameters
    Parameters:
        mean_h_b: mean of building heights, 
        std_h_b: std of building heights
    returns:
        standard deviation of log-normal distribution
        $\sigma_0^2 =\ln (1+ \frac{v_0}{m_0^2}) $
    """
    return np.sqrt(np.log(1 + (std_h_b/mean_h_b**2)) )

def logNormal_CDF(h, mu, std):
    from scipy.special import erf
    return 0.5 + 0.5 * erf((np.log(h) - mu)/(np.sqrt(2)*std))

def P_LoS_AlHourani2020(d2d, h_tx, h_rx, 
                        alpha, beta, gamma, 
                        r0 = None, mu = None, std = None):
    """ implementation of LoS probability in 
        A. Al-Hourani, “On the Probability of Line-of-Sight in Urban Envi-
        ronments,” IEEE Wireless Communications Letters, vol. 9, no. 8, pp.
        1178–1181, aug 2020._summary_

    Parameters:
        d2d (float): horizontal distance
        h_tx (float): transmitter height
        h_rx (float): receiver height
        alpha (float): ratio of built-up area in urban
        beta (int): number of buildings in square kilometer
        gamma (float): Rayleigh parameter, used to convert from Rayleigh
        to log-normal
        r0 (float, optional): radius of cylinder building. Defaults to None.
        mu (float, optional): log-normal mean parameter. Defaults to None.
        std (float, optional): log-normal std parameter. Defaults to None.

    Returns:
        P_{LoS}: line-of-sight probability
    """
    beta = beta 
    if not r0:
        w =  1000*np.sqrt(alpha / beta)
        r0 = np.sqrt(w*w / np.pi)
    if d2d <= r0 * np.pi/2:
        return 1
    else:
        if not mu:
            mu0 = gamma * np.sqrt(np.pi/2)
            var = gamma * np.sqrt(2 - np.pi/2)
            mu = log_normal_mu(mu0, var)
        if not std:
            mu0 = gamma * np.sqrt(np.pi/2)
            var = gamma * np.sqrt(2 - np.pi/2)
            std = log_normal_std(mu0, var)
        # 
        return np.exp(- 2 * r0 * beta * (d2d - np.pi/2 * r0) * (1 - logNormal_CDF(h_tx -h_rx, mu, std)))

def P_LoS_Zhou2018(d2d, h_tx, h_rx, alpha, beta, gamma):
    """ implementation of LoS probability in the study:
        L. Zhou, Z. Yang, G. Zhao, S. Zhou, and C.-X. Wang, “Propagation
        Characteristics of Air-to-Air Channels in Urban Environments,” in 2018
        IEEE Global Communications Conference (GLOBECOM), dec 2018,
        pp. 1–6.
        $P_{LoS} (h_{rx}, \theta) =  exp(4 \gamma \sqrt{\frac{2 \alpha \beta}{\pi}} 
        \, Q(\frac{h_{rx}}{\gamma}) \,  \frac{1}{tan(\theta)})$
    parameters:
        d2d (float): horizontal distance
        h_tx (float): transmitter height
        h_rx (float): receiver height
        alpha (float): ratio of built-up area in urban
        beta (int): number of buildings in square kilometer
        gamma (float): Rayleigh parameter, 
        
    Returns:
        P_{LoS}: line-of-sight probability
    """
    # implement Q-function as erf
    from scipy.special import erf, erfc
    # Q = 0.5 - 0.5 * erf((h_rx/gamma)/np.sqrt(2))
    Q = 0.5 * erf((h_rx/gamma)/np.sqrt(2))
    if d2d == 0:
        return 1.0
    theta = np.arctan((h_tx - h_rx)/d2d)
    return np.exp(-(4 * gamma / np.tan(theta))  * np.sqrt(2 * alpha * beta /np.pi) *Q)
    
def P_LoS_3GPP(d3d):
    """ implementation of P_LoS of 3GPP TR 38.900 and 3GPP TR 38.900
        [1]  3GPP TR 38.900 - V14.2.0, “Study on channel model for frequency
            spectrum above 6 GHz,” Tech. Rep., 2017.
            
        [2] 3GPP TR 38.901 V16.1.0, “Study on channel model for frequencies
        from 0.5 to 100 GHz,” Tech. Rep., 2020.
        
    Parameters: 
        d3d (float): ray distance 
    return:
        P_{LoS}: line-of-sight probability
    """
    return min(18/d3d, 1) * (1 - np.exp(-d3d/36)) + np.exp(-d3d/36)

def P_LoS_Saboor2024(h_tx, theta, alpha, beta, gamma):
    """ implementation of Saboor2024
    A. Saboor, E. Vinogradov, Z. Cui, A. Al-Hourani, and S. Pollin, “A
    Geometry-Based Modelling Approach for the Line-of-Sight Probability
    in UAV Communications,” IEEE Open Journal of the Communications
    Society, vol. 5, pp. 364–378, 2024.

    parameters:
        h_tx (float): transmitter height
        theta (float): elevation angle
        alpha (float): ratio of built-up area in urban
        beta (int): number of buildings in square kilometer
        gamma (float): Rayleigh parameter, 
        
    Returns:
        P_{LoS}: line-of-sight probability
    """
    from scipy.special import erf
    
    W = 1000 * np.sqrt(alpha/beta)
    S = 1000 / np.sqrt(alpha) - W
    theta = np.deg2rad(theta)
    n = np.floor(h_tx/(np.tan(theta) * (S+W)))
    h_Bmax1 = S * np.tan(theta)
    k1i = (np.arange(n) -1) * (S+W)
    k2i = k1i + S
    h_Bmini = k1i * np.tan(theta)
    h_Bmaxi = k2i * np.tan(theta)
    G1i = (np.arange(n) - 1)/ np.sqrt(beta)
    G2i = (np.arange(n) - np.sqrt(alpha))/ np.sqrt(beta)
    t1 = (gamma * np.sqrt(np.pi/2))/ h_Bmax1
    t2 = erf(h_Bmaxi/(np.sqrt(2)*gamma))
    t3 = erf(h_Bmini/(np.sqrt(2)*gamma))
    return np.prod( 1 - t1 * (t2 - t3))
    
def P_LoS_Saboor(d2d, h_tx, h_rx, alpha, beta, gamma, theta=None):
    """ implementation of Saboor2024
    A. Saboor, E. Vinogradov, Z. Cui, A. Al-Hourani, and S. Pollin, “A
    Geometry-Based Modelling Approach for the Line-of-Sight Probability
    in UAV Communications,” IEEE Open Journal of the Communications
    Society, vol. 5, pp. 364–378, 2024.

    parameters:
        h_tx (float): transmitter height
        theta (float): elevation angle
        alpha (float): ratio of built-up area in urban
        beta (int): number of buildings in square kilometer
        gamma (float): Rayleigh parameter, 
        
    Returns:
        P_{LoS}: line-of-sight probability
    """
    W = 1000 * np.sqrt(alpha/beta)
    S = 1000 / np.sqrt(alpha) - W
    
    if d2d == 0:
        d2d += 0.001
    if theta:
        theta = np.deg2rad(theta)
    else:
        theta = np.arctan((h_tx - h_rx)/d2d)
    n = np.floor(h_tx/(np.tan(theta) * (S+W)))
    if n == 0:
        return 1
    # elif n <4:
    #     Sr = [S + 2 * S * np.tan(phi),
    #         S + 2 * S * ( 1/ np.tan(phi)),
    #         S + 2 * S * ( 1/ np.tan(phi))]
    #     Wr = [W/np.cos(phi),
    #           W/np.cos(phi),
    #           W/np.cos(phi)]
        
    # else:
    #     Sr = S
    #     Wr = W
    h_Bmax1 = S * np.tan(theta)
    k1i = (np.arange(n) -1) * (S+W)
    k2i = k1i + S
    # 
    
    h_Bmini = k1i * np.tan(theta)
    h_Bmaxi = k2i * np.tan(theta)
    G1i = (np.arange(n) - 1)/ np.sqrt(beta)
    G2i = (np.arange(n) - np.sqrt(alpha))/ np.sqrt(beta)
    t1 = (gamma * np.sqrt(np.pi/2))/ h_Bmax1
    t2 = erf(h_Bmaxi/(np.sqrt(2)*gamma))
    t3 = erf(h_Bmini/(np.sqrt(2)*gamma))
    # Pn = np.prod( 1 - t1 * (t2 - t3))
    # t1r = (gamma * np.sqrt(np.pi/2))/ h_Bmax1
    # t2r = erf(h_Bmaxi/(np.sqrt(2)*gamma))
    # t3r = erf(h_Bmini/(np.sqrt(2)*gamma))
    # Pr = 
    return np.prod( 1 - t1 * (t2 - t3))


# Line of sight probability for simulation scenarios and distances
# 3GPP TR 38.901 V16.0.0 (2020-01) Table 7.2-1: Evaluation parameters for RMa
def probability_of_los(scenario, d2d, d3d, indoor):
    """
    """
    if scenario == 'RMa':
        return probability_of_los_rma(d2d)
    
def probability_of_los_rma(d2d):
    """
    """
    if d2d <= 10:
        return 1
    else:
        return np.exp(-((d2d - 10)/1000))
    
def c_apostrophe(h_ut):
    """
    """
    if h_ut <= 13:
        return 0
    elif 13 < h_ut and h_ut <= 23:
        return ((h_ut - 13)/10)**1.5
    
def probability_of_los_uma(d2d, h_ut):
    """
    """
    if d2d <= 18:
        return 1
    else:
        return ((18/d2d) + np.exp(-(d2d/63)) * (1 - (18/d2d)) 
        * (1 + c_apostrophe(h_ut) *(5/4) * ((d2d/100)**3)*np.exp(-(d2d/150)))
        )
        
def probability_of_los_umi(d2d):
    """
    """
    if d2d <= 18:
        return 1
    else:
        return ((18/d2d) + np.exp(-(d2d/36)) * (1 - (18/d2d)))
    
def probability_of_los_inh_mixed(d2d):
    """
    """
    if d2d <= 1.2:
        return 1
    elif 1.2 < d2d and d2d < 6:
        return np.exp(-(d2d - 1.2)/4.7) 
    else:
        return np.exp(-(d2d - 6.5)/32.6) * 0.32
    
def probability_of_los_inh_open(d2d):
    """
    """
    if d2d <= 5:
        return 1
    elif 5 < d2d and d2d <= 49:
        return np.exp(-(d2d - 5)/70.8) 
    else:
        return np.exp(-(d2d - 49)/32.6) * 0.54
    
def probability_of_los_inf(scenario, d2d, d_clutter, hbs, hut, r, hc):
    """ LOS probability for indoor hotspot scenarios
        for parameters see 3GPP TR 38.901 V16.0.0 (2020-01) Table 7.2-4: Evaluation parameters for InF
    """
    k_subsce = calculate_k_subsce(scenario, d_clutter, hbs, hut, r, hc)
    return np.exp(-(d2d / k_subsce))
    
def calculate_k_subsce(scenario, d_clutter, hbs, hut, r, hc):
    """ Calculate k_subsce for indoor hotspot scenarios
        for parameters see 3GPP TR 38.901 V16.0.0 (2020-01) Table 7.2-4: Evaluation parameters for InF
    """
    if scenario == 'InF-SL' or scenario == 'InF-DL':
        return -(d_clutter / np.ln(1 - r))
    if scenario == 'InF-SH' or scenario == 'InF-DH':
        return -(d_clutter / np.ln(1 - r)) * ((hbs - hut) / (hc - hbs))


def get_building_width(alpha, beta):
    """
    returns width of building and width of street
    
    Parameters:
        alpha: ratio of built-up area
        beta: number of buildings in square kilometers
    return:
        w_b: width of building
    """
    wb = 1000 * np.sqrt(alpha /beta)
    return wb

def get_street_width(alpha, beta):
    """
    returns width of building and width of street
    
    Parameters:
        alpha: ratio of built-up area
        beta: number of buildings in square kilometers
    return:
        w_s: width of street
    """
    wb = get_building_width(alpha, beta)
    ws = 1000 / np.sqrt(beta) - wb
    return ws

def Rayleigh_mean(sigma):
    """
    The mean of a Rayleigh random variable is :
    {\displaystyle \mu (X)=\sigma {\sqrt {\frac {\pi }{2}}}\ \approx 1.253\ \sigma .}
    """
    return sigma * np.sqrt(np.pi / 2)

def Rayleigh_std(sigma):
    """
    The mean of a Rayleigh random variable is :
    {\displaystyle \mu (X)=\sigma {\sqrt {\frac {\pi }{2}}}\ \approx 1.253\ \sigma .}
    """
    return sigma * np.sqrt(2 - (np.pi / 2))


def get_min_theta(alpha, beta, gamma):
    """
    { \theta_{min} = arctan (\mu_{Rayleigh} / W_s)
    where $\mu_{Rayleigh}$: mean of Rayleigh distribution with parameter gamma
    $W_s$: the street width
    """
    return np.arctan(Rayleigh_mean(gamma) / get_street_width(alpha, beta))
    
def get_max_theta(alpha, beta, gamma):
    """
    { \theta_{min} = arctan (\mu_{Rayleigh} / W_s)
    where $\mu_{Rayleigh}$: mean of Rayleigh distribution with parameter gamma
    $W_s$: the street width
    """
    return np.arctan(Rayleigh_mean(gamma) / get_street_width(alpha, beta))

def get_num_buildings(alpha, beta, d2D = 1000):
    """ number of buildings
    d2d (float): horizontal distance 
    """
    d2D = d2D/1000 # distance in km
    return np.floor((d2D * np.sqrt(alpha * beta)))

def filter_data(data, by_columns, wanted_columns, k_v=None):
    where = np.where(np.ones(len(data))==1, 1,0)
    if k_v:
        for k, v in k_v.items():
            where = np.where(data[k] == v , 1,0) & where
    #
    filtered = data.loc[where==1].groupby(by_columns)[wanted_columns].mean()
    filtered = filtered.reset_index()
    return filtered

# new models
def P_LoS_Bash(d2d, h_tx, h_rx, alpha, beta, gamma): 
    """ proposed 
    
    parameters:
        d2d (float): horizontal distance
        h_tx (float): transmitter height
        h_rx (float): receiver height
        alpha (float): ratio of built-up area in urban
        beta (int): number of buildings in square kilometer
        gamma (float): Rayleigh parameter, 
        
    Returns:
        P_{LoS}: line-of-sight probability
    """
    mu0 = gamma * np.sqrt(np.pi/2)
    var = gamma * np.sqrt(2 - np.pi/2)
    mu = log_normal_mu(mu0, var)
    std = log_normal_std(mu0, var)
    
    h = abs(h_tx -h_rx)
    # 
    return np.exp(- 2* beta * (d2d - np.pi/2 ) * (1 - logNormal_CDF(h, mu, std))) # 86, 86, 87 %

def P_LoS_Bash_multivariate(d2d, h_tx, h_rx, alpha, beta, gamma):
    from scipy.stats import rayleigh
    w = 1000 *np.sqrt(alpha/beta)
    s = 1000 / np.sqrt(beta) - w
    # a = 0.1* s/(2*(w+s))
    if d2d == 0:
        d2d += 0.01
    # theta = np.arctan(h_tx/d2d)
    n = np.floor(d2d/1000 * np.sqrt(alpha * beta))
    if n == 0:
        return 1
    # di =  np.arange(n) * np.tan(theta)
    h = (h_tx-h_rx)
    
    #
    # rayleigh = (d2d/(2*alpha*beta)) * np.exp(-d2d**2/(2*alpha*beta**2))
    # sin = (1+a*np.sin(2 * np.pi * (h_tx-h_rx)/ np.sqrt(beta)))
    # 
    # exp = np.exp(-d2d / (np.sqrt(2)*(alpha*beta)**2))
    exp = np.exp(-d2d *(2)/((np.sqrt(np.pi))*(alpha*beta)**2))
    #
    # mu = np.pi/2 * gamma
    std = (2 -np.pi/2) * gamma
    # err = (erf((h_tx-h_rx) /( 2* gamma**2)))
    err =  (erf(( h )  /(s  * std))) 
    # 
    return   exp * err


######################################
# Line of sight calculator
# pandas apply function
def calc_P_LoS(transData, alpha, beta, gamma, env, freq):
    """ Pandas apply function for calculating the LoS probability
        for different types of LoS probability models.
    """
    # transformProcessor
    TrP = TransformProcessor()
    # 
    # transData['Environment'] = env
    # transData['Frequency'] = freq
    transData['d2D'] = transData.apply(
        lambda row: TrP.getd2D(row['Transforms']), axis=1)
    transData['h_tx'] = transData.apply(
        lambda row: TrP.getTxZ(row['Transforms']), axis=1)
    transData['h_rx'] = transData.apply(
        lambda row: TrP.getRxZ(row['Transforms']), axis=1)
    # ITU-R P.1410-5
    transData['ITU_R'] = transData.apply(
        lambda row: P_LoS_Rec_ITU_R_P_1410_5(
            row['d2D'], row['h_tx'], row['h_rx'], 
            alpha, beta, gamma), axis=1)
    # Kang2019
    transData['Kang2019Secrecy'] = transData.apply(
        lambda row: P_LoS_Kang2019Secrecy(
            row['d2D'], row['h_tx'], row['h_rx'], 
            alpha, beta, gamma), axis=1)
    # Mohammed2023
    transData['Mohammed2023Closed'] = transData.apply(
        lambda row: P_LoS_Mohammed2023Closed(
            row['d2D'], row['h_tx'], row['h_rx'], 
            alpha, beta, gamma), axis=1)
    # Pang2023
    wavelength = 3e8 / (freq * 1e9) # in meter
    transData['Pang2023Geometry'] = transData.apply(
        lambda row: P_LoS_Pang2023Geometry(
            row['d2D'], row['h_tx'], row['h_rx'], 
            alpha, beta, gamma, wavelength), axis=1)
    
    return transData


TruePositive = lambda data, model: (np.where(data['RayLoS'] == 1, 1, 0) & np.where(data[model] >=0.5, 1, 0)).sum()
FalsePositive = lambda data, model: (np.where(data['RayLoS'] == 0, 1, 0) & np.where(data[model] >=0.5, 1, 0)).sum()
FalseNegative = lambda data, model: (np.where(data['RayLoS'] == 1, 1, 0) & np.where(data[model] <0.5, 1, 0)).sum()
TrueNegative = lambda data, model: (np.where(data['RayLoS'] == 0, 1, 0) & np.where(data[model] <0.5, 1, 0)).sum()


def accuracy(data, model, latex='No'):
    TP = TruePositive(data, model)
    FP = FalsePositive(data, model)
    FN = FalseNegative(data, model)
    TN = TrueNegative(data, model)
    # false positive rate
    fpr = round(FP/(FP+TN),2)
    # false negative rate
    fnr = round(FN/(FN+TP),2)
    # sensitivity
    sensitivity = round(TP/(TP+FN),2)
    # specificity
    specificity = round(TN/(TN+FP),2)
    # Youden index
    Y = round(sensitivity + specificity -1,2)
    # accuracy
    accuracy = round((TP+TN)/(TP + TN+FP+FN),2)
    # return model, TP, FP, FN, TN, fpr, fnr, sensitivity, specificity, Y, accuracy
    if latex.lower() == 'no':
        return {'model': model, 'TP':TP, 'FP': FP, 'FN':FN, 'TN':TN, 'fpr':fpr, 'fnr':fnr, 'Sensitivity':sensitivity, 'Specificity':specificity, 'Y':Y, 'Accuracy':accuracy}
    else:
        return f'{model} & {TP} & {FP} & {FN} & {TN} & {fpr} & {fnr} & {sensitivity} & {specificity} & {Y} & {accuracy} \\\\ \hline'

