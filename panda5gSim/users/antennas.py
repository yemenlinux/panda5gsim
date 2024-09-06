import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import pyvista as pv
import os
import matplotlib.pyplot as plt 

working_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(working_dir, os.pardir,os.pardir ))
results_dir = os.path.abspath(os.path.join(parent_dir, 'output', 'results' ))


def get_antenna_gain(phi, theta, 
                     phi_3db = 65, 
                     theta_3db = 65, 
                     sla = 30, 
                     att = 30):
    a_v = - np.min([12* ((theta-90)/theta_3db)**2, sla])
    a_h = - np.min([12 * ((phi )/phi_3db)**2, att])
    return  -np.minimum(- (a_v + a_h), att)


def get_3d_power_radiation_pattern(
                                transformed_phi = None,
                                transformed_theta = None,
                                theta_3db = 65, 
                                phi_3db = 65, 
                                SLA_v = 30, 
                                A_max = 30):
    """ Generate 3D power radiation pattern for 3GPP antenna for 5G
    """
    A_double_abostroph_theta = - np.minimum(12 * 
            (transformed_theta / theta_3db)**2, SLA_v)
    A_double_abostroph_phi = -np.minimum(12 * 
            (transformed_phi / phi_3db)**2, A_max)
    A_double_abostroph = - np.minimum(
        -(A_double_abostroph_theta + A_double_abostroph_phi), A_max)
    return A_double_abostroph
    


def generate_3d_power_radiation_pattern(requested_phi = None,
                                        requested_theta = None,
                                        theta_3db = 15, 
                                        phi_3db = 65, 
                                        antenna_down_tilt = 0, 
                                        SLA_v = 10, 
                                        A_max = 30, 
                                        fron_back_ratio = -40):
    """ Generate 3D power radiation pattern for 3GPP antenna for 5G
    """
    theta = np.linspace(0, 180, 181)
    #theta = np.where(theta<181, theta, theta*0)
    phi = np.linspace(-180, 180, 361)
    theta_mesh, phi_mesh = np.meshgrid(phi, theta)
    # print(theta_mesh.shape, phi_mesh.shape)
    # calculate the the antenna pattern
    # source: 3GPP TR 38.901 version 14.0.0 (Release 14) 
    # Vertical cut of the radiation power pattern (dB)
    A_double_abostroph_theta = - np.minimum(12 * ((theta_mesh - 90 - antenna_down_tilt)/theta_3db)**2, SLA_v) 
    A_double_abostroph_phi = -np.minimum(12 * (phi_mesh/phi_3db)**2, A_max) 
    #
    A_double_abostroph = - np.minimum(-(A_double_abostroph_theta + A_double_abostroph_phi), A_max)
    if requested_theta is not None and requested_phi is not None:
        return A_double_abostroph[int(requested_theta)][int(requested_phi)]
    else:
        return A_double_abostroph
    
def theta_bar(a, b, g, ph, th):
     # 3GPP TR 38.900 equation (7.1-18) 
    # theta_bar = arccos(cos(phi) sin(theta) sin(beta) + cos(theta) cos(beta)
    a = np.deg2rad(a)
    b = np.deg2rad(b)
    g = np.deg2rad(g)
    ph = np.deg2rad(ph)
    th = np.deg2rad(th)
    return np.arccos(np.cos(ph) * np.sin(th) * np.sin(b) 
                          + np.cos(th) * np.cos(b))

def phi_bar(a, b, g, ph, th):
     # 3GPP TR 38.900 equation (7.1-18) 
    # theta_bar = arccos(cos(phi) sin(theta) sin(beta) + cos(theta) cos(beta)
    a = np.deg2rad(a)
    b = np.deg2rad(b)
    g = np.deg2rad(g)
    ph = np.deg2rad(ph)
    th = np.deg2rad(th)
    #phi_bar = arg(cos(phi) sin(theta) cos(beta) - cos(theta) sine(beta) + j sin(phi) sine(theta))
    ex = (np.cos(ph) * np.sin(th) * np.cos(b) - np.cos(th) * np.sin(b)
                         + 1j *np.sin(ph) * np.sin(th))
    return np.arctan2(ex.imag, ex.real)

def psi_(a, b, g, ph, th):
    # eq 7.1-15 of 3GPP TR 38.900
    a = np.deg2rad(a)
    b = np.deg2rad(b)
    g = np.deg2rad(g)
    ph = np.deg2rad(ph)
    th = np.deg2rad(th)
    ex = (np.sin(th) * np.cos(b) - np.cos(ph) * np.cos(th) * np.sin(b) + 1j * np.sin(ph) * np.sin(b)
        )
    return np.arctan2(ex.imag, ex.real)
    
def cos_psi(a, b, g, ph, th):
    # eq 7.1-16 of 3GPP TR 38.900
    a = np.deg2rad(a)
    b = np.deg2rad(b)
    g = np.deg2rad(g)
    ph = np.deg2rad(ph)
    th = np.deg2rad(th)
    return ( (np.cos(b) * np.cos(g) * np.sin(th) - (np.sin(b) * np.cos(g) * np.cos(ph-a) -np.sin(g) * np.sin(ph-a))* np.cos(th))
    / np.sqrt(1-(np.cos(b) * np.cos(g) * np.cos(th) + ((np.sin(b) * np.cos(g) * np.cos(ph-a) -np.sin(g) * np.sin(ph-a)) * np.sin(th))**2))
    )
    
def sin_psi(a, b, g, ph, th):
    # eq 7.1-17 of 3GPP TR 38.900
    a = np.deg2rad(a)
    b = np.deg2rad(b)
    g = np.deg2rad(g)
    ph = np.deg2rad(ph)
    th = np.deg2rad(th)
    return ( (np.sin(b) * np.cos(g) * np.sin(ph-a) + np.sin(g) * np.cos(ph-a))
    / np.sqrt(1-(np.cos(b) * np.cos(g) * np.cos(th) + ((np.sin(b) * np.cos(g) * np.cos(ph-a) -np.sin(g) * np.sin(ph-a)) * np.sin(th))**2))
    )
def f_th_bar(ph, th, ps, att):
    # 3GPP TR 38.900 eq: 7.1-21
    # ph = np.deg2rad(ph)
    # th = np.deg2rad(th)
    ps = np.deg2rad(ps)
    return get_antenna_gain(0, th, att = att) * np.cos(ps) - get_antenna_gain(ph, 90, att = att) * np.sin(ps)
    
def f_ph_bar(ph, th, ps, att):
    # 3GPP TR 38.900 eq: 7.1-22
    # ph = np.deg2rad(ph)
    # th = np.deg2rad(th)
    ps = np.deg2rad(ps)
    return get_antenna_gain(0, th, att = att) * np.sin(ps) - get_antenna_gain(ph, 90, att = att) * np.cos(ps)
    
def F_ph_th(ph, th, psi, att = 30):
    #
    # ph = np.deg2rad(ph)
    # th = np.deg2rad(th)
    # psi = np.deg2rad(psi)
    ex = f_ph_bar(ph, th, psi, att) + 1j * f_th_bar(ph, th, psi, att)
    return np.arctan2(ex.imag, ex.real)
    # return - np.minimum(-(f_th_bar(ph, th, psi)+ f_ph_bar(ph, th, psi)), att)
    
def antenna_gain_dBi(phi, theta, 
                     phi_3db = 65, 
                     theta_3db = 65, 
                     sla = 30, 
                     att = 30):
    # 3GPP TR 38.900 Table 7.3-1
    return round(get_antenna_gain(phi, theta, phi_3db, theta_3db, sla, att)
                 + 10 * np.log10(1000), 2) # convert to dBi
    
#pv.set_jupyter_backend('trame')

def generate_3gpp_antenna_gain_db(requested_theta = None, 
                                  requested_phi = None, 
                                  theta_3dB = 65, 
                                  phi_3dB = 65, 
                                  SLA_v = -30, 
                                  G_max = 30):
    """ Generate 3D power radiation pattern for 3GPP antenna for 5G
        source: 3GPP TR 38.901 version 14.0.0 (Release 14) 
        theta_3dB :65,# half power beamwidth v
        phi_3dB :65, # half power beamwidth h
        SLA_v : -30, # side lobe level
        G_max: 18,    # maximum antenna gain
        
    """
    theta = np.linspace(0, 180, 181)
    #theta = np.where(theta<181, theta, theta*0)
    phi = np.linspace(-180, 180, 361)
    
    theta_mesh, phi_mesh = np.meshgrid(phi, theta)
    #print(theta_mesh.shape, phi_mesh.shape)
    # calculate the the antenna pattern
    # source: 3GPP TR 38.901 version 14.0.0 (Release 14) 
    A_double_abostroph_theta = - np.minimum(12 * ((theta_mesh - 90) /theta_3dB)**2, SLA_v) 
    A_double_abostroph_phi = -np.minimum(12 * (phi_mesh/phi_3dB)**2, G_max) 
    # 3D radiation power pattern (dB)
    gain_db = - np.minimum(-(A_double_abostroph_theta + A_double_abostroph_phi), G_max)
    # convert from dB to linear scale
    #gain = np.sqrt(10**(gain_db/10.))
    # squate_root_gain = np.sqrt(10**(gain_db/10.))
    #
    if requested_theta is not None and requested_phi is not None:
        # if requested_theta == -180: 
        #     requested_theta = 180
        if requested_theta < 0 : 
            requested_theta = abs(requested_theta)
        if requested_theta > 180: 
            requested_theta %= 180
        if requested_phi < -180 or requested_phi > 180:
            requested_phi %= 180
        th_idx = np.where(theta == requested_theta)[0][0]
        ph_idx = np.where(phi == requested_phi)[0][0]
        return gain_db[th_idx][ph_idx]
    else:
        return gain_db

def generate_3gpp_antenna_poewr_radiation_pattern(requested_theta = None, requested_phi = None, theta_3dB = 65, phi_3dB = 65, SLA_v = -30, G_max = 30):
    """ Generate 3D power radiation pattern for 3GPP antenna for 5G
        source: 3GPP TR 38.901 version 14.0.0 (Release 14) 
        theta_3dB :65,# half power beamwidth v
        phi_3dB :65, # half power beamwidth h
        SLA_v : -30, # side lobe level
        G_max: 18,    # maximum antenna gain
        
    """
    theta = np.linspace(0, 180, 181)
    #theta = np.where(theta<181, theta, theta*0)
    phi = np.linspace(-180, 180, 361)
    
    theta_mesh, phi_mesh = np.meshgrid(phi, theta)
    #print(theta_mesh.shape, phi_mesh.shape)
    # calculate the the antenna pattern
    # source: 3GPP TR 38.901 version 14.0.0 (Release 14) 
    # Vertical cut of the radiation power pattern (dB)
    
    A_double_abostroph_theta = - np.minimum(12 * ((theta_mesh - 90) /theta_3dB)**2, SLA_v) 
    A_double_abostroph_phi = -np.minimum(12 * (phi_mesh/phi_3dB)**2, G_max) 
    # 3D radiation power pattern (dB)
    gain_db = - np.minimum(-(A_double_abostroph_theta + A_double_abostroph_phi), G_max)
    # convert from dB to linear scale
    #gain = np.sqrt(10**(gain_db/10.))
    squate_root_gain = np.sqrt(10**(gain_db/10.))
    #
    if requested_theta is not None and requested_phi is not None:
        return squate_root_gain[requested_theta][requested_phi]
    else:
        return squate_root_gain

def radF(RadPatternFunction, polarization='horizontal'):
    """ radiation pattern
    pol    : h | v | c polarization (horizontal, vertical, circular)
    """
    assert polarization in ['horizontal','vertical','circular']
    squate_root_gain = RadPatternFunction()
    if polarization == 'horizontal':
        Fp = squate_root_gain
        Ft = np.zeros(Fp.shape)
    elif polarization == 'vertical':
        Ft = squate_root_gain
        Fp =np.zeros(Ft.shape)
    elif polarization == 'circular':
        Fp = (1./np.sqrt(2))*squate_root_gain
        Ft = (1j/np.sqrt(2))*squate_root_gain
    else:
        raise ValueError("Invalid polarization. Must be 'Vertical', 'Horizontal', or 'Circular'.")
    return Ft,Fp
#
def get_linear_gain(RadPatternFunction, polarization='h'):
    """ get linear gain
    """
    # calculate linear gain
    Ftheta, Fphi = radF(RadPatternFunction, polarization = polarization)
    linear_gain = np.real( Fphi * np.conj(Fphi)
                   +  Ftheta * np.conj(Ftheta))
    return linear_gain

def calcualte_directivity(Efficiency, RadPatternFunction, *args):
    """
    Based on calc_directivity.m from ArrayCalc.
    Calculates peak directivity in dBi value using numerical integration.
    If the array efficiency is set to below 100% then the returned value is referred to as Gain (dB).
    Usage: ThetaMax, PhiMax = CalcDirectivity(RadPatternFunction, Efficiency)
    RadPatternFunction - antennas radiation pattern function. F(Theta, Phi)
    Efficiency - Efficiency of antenna in %. Default 100%.
    Returned values:
    ThetaMax - Theta value for direction of maximum directivity (Deg)
    PhiMax - Phi value for direction of maximum directivity (Deg)
    Integration is of the form :
    %
    %       360   180
    %     Int{  Int{  (E(theta,phi)*conj(E(theta,phi))*sin(theta) d(theta) d(phi)
    %        0     0
    %
    %         z
    %         |-theta   (theta 0-180 measured from z-axis)
    %         |/
    %         |_____ y
    %        /\
    %       /-phi       (phi 0-360 measured from x-axis)
    %      x
    %
    """
    # print("Calculating Directivity for " + RadPatternFunction.__name__)
    deltheta = 1                                                                # Step value of theta (Deg)
    delphi = 1                                                                  # Step value for phi (Deg)
    #
    dth = np.radians(deltheta)
    dph = np.radians(delphi)
    #
    Psum = 0
    Pmax = 0
    Thmax = 0
    Phmax = 0
    #
    fields = RadPatternFunction()
    for theta in range(0, 181, deltheta):
        for phi in range(0, 361, delphi):
            eField = fields[theta][phi]                                       # Total E-field at point
            Pthph = eField * np.conjugate(eField)                                                                             # Convert to power

            if Pthph > Pmax:
                Pmax = Pthph                                                                                # Store peak value
                Thmax = theta #+90                                                                              # Store theta value for the maximum
                Phmax = phi                                                                                 # Store phi value for the maximum

            # print(str(theta) + "," + str(phi) + ": " + str(Pthph))
            Psum = Psum + Pthph * np.sin(np.radians(theta)) * dth * dph                                           # Summation

    Pmax = Pmax * (Efficiency / 100)                                                                        # Apply antenna efficiency
    #
    directivity_lin = Pmax / (Psum / (4 * np.pi))                                                              # Directivity (linear ratio)
    directivity_dBi = 10 * np.log10(directivity_lin)                                                           # Directivity (dB wrt isotropic)

    if Efficiency < 100:                                                                                    # Gain case
        dBdiff = 10 * np.log10(abs(100 / Efficiency))                                                          # Difference between gain and directivity
        # print(f'Directivity = {directivity_dBi + dBdiff} dBi')                                     # Display what directivity would be for ref.
        # print(f'Efficiency = {Efficiency} %')
        # print(f'Gain = {directivity_dBi} dB')
    # else:                                                                                                   # Directivity case
    #     print(f'Directivity = {directivity_dBi} dBi')

    # print(f'At Theta = {Thmax}, Phi = {Phmax}, Directivity = {directivity_dBi} dBi')
    return Thmax, Phmax, directivity_dBi

def get_gain_dB(RadPatternFunction, polarization='horizontal', efficiency = 100):
    """ get antenna gain in dB
    """
    th, ph, directivity_gain_dBi = calcualte_directivity(efficiency, RadPatternFunction)
    linear_gain = get_linear_gain(RadPatternFunction, polarization = polarization)
    # square_root_gain = np.sqrt(linear_gain)
    #gain = directivity_gain_dBi + 10 * np.log10(linear_gain +1e-12)
    gain = 10 * np.log10(linear_gain +1e-12)
    return gain

def get_gain_dB_by_angles(theta, phi, RadPatternFunction, polarization='horizontal', efficiency = 100):
    """ get antenna gain in dB
    """
    th, ph, directivity_gain_dBi = calcualte_directivity(efficiency, RadPatternFunction)
    linear_gain = get_linear_gain(RadPatternFunction, polarization = polarization)
    # square_root_gain = np.sqrt(linear_gain)
    gain = 10 * np.log10(linear_gain +1e-12)
    return gain[theta][phi]

def get_antenna_gain_db(requested_theta, requested_phi, RadPatternFunction):
    """ get antenna gain from a gain array or function
        inputs:
        requested_theta: theta angle in degrees
        requested_phi: phi angle in degrees
        RadPatternFunction: a function that returns a gain array
        
        example:
        gain_db = get_antenna_gain_db(30, 45, RadPatternFunction)
        th = 30
        ph = 45
        gain_db = get_antenna_gain_db(th, ph, generate_3gpp_antenna_gain_db())
        gain_array = generate_3gpp_antenna_gain_db()
        gain_db = get_antenna_gain_db(th, ph, gain_array)
    """
    theta = np.linspace(0, 180, 181)
    #theta = np.where(theta<181, theta, theta*0)
    phi = np.linspace(-180, 180, 361)
    
    # 3D radiation power pattern (dB)
    # check if RadPatternFunction is a function or an array
    if callable(RadPatternFunction):
        gain_db = RadPatternFunction()
    else:
        gain_db = RadPatternFunction
    #
    if requested_theta < 0 : 
        requested_theta = abs(requested_theta)
    if requested_theta > 180: 
        requested_theta %= 180
    if requested_phi < -180 or requested_phi > 180:
        requested_phi %= 180
    th_idx = np.where(theta == requested_theta)[0][0]
    ph_idx = np.where(phi == requested_phi)[0][0]
    return gain_db[th_idx, ph_idx]

def generate_omni_antenna_gain_db(theta = None, phi=None):
    """ generate omni antenna gain in dB
    """
    theta = np.linspace(0, 180, 181)
    #theta = np.where(theta<181, theta, theta*0)
    phi = np.linspace(-180, 180, 361)
    # 3D radiation power pattern (dB)
    gain_db = np.zeros((len(theta), len(phi)))
    return gain_db


def get_antenna_pattern(RadPatternFunction = '3gpp'):
    """_summary_

    Args:
        RadPatternFunction (str, optional): _description_. Defaults to '3gpp'.
    """
    if RadPatternFunction == '3gpp':
        return generate_3gpp_antenna_gain_db()
    if RadPatternFunction == 'omni':
        return generate_omni_antenna_gain_db()

def polar2cart(v):
    theta = np.linspace(0, 181, 181) #-90
    phi = np.linspace(-180, 180, 361)
    
    #
    vt = np.ones(v.shape[0])
    vp = np.ones(v.shape[1])
    Th = np.radians(np.outer(theta, vp))
    Ph = np.radians(np.outer(vt, phi))
    x = abs(v) * np.cos(Ph) * np.sin(Th)
    y = abs(v) * np.sin(Ph) * np.sin(Th)
    z = abs(v) * np.cos(Th)
    return x, y, z

class Antenna3GPP:
    max_attenuation = 30
    side_lobe_attenuation = 30
    phi_3dB = 65
    theta_3dB = 65
    max_gain = 30 
    polarization = 'circular'
    def __init__(self, **kwargs):
        """ generate 3GPP antenna
        Parameters:
        max_attenuation (float): maximum attenuation dB
        side_lobe_attenuation 
        """
        #
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.generate_pattern()
        
    def generate_pattern(self):
        self.theta_d = np.linspace(0, 180, 361)
        self.phi_d =  np.linspace(-180, 180, 361)
        self.theta_r = np.deg2rad(self.theta_d)
        self.phi_r = np.deg2rad(self.phi_d)
        self.th_2d, self.phi_2d = np.meshgrid(self.theta_d, self.phi_d)
        sla = np.ones(self.th_2d.shape) * self.side_lobe_attenuation
        att = np.ones(self.th_2d.shape) *self.side_lobe_attenuation
        self.A_v = - np.minimum(12* ((self.th_2d-90)/self.theta_3dB)**2, att) 
        #
        self.A_h = - np.minimum(12 * ((self.phi_2d )/self.phi_3dB)**2, att) 
        self.pattern = -np.minimum( -(self.A_v + self.A_h), sla)

        self.squate_root_gain = np.sqrt(10**(self.pattern/10))
        if self.polarization == 'horizontal':
            self.Fphi = self.squate_root_gain
            self.Ftheta = np.zeros(self.Fphi.shape)
        elif self.polarization == 'vertical':
            self.Ftheta = self.squate_root_gain
            self.Fphi =np.zeros(self.Ftheta.shape)
        elif self.polarization == 'circular':
            polrization_degree = np.deg2rad(45)
            self.Fphi = np.sin(polrization_degree)*self.squate_root_gain
            self.Ftheta = 1j * np.cos(polrization_degree)*self.squate_root_gain
            # self.Fphi = (1./np.sqrt(2))*self.squate_root_gain
            # self.Ftheta = (1j/np.sqrt(2))*self.squate_root_gain
        # calculate linear gain
        self.linear_gain = np.real( self.Fphi * np.conj(self.Fphi)
                       +  self.Ftheta * np.conj(self.Ftheta))
        
        self.dB_gain =  10*np.log10(self.linear_gain) 
        
    def get_dB_gain(self, phi = None, theta = None):
        if phi != None and theta != None:
            phi = int(round(abs(phi)))
            theta = int(round(abs(theta+90)))
            i, j = np.where(self.phi_d.astype('int') == phi)[0][0], np.where(self.theta_d.astype('int') == theta)[0][0]
            gain = self.dB_gain + self.max_gain
            return gain[i, j]
        return self.dB_gain + self.max_gain
    
    def get_linear_gain(self, phi = None, theta = None):
        if phi != None and theta != None:
            phi = int(round(abs(phi)))
            theta = int(round(abs(theta+90)))
            i, j = np.where(self.phi_d.astype('int') == phi)[0][0], np.where(self.theta_d.astype('int') == theta)[0][0]
            gain = self.linear_gain + 10**(self.max_gain/10)
            return gain[i, j]
        return self.linear_gain + 10**(self.max_gain/10)
    
    def polarplot(self):
        gain = self.get_dB_gain()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='polar')
        # 3GPP directional Antenna elevation plane pattern
        ax.plot(self.theta_r, gain[181,:], label='Vertical cut at $\\phi = 0^0$, dB')
        ax.plot(self.phi_r, gain[:,181], label='Horizontal cut at $\\theta = 90^o$, dB')
        # ax.set_theta_zero_location('N')
        # ax.set_theta_direction(1)
        # ax.set_rlabel_position(0)
        # ax.set_yticks(np.arange(0, 1.1, 0.2))
        # ax.set_ylim((0, 1))
        # ax.set_title('Antenna Pattern (Beamwidth = {}°, Polarization = {}, Downtilt = {}°)'.format(beamwidth, polarization, downtilt))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
        # save
        file1 = os.path.abspath(os.path.join(results_dir, 'antennaPattern.jpg' ))
        fig.savefig(file1, dpi=300)
        plt.show()
        
    def polar2cart(self):
        v = self.get_dB_gain()
        
        theta = np.linspace(0, 180, 361) #-90
        phi = np.linspace(-180, 180, 361)

        #
        vt = np.ones(v.shape[0])
        vp = np.ones(v.shape[1])
        Th = np.radians(np.outer(theta, vp))
        Ph = np.radians(np.outer(vt, phi))
        
        x = abs(v) * np.cos(Ph) * np.sin(Th)
        y = abs(v) * np.sin(Ph) * np.sin(Th)
        z = abs(v) * np.cos(Th)
        return x, y, z
    
    def plot3d(self):
        # test the antenna pattern
        X, Y, Z = self.polar2cart()
        # Z[Z<-15] += 8
        #print(np.unravel_index(np.argmax(rp), rp.shape))
        # Create and plot structured grid
        grid = pv.StructuredGrid(X, Y, Z)

        plotter = pv.Plotter(window_size= pv.global_theme.window_size)
        # plotter.add_mesh(grid, scalars=grid.points[:, -1], show_edges=True, scalar_bar_args={'vertical': True})
        plotter.add_mesh(grid, scalars=grid.points[:, -1], scalar_bar_args={'vertical': True})
        plotter.show_grid()

        plotter.show()




# test the antenna pattern
if __name__ == '__main__':
    # example 1
    rp = get_gain_dB(generate_3gpp_antenna_poewr_radiation_pattern, polarization="horizontal", efficiency = 100)
    X, Y, Z = polar2cart(rp)
    #print(np.unravel_index(np.argmax(rp), rp.shape))
    # Create and plot structured grid
    grid = pv.StructuredGrid(X, Y, Z)

    plotter = pv.Plotter(window_size= pv.global_theme.window_size)
    # plotter.add_mesh(grid, scalars=grid.points[:, -1], show_edges=True, scalar_bar_args={'vertical': True})
    plotter.add_mesh(grid, scalars=grid.points[:, -1], scalar_bar_args={'vertical': True})
    plotter.show_grid()

    plotter.show()
    # Add a blocking loop to keep the window open
    # show poltter window
    
    # example 2
    # a = Antenna3GPP()
    # a.linear_gain.shape
    # a.plot3d()
    # a.polarplot()