# This module contains classes for collecting metrics
import numpy as np



from panda5gSim.metrics.distance import *
from panda5gSim.core.scene_graph import (
    getRayLoS, findNPbyTag)
from panda5gSim.core.helpers import pairwise
from panda5gSim.core.transformations import TransformProcessor
from panda5gSim.metrics.pathloss import *
from panda5gSim.metrics.los_probability import *
from panda5gSim.metrics.writer import Writer


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


header = [
    'Iter',
    'Environment',
    'Frequency (GHz)',
    'Bandwidth (MHz)',
    'Generation',
    'Average building height',
    'Average street width',
    'Average roof height',
    'Above roof',
    #
    'Rx ID',
    'Tx ID',
    'Rx position',
    'Tx position',
    'Tx altitude',
    'Rx altitude',
    'Distance',
    'LoS type',
    'LoS value',
    'Path loss (dB)',
    'Received power (dB)',
    'Noise (dB)',
    'Interference (dB)',
    'SINR (dB)',
    'Spectral efficiency (bps/Hz)',
    'Capacity (Mbps)',
    'Capacity (Mbps/km^2)',
]
class MetricsCollector(TransformProcessor):
    # This class updates transforms 
    def __init__(self, txTags, RxTags):
        self.TxTags = txTags
        self.RxTags = RxTags
        # find Tx and Rx nodes
        self.findTags()
        self.initTransformations()
        
    def findTags(self):
        self.TxNodes = []
        self.RxNodes = []
        for tag in self.TxTags:
            for tx in findNPbyTag(tag, 'type'):
                self.TxNodes.append(tx)
        for tag in self.RxTags:
            for rx in findNPbyTag(tag, 'type'):
                self.RxNodes.append(rx)
        # self.TxNodes = np.array(self.TxNodes, dtype=object)
        # self.RxNodes = np.array(self.RxNodes, dtype=object)
        
    def initTransformations(self):
        # Create Transformation matrix
        size = (len(self.RxNodes), len(self.TxNodes))
        # Transform of rx relative to tx
        self.Transforms = np.zeros(size, dtype=object) 
        # For handover and if MIMO is used
        self.connectedToArray = np.zeros(size)
        
        
    def updateDynamics(self):
        # update Transformation matrix
        for i in range(len(self.RxNodes)):
            for j in range(len(self.TxNodes)):
                # a Transform = (rx.getTransform(tx), rx.getTransform(), tx.getTransform)
                self.Transforms[i,j] = (self.RxNodes[i].getTransform(self.TxNodes[j]),
                                        self.RxNodes[i].getTransform(render),
                                        self.TxNodes[j].getTransform(render),
                                        i,j)
        return True
    

# rewrite the CapacityMetrics class to reduce execution time
class RadioLinkBudgetMetricsCollector(MetricsCollector):
    def __init__(self, scenario_name, txTags, RxTags, filename):
        MetricsCollector.__init__(self, txTags, RxTags)
        self.header = header
        self.scenario_name = scenario_name
        self.filename = filename
        self.percentileFileName = f'{self.filename}_percentile'
        # update statics
        self.updateStatic()
        # init the writers
        self.initWriters()
        
        
    def updateStatic(self):
        # update static variables that are fix during simulation
        self.StaticParam = {
            'Environment': self.scenario_name,
            # data['Inter site distance m'] = SimData['ISD'],
            # 'number_of_sectors': SimData['Number of sectors'],
            'Average building height': SimData['Building Defaults']['Gamma'],
            'Average street width': SimData['Building Defaults']['Average street width'],
            'Average roof height': SimData['Building Defaults']['Floor height'],
            'Above roof': SimData['Building Defaults']['Above roof'],
        }
        
    def initWriters(self):
        # init the writers
        # init raw data writers
        self.rawWriter = Writer(self.filename, self.header)
        # init percentile writers
        # self.PercentileHeader = [
        #     'Percentile',
        # ]
        # self.percentileWriter = Writer(
        #             self.percentileFileName, 
        #             self.PercentileHeader)
    
    def getLOSProbability(self, transform):
        # get the line of sight (LOS) probability for the UMa, 
        # 2D distance
        d = self.getd2D(transform)
        # rx_height
        rx_height = self.getRxZ(transform)
        # get the LOS probability
        if self.scenario_name == "RMa":
            if d <= 10:
                return 1
            else:
                return round(np.exp(-(d-10)/1000),2)
        elif self.scenario_name == "UMa":
            if d <= 18:
                return 1
            else:
                if rx_height <= 13:
                    chUT = 0
                else:
                    chUT = ((rx_height - 13)/10)**1.5
                return round(((18/d) + np.exp(-(d/36))*(1-(18/d))) \
                        *(((1-chUT)*(4/5)*(d/100)**3) * np.exp(-d/150)),2)
                
        elif self.scenario_name == "UMi":
            if d <= 18:
                return 1
            else:
                return round((18/d) + np.exp(-(d)/36)*(1-(18/d)),2)
        elif self.scenario_name == "InH":
            if d <= 1.2:
                return 1
            elif d > 1.2 and d <= 6.5:
                return round(np.exp(-((d-1.2)/4.7)),2)
            else:
                return round(np.exp(-((d-6.5)/32.6))* 0.32,2)
        else:
            return 0.1
    
    def getLOSRay(self, transform):
        # get trace ray Line of sight
        rx = self.RxNodes[transform[3]]
        tx = self.TxNodes[transform[4]]
        los = getRayLoS(tx, rx)
        return los
    
    def estimateRxPower(self, Rx_index, transform_vec, freq, los='prob'):
        # ETSI TR 138 901 V14.0.0 (2017-10)
        Rx_subclass = self.RxNodes[Rx_index].getPythonTag('subclass')
        Result = []
        for j, transform in enumerate(transform_vec):
            if los == 'ray':
                LOSvalue = self.getLOSRay(transform)
                LOS = 1 if LOSvalue >= 0.5 else 0
            elif los == 'prob':
                LOSvalue = self.getLOSProbability(transform)
                LOS = 1 if LOSvalue >= 0.5 else 0
            else:
                raise ValueError('los must be either ray or prob')
            PL, _ = calculate_path_loss(
                self.scenario_name,
                freq, 
                self.getd2D(transform), 
                self.getd3D(transform), 
                self.getTxZ(transform),
                self.getRxZ(transform),
                SimData['Building Defaults']['Average building height'],
                LOS,
                False, #TODO: add indoor to the actors
                SimData['Building Defaults']['Average street width'],
                SimData['Seed value'],
                SimData['Iterations']
            )
            # get azimuth and zenith angels as seen from tx
            phi, theta = self.getRelativePhiTheta(transform)
            #
            Tx_subclass = self.TxNodes[j].getPythonTag('subclass')
            # get Tx_power in dB
            Tx_power = Tx_subclass.getTxPowerdB(phi, theta)
            # get the received power in dB after adding rx antenna's gain
            Rx_power = round(Rx_subclass.getRxPowerdB(Tx_power - PL),2)
            
            Result.append([Rx_power, PL, LOSvalue, transform])
        return Result
    
    def estimateNoisedB(self, bandwidth_MHz):
        """
        Estimates the potential noise at the UE receiver.
        
        Terminal noise can be calculated as:

        “K (Boltzmann constant) x T (290K) x bandwidth”.

        The bandwidth depends on bit rate, which defines the number
        of resource blocks. We assume 50 resource blocks, equal 9 MHz,
        transmission for 1 Mbps downlink.

        Required SNR (dB)
        Detection bandwidth (BW) (Hz)
        k = Boltzmann constant
        T = Temperature (Kelvins) (290 Kelvin = ~16 degrees celcius)
        NF = Receiver noise figure (dB)

        NoiseFloor (dBm) = 10log10(k * T * 1000) + NF + 10log10BW

        NoiseFloor (dBm) = (
            10log10(1.38 x 10e-23 * 290 * 1x10e3) + 1.5 + 10log10(10 x 10e6)
        )

        Parameters
        ----------
        bandwidth : int
            The bandwidth of the carrier frequency (MHz).

        Returns
        -------
        noise : float (dBm)
            Received noise at the UE receiver in decibels

        """
        k = 1.38e-23
        T = 290
        noise_figure = 1.5
        # convert MHz to Hz
        BW_Hz = bandwidth_MHz * 1e6
        # noise in dB
        return (10 * np.log10(k * T * 1e3) 
                + noise_figure + 10 * np.log10(BW_Hz))
    
    def estimateSINR1(self, rx_power_list, bandwidth_MHz):
        # get the SINR for received power from each Tx 
        _power = []
        for item in rx_power_list:
            _power.append(item[0])
            
        i_summed = np.sum(_power)
        noise = self.estimateNoisedB(bandwidth_MHz)
        network_load = SimData['Network load']
        # network_load = 1
        i_sum = i_summed * network_load
        SINR = noise
        # result = [tx_index, Rx_power (dB), PL  (dB), 
        # los, transform, sinr, i_plus_n (dB), noise]
        result = None
        for j, Rx_power in enumerate(_power):
            # get the interference in dB
            i_plus_n = round(i_sum - Rx_power + noise,2)
            sinr = round(Rx_power / i_plus_n, 2)
            # sinr = round(10*np.log10(abs(Rx_power / i_plus_n)), 2)
            #
            # 
            if sinr > SINR:
                SINR = sinr
                # j = i
                result = [
                    j,
                    rx_power_list[j][0], # rx_power
                    rx_power_list[j][1], # PL
                    rx_power_list[j][2], # LOS value
                    rx_power_list[j][3], # transform
                    sinr,
                    i_plus_n,
                    round(noise,2),
                ]
        return result
    
    def estimateSINR(self, rx_power_list, bandwidth_MHz):
        # rewrite estimateSINR using sinr = np.log2(1+ snr)
        _power = []
        for item in rx_power_list:
            # convert rx_power from dB to W
            _power.append(10**(item[0]/10))
        # get noise in W
        noise_db = self.estimateNoisedB(bandwidth_MHz)
        noise = 10**(noise_db/10)
        # network_load = SimData['Network load']
        # get total interference in W
        i_summed = np.sum(_power)
        # get the interference plus noise
        SINR = noise_db
        for j, Rx_power in enumerate(_power):
            i_plus_n = i_summed - Rx_power + noise
            # get the SINR
            sinr = round(10 * np.log10(np.log2(1 + Rx_power / i_plus_n)),2)
            if sinr > SINR:
                SINR = sinr
                result = [
                    j,
                    rx_power_list[j][0], # rx_power
                    rx_power_list[j][1], # PL
                    rx_power_list[j][2], # LOS value
                    rx_power_list[j][3], # transform
                    sinr,
                    round(10*np.log10(i_plus_n),2),
                    round(noise_db,2)
                ]
        return result
        
    def estimateSpectralEfficiency(self, sinr, generation, MCLut = None):
        """
        Uses the SINR to determine spectral efficiency given the relevant
        modulation and coding scheme.

        Parameters
        ----------
        sinr : float
            Signal-to-Interference-plus-Noise-Ratio (SINR) in decibels.
        generation : string
            Either 4G or 5G dependent on technology.
        MCLut : list of tuples of modulation_and_coding_lut  
            A lookup table containing modulation and coding rates,
            spectral efficiencies and SINR estimates.

        Returns
        -------
        spectral_efficiency : float
            Efficiency of information transfer in Bps/Hz

        """
        if MCLut == None:
            MCLut = MODULATION_AND_CODING_LUT
        #
        spectral_efficiency = 0.1
        for lower, upper in pairwise(MCLut):
            if lower[0] and upper[0] == generation:
                lower_sinr = lower[6]
                upper_sinr = upper[6]
                if sinr >= lower_sinr and sinr < upper_sinr:
                    spectral_efficiency = lower[5]
                    return round(spectral_efficiency,2)
        #
        highest_value = MCLut[-1]
        if sinr >= highest_value[6]:
            spectral_efficiency = highest_value[5]
            return round(spectral_efficiency,2)
        lowest_value = MCLut[0]
        if sinr < lowest_value[6]:
            spectral_efficiency = 0
            return round(spectral_efficiency,2)
        
    def estimateAverageCapacity(self, bandwidth, spectral_efficiency):
        """
        Estimate link capacity based on bandwidth and received signal.

        Parameters
        ----------
        bandwidth : int
            Channel bandwidth in MHz
        spectral_efficiency : float
            Efficiency of information transfer in Bps/Hz

        Returns
        -------
        capacity_mbps : float
            Average link budget capacity in Mbps.
        capacity_mbps_km2 : float
            Average site area capacity in Mbps km^2.

        """
        bandwidth_in_hertz = bandwidth * 1e6 #MHz to Hz
        capacity_mbps = round((
            (bandwidth_in_hertz * spectral_efficiency) / 1e6),2)
        area_km2 = SimData['Simulation area per square km']
        capacity_mbps_km2 = round((
            capacity_mbps / area_km2),2)
        return capacity_mbps, capacity_mbps_km2
        
    def collect(self, iteration):
        #
        generations = ['4G', '5G']
        frequencies = SimData['Dynamic ranges']['Frequency range (GHz)']
        bandwidths = SimData['Dynamic ranges']['Bandwidth range (MHz)']
        type_of_sight = ['prob', 'ray']
        n_rx = len(self.RxNodes)
        # 
        for i in range(n_rx):
            for f in frequencies:
                for b in bandwidths:
                    for g in generations:
                        for s in type_of_sight:
                            transform_vec = self.Transforms[i,:]
                            rx_power = self.estimateRxPower(
                                i, 
                                transform_vec, 
                                f, 
                                s)
                            # if i == 0 :
                            #     print(f'F:{f}B:{b}G:{g}LoS:{s}, Rx: {rx_power}')
                            (j, rxPower, PL, LOSvalue, _transform,
                            sinr, i_plus_n, noise) = \
                                self.estimateSINR(
                                rx_power, b)
                            SE = self.estimateSpectralEfficiency(
                                    sinr, g)
                            (capacity, capacity_km2) = \
                                self.estimateAverageCapacity(
                                    b, SE)
                                
                            # write result
                            # _transform = transform_vec[j]
                            data = {
                                'Iter': iteration,
                                'Environment': self.scenario_name,
                                'Frequency (GHz)': f,
                                'Bandwidth (MHz)': b,
                                'Generation': g,
                                'Average building height': self.StaticParam['Average building height'],
                                'Average street width': self.StaticParam['Average street width'],
                                'Average roof height': self.StaticParam['Average roof height'],
                                'Above roof': self.StaticParam['Above roof'],
                                #
                                'Rx ID': i,
                                'Tx ID': j,
                                'Rx position': self.getRxPos(_transform),
                                'Tx position': self.getTxPos(_transform),
                                'Tx altitude': self.getTxZ(_transform),
                                'Rx altitude': self.getRxZ(_transform),
                                'Distance': self.getd3D(_transform),
                                'LoS type': s,
                                'LoS value': LOSvalue,
                                'Path loss (dB)': PL,
                                'Noise (dB)': noise,
                                'Interference (dB)': i_plus_n,
                                'Received power (dB)': rxPower,
                                # 
                                'SINR (dB)': sinr,
                                'Spectral efficiency (bps/Hz)': SE,
                                'Capacity (Mbps)': capacity,
                                'Capacity (Mbps/km^2)': capacity_km2,
                            }
                            self.rawWriter.writerow(data)
                            # result.append(data)
                            # # collect percentiles
                            # data_percentile[s][f][b][g]['Path loss (dB)'].append(PL)
                            # data_percentile[s][f][b][g]['Received power (dB)'].append(rxPower)
                            # data_percentile[s][f][b][g]['SINR'].append(sinr)
                            # data_percentile[s][f][b][g]['I_plus_n (dB)'].append(i_plus_n)
                            # data_percentile[s][f][b][g]['Spectral efficiency (bps/Hz)'].append(SE)
                            # data_percentile[s][f][b][g]['Capacity (Mbps)'].append(capacity)
                            # data_percentile[s][f][b][g]['Capacity (Mbps/km^2)'].append(capacity_km2)
                            
        #
        return True
        
    # def collect(self):
    #     """ Collect metrics  and write to files """
    #     result = self._collect()
    #     for data in result:
    #         self.rawWriter.writerow(data)
            
    #     return True
