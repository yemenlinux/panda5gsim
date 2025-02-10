import numpy as np
import pandas as pd
import time

from panda5gSim.core.scene_graph import (
    getRayLoS_nBuildings, findNPbyTag)
# from panda5gSim.core.helpers import pairwise
from panda5gSim.core.transformations import TransformProcessor
# from panda5gSim.metrics.writer import Writer
from panda5gSim.users.antennas import (
    # theta_bar, phi_bar, psi_, 
    antenna_gain_dBi,
    Antenna3GPP)


class TransformReader(TransformProcessor):
    # This class updates transforms 
    def __init__(self, txTags, RxTags):
        self.TxTags = txTags
        self.RxTags = RxTags
        # find Tx and Rx nodes
        self.findTags()
        #
        self.antenna = Antenna3GPP()
        
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
        
    def getTransformsDF(self):
        # returns a pandas dataframe of transforms
        # make a pandas dataframe and update transforms to it
        Transforms = pd.DataFrame(columns=['Transforms', 'RayLoS', 'nBuildings'])
        # update Transformation matrix
        for i in range(len(self.RxNodes)):
            for j in range(len(self.TxNodes)):
                # a Transform = (rx.getTransform(tx), rx.getTransform(), tx.getTransform)
                try:
                    los, n = getRayLoS_nBuildings(self.TxNodes[j], self.RxNodes[i])
                    transform =  [(
                                self.RxNodes[i].getTransform(self.TxNodes[j]),
                                self.RxNodes[i].getTransform(render), # type: ignore
                                self.TxNodes[j].getTransform(render), # type: ignore
                                i,j),
                                # get ray LoS
                                los,
                                n
                                ]
                    Transforms.loc[len(Transforms)] = transform
                except:
                    print(f"Error in getting transforms {i}, {j}")
                # transform =  (
                #             self.RxNodes[i].getTransform(self.TxNodes[j]),
                #             self.RxNodes[i].getTransform(render), # type: ignore
                #             self.TxNodes[j].getTransform(render), # type: ignore
                #             i,j)
                # Transforms.loc[len(self.Transforms)] = transform
        return Transforms
    
    # def _getTimedTransformsDF(self):
    #     # returns a pandas dataframe of transforms
    #     # make a pandas dataframe and update transforms to it
    #     cols = ['Time', 'RxNode', 'TxNode',
    #             'dRayLoS', 
    #             #'nBuildings', #'v_rx', 'v_tx',
    #             'phi', 'theta', #'psi', 
    #             'd3D',
    #             'd2D', 'Gain',
    #             # 'Transforms', 
    #             'h']
    #     phi_3dB = 65
    #     theta_3dB = 65
    #     Transforms = pd.DataFrame(columns=cols)
    #     # update Transformation matrix
    #     t = time.time()
    #     for i in range(len(self.RxNodes)):
    #         for j in range(len(self.TxNodes)):
    #             # a Transform = (rx.getTransform(tx), rx.getTransform(), tx.getTransform)
    #             # get time 
    #             dt = globalClock.getDt() # type: ignore
    #             # print(dt)
    #             # get the actor's velocity
    #             class_rx = self.RxNodes[i].getParent().getPythonTag("subclass")
    #             v_rx = round(class_rx.getVelocity().length()/dt,2)
    #             class_tx = self.TxNodes[j].getParent().getPythonTag("subclass")
    #             v_tx = round(class_tx.getVelocity().length()/dt, 2)
    #             # get the actor's name
    #             rx_name = self.RxNodes[i].getParent().getPythonTag("subclass").getActorName()
    #             tx_name = self.TxNodes[j].getParent().getPythonTag("subclass").getActorName()
    #             # print(f"dt: {dt}, v_rx: {v_rx}, v_tx: {v_tx}")
    #             #
    #             # a, b, g = self.TxNodes[j].getHpr()
    #             # ph, th, ps = self.RxNodes[i].getHpr()
    #             # theta = theta_bar(a, b, g, ph, th)
    #             # phi = phi_bar(a, b, g, ph, th)
    #             # psi_ = psi(a, b, g, ph, th)
    #             # F_ph_th_ = F_ph_th(np.rad2deg(phi), np.rad2deg(theta), np.rad2deg(psi_), att = 30)
    #             phi, theta, ps = self.RxNodes[i].getTransform(self.TxNodes[j]).getHpr()
    #             #
                
    #             # F_ph_th_ = antenna_gain_dBi(phi, theta)
    #             F_ph_th_ = self.antenna.get_dB_gain(phi, theta) # dBi
    #             try:
    #                 los, n = getRayLoS_nBuildings(self.TxNodes[j], 
    #                                             self.RxNodes[i])
    #                 #
    #                 if (abs(phi) <= phi_3dB/2 
    #                     and abs(theta) <= theta_3dB/2
    #                     and los == 1):
    #                     dlos = 1
    #                 else:
    #                     dlos = 0
    #                 # 
    #                 transform =  {
    #                     'Time': t,
    #                     'RxNode': rx_name,
    #                     'TxNode': tx_name,
    #                     # get ray LoS
    #                     'RayLoS': los,
    #                     'dRayLoS': dlos,
    #                     # n,
    #                     # v_rx,
    #                     # v_tx,
    #                     # 
    #                     'phi': round(phi,0),
    #                     'theta': round(theta,0),
    #                     # round(ps,0),
    #                     'd3D': round((self.RxNodes[i].getTransform(render).getPos() 
    #                            - self.TxNodes[j].getTransform(render).getPos()).length(),0),
    #                     'd2D': round((self.RxNodes[i].getTransform(render).getPos() 
    #                            - self.TxNodes[j].getTransform(render).getPos()).getXy().length(),0),
    #                     'Gain': F_ph_th_,
    #                     # 'Transforms',
    #                     # (self.RxNodes[i].getTransform(self.TxNodes[j]),
    #                     # self.RxNodes[i].getTransform(render), # type: ignore
    #                     # self.TxNodes[j].getTransform(render), # type: ignore
    #                     # i,j),
    #                     # h
    #                     'h': round(self.TxNodes[j].getZ(render) - self.RxNodes[i].getZ(render),2)
    #                 }
    #                 Transforms.loc[len(Transforms)] = transform
    #             except:
    #                 print(f"Error in getting transforms {i}, {j}")
    #             # transform =  (
    #             #             self.RxNodes[i].getTransform(self.TxNodes[j]),
    #             #             self.RxNodes[i].getTransform(render), # type: ignore
    #             #             self.TxNodes[j].getTransform(render), # type: ignore
    #             #             i,j)
    #             # Transforms.loc[len(self.Transforms)] = transform
    #     return Transforms
    
    def getTimedTransformsDF(self, columns=None):
        # returns a pandas dataframe of transforms
        # Transforms = pd.DataFrame()
        # make a pandas dataframe and update transforms to it
        if columns is None:
            columns = ['Time', 'RxNode', 'TxNode',
                    'RayLoS', 'dRayLoS', 
                    'nBuildings', 'v_rx', 'v_tx',
                    'phi', 'theta', #'psi', 
                    'd3D',
                    'd2D', 'Gain',
                    'Transforms', 
                    'h']
        phi_3dB = 65
        theta_3dB = 65
        Transforms = pd.DataFrame(columns=columns)
        # update Transformation matrix
        t = time.time()
        for i in range(len(self.RxNodes)):
            for j in range(len(self.TxNodes)):
                # a Transform = (rx.getTransform(tx), rx.getTransform(), tx.getTransform)
                # get time 
                dt = globalClock.getDt() # type: ignore
                # print(dt)
                # get the actor's velocity
                class_rx = self.RxNodes[i].getParent().getPythonTag("subclass")
                v_rx = round(class_rx.getVelocity().length()/dt,2)
                class_tx = self.TxNodes[j].getParent().getPythonTag("subclass")
                v_tx = round(class_tx.getVelocity().length()/dt, 2)
                # get the actor's name
                rx_name = self.RxNodes[i].getParent().getPythonTag("subclass").getActorName()
                tx_name = self.TxNodes[j].getParent().getPythonTag("subclass").getActorName()
                # print(f"dt: {dt}, v_rx: {v_rx}, v_tx: {v_tx}")
                #
                # a, b, g = self.TxNodes[j].getHpr()
                # ph, th, ps = self.RxNodes[i].getHpr()
                # theta = theta_bar(a, b, g, ph, th)
                # phi = phi_bar(a, b, g, ph, th)
                # psi_ = psi(a, b, g, ph, th)
                # F_ph_th_ = F_ph_th(np.rad2deg(phi), np.rad2deg(theta), np.rad2deg(psi_), att = 30)
                phi, theta, ps = self.RxNodes[i].getTransform(self.TxNodes[j]).getHpr()
                #
                
                # F_ph_th_ = antenna_gain_dBi(phi, theta)
                F_ph_th_ = self.antenna.get_dB_gain(phi, theta) # dBi
                try:
                    los, n = getRayLoS_nBuildings(self.TxNodes[j], 
                                                self.RxNodes[i])
                    #
                    if (abs(phi) <= phi_3dB/2 
                        and abs(theta) <= theta_3dB/2
                        and los == 1):
                        dlos = 1
                    else:
                        dlos = 0
                    # 
                    transform =  {}
                    if 'Time' in columns:
                        transform['Time'] = t
                    if 'RxNode' in columns:
                        transform['RxNode'] = rx_name
                    if 'TxNode' in columns:
                        transform['TxNode'] = tx_name
                    if 'RayLoS' in columns:
                        transform['RayLoS'] = los
                    if 'dRayLoS' in columns:
                        transform['dRayLoS'] = dlos
                    if 'phi' in columns:
                        transform['phi'] = round(phi,0)
                    if 'theta' in columns:
                        transform['theta'] = round(theta,0)
                    if 'd3D' in columns:
                        transform['d3D'] = round((self.RxNodes[i].getTransform(render).getPos() 
                               - self.TxNodes[j].getTransform(render).getPos()).length(),0)
                    if 'd2D' in columns:
                        transform['d2D'] = round((self.RxNodes[i].getTransform(render).getPos() 
                               - self.TxNodes[j].getTransform(render).getPos()).getXy().length(),0)
                    if 'Gain' in columns:
                        transform['Gain'] = F_ph_th_
                    if 'h' in columns:
                        transform['h'] = round(self.TxNodes[j].getZ(render) - self.RxNodes[i].getZ(render),2)
                    if 'Transforms' in columns:
                        transform['Transforms'] = (self.RxNodes[i].getTransform(self.TxNodes[j]),
                                                    self.RxNodes[i].getTransform(render), # type: ignore
                                                    self.TxNodes[j].getTransform(render), # type: ignore
                                                    i,j)
                    if 'v_rx' in columns:
                        transform['v_rx'] = v_rx
                    if 'v_tx' in columns:
                        transform['v_tx'] = v_tx
                    if 'nBuildings' in columns:
                        transform['nBuildings'] = n
                    
                    Transforms.loc[len(Transforms)] = transform
                except:
                    print(f"Error in getting transforms {i}, {j}")
                # transform =  (
                #             self.RxNodes[i].getTransform(self.TxNodes[j]),
                #             self.RxNodes[i].getTransform(render), # type: ignore
                #             self.TxNodes[j].getTransform(render), # type: ignore
                #             i,j)
                # Transforms.loc[len(self.Transforms)] = transform
        return Transforms
    