import numpy as np
import pandas as pd

from panda5gSim.core.scene_graph import (
    getRayLoS_nBuildings, findNPbyTag)
from panda5gSim.core.helpers import pairwise
from panda5gSim.core.transformations import TransformProcessor
from panda5gSim.metrics.writer import Writer

class TransformReader(TransformProcessor):
    # This class updates transforms 
    def __init__(self, txTags, RxTags):
        self.TxTags = txTags
        self.RxTags = RxTags
        # find Tx and Rx nodes
        self.findTags()
        
        
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
    