#
from panda3d.core import TransformState
import numpy as np
import re
from panda3d.core import (
    NodePath,
    LPoint3f,
    LVecBase3)

class TransformProcessor:
    # Transform format:
    # a Transform = (rx.getTransform(tx), rx.getTransform(), tx.getTransform(), rx_id, tx_id)
    def getRxPos(self, transform):
        pos = transform[1].getPos()
        return (round(pos[0],2), round(pos[1],2), round(pos[2],2))
    def getRxZ(self, transform):
        return round(transform[1].getPos().getZ(),2)
    def getRxX(self, transform):
        return transform[1].getPos().getX()
    def getRxY(self, transform):
        return transform[1].getPos().getY()
    def getRxHpr(self, transform):
        return transform[1].getHpr()
    def getRxH(self, transform):
        return transform[1].getHpr().getX()
    def getRxP(self, transform):
        return transform[1].getHpr().getY()
    def getRxR(self, transform):
        return transform[1].getHpr().getZ()
    def getRxPhiTheta(self, transform):
        return transform[1].getHpr().getXy()
    def getRxID(self, transform):
        return transform[3]
    #
    def getTxPos(self, transform):
        pos = transform[2].getPos()
        return (round(pos[0],2), round(pos[1],2), round(pos[2],2)) 
    def getTxZ(self, transform):
        return round(transform[2].getPos().getZ(),2)
    def getTxX(self, transform):
        return transform[2].getPos().getX()
    def getTxY(self, transform):
        return transform[2].getPos().getY()
    def getTxHpr(self, transform):
        return transform[2].getHpr()
    def getTxH(self, transform):
        return transform[2].getHpr().getX()
    def getRxP(self, transform):
        return transform[2].getHpr().getY()
    def getTxR(self, transform):
        return transform[2].getHpr().getZ()
    def getTxPhiTheta(self, transform):
        return transform[2].getHpr().getXy()
    def getTxID(self, transform):
        return transform[4]
    #
    # def getd3D(self, transform):
    #     return transform[0].getPos().length()
    def getd3D(self, transform):
        return round((transform[1].getPos() - transform[2].getPos()).length(),2)
    def getd2D(self, transform):
        return round((transform[1].getPos() - transform[2].getPos()).getXy().length(),2)
    #
    def getPhiTheta(self, transform):
        return transform[0].getHpr().getXy()
    def getAngles(self, transform):
        return transform[0].getHpr()
    #
    def linear2dB(self, W):
        return 10*np.log10(W) 
    def dB2Linear(self, dB):
        return 10**(dB/10)
    
    #
    def getRelativePhiTheta(self, transform):
        # telative angles from tx to rx
        return transform[0].getHpr().getXy()
    
    def getPhi_Transform(self, transform):
        return round(self.getRelativePhiTheta(transform).getX())
    
    def getTheta_Transform(self, transform):
        return round(self.getRelativePhiTheta(transform).getY())
    
    def getRelativePos(self, transform):
        return transform[0].getPos()
    
    def getPhiTheta_Ray(self, transform):
        rx = NodePath('rx')
        tx = NodePath('tx')
        rx.setTransform(transform[1])
        tx.setTransform(transform[2])
        rx.lookAt(tx.getPos())
        trs = (rx.getTransform(tx), rx.getTransform(), tx.getTransform(), transform[3], transform[4])
        return self.getRelativePhiTheta(trs)
    
    def getPhi_DirectRay(self, transform):
        return round(self.getPhiTheta_Ray(transform).getX())
        
    def getTheta_DirectRay(self, transform):
        return round(self.getPhiTheta_Ray(transform).getY())
    
    def makeTransformState(self, instr):
        # ts = TransformState()
        # pattern = r'[-+]?\d*\.?\d+'
        # ppos = r'pos ([-+]?\d*\.?\d+) ([-+]?\d*\.?\d+) ([-+]?\d*\.?\d+)'
        ppos = r'pos ([-+]?\d*\.?\d*e?[-+]?\d+) ([-+]?\d*\.?\d*e?[-+]?\d+) ([-+]?\d*\.?\d*e?[-+]?\d+)'
        mpos = re.findall(ppos, instr)
        phpr = r'hpr ([-+]?\d*\.?\d+) ([-+]?\d*\.?\d+) ([-+]?\d*\.?\d+)'
        mhpr = re.findall(phpr, instr)
        pscale1 = r'scale ([-+]?\d*\.?\d+) ([-+]?\d*\.?\d+) ([-+]?\d*\.?\d+)'
        mscale1 = re.findall(pscale1, instr)
        if len(mscale1) == 0:
            pscale1 = r'scale ([-+]?\d*\.?\d+)'
            mscale = re.findall(pscale1, instr)
        else:
            mscale = mscale1
        if len(mscale) > 0 and len(mpos) >0 and len(mhpr)>0:
            if type(mscale[0]) is tuple:
                return TransformState.makePosHprScale(
                    LVecBase3(float(mpos[0][0]), float(mpos[0][1]), float(mpos[0][2])),
                    LVecBase3(float(mhpr[0][0]), float(mhpr[0][1]), float(mhpr[0][2])),
                    LVecBase3(float(mscale[0][0]), float(mscale[0][1]), float(mscale[0][2])))
            else:
                return TransformState.makePosHprScale(
                    LVecBase3(float(mpos[0][0]), float(mpos[0][1]), float(mpos[0][2])),
                    LVecBase3(float(mhpr[0][0]), float(mhpr[0][1]), float(mhpr[0][2])),
                    LVecBase3(float(mscale[0]), float(mscale[0]), float(mscale[0])))
        elif len(mpos) >0 and len(mhpr)>0:
            return TransformState.makePosHpr(
                    LVecBase3(float(mpos[0][0]), float(mpos[0][1]), float(mpos[0][2])),
                    LVecBase3(float(mhpr[0][0]), float(mhpr[0][1]), float(mhpr[0][2]))
                    )
        elif len(mpos) >0 :
            return TransformState.makePos(
                    LVecBase3(float(mpos[0][0]), float(mpos[0][1]), float(mpos[0][2])))
        else:
            print(instr)
    
    def text_to_Transforms(self, instr):
        inlist = instr[1:][:-1].split(', ')
        return (self.makeTransformState(inlist[0]),
                self.makeTransformState(inlist[1]),
                self.makeTransformState(inlist[2]),
                float(inlist[3]),
                float(inlist[4]))
    
    def relative_Ray_Angles(self, transform):
        """ Get the relative angles between rx and tx
        that are used in calculating the antenna gain
        for a directional antenna.
        """
        rx = NodePath('rx')
        tx = NodePath('tx')
        rx.setTransform(transform[1])
        tx.setTransform(transform[2])
        rx.lookAt(tx.getPos())
        # 
        # equation (7.1-18), (7.1-19) 3GPP TR 38.900
        # theta_bar = arccos(cos(phi) sin(theta) sin(beta) + cos(theta) cos(beta)
        #phi_bar = arg(cos(phi) sin(theta) cos(beta) - cos(theta) sine(beta) + j sin(phi) sine(theta))
        relative_trans = rx.getTransform(tx)
        phi, theta, slant = relative_trans.getHpr()
        return round(phi, 2), round(theta, 2), round(slant, 2)
    
    

    
