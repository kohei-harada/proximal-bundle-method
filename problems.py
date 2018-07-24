import sys
sys.path.append("problems")
import GeneralizedMAXQ_10
import GeneralizedMXHILB_10
import GeneralizedMXHILB_100
import GeneralizedMXHILB_1000
import ChainedLQ_10
import ChainedLQ_100
import ChainedLQ_1000
import ChainedCB3_I_10
import ChainedCB3_II_10
import ChainedCB3_II_100
import ChainedCB3_II_1000
import EVD52 
import RosenSuzuki 
import maxquad 
import DEM
import Shor
import TiltedNorm_10
import TiltedNorm_50
import TiltedNorm_100
import TiltedNorm_500
import MQ_10
import MQ_50
import MQ_100
import MQ_500
import bays29
import hk48
import ch130
import pcb442
import sample

testset = [EVD52, RosenSuzuki, maxquad, DEM, Shor, \
           GeneralizedMAXQ_10, \
           GeneralizedMXHILB_10, GeneralizedMXHILB_100, GeneralizedMXHILB_1000, \
           ChainedLQ_10, ChainedLQ_100, ChainedLQ_1000, \
           ChainedCB3_I_10, \
           ChainedCB3_II_10, ChainedCB3_II_100, ChainedCB3_II_1000, \
           MQ_10, MQ_50, MQ_100, MQ_500, \
           TiltedNorm_10, TiltedNorm_50, TiltedNorm_100, TiltedNorm_500,
           bays29, hk48, ch130, pcb442]
#testset = [maxquad]
testset = [sample]
