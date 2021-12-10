#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import lib.EventSelection as es
import lib.ProfileLikelihoodBuilder as plb
import lib.AmBePlots as abp
import lib.BeamPlots as bp
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as scp
import numpy as np
import scipy.misc as scm
from pylab import figure, axes, pie, title, show

plt.rc('font', family='Times', size=12)
import pylab
pylab.rcParams['figure.figsize'] = 10, 7.6


#SIGNAL_DIRS = ["./Data/V3_5PE100ns/BeamData/"]
SIGNAL_DIRS = ["/Users/edrakopo/work/ANNIETools/ANNIENtupleAnalysis/Data/"]
SIGNAL_DIRS = ["/Users/edrakopo/work/ANNIETools/ANNIENtupleAnalysis/Data/ProcessedNTuples/"]
#SIGNAL_DIRS = ["/Users/edrakopo/work/ANNIETools/ANNIENtupleAnalysis/Data/ProcessedNTuples/ProcessedBeamR1623S0p*"]
SIGNAL_LABELS = ['Beam']
#BKG_DIR = "./Data/BkgCentralData/"

#PEPERMEV = 12.

#expoPFlat= lambda x,C1,tau,mu,B: C1*np.exp(-(x-mu)/tau) + B
#mypoisson = lambda x,mu: (mu**x)*np.exp(-mu)/scm.factorial(x)

def GetDataFrame(mytreename,mybranches,filelist):
    RProcessor = rp.ROOTProcessor(treename=mytreename)
    for f1 in filelist:
        RProcessor.addROOTFile(f1,branches_to_get=mybranches)
    data = RProcessor.getProcessedData()
    df = pd.DataFrame(data)
    return df

def BeamPlotDemo(PositionDict,MCdf):
    Sdf = PositionDict["Beam"][0]
    Sdf_trig = PositionDict["Beam"][1]
    Sdf_mrd = PositionDict["Beam"][2]
    print("Sdf ",Sdf.head()," ",Sdf.shape)
    print("All columns are: ", Sdf.columns.values.tolist())

    Sdf = Sdf.loc[Sdf["eventTimeTank"]>-9].reset_index(drop=True)
    Sdf_trig = Sdf_trig.loc[Sdf_trig["eventTimeTank"]>-9].reset_index(drop=True)
    Sdf_mrd = Sdf_mrd.loc[Sdf_mrd["eventTimeTank"]>-9].reset_index(drop=True)
    
    Sdf_TankVeto = es.HasVetoHit_TankClusters(Sdf,Sdf_trig)
    print("NUM TANK CLUSTERS WITH VETO HIT: " + str(len(Sdf_TankVeto)))

    print("NUM TRIGS: " + str(len(Sdf_trig)))
    HasVetoHit = np.where(Sdf_mrd["vetoHit"].values==1)[0]
    print("NUM MRD CLUSTERS WITH VETO HIT: " + str(len(HasVetoHit)))

    #---- My Plots:
    Sdf_prompt=Sdf.loc[Sdf['clusterTime']<2000].reset_index(drop=True) #prompt events
    plt.hist(Sdf_prompt['clusterTime'],bins=100,range=(0,2000))
    plt.title("Prompt window Tank cluster times - no cuts")
    plt.xlabel("Cluster time [ns]")
    plt.show()
#    plt.savefig("plots/time_prompt.png")

    Sdf_del=Sdf.loc[Sdf['clusterTime']>=2000].reset_index(drop=True) #delayed events
    plt.hist(Sdf_del['clusterTime'])#,bins=100,range=(10000,70000))
    plt.title("Delayed window Tank cluster times - no cuts")
    plt.xlabel("Cluster time [ns]")
    plt.show()
#    plt.savefig("plots/time_del.png")

    #--- CB to cluster Time:   
    labels = {'title': 'Charge balance parameters in time window \n (Beam data, $t_{c}>=2 \, \mu s$)', 
             'xlabel': 'Cluster time (ns)', 'ylabel': 'Charge balance'}
    ranges = {'xbins': 58, 'ybins':50, 'xrange':[2000,60000],'yrange':[0,1]}
    abp.Make2DHist(Sdf_del,'clusterTime','clusterChargeBalance',labels,ranges)
    plt.show()
#    plt.savefig("plots/CB_time_del.png")
    labels = {'title': 'Charge balance parameters in time window \n (Beam data, $t_{c}<2 \, \mu s$)',
              'xlabel': 'Cluster time (ns)', 'ylabel': 'Charge balance'}
    ranges = {'xbins': 20, 'ybins':50, 'xrange':[0,2000],'yrange':[0,1]}
    abp.Make2DHist(Sdf_prompt,'clusterTime','clusterChargeBalance',labels,ranges)
    plt.show()
#    plt.savefig("plots/CB_time_prompt.png")

    #--- CB to clusterPE: 
    labels = {'title': 'Charge balance parameters in time window \n (Beam data, $t_{c}>=2 \, \mu s$)',
              'xlabel': 'Cluster PE', 'ylabel': 'Charge balance'}
    ranges = {'xbins': 58, 'ybins':50, 'xrange':[0,500],'yrange':[0,1]}
    abp.Make2DHist(Sdf_del,'clusterPE','clusterChargeBalance',labels,ranges)
    plt.show()
#    plt.savefig("plots/CB_PE_del.png")
    labels = {'title': 'Charge balance parameters in time window \n (Beam data, $t_{c}<2 \, \mu s$)',
              'xlabel': 'Cluster PE', 'ylabel': 'Charge balance'}
    ranges = {'xbins': 20, 'ybins':50, 'xrange':[0,500],'yrange':[0,1]}
    abp.Make2DHist(Sdf_prompt,'clusterPE','clusterChargeBalance',labels,ranges)
    plt.show()
#    plt.savefig("plots/CB_PE_prompt.png")

    #splitting to CB categories:
    #--- CB>=0.9 
    Sdf_prompt_highCB = Sdf_prompt.loc[Sdf_prompt['clusterChargeBalance']>=0.9].reset_index(drop=True) 
    Sdf_del_highCB = Sdf_del.loc[Sdf_del['clusterChargeBalance']>=0.9].reset_index(drop=True)

    labels = {'title': 'Total PE vs Maximum PE in Cluster for \n (Beam data, $t_{c}<2 \, \mu s$) \n CB>=0.9 ',
              'xlabel': 'Cluster PE', 'ylabel': 'Maximum PE in Cluster'}
    ranges = {'xbins': 200, 'ybins':200, 'xrange':[0,200],'yrange':[0,200]}
    #abp.Make2DHist(Sdf_prompt_highCB,'clusterPE','clusterMaxPE',labels,ranges)
    abp.Make2DHist(Sdf_prompt_highCB,'clusterPE','clusterMaxPE',labels,ranges)
    plt.show()
#    plt.savefig("plots/PE_maxPE_prompt_highCB.png")

    #PE = np.hstack(Sdf_del_highCB['hitPE'])
    #ID = np.hstack(Sdf_del_highCB['hitDetID'])
    #T = np.hstack(Sdf_del_highCB['hitT'])
    #maxPE_highCB = max(np.hstack(Sdf_prompt_highCB.hitPE))
    #print("maxPE_highCB ",maxPE_highCB," clusterMaxPE ",Sdf_prompt_highCB.clusterMaxPE)

    highCB_PE = np.hstack(Sdf_prompt_highCB.hitPE)
    highCB_DetID = np.hstack(Sdf_prompt_highCB.hitDetID)
#    highCB_PE = np.hstack(Sdf_del_highCB.hitPE)
#    highCB_DetID = np.hstack(Sdf_del_highCB.hitDetID)
    plt.hist2d(highCB_DetID,highCB_PE)
    plt.title("PE distribution for all hits in clusters, CB>=0.9)")
    plt.xlabel("Tube ID")
    plt.ylabel("PE")
    plt.show()
#    plt.savefig("plots/TubeID_PE_prompt_highCB.png")
    
    #--- 0.6<CB<0.9
    Sdf_prompt_upperCB = Sdf_prompt.loc[(Sdf_prompt['clusterChargeBalance']<0.9) & (Sdf_prompt['clusterChargeBalance']>=0.6)].reset_index(drop=True)
    Sdf_del_upperCB = Sdf_del.loc[(Sdf_del['clusterChargeBalance']<0.9) & (Sdf_prompt['clusterChargeBalance']>=0.6)].reset_index(drop=True)

    labels = {'title': 'Total PE vs Maximum PE in Cluster for \n (Beam data, $t_{c}<2 \, \mu s$) \n 0.6<=CB<0.9',
              'xlabel': 'Cluster PE', 'ylabel': 'Maximum PE in Cluster'}
    ranges = {'xbins': 200, 'ybins':200, 'xrange':[0,200],'yrange':[0,200]}
    abp.Make2DHist(Sdf_prompt_upperCB,'clusterPE','clusterMaxPE',labels,ranges)
    plt.show()
#    plt.savefig("plots/PE_maxPE_prompt_upperCB.png")

    upperCB_PE = np.hstack(Sdf_prompt_upperCB.hitPE)
    upperCB_DetID = np.hstack(Sdf_prompt_upperCB.hitDetID)
    plt.hist2d(upperCB_DetID,upperCB_PE)
    plt.title("PE distribution for all hits in clusters, 0.6=<CB<0.9)")
    plt.xlabel("Tube ID")
    plt.ylabel("PE")
    plt.show()
#    plt.savefig("plots/TubeID_PE_prompt_upperCB.png")

    #--- 0.4<CB<0.6
    Sdf_prompt_midCB = Sdf_prompt.loc[(Sdf_prompt['clusterChargeBalance']<0.6) & (Sdf_prompt['clusterChargeBalance']>=0.4)].reset_index(drop=True)
    Sdf_del_midCB = Sdf_del.loc[(Sdf_del['clusterChargeBalance']<0.6) & (Sdf_prompt['clusterChargeBalance']>=0.4)].reset_index(drop=True)
   
    labels = {'title': 'Total PE vs Maximum PE in Cluster for \n (Beam data, $t_{c}<2 \, \mu s$)\n 0.4<=CB<0.6',
              'xlabel': 'Cluster PE', 'ylabel': 'Maximum PE in Cluster'}
    ranges = {'xbins': 200, 'ybins':200, 'xrange':[0,200],'yrange':[0,200]}
    abp.Make2DHist(Sdf_prompt_midCB,'clusterPE','clusterMaxPE',labels,ranges)
    plt.show()
#    plt.savefig("plots/PE_maxPE_prompt_midCB.png")
     
    midCB_PE = np.hstack(Sdf_prompt_midCB.hitPE)
    midCB_DetID = np.hstack(Sdf_prompt_midCB.hitDetID)
    plt.hist2d(midCB_DetID,midCB_PE)
    plt.title("PE distribution for all hits in clusters, 0.4=<CB<0.6)")
    plt.xlabel("Tube ID")
    plt.ylabel("PE")
    plt.show()
#    plt.savefig("plots/TubeID_PE_prompt_midCB.png")

    #--- CB<0.4
    Sdf_prompt_lowCB = Sdf_prompt.loc[Sdf_prompt['clusterChargeBalance']<0.4].reset_index(drop=True)
    Sdf_del_lowCB = Sdf_del.loc[Sdf_del['clusterChargeBalance']<0.4].reset_index(drop=True)
     
    labels = {'title': 'Total PE vs Maximum PE in Cluster for \n (Beam data, $t_{c}<2 \, \mu s$) \n CB<0.4',
              'xlabel': 'Cluster PE', 'ylabel': 'Maximum PE in Cluster'}
    ranges = {'xbins': 200, 'ybins':200, 'xrange':[0,200],'yrange':[0,200]}
    abp.Make2DHist(Sdf_prompt_lowCB,'clusterPE','clusterMaxPE',labels,ranges)
    plt.show()
#    plt.savefig("plots/PE_maxPE_prompt_lowCB.png")
      
    lowCB_PE = np.hstack(Sdf_prompt_lowCB.hitPE)
    lowCB_DetID = np.hstack(Sdf_prompt_lowCB.hitDetID)
    plt.hist2d(lowCB_DetID,lowCB_PE)
    plt.title("PE distribution for all hits in clusters, CB<=0.4)")
    plt.xlabel("Tube ID")
    plt.ylabel("PE")
    plt.show()
#    plt.savefig("plots/TubeID_PE_prompt_lowCB.png")

'''
#########
    Sdf_prompt_noCB = Sdf.loc[Sdf['clusterTime']<2000].reset_index(drop=True)
    Sdf_prompt = Sdf_prompt_noCB.loc[Sdf_prompt_noCB['clusterChargeBalance']<0.9].reset_index(drop=True) ##for flasherscheck clusters with CB>0.9
#    plt.hist(Sdf_prompt['clusterTime'],bins=100,range=(0,2000))
#    plt.title("Prompt window Tank cluster times")
#    plt.xlabel("Cluster time [ns]")
#    plt.show()
    print("TOTAL PROMPT TANK CLUSTERS, NO CB: " + str(len(Sdf_prompt_noCB)))
    print("TOTAL PROMPT TANK CLUSTERS: " + str(len(Sdf_prompt)))
    print("TOTAL PROMPT MRD CLUSTERS: " + str(len(Sdf_mrd)))
   
    labels = {'title': 'Charge balance parameters in time window \n (Beam data, $t_{c}<2 \, \mu s$)', 
            'xlabel': 'Cluster time (ns)', 'ylabel': 'Charge balance'}
    ranges = {'xbins': 40, 'ybins':40, 'xrange':[0,2000],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist(Sdf_prompt_noCB,'clusterTime','clusterChargeBalance',labels,ranges)
    plt.show()

    labels = {'title': 'Tank PMT hit cluster count as a function of time \n (Beam data, $t_{c}<2 \, \mu s$)', 
            'xlabel': 'Cluster time (ns)', 'ylabel': 'Cluster PE'}
    ranges = {'xbins': 40, 'ybins':25, 'xrange':[0,2000],'yrange':[0,5000]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist(Sdf_prompt_noCB,'clusterTime','clusterPE',labels,ranges)
    plt.show()


    #Get largest cluster in each acquisition in prompt window
    Sdf_maxPE = es.MaxPEClusters(Sdf_prompt)
    print("TOTAL HIGHEST PE PROMPT CLUSTERS: " + str(len(Sdf_maxPE)))
    Sdf_mrd_maxhit = es.MaxHitClusters(Sdf_mrd)
    print("TOTAL MOST PADDLE MRD CLUSTERS: " + str(len(Sdf_mrd_maxhit)))
'''
'''
    #Now, get the index number for clusterTime pairs in the same triggers 
    TankIndices, MRDIndices = es.MatchingEventTimes(Sdf_maxPE,Sdf_mrd_maxhit)
    TankTimes = Sdf_maxPE["clusterTime"].values[TankIndices]
    MRDTimes = Sdf_mrd_maxhit["clusterTime"].values[MRDIndices]
    Pairs_HaveVeto = Sdf_mrd_maxhit.loc[(Sdf_mrd_maxhit["vetoHit"].values[MRDIndices]==1)]
    print("NUM OF MRD CLUSTERS IN TRIG WITH A TANK CLUSTER: " + str(len(MRDTimes)))
    print("NUM OF MRD CLUSTERS WITH VETO IN SUBSET: " + str(len(Pairs_HaveVeto)))
    plt.scatter(TankTimes,MRDTimes,marker='o',s=15,color='blue',alpha=0.7)
    plt.title("Tank and MRD cluster times in prompt window \n (Largest PE tank clusters, largest paddle count MRD clusters)")
    plt.xlabel("Tank Cluster time [ns]")
    plt.ylabel("MRD Cluster time [ns]")
    plt.show()

    plt.hist(MRDTimes - TankTimes, bins = 160, color='blue', alpha=0.7)
    plt.axvline(x=700,color='black',linewidth=6)
    plt.axvline(x=800,color='black',linewidth=6)
    plt.title("Difference in MRD and Tank cluster times in acquisitions \n (Largest PE tank clusters, largest paddle count MRD clusters)")
    plt.xlabel("MRD cluster time - Tank cluster time [ns]")
    plt.show()


    print("CLUSTER COUNT IN EVENTS BEFORE 2 US: " + str(len(Sdf_ClustersInPromptCandidates.loc[Sdf_ClustersInPromptCandidates["clusterTime"]<2000].values)))
    Sdf_ValidDelayedClusters = Sdf_ClustersInPromptCandidates.loc[Sdf_ClustersInPromptCandidates['clusterTime']>12000].reset_index(drop=True)
    Sdf_ValidDelayedClustersCB = Sdf_ClustersInPromptCandidates.loc[Sdf_ClustersInPromptCandidates['clusterChargeBalance']<0.4].reset_index(drop=True)
    print("CLUSTER COUNT IN EVENTS WITH PMT/MRD ACTIVITY PAST 12 US: " + str(len(Sdf_ValidDelayedClusters)))

    plt.hist(Sdf.loc[Sdf["clusterTime"]>12000,"clusterTime"],bins=20,range=(12000,65000),label='No PMT/MRD pairing in prompt',alpha=0.8)
    plt.hist(Sdf_ValidDelayedClusters["clusterTime"], bins=20, range=(12000,65000),label='PMT/MRD pair required in prompt',alpha=0.8)
    plt.hist(Sdf_ValidDelayedClustersCB["clusterTime"], bins=20, range=(12000,65000),label=' + CB < 0.4',alpha=0.8)
    plt.title("Delayed cluster times in beam runs")
    plt.ylabel("Number of clusters")
    plt.xlabel("Cluster time [ns]")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

'''

if __name__=='__main__':

    mybranches = ['eventNumber','eventTimeTank','clusterTime','hitT','hitQ','hitPE','hitDetID','clusterChargeBalance','clusterPE','clusterMaxPE','clusterHits']
    mytrigbranches = ['eventNumber','eventTimeTank','eventTimeMRD','vetoHit']

    myMRDbranches = ['eventNumber','eventTimeTank','eventTimeMRD','clusterTime','clusterHits','vetoHit',
            'numClusterTracks','MRDTrackAngle','MRDPenetrationDepth','MRDEntryPointRadius','MRDEnergyLoss','MRDEnergyLossError','MRDTrackLength']

    MCbranches = ['trueMuonEnergy','Pi0Count','PiPlusCount','PiMinusCount']
    mclist = ["../Data/V3_5PE100ns/MCProfiles/PMTVolumeReco_Full_06262019.ntuple.root"]

    #mybkgbranches = ['eventNumber','eventTimeTank','clusterTime','hitT','hitQ','hitPE','clusterChargeBalance','clusterPE','clusterMaxPE','clusterChargePointZ','SiPM1NPulses','SiPM2NPulses','clusterHits']
    #mybkgtrigbranches = ['eventNumber','eventTimeTank','eventTimeMRD','vetoHit','SiPM1NPulses','SiPM2NPulses']
    #blist = glob.glob(BKG_DIR+"*.ntuple.root")
    #Bdf = GetDataFrame("phaseIITankClusterTree",mybkgbranches,blist)
    #Bdf_trig = GetDataFrame("phaseIITriggerTree",mybranches,blist)

    MCdf = GetDataFrame("phaseII",MCbranches,mclist)
    PositionDict = {}
    for j,direc in enumerate(SIGNAL_DIRS):
        direcfiles = glob.glob(direc+"*.ntuple.root")

        livetime_estimate = es.EstimateLivetime(direcfiles)
        print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
        PositionDict[SIGNAL_LABELS[j]] = []
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIITankClusterTree",mybranches,direcfiles))
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIITriggerTree",mytrigbranches,direcfiles))
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIIMRDClusterTree",myMRDbranches,direcfiles))

    print("THE GOOD STUFF")
    BeamPlotDemo(PositionDict,MCdf)
    #EstimateNeutronEfficiencyAllPosns(PositionDict,Bdf,Bdf_trig)


