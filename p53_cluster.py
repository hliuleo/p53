# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:33:37 2016

@author: hliu
"""

import mdtraj as md
import pandas as pd
import numpy as np
from plot_set import *
import glob
import os
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.preprocessing import RobustScaler
from msmbuilder.decomposition import tICA
from msmbuilder.dataset import dataset
from msmbuilder.cluster import MiniBatchKMeans

os.chdir(os.environ['p53']+os.sep+'data/analyzable_data')

def getTraj(trajDir, trajNameType, topFile):
    trajs = []
    trajNames = glob.glob1(trajDir, trajNameType)
    trajNames.sort()
    for n in trajNames:
        f = trajDir+os.sep+n
        trajs.append(md.load(f, top=topFile))
    return trajs

t = md.load('RSFF/rsff_skip20.xtc', top='p53_prot.pdb')
xyz = dataset('RSFF/rsff_skip20.xtc', topology='p53_prot.pdb')

featurizer = DihedralFeaturizer(types=['phi', 'psi'])
diheds = xyz.fit_transform_with(featurizer, 'diheds/', fmt='dir-npy')

scaler = RobustScaler()
scaled_diheds = diheds.fit_transform_with(scaler, 'scaled_diheds/', fmt='dir-npy')

tica_model = tICA(lag_time=1, n_components=4)
tica_model = scaled_diheds.fit_with(tica_model)
tica_trajs = scaled_diheds.transform_with(tica_model, 'ticas/', fmt='dir-npy')

clusterer = MiniBatchKMeans(n_clusters=100, random_state=42)
clustered_trajs = tica_trajs.fit_transform_with(
    clusterer, 'kmeans/', fmt='dir-npy'
)

txx = np.concatenate(tica_trajs)

plt.hexbin(txx[:,0], txx[:,1], bins='log', mincnt=1, cmap='viridis')
plt.scatter(clusterer.cluster_centers_[:,0],
            clusterer.cluster_centers_[:,1], 
            s=300, c='w')



from msmbuilder.msm import MarkovStateModel
from msmbuilder.utils import dump
import msmexplorer as msme
msm = MarkovStateModel(lag_time=1, n_timescales=20)
msm.fit(clustered_trajs)


assignments = clusterer.partial_transform(txx)
assignments = msm.partial_transform(assignments)
ax=msme.plot_free_energy(txx, obs=(0, 1), n_samples=10000,
                      pi=msm.populations_[assignments],
                      cmap='bone', alpha=0.5,vmin=-.001,
                      xlabel='tIC 1', ylabel='tIC 2')

plt.scatter(clusterer.cluster_centers_[msm.state_labels_, 0],
            clusterer.cluster_centers_[msm.state_labels_, 1],
            s=1e4 * msm.populations_,       # size by population
            c=msm.left_eigenvectors_[:, 1], # color by eigenvector
            cmap="coolwarm",
            zorder=3) 
plt.colorbar(label='First dynamical eigenvector')
plt.tight_layout()


msm.timescales_

msme.plot_timescales(msm, n_timescales=5,
                     ylabel='Implied Timescales ($ns$)')

from msmbuilder.lumping import PCCAPlus
pcca = PCCAPlus.from_msm(msm, n_macrostates=2)
macro_trajs = pcca.transform(clustered_trajs)

cm = plt.cm.get_cmap('RdYlBu')
msme.plot_free_energy(txx, obs=(0, 1), n_samples=10000, vmin=-0.001,
                      pi=msm.populations_[assignments],
                      xlabel='tIC 1', ylabel='tIC 2')
sc =plt.scatter(clusterer.cluster_centers_[msm.state_labels_, 0],
                clusterer.cluster_centers_[msm.state_labels_, 1],
                s=300,
                c=pcca.microstate_mapping_,
                zorder=3,
                cmap=cm
               )
#plt.colorbar(sc)
plt.tight_layout()