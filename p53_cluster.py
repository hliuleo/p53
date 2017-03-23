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
from multiprocessing import Pool
from msmbuilder.msm import MarkovStateModel
from msmbuilder.utils import dump
import msmexplorer as msme
import matplotlib
from msmbuilder.lumping import PCCAPlus

os.chdir(os.environ['p53']+os.sep+'data/analyzable_data')

#t = md.load(trajNames, top='p53_prot.pdb')


def featurizeData(xyz, tica_dim):
    featurizer = DihedralFeaturizer(types=['phi', 'psi'])
    if os.path.exists('diheds'):
        os.system('rm -rf diheds')
    diheds = xyz.fit_transform_with(featurizer, 'diheds/', fmt='dir-npy')
    
    scaler = RobustScaler()
    if os.path.exists('scaled_diheds'):
        os.system('rm -rf scaled_diheds')
    scaled_diheds = diheds.fit_transform_with(scaler, 'scaled_diheds/', fmt='dir-npy')
    
    tica_model = tICA(lag_time=1, n_components=tica_dim)
    tica_model = scaled_diheds.fit_with(tica_model)
    if os.path.exists('ticas'):
        os.system('rm -rf ticas')
    tica_trajs = scaled_diheds.transform_with(tica_model, 'ticas/', fmt='dir-npy')
    return tica_trajs


def clusterData(tica_trajs, micro_num):
    clusterer = MiniBatchKMeans(n_clusters=micro_num, random_state=42)

    if os.path.exists('kmeans'):
        os.system('rm -rf kmeans')
    clustered_trajs = tica_trajs.fit_transform_with(
                          clusterer, 'kmeans/', fmt='dir-npy')
    return clustered_trajs, clusterer


def drawMicroCluster():
    plt.hexbin(txx[:,0], txx[:,1], bins='log', mincnt=1, cmap='viridis')
    plt.scatter(clusterer.cluster_centers_[:,0],
                clusterer.cluster_centers_[:,1], 
                s=300, c='w')


## Define what to do for parallel execution
def at_lagtime(lt):
    msm = MarkovStateModel(lag_time=lt, n_timescales=20, verbose=False)
    msm.fit(clustered_trajs)
    ret = {
        'lag_time': lt,
        'percent_retained': msm.percent_retained_,
    }
    for i in range(msm.n_timescales):
        ret['timescale_{}'.format(i)] = msm.timescales_[i]
    return ret


def calc_ImpliedTimescale():
    with Pool() as p:
           results = p.map(at_lagtime, lagtimes)
    
    timescales = pd.DataFrame(results)
    
    n_timescales = len([x for x in timescales.columns
                        if x.startswith('timescale_')])
    return timescales, n_timescales


## Implied timescales vs lagtime
def plot_timescales():
    for i in range(n_timescales):
        plt.plot(timescales['lag_time'],
                 timescales['timescale_{}'.format(i)],
                 c=colors[0]
                )

    xmin, xmax = ax.get_xlim()
    xx = np.linspace(xmin, xmax)
    #plt.plot(xx, xx, color=colors[2], label='$y=x$')
    #plt.legend(loc='best', fontsize=14)
    plt.xlabel('Lag Time / step')
    plt.ylabel('Implied Timescales / ps')
    #ax.set_xscale('log')
    plt.yscale('log')

## Percent trimmed vs lagtime
def plot_trimmed(ax):
    ax.plot(timescales['lag_time'],
            timescales['percent_retained'],
            'o-',
            label=None,  # pandas be interfering
            )
    ax.axhline(100, color='k', ls='--', label='100%')
    ax.legend(loc='best', fontsize=14)
    ax.set_xlabel('Lag Time / todo:units', fontsize=18)
    ax.set_ylabel('Retained / %', fontsize=18)
    ax.set_xscale('log')
    ax.set_ylim((0, 110))

def draw_ImpliedTimescale():
## Plot timescales
    #matplotlib.use('Agg')
    import seaborn as sns
    #sns.set_style('ticks')
    colors = sns.color_palette()
    #fig = plt.subplots(figsize=(7, 5))
    plot_timescales()
    plt.tight_layout()
#plt.xlim([0, 400])
#plt.ylim([50, 1600])
#fig.savefig('implied-timescales.pdf')
# 

## Plot trimmed
#fig, ax = plt.subplots(figsize=(7,5))
#plot_trimmed(ax)
#fig.tight_layout()

def draw_MicroCluster_FreeEnergy():
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

    
def check_MacroNum(n_timescales):
    msme.plot_timescales(msm, n_timescales=n_timescales,
                         ylabel='Implied Timescales ($ns$)')


class MacroCluster():

    def __init__(self, pcca, macro_num):
        self.macro_num = macro_num
        self.macro_trajs = pcca.transform(clustered_trajs)
        self.microstate_mapping_ = pcca.microstate_mapping_
        self.macro_index = np.concatenate(self.macro_trajs)
        self.macro_pop = self.getPop()

    def do_lump(self, pcca):
#        pcca = PCCAPlus.from_msm(msm,
#                                 n_macrostates=self.macro_num)
# if self.pcca then will have same pcca even though using diff macro_num
        self.macro_trajs = pcca.transform(clustered_trajs)
        return pcca.microstate_mapping_

    def getPop(self):
        num = np.unique(self.macro_index).shape[0]
        pop = [np.where(self.macro_index == i)[0].shape[0] for i in range(num)]
        return np.array(pop)

    def draw_MacroCluster_FreeEnergy(self, pcca):
        cm = plt.cm.get_cmap('RdYlBu')
        msme.plot_free_energy(txx, obs=(0, 1), n_samples=10000, vmin=-0.001,
                              pi=pcca.populations_[assignments],
                              xlabel='tIC 1', ylabel='tIC 2')
        sc =plt.scatter(clusterer.cluster_centers_[pcca.state_labels_, 0],
                        clusterer.cluster_centers_[pcca.state_labels_, 1],
                        s=300,
                        c=self.microstate_mapping_,
                        zorder=3,
                        cmap=cm
                       )
        cbar = plt.colorbar(sc)
        pop_percent = self.macro_pop/self.macro_pop.sum()*100
        cluster_index = np.array(list(range(self.macro_num)))
        cbar.set_ticks(cluster_index)
        cbar.set_ticklabels(['{0} {1: .2f}%'.format(i[0]+1, i[1]) for i in zip(cluster_index, pop_percent)])
        plt.tight_layout()


def saveTrj(xyz_all, macro_index, savePth):
    macro_num = macro_index.max() + 1
    for i in range(macro_num):
        traj_index = np.where(macro_index == i)[0]
        traj = xyz_all[0][traj_index]
        traj.save_xtc(savePth+os.sep+'macro_%d.xtc' % (i+1))

xyz = dataset('RSFF/*_skip5.xtc', topology='p53_prot.pdb')
xyz_all = dataset('RSFF/combine/rsff_skip5.xtc', topology='p53_prot.pdb')
dt = 50

micro_num = 100
tica_dim = 2
tica_trajs = featurizeData(xyz, tica_dim)
txx = np.concatenate(tica_trajs)
clustered_trajs, clusterer = clusterData(tica_trajs, micro_num)
lagtimes = list(range(1, 41, 2))
drawMicroCluster()

micro_index = np.concatenate(clustered_trajs)
micro_pop = getPop(micro_index)

timescales, n_timescales = calc_ImpliedTimescale()
timescale = timescales*dt 
draw_ImpliedTimescale()
plt.ylim([5,10000])

msm = MarkovStateModel(lag_time=1, n_timescales=20)
msm.fit(clustered_trajs)
assignments = clusterer.partial_transform(txx)
assignments = msm.partial_transform(assignments)
draw_MicroCluster_FreeEnergy()

msm = MarkovStateModel(lag_time=30, n_timescales=20)
msm.fit(clustered_trajs)

for i in range(2, 11):
    fig = plt.figure(i)
    n_timescales = i
    check_MacroNum(n_timescales)
    fig.savefig('%d_timescales' % i)

msm_build_macro = MarkovStateModel(lag_time=1, n_timescales=20)
msm_build_macro.fit(clustered_trajs)
macro_clusters = {}
pcca_clusters = {}
for i in range(2, 11):
    fig = plt.figure(i)
    pcca = PCCAPlus.from_msm(msm_build_macro, i)
    pcca_clusters[i] = pcca
    macro_clusters[i] = MacroCluster(pcca, i)
    macro_clusters[i].draw_MacroCluster_FreeEnergy(pcca_clusters[i])
    fig.savefig('%d_macrostates.png' % i)

macro_num = 6
savePth = '{}_macro_trajs'.format(macro_num)
if os.path.exists(savePth):
    os.system('rm -rf {}'.format(savePth))
os.mkdir(savePth)
saveTrj(xyz_all, macro_clusters[macro_num].macro_index, savePth)