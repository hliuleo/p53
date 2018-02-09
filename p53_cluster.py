# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:33:37 2016

@author: hliu
"""

import mdtraj as md
import pandas as pd
from researchcode.plotting.plot_set import *
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
import seaborn as sns

#sns.set_style('ticks')
colors = sns.color_palette()


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


def drawMicroCluster(txx, clusterer):
    plt.hexbin(txx[:,0], txx[:,1], bins='log', mincnt=1, cmap='viridis')
    plt.scatter(clusterer.cluster_centers_[:,0],
                clusterer.cluster_centers_[:,1], 
                s=300, c='w')


## Define what to do for parallel execution
def at_lagtime(lt, clustered_trajs):
    msm = MarkovStateModel(lag_time=lt, n_timescales=20, verbose=False)
    msm.fit(clustered_trajs)
    ret = {
        'lag_time': lt,
        'percent_retained': msm.percent_retained_,
    }
    for i in range(msm.n_timescales):
        ret['timescale_{}'.format(i)] = msm.timescales_[i]
    return ret


def calc_ImpliedTimescale(lagtimes, clustered_trajs):
    with Pool() as p:
        results = p.map(lambda x: at_lagtime(x, clustered_trajs),
                        lagtimes)
    
    timescales = pd.DataFrame(results)
    
    n_timescales = len([x for x in timescales.columns
                        if x.startswith('timescale_')])
    return timescales, n_timescales


## Implied timescales vs lagtime
def plot_timescales(timescales, n_timescales):
    for i in range(n_timescales):
        plt.plot(timescales['lag_time'],
                 timescales['timescale_{}'.format(i)],
                 c=colors[0]
                )
    ax = plt.gca()
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

def draw_ImpliedTimescale(timescales, n_timescales):
## Plot timescales
    #matplotlib.use('Agg')

    #fig = plt.subplots(figsize=(7, 5))
    plot_timescales(timescales, n_timescales)
    plt.tight_layout()
#plt.xlim([0, 400])
#plt.ylim([50, 1600])
#fig.savefig('implied-timescales.pdf')
# 

## Plot trimmed
#fig, ax = plt.subplots(figsize=(7,5))
#plot_trimmed(ax)
#fig.tight_layout()

def draw_MicroCluster_FreeEnergy(txx, msm, clusterer, assignments):
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

    
def check_MacroNum(msm, n_timescales):
    msme.plot_timescales(msm, n_timescales=n_timescales,
                         ylabel='Implied Timescales ($ns$)')


class MacroCluster():

    def __init__(self, pcca, macro_num, clustered_trajs):
        self.macro_num = macro_num
        self.microstate_mapping_ = pcca.microstate_mapping_
        self.getMacroTraj(pcca, clustered_trajs)
        self.macro_pop = self.getPop()

    def getMacroTraj(self, pcca, clustered_trajs):
        self.macro_trajs = pcca.transform(clustered_trajs)
        self.macro_index = np.concatenate(self.macro_trajs)

    def getPop(self):
        num = np.unique(self.macro_index).shape[0]
        pop = [np.where(self.macro_index == i)[0].shape[0] for i in range(num)]
        return np.array(pop)

    def draw_MacroCluster_FreeEnergy(self, txx, clusterer, pcca, assignments):
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
        np.savetxt(savePth+os.sep+'macro_%d_cluster_index.dat'%(i+1),
                   macro_index,
                   fmt='%d')


def getPop(index):
    num = np.unique(index).shape[0]
    pop = [np.where(index == i)[0].shape[0] for i in range(num)]
    return np.array(pop)