# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 16:10:31 2016

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


os.chdir(os.environ['p53']+os.sep+'analyzable_data')

struct_funct = {'ss': lambda x: md.compute_dssp(x),
                'rg': lambda x: md.compute_rg(x),
                'heli': lambda x: calSSPercent(x, 'H'),
                'beta': lambda x: calSSPercent(x, 'E'),
                'rmsd': lambda x: rmsds[x.name]
              }


def addProperty2Traj(traj, props):
    for key in props:
        if not hasattr(traj, key):
            setattr(traj, key, props[key](traj))
        else:
            continue
    return trajs


def getTraj(trajNameType, topFile):
    trajs = []
    trajNames = glob.glob(trajNameType)
    trajNames.sort()
    for n in trajNames:
        trajs.append(md.load(n, top=topFile))
    return trajs


def calSSPercent(traj, ss_type):
    if not hasattr(traj, 'ss'):
       traj.ss = md.compute_ss(traj)
    percent =  np.where(traj.ss==ss_type, 1, 0).mean(axis=0)*100
    return percent


def ave(prop, skip):
    rep = prop.shape[0]/skip
    prop = prop[: rep*skip].reshape((rep, skip))
    prop = prop.mean(axis=1)
    return prop


rmsd_dir = '/home/hliu/Project/p53/analysis/RMSD_next'
rmsd_files = glob.glob1(rmsd_dir, '*.dat')
rmsd_files.sort()
rmsds = {}
for idx, rmsd_f in enumerate(rmsd_files):
    rmsds['traj %d' % (idx+1)] = np.loadtxt(rmsd_dir+os.sep+rmsd_f)


for idx, traj in enumerate(trajs):
    traj.name = 'traj %d' % (idx+1)
    addProperty2Traj(traj, struct_funct)


def plot_Time(trajs, prop):
    skip = 1
    for traj in trajs:
        y = ave(getattr(traj, prop), skip)
        x = np.arange(len(y))*skip
        plt.plot(x, y, label=traj.name)
    l = plt.legend(ncol=3, bbox_to_anchor=(0, 1.02, 1, 0.102), loc=3, mode='expand', borderaxespad=0.)
    l.get_frame().set_linewidth(mpl.rcParams['axes.linewidth'])
    plt.xlabel('Simulation steps')
    
    if prop == 'rg':
        plt.ylabel('Rg (nm)')
    elif prop == 'rmsd':
        plt.ylabel('RMSD (A)')   


def plot_Hist(trajs, prop):

    def to_percentage(y, position):
        s = str(100*y)
        return s+'%'

    binwidth = 100
    histtype = 'step'
    for traj in trajs:
        prop_value = getattr(traj, prop)
        w = np.ones_like(prop_value)/len(prop_value)
        plt.hist(prop_value, binwidth, histtype=histtype, label=traj.name, weights=w)
    fmt = FuncFormatter(to_percentage)
    plt.gca().yaxis.set_major_formatter(fmt)
    plt.legend(frameon=False)
    if prop == 'rg':
        plt.ylabel('Percentage')
        plt.xlabel('Rg (nm)')
    elif prop == 'rmsd':
        plt.ylabel('Percentage')
        plt.xlabel('RMSD (A)')        


def plot_SSPercent(trajs, ss_type):
    for traj in trajs:
        y = getattr(traj, ss_type)
        x = np.arange(len(y))+1
        plt.plot(x, y, label=traj.name)
    if ss_type == 'heli':
        plt.ylabel('Helicity (%)')
    elif ss_type == 'beta':
        plt.ylabel('Beta content (%)')
    plt.xlabel('Residue #')
    plt.legend(frameon=False)
    fmt = FuncFormatter(lambda y, position: str(y)+'%')
    plt.gca().yaxis.set_major_formatter(fmt)
