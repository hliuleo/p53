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

os.chdir(os.environ['p53']+os.sep+'analyzable_data')

struct_funct = {'ss': lambda x: md.compute_dssp(x),
                'rg': lambda x: md.compute_rg(x),
                'heli': lambda x: _calHeli(x),
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


def _calHeli(traj):
    if not hasattr(traj, 'ss'):
       traj.ss = md.compute_ss(traj)
    heli =  np.where(traj.ss=='H', 1, 0).mean(axis=0)*100
    return heli


def getHeli(traj):
    if hasattr(traj, 'heli'):
        return traj.heli
    else:
        traj.heli = traj._calHeli(traj)


def ave(prop, skip):
    rep = prop.shape[0]/skip
    prop = prop[: rep*skip].reshape((rep, skip))
    prop = prop.mean(axis=0)
    return prop


for idx, traj in enumerate(trajs):
    traj.name = 'traj %d' % (idx+1)
    addProperty2Traj(traj, struct_prop)


def plot_Rg(trajs):
    for traj in trajs:
        plt.plot()
        
