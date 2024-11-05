############################################################################
#        AUTOMATED ANISOTROPIC RESISTIVITY INVERSION FOR EFFICIENT         #
#           FORMATION EVALUATION AND UNCERTAINTY QUANTIFICATION            #
############################################################################
# Author: Misael M. Morales (github.com/misaelmmorales)                    #
# Co-Authors: Oriyomi Raheem, Ali Eghbali                                  #
# Co-Authors: Dr. Michael Pyrcz, Dr. Carlos Torres-Verdin                  #
# Date: 2024                                                               #
############################################################################
# Copyright (c) 2024, Misael M. Morales                                    #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
############################################################################

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

import lasio
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy import linalg, optimize
from scipy.io import loadmat
from numdifftools import Jacobian, Hessian
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

my_box = dict(facecolor='lightgrey', edgecolor='k', alpha=0.5)

def check_torch():
    torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
    count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
    print('\n'+'-'*60)
    print('----------------------- VERSION INFO -----------------------')
    print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
    print('# Device(s) available: {}, Name(s): {}'.format(count, name))
    print('-'*60+'\n')
    return None

def load_all_data():
    column_names = ['CAL','AT10', 'AT30', 'AT60', 'AT90', 'GR', 'Rv', 'Rh']
    index_name   = 'DEPTH'
    # well 1
    well1 = lasio.read('cases/well1.las').df()
    case1 = well1[['CALI','AT10','AT30','AT60','AT90','GR','RV72H_1D_FLT','RH72H_1D_FLT']].dropna()
    case1.columns = column_names
    case1.index.name = index_name
    case1.loc[:,'Rvsh'] = 2.813
    case1.loc[:,'Rhsh'] = 0.775
    case1.loc[:,'WNAME'] = 'Field_1'
    case1.loc[:,'WIDX'] = 1
    # well 2
    well2 = lasio.read('cases/well2.LAS').df()
    case2 = well2[['HCAL','AT10','AT30','AT60','AT90','HCGR','RV72_1DF','RH72_1DF']].dropna()
    case2.columns = column_names
    case2.index.name = index_name
    case2.loc[:,'Rvsh'] = 2.78
    case2.loc[:,'Rhsh'] = 0.58
    case2.loc[:,'WNAME'] = 'Field_2'
    case2.loc[:,'WIDX'] = 2
    # synthetic 1
    synthetic1_raw = lasio.read('cases/Case1.las').df()
    synthetic1_raw = synthetic1_raw.join(lasio.read('cases/Case1_RvRh.las').df())
    synthetic1_names = ['GR','DPHI','NPHI','PEF','AT10', 'AT30', 'AT60', 'AT90', 'Rh', 'Rv']
    synthetic1 = synthetic1_raw[['ECGR','DPHI','NPHI','PEF','RF10', 'RF30', 'RF60', 'RF90', 
                            'RESISTIVITY FORMATION (UNINVADED)', 'RESISTIVITY (PERPENDICULAR) FORMATION (UNINVADED)']]
    synthetic1.columns = synthetic1_names
    synthetic1 = synthetic1.loc[5479.9:5680.1]
    synthetic1.loc[:,'Rvsh'] = 10
    synthetic1.loc[:,'Rhsh'] = 1
    synthetic1.loc[:,'WNAME'] = 'Synthetic1'
    synthetic1.loc[:,'WIDX'] = 3
    # synthetic 2
    synthetic2_raw = lasio.read('cases/Case2.las').df()
    synthetic2_names = ['GR','RHOZ','NPOR','PEFZ','Rv','Rh']
    synthetic2 = synthetic2_raw[['GR','RHOZ','NPOR','PEFZ','RD_V','RD_H']].dropna()
    synthetic2.columns = synthetic2_names
    synthetic2 = synthetic2.loc[:5195]
    synthetic2.loc[:,'Rvsh'] = 10
    synthetic2.loc[:,'Rhsh'] = 1
    synthetic2.loc[:,'WNAME'] = 'Synthetic2'
    synthetic2.loc[:,'WIDX'] = 4
    # return
    print('Name              : Source                : Shape')
    print('----------------- : --------------------- : -----------')
    print('Field Case 1      : (North Africa)        : {}'.format(case1.shape))
    print('Field Case 2      : (North Sea)           : {}'.format(case2.shape))
    print('Synthetic Case 1  : (Laminated)           : {}'.format(synthetic1.shape))
    print('Synthetic Case 2  : (Laminated+Dispersed) : {}'.format(synthetic2.shape))
    return case1, case2, synthetic1, synthetic2

def error_metrics(df):
    mse_rv = mean_squared_error(df['Rv'], df['Rv_sim'])
    mse_rh = mean_squared_error(df['Rh'], df['Rh_sim'])
    r2_rv = r2_score(df['Rv'], df['Rv_sim'])*100
    r2_rh = r2_score(df['Rh'], df['Rh_sim'])*100
    sterr_rv = np.mean(np.abs(df['Rv']-df['Rv_sim'])) / np.std(np.abs(df['Rv']-df['Rv_sim']))
    sterr_rh = np.mean(np.abs(df['Rh']-df['Rh_sim'])) / np.std(np.abs(df['Rh']-df['Rh_sim']))
    mape_rv = mean_absolute_percentage_error(df['Rv'], df['Rv_sim']) * 100
    mape_rh = mean_absolute_percentage_error(df['Rh'], df['Rh_sim']) * 100
    print('Mean Squared Error - Rv: {:.4f}  | Rh: {:.4f}'.format(mse_rv, mse_rh))
    print('R2 Score           - Rv: {:.3f}  | Rh: {:.3f}'.format(r2_rv, r2_rh))
    print('Standard Error     - Rv: {:.4f}  | Rh: {:.4f}'.format(sterr_rv, sterr_rh))
    print('MAPE               - Rv: {:.3f}%  | Rh: {:.3f}%'.format(mape_rv, mape_rh))
    return None

def quadratic_inversion(df, Rvsh=None, Rhsh=None):
    quad_inv = []
    for _, row in df.iterrows():
        Rv, Rh = row['Rv'], row['Rh']
        Rvsh, Rhsh = row['Rvsh'], row['Rhsh']
        a = Rh*Rvsh - Rh*Rhsh
        b = Rv**2 + Rvsh*Rhsh - 2*Rh*Rhsh
        c = Rv*Rhsh - Rh*Rhsh
        qsol = np.roots([a,b,c])
        if len(qsol) == 1:
            quad_inv.append({'Rss_q':qsol[0], 'Csh_q':np.nan})
        elif len(qsol) == 2:
            quad_inv.append({'Rss_q':qsol[0], 'Csh_q':qsol[1]})
        else:
            quad_inv.append({'Rss_q':np.nan, 'Csh_q':np.nan})
    return pd.DataFrame(quad_inv, index=df.index)

def newton_inversion(data, x0=[0.6, 1.2], method='hybr', tol=1e-10, maxiter=1000, clip:bool=True):
    def quad_fun(x, *args):
        Csh, Rss = x
        Rv, Rh, Rvsh, Rhsh = args[0], args[1], args[2], args[3]
        eq1 = (Csh*Rvsh + (1-Csh)*Rss) - Rv
        eq2 = (Csh/Rhsh + (1-Csh)/Rss) - (1/Rh)
        return np.array([eq1, eq2])
    def quad_jac(x, *args):
        Csh, Rss = x
        Rv, Rh, Rvsh, Rhsh = args[0], args[1], args[2], args[3]
        J11 = -(1-Csh)
        J12 = Rvsh - Rss
        J21 = -Csh/Rss**2
        J22 = 1/Rhsh - 1/Rss
        return np.array([[J11, J12],[J21, J22]])
    Csh_pred, Rss_pred = [], []
    for i in range(data.shape[0]):
        Rv, Rh, Rvsh, Rhsh = data.iloc[i][['Rv', 'Rh', 'Rvsh', 'Rhsh']]
        sol = optimize.root(quad_fun, 
                            x0      = x0, 
                            args    = (Rv, Rh, Rvsh, Rhsh), 
                            method  = method, 
                            jac     = quad_jac,
                            tol     = tol,
                            options = {'maxfev': maxiter})
        Rss_pred.append(sol.x[0])
        Csh_pred.append(np.clip(sol.x[1],0,1)) if clip else Csh_pred.append(sol.x[1])
    return pd.DataFrame(np.array([Csh_pred, Rss_pred]).T, columns=['Csh_q', 'Rss_q'], index=data.index)

def plot_curve(ax, df, curve, lb=None, ub=None, color='k', pad=0, s=2, mult=1,
            units:str=None, mask=None, offset:int=0, title:str=None, label:str=None,
            semilog:bool=False, bar:bool=False, fill:bool=None, rightfill:bool=False,
            marker=None, edgecolor=None, ls=None, alpha=None):
        if mask is None:
            x, y = -offset+mult*df[curve], df.index
        else:
            x, y = -offset+mult*df[curve][mask], df.index[mask]
        lb = x[~np.isnan(x)].min() if lb is None else lb
        ub = x[~np.isnan(x)].max() if ub is None else ub
        if semilog:
            ax.semilogx(x, y, c=color, label=curve, alpha=alpha,
                        marker=marker, markersize=s, markeredgecolor=edgecolor, linestyle=ls, linewidth=s)
        else:
            if bar:
                ax.barh(y, x, color=color, label=curve, alpha=alpha)
            else:
                ax.plot(x, y, c=color, label=curve, alpha=alpha,
                        marker=marker, markersize=s, markeredgecolor=edgecolor, linewidth=s, linestyle=ls)
        if fill:
            if rightfill:
                ax.fill_betweenx(y, x, ub, alpha=alpha, color=color)
            else:
                ax.fill_betweenx(y, lb, x, alpha=alpha, color=color)
        if units is None:
            if hasattr(df, 'curvesdict'):
                units = df.curvesdict[curve].unit
            else:
                units = ''
        ax.set_xlim(lb, ub)
        ax.grid(True, which='both')
        ax.set_title(title, weight='bold') if title != None else None
        xlab = label if label is not None else curve
        if offset != 0:
            ax.set_xlabel('{} [{}] with {} offset'.format(xlab, units, offset), color=color, weight='bold')
        else:
            ax.set_xlabel('{} [{}]'.format(xlab, units), color=color, weight='bold')
        ax.xaxis.set_label_position('top'); ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_tick_params(color=color, width=s)
        ax.spines['top'].set_position(('axes', 1+pad/100))
        ax.spines['top'].set_edgecolor(color); ax.spines['top'].set_linewidth(2)
        if ls is not None:
            ax.spines['top'].set_linestyle(ls)
        return None

def plot_uq(ax, data, data_uq, ls='-', lw=2, color='r', qcolor='gray', alpha=0.4, galpha=0.66, 
            eb_sampling=None, eb_stretch=2, semilog=False, xlim=None,
            xlabel='mean $C_{sh}$', units='v/v', title='Case N'):
    ax.fill_betweenx(data.index,
                    data_uq.max(axis=0),
                    data_uq.min(axis=0),
                    color=qcolor, alpha=alpha)
    ax.plot(data_uq.mean(axis=0), data.index, color, ls=ls, lw=lw, label='Mean')
    ax.grid(True, which='both', alpha=galpha)
    ax.invert_yaxis()
    ax.set_xscale('log') if semilog else None
    ax.set_xlim(xlim) if xlim is not None else None

    ebar_sampling = len(data)//eb_sampling
    ax.errorbar(data_uq.mean(axis=0)[::ebar_sampling],
                    data.index[::ebar_sampling],
                    xerr=eb_stretch*data_uq.std(axis=0)[::ebar_sampling],
                    fmt='.', color='k', label='Error')

    ax.set_xlabel('{} [{}]'.format(xlabel, units), color=color, weight='bold')
    ax.xaxis.set_label_position('top'); ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_tick_params(color=color, width=lw)
    ax.spines['top'].set_position(('axes', 1))
    ax.spines['top'].set_edgecolor(color); ax.spines['top'].set_linewidth(lw)
    ax.spines['top'].set_linestyle(ls)
    ax.set_title(title, weight='bold')
    return None

def hist_uq(ax, data_uq, bins=30, alpha=0.7, color='r', label='$C_{sh}$', galpha=0.66, density=False, semilog=False):
    ax.hist(data_uq.mean(axis=0), bins=bins, color=color, alpha=alpha, edgecolor='k', density=density)
    ax.set_xlabel('Mean {}'.format(label))
    ax.set_xscale('log') if semilog else None
    ax.grid(True, which='both', alpha=galpha)
    return None

def plot_inversion_solution(data, sol, ali, figsize=(16.5,10)):
    _, axs = plt.subplots(1,5, figsize=figsize, sharey=True, width_ratios=[0.6,0.6,0.6,1,1])
    ax1, ax2, ax3, ax4, ax5 = axs
    colors = ['darkviolet','royalblue','firebrick']
    ax1.plot(data['GR'], data.index, c='g', label='GR')
    ax1.grid(True, which='both')
    ax1.set(xlim=(20,120), title='GR')
    ax1.hlines(10188.75, 0, data['GR'].max(), color='k', lw=5)
    ax2.plot(data['Rv'], data.index, c='k', label='Rv')
    ax2.plot(ali['Rv_sim'], ali['df'].iloc[1:2156,0], c='b', ls='--', label='Ali_sim')
    ax2.plot(sol['Rv_sim'], sol.index, c='r', ls='--', label='Simulated')
    ax2.set(xscale='log', title='Rv')
    ax2.grid(True, which='both')
    ax2.legend(loc='upper left', facecolor='lightgrey', edgecolor='k')
    ax2.hlines(10188.75, 0, sol['Rss_pred'].max(), color='k', lw=5)
    ax3.plot(data['Rh'], data.index, c='k', label='Rh')
    ax3.plot(ali['Rh_sim'], ali['df'].iloc[1:2156,0], c='b', ls='--', label='Ali_sim')
    ax3.plot(sol['Rh_sim'], sol.index, c='r', ls='--', label='Simulated')
    ax3.set(xscale='log', title='Rh')
    ax3.grid(True, which='both')
    ax3.legend(loc='upper left', facecolor='lightgrey', edgecolor='k')
    ax3.hlines(10188.75, 0, sol['Rss_pred'].max(), color='k', lw=5)
    ax4.plot(ali['Csh'], ali['df'].iloc[1:2156,0], label='Ali_1', c='b')
    ax4.plot(ali['df'].iloc[:,14], ali['df'].iloc[:,0], label='Ali_2')
    ax4.plot(sol['Csh_pred'], sol.index, label='Mine', ls='--', c='r')
    ax4.set(title='Csh', xlim=(0,1))
    ax4.legend(loc='upper left', facecolor='lightgrey', edgecolor='k')
    ax4.grid(True, which='both')
    ax4.hlines(10188.75, 0, 1, color='k', lw=5)
    ax5.plot(ali['Rss'], ali['df'].iloc[1:2156,0], label='Ali_1', c='b')
    ax5.plot(ali['df'].iloc[:,15], ali['df'].iloc[:,0], label='Ali_2')
    ax5.plot(sol['Rss_pred'], sol.index, label='Mine', ls='--', c='r')
    ax5.set(xscale='log', title='Rss'); 
    ax5.legend(loc='upper left', facecolor='lightgrey', edgecolor='k')
    ax5.grid(True, which='both')
    ax5.hlines(10188.75, 0, sol['Rss_pred'].max(), color='k', lw=5)
    ax1.invert_yaxis()
    plt.tight_layout()
    plt.show()
    return None

def plot_inversion_comparison(sol, ali, figsize=(15,10)):
    _, axs = plt.subplots(1, 6, figsize=figsize, sharey=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axs
    ax1.plot(sol['Rv'], sol.index, c='k', lw=2.75, label='Rv_true')
    ax1.plot(sol['Rv_sim'], sol.index, c='r', ls='--', label='Rv_sim')
    ax1.set(title='Rv_sim'); ax1.grid(True, which='both'); ax1.set_xscale('log')
    ax1.legend(facecolor='lightgrey', edgecolor='k', loc='upper right')
    ax2.plot(sol['Rh'], sol.index, c='k', lw=2.75, label='Rh_true')
    ax2.plot(sol['Rh_sim'], sol.index, c='r', ls='--', label='Rh_sim')
    ax2.set(title='Rh_sim'); ax2.grid(True, which='both'); ax2.set_xscale('log')
    ax2.legend(facecolor='lightgrey', edgecolor='k', loc='upper right')
    ax3.plot(sol['Rv'], sol.index, c='k', lw=2.75, label='Rv_true')
    ax3.plot(ali['Rv_sim'], ali['df'].iloc[:2155,0], c='b', ls='--', label='ALI_Rv_sim')
    ax3.set(title='Rv_sim'); ax3.grid(True, which='both'); ax3.set_xscale('log')
    ax3.legend(facecolor='lightgrey', edgecolor='k', loc='upper right')
    ax4.plot(sol['Rh'], sol.index, c='k', lw=2.75, label='Rh_true')
    ax4.plot(ali['Rh_sim'], ali['df'].iloc[:2155,0], c='b', ls='--', label='ALI_Rh_sim')
    ax4.set(title='Rh_sim'); ax4.grid(True, which='both'); ax4.set_xscale('log')
    ax4.legend(facecolor='lightgrey', edgecolor='k', loc='upper right')
    ax5.plot(sol['Rv'], sol.index, c='k', lw=2.75, label='Rv_true')
    ax5.plot(ali['Rv_sim'], ali['df'].iloc[:2155,0], c='b', ls='--', label='ALI_Rv_sim')
    ax5.plot(sol['Rv_sim'], sol.index, c='r', ls='--', label='Rv_sim')
    ax5.set(title='Rv_sim'); ax5.grid(True, which='both'); ax5.set_xscale('log')
    ax5.legend(facecolor='lightgrey', edgecolor='k', loc='upper right')
    ax6.plot(sol['Rh'], sol.index, c='k', lw=2.75, label='Rh_true')
    ax6.plot(ali['Rh_sim'], ali['df'].iloc[:2155,0], c='b', ls='--', label='ALI_Rh_sim')
    ax6.plot(sol['Rh_sim'], sol.index, c='r', ls='--', label='Rh_sim')
    ax6.set(title='Rh_sim'); ax6.grid(True, which='both'); ax6.set_xscale('log')
    ax6.legend(facecolor='lightgrey', edgecolor='k', loc='upper right')
    for ax in axs:
        ax.hlines(10188.75, 0, sol['Rss_pred'].max(), color='k', lw=3.5)
    ax1.invert_yaxis()
    plt.tight_layout()
    plt.show()
    return None

def plot_inversion_fullsuite(data, sol, ali_sol, figsize=(17.5,10)):
    df = lasio.read('well1.las').df()
    _, axs = plt.subplots(1, 10, figsize=figsize, sharey=True)
    ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axs
    ax01, ax02 = ax0.twiny(), ax0.twiny()
    plot_curve(ax0, df, 'CALI', 12, 24, 'dodgerblue', units='in', fill=True, semilog=True, pad=0)
    plot_curve(ax01, df, 'GR', 10, 150, 'g', units='API', pad=8)
    plot_curve(ax02, sol, 'Csh_pred', 0, 1, 'r', units='v/v', s=1, pad=16)
    ax11, ax12 = ax1.twiny(), ax1.twiny()
    plot_curve(ax1, data, 'Csh_lin', 0, 1, 'k', units='v/v', pad=0)
    plot_curve(ax11, data, 'Csh_q', 0, 1, 'gray', units='v/v', pad=8)
    plot_curve(ax12, sol, 'Csh_pred', 0, 1, 'r', units='v/v', pad=16)
    ax21, ax22 = ax2.twiny(), ax2.twiny()
    plot_curve(ax2, df, 'TNPH', 1, 0, 'b', units='PU', pad=0)
    plot_curve(ax21, df, 'RHOZ', 1.65, 2.65, 'maroon', units='g/cc', pad=8)
    plot_curve(ax22, df, 'PE', -5, 5, 'm', units='b/e', pad=16)
    ax31, ax32 = ax3.twiny(), ax3.twiny()
    plot_curve(ax3, df, 'AT10', 0.2, 100, 'darkred', semilog=True, units='$\Omega\cdot m$', pad=0)
    plot_curve(ax31, df, 'AT90', 0.2, 100, 'darkblue', semilog=True, units='$\Omega\cdot m$', pad=8)
    plot_curve(ax32, sol, 'Rss_pred', 0.2, 100, 'r', semilog=True, units='$\Omega\cdot m$', pad=16)
    ax41 = ax4.twiny()
    plot_curve(ax4, ali_sol, 'Csh_ALI', 0, 1, 'b', units='v/v', pad=0)
    plot_curve(ax41, sol, 'Csh_pred', 0, 1, 'r', ls='--', units='v/v', pad=8)
    ax51 = ax5.twiny()
    plot_curve(ax5, ali_sol, 'Rss_ALI', 0.2, 100, 'b', semilog=True, units='$\Omega\cdot m$', pad=0)
    plot_curve(ax51, sol, 'Rss_pred', 0.2, 100, 'r', ls='--', semilog=True, units='$\Omega\cdot m$', pad=8)
    ax61, ax62 = ax6.twiny(), ax6.twiny()
    plot_curve(ax6, df, 'RV72H_1D_FLT', 0.2, 100, 'k', s=4, semilog=True, units='$\Omega\cdot m$', pad=0)
    plot_curve(ax61, ali_sol, 'Rv_sim_ALI', 0.2, 100, 'b', semilog=True, units='$\Omega\cdot m$', pad=8)
    plot_curve(ax62, sol, 'Rv_sim', 0.2, 100, 'r', ls='--', semilog=True, units='$\Omega\cdot m$', pad=16)
    ax71, ax72 = ax7.twiny(), ax7.twiny()
    plot_curve(ax7, df, 'RH72H_1D_FLT', 0.2, 100, 'k', s=4, semilog=True, units='$\Omega\cdot m$', pad=0)
    plot_curve(ax71, ali_sol, 'Rh_sim_ALI', 0.2, 100, 'b', semilog=True, units='$\Omega\cdot m$', pad=8)
    plot_curve(ax72, sol, 'Rh_sim', 0.2, 100, 'r', ls='--', semilog=True, units='$\Omega\cdot m$', pad=16)
    ax81, ax82, ax83 = ax8.twiny(), ax8.twiny(), ax8.twiny()
    plot_curve(ax8, data, 'Rv', 0.2, 100, 'k', s=4, semilog=True, units='$\Omega\cdot m$', pad=0)
    plot_curve(ax81, sol, 'Rv_sim', 0.2, 100, 'darkred', ls='--', semilog=True, units='$\Omega\cdot m$', pad=8)
    plot_curve(ax82, data, 'Rh', 0.2, 100, 'k', s=4, semilog=True, units='$\Omega\cdot m$', pad=16)
    plot_curve(ax83, sol, 'Rh_sim', 0.2, 100, 'darkblue', ls='--', semilog=True, units='$\Omega\cdot m$', pad=24)
    ax91, ax92 = ax9.twiny(), ax9.twiny()
    plot_curve(ax9, sol, 'fun', 1e-6, 1e0, 'k', s=1, semilog=True, units='/', pad=0)
    plot_curve(ax91, sol, 'jac_norm', 0, 2.5, 'darkmagenta', s=1, units='/', pad=8, alpha=0.5)
    plot_curve(ax92, sol, 'nfev', 10, 150, 'darkgreen', s=1, units='/', pad=16, alpha=0.5)
    ax1.set_ylim(10190, 9650)
    plt.tight_layout()
    plt.show()
    return None

def plot_short_results(data, sol, figsize=(6,8), cfactor=5):
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)
    ax1, ax2, ax3 = axs
    c = cfactor*(data['CALI'] - data['CALI'].mean()) / data['CALI'].std()
    ax1.plot(data['GR'], data.index, c='g', label='GR')
    ax1.plot(c, data.index, c='dodgerblue', label='CALI')
    ax2.plot(data['Rv'], data.index, c='k', label='Rv')
    ax2.plot(sol['Rv_sim'], sol.index, c='r', ls='--', label='Rv_sim')
    ax2.plot(data['Rh'], data.index, c='k', label='Rh')
    ax2.plot(sol['Rh_sim'], sol.index, c='b', ls='--', label='Rh_sim')
    ax2.set_xscale('log')
    ax3.plot(sol['Csh_pred'], sol.index, c='k', label='Csh_pred')
    for ax in axs:
        ax.grid(True, which='both')
        ax.legend(loc='upper right', facecolor='lightgrey', edgecolor='k')
    ax1.invert_yaxis()
    plt.tight_layout()
    plt.show()
    return None

def plot_crossplot(sol, figsize=(10,4), cmap='jet', alpha=0.66, vlim:tuple=(0.2,100), hlim:tuple=(0.25,10), axlim:list=[0.1,100]):
    fig, axs = plt.subplots(1, 3, figsize=figsize, width_ratios=[1,1,0.1])
    ax1, ax2, cax = axs
    im1 = ax1.scatter(sol['Rv'], sol['Rv_sim'], c=sol.index, alpha=alpha, cmap=cmap)
    im2 = ax2.scatter(sol['Rh'], sol['Rh_sim'], c=sol.index, alpha=alpha, cmap=cmap)
    r2_rv = r2_score(sol['Rv'], sol['Rv_sim'])*100
    r2_rh = r2_score(sol['Rh'], sol['Rh_sim'])*100
    for i, ax in enumerate([ax1, ax2]):
        ax.plot(axlim, axlim, 'k--')
        ax.set(xscale='log', yscale='log', xlabel='Measured', ylabel='Simulated', title=['$R_v$', '$R_h$'][i])
        ax.grid(True, which='both')
        ax.text(3, 1.25, ['$R^2$: {:.2f}%'.format(r2_rv), '$R^2$: {:.2f}%'.format(r2_rh)][i], bbox=my_box)
    ax1.set(xlim=vlim, ylim=vlim)
    ax2.set(xlim=hlim, ylim=hlim)
    cb = plt.colorbar(im1, cax=cax); cb.set_label('Depth [ft]', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.show()
    return None

def plot_loss(losses, figsize=(5,3)):
    train_loss, valid_loss = losses
    epochs = len(train_loss)
    plt.figure(figsize=figsize)
    plt.plot(range(epochs), train_loss, label='Trianing', c='tab:blue')
    plt.plot(range(epochs), valid_loss, label='Validation', c='tab:orange')
    plt.legend(facecolor='lightgrey', edgecolor='k')
    plt.grid(True, which='both')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.xlim(-2, epochs+2)
    plt.tight_layout()
    plt.show()
    return None

def plot_pinn_results(results, figsize=(12,12), height_ratios=[1, 0.3], suptitle:str=None,
                      gr_lim=[0,150], res_lim=[0.2,50], r_lim=[0.15,120], h_lim=[0.2,10],
                      csh_c='k', rss_c='k', bins=50, cmaps=['Reds','Blues'],
                      at_flag:bool=True):

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig, height_ratios=height_ratios)

    ax11 = fig.add_subplot(gs[0, 0]); ax11.set(ylabel='Depth [ft]')
    ax12 = fig.add_subplot(gs[0, 1])
    ax13 = fig.add_subplot(gs[0, 2])
    ax14 = fig.add_subplot(gs[0, 3])

    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])
    ax23 = fig.add_subplot(gs[1, 2])
    ax24 = fig.add_subplot(gs[1, 3])

    ax11b = ax11.twiny()
    plot_curve(ax11, results, 'GR', gr_lim[0], gr_lim[1], 'g', units='API', pad=8)
    plot_curve(ax11b, results, 'Csh_pred', 0, 1, csh_c, units='v/v')

    ax12b, ax12c = ax12.twiny(), ax12.twiny()
    if at_flag:
        plot_curve(ax12, results, 'AT10', res_lim[0], res_lim[1], 'r', semilog=True, units='$\Omega\cdot m$', pad=8)
        plot_curve(ax12b, results, 'AT90', res_lim[0], res_lim[1], 'b', semilog=True, units='$\Omega\cdot m$', pad=16)
    else:
        plot_curve(ax12, results, 'RHOZ', 1.81, 2.81, 'r', units='g/cc', pad=8)
        plot_curve(ax12b, results, 'NPOR', 0.6, 0.0, 'b', units='PU', pad=16)
    plot_curve(ax12c, results, 'Rss_pred', res_lim[0], res_lim[1], rss_c, units='$\Omega\cdot m$', semilog=True)

    ax13b, ax13c = ax13.twiny(), ax13.twiny()
    plot_curve(ax13, results, 'Rv', res_lim[0], res_lim[1], 'r', semilog=True, units='$\Omega\cdot m$', pad=8)
    plot_curve(ax13b, results, 'Rv_sim', res_lim[0], res_lim[1], 'k', units='$\Omega\cdot m$', semilog=True)
    plot_curve(ax13c, results, 'Rv_err', 0, 100, 'darkred', alpha=0.5, units='%', pad=16)

    ax14b, ax14c = ax14.twiny(), ax14.twiny()
    plot_curve(ax14, results, 'Rh', res_lim[0], res_lim[1], 'b', semilog=True, units='$\Omega\cdot m$', pad=8)
    plot_curve(ax14b, results, 'Rh_sim', res_lim[0], res_lim[1], 'k', units='$\Omega\cdot m$', semilog=True)
    plot_curve(ax14c, results, 'Rh_err', 0, 100, 'darkblue', alpha=0.5, units='%', pad=16)

    [ax.grid(True, which='both') for ax in [ax11, ax12, ax13, ax14]]
    [ax.invert_yaxis() for ax in [ax11, ax12, ax13, ax14]]

    ax21.hist(results['Csh_pred'], bins=bins, color=csh_c, edgecolor='k', alpha=0.6, density=True)
    ax21.set(xlabel='Csh', ylabel='Density', xlim=(0,1))

    ax22.hist(results['Rss_pred'], bins=bins, color=rss_c, edgecolor='k', alpha=0.6, density=True)
    ax22.set(xlabel='Rss [$\Omega\cdot m$]', ylabel='Density')

    ax23.scatter(results['Rv'], results['Rv_sim'], c=results.index, cmap=cmaps[0], edgecolor='gray', alpha=0.6)
    ax23.set(xlabel='Rv true [$\Omega\cdot m$]', ylabel='Rv simulated [$\Omega\cdot m$]',
                xlim=r_lim, ylim=r_lim, xscale='log', yscale='log')
    ax23.plot(r_lim, r_lim, 'k--')

    ax24.scatter(results['Rh'], results['Rh_sim'], c=results.index, cmap=cmaps[1], edgecolor='gray', alpha=0.6)
    ax24.set(xlabel='Rh true [$\Omega\cdot m$]', ylabel='Rh simulated [$\Omega\cdot m$]',
                xlim=h_lim, ylim=h_lim, xscale='log', yscale='log')
    ax24.plot(h_lim, h_lim, 'k--')

    [ax.grid(True, which='both', alpha=0.4) for ax in [ax21, ax22, ax23, ax24]]
    
    fig.suptitle(suptitle, weight='bold', fontsize=14) if suptitle else None
    plt.tight_layout()
    plt.show()
    return None

def plot_pinn_gb_comparison(pinn_results, gb_results, figsize=(12,12), suptitle:str=None,
                            height_ratios=[1, 0.3], gr_lim=[0,150], res_lim=[0.2,50], at_flag:bool=False,
                            pinn_c='k', gb_c='dimgrey', cmaps=['Greens', 'Oranges', 'Reds','Blues']):

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig, height_ratios=height_ratios)

    if 'RHOZ' in pinn_results.columns:
        rho = 'RHOZ'
    elif 'RHOB' in pinn_results.columns:
        rho = 'RHOB'
    else:
        rho = 'DPHI'

    if 'TNPH' in pinn_results.columns:
        nph = 'TNPH'
    elif 'NPHI' in pinn_results.columns:
        nph = 'NPHI'
    else:
        nph = 'DPHI'

    ax11 = fig.add_subplot(gs[0, 0]); ax11.set(ylabel='Depth [ft]')
    ax12 = fig.add_subplot(gs[0, 1])
    ax13 = fig.add_subplot(gs[0, 2])
    ax14 = fig.add_subplot(gs[0, 3])

    ax21 = fig.add_subplot(gs[1, 0]); ax21.set(xlim=(0,1), ylim=(0,1))
    ax22 = fig.add_subplot(gs[1, 1]); ax22.set(xlim=res_lim, ylim=res_lim)
    ax23 = fig.add_subplot(gs[1, 2]); ax23.set(xlim=res_lim, ylim=res_lim)
    ax24 = fig.add_subplot(gs[1, 3]); ax24.set(xlim=res_lim, ylim=res_lim)

    axs = [ax11, ax12, ax13, ax14, ax21, ax22, ax23, ax24]

    ax11b, ax11c = ax11.twiny(), ax11.twiny()
    plot_curve(ax11, pinn_results, 'GR', gr_lim[0], gr_lim[1], 'g', units='API')
    plot_curve(ax11b, gb_results, 'Csh_pred', 0, 1, gb_c, ls='--', units='v/v', pad=8)
    plot_curve(ax11c, pinn_results, 'Csh_pred', 0, 1, pinn_c, ls='--', units='v/v', pad=16)

    ax12b, ax12c, ax12d = ax12.twiny(), ax12.twiny(), ax12.twiny()
    if at_flag:
        plot_curve(ax12, pinn_results, 'AT10', res_lim[0], res_lim[1], 'r', semilog=True, units='$\Omega\cdot m$', pad=8)
        plot_curve(ax12b, pinn_results, 'AT90', res_lim[0], res_lim[1], 'b', semilog=True, units='$\Omega\cdot m$', pad=16)
    else:
        plot_curve(ax12, pinn_results, rho, 1.81, 2.81, 'r', units='g/cc')
        plot_curve(ax12b, pinn_results, nph, 0.6, 0.0, 'b', units='PU', pad=8)
    plot_curve(ax12c, gb_results, 'Rss_pred', res_lim[0], res_lim[1], gb_c, units='$\Omega\cdot m$', semilog=True, pad=16)
    plot_curve(ax12d, pinn_results, 'Rss_pred', res_lim[0], res_lim[1], pinn_c, semilog=True, units='$\Omega\cdot m$', pad=24)

    ax13b, ax13c = ax13.twiny(), ax13.twiny()
    plot_curve(ax13, pinn_results, 'Rv', res_lim[0], res_lim[1], 'r', semilog=True, units='$\Omega\cdot m$')
    plot_curve(ax13b, gb_results, 'Rv_sim', res_lim[0], res_lim[1], gb_c, ls='--', semilog=True, units='$\Omega\cdot m$', pad=8)
    plot_curve(ax13c, pinn_results, 'Rv_sim', res_lim[0], res_lim[1], pinn_c, ls='--', semilog=True, units='$\Omega\cdot m$', pad=16)

    ax14b, ax14c = ax14.twiny(), ax14.twiny()
    plot_curve(ax14, pinn_results, 'Rh', res_lim[0], res_lim[1], 'b', semilog=True, units='$\Omega\cdot m$')
    plot_curve(ax14b, gb_results, 'Rh_sim', res_lim[0], res_lim[1], gb_c, ls='--', semilog=True, units='$\Omega\cdot m$', pad=8)
    plot_curve(ax14c, pinn_results, 'Rh_sim', res_lim[0], res_lim[1], pinn_c, ls='--', semilog=True, units='$\Omega\cdot m$', pad=16)

    ax21.scatter(pinn_results['Csh_pred'], gb_results['Csh_pred'], c=pinn_results.index, cmap=cmaps[0], edgecolor='gray', alpha=0.6)
    ax22.scatter(pinn_results['Rss_pred'], gb_results['Rss_pred'], c=pinn_results.index, cmap=cmaps[1], edgecolor='gray', alpha=0.6)
    ax23.scatter(pinn_results['Rv_sim'], gb_results['Rv_sim'], c=pinn_results.index, cmap=cmaps[2], edgecolor='gray', alpha=0.6)
    ax24.scatter(pinn_results['Rh_sim'], gb_results['Rh_sim'], c=pinn_results.index, cmap=cmaps[3], edgecolor='gray', alpha=0.6)

    [ax.invert_yaxis() for ax in [ax11, ax12, ax13, ax14]]
    [ax.grid(True, which='both', alpha=0.4) for ax in axs]
    [ax.set(xlabel='PINN', ylabel='Gradient-Based') for ax in [ax21, ax22, ax23, ax24]]
    [ax.set(xscale='log', yscale='log') for ax in axs[-3:]]
    [ax.axline((0,0), (1,1), c='k', ls='--') for ax in axs[-4:]]

    fig.suptitle(suptitle, weight='bold', fontsize=14) if suptitle else None
    plt.tight_layout()
    plt.show()
    return None
