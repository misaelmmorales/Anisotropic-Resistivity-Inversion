############################################################################
#        AUTOMATED ANISOTROPIC RESISTIVITY INVERSION FOR EFFICIENT         #
#           FORMATION EVALUATION AND UNCERTAINTY QUANTIFICATION            #
############################################################################
# Author: Misael M. Morales (github.com/misaelmmorales)                    #
# Co-Authors: Oriyomi Raheem, Ali Eghbali - UT Austin                      #
# Co-Authors: Dr. Michael Pyrcz, Dr. Carlos Torres-Verdin - UT Austin      #
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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

import lasio
import pywt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import linalg, optimize
from scipy.io import loadmat
from numdifftools import Jacobian, Hessian
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

class ARI:
    def __init__(self):
        self.verbose     = True
        self.save_data   = True
        self.return_data = True
        self.check_torch_gpu()

        self.n_ensemble = 100
        self.noise_lvl  = 10


    def check_torch_gpu(self):
        '''
        Check if Torch is successfully built with GPU support
        '''
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        if self.verbose:
            print('\n'+'-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
            print('# Device(s) available: {}, Name(s): {}'.format(count, name))
            print('-'*60+'\n')
        return None
    
    def load_data(self):
        column_names = ['AT10', 'AT30', 'AT60', 'AT90', 'GR', 'Rv', 'Rh']
        index_name   = 'DEPTH'
        # well 1
        well1 = lasio.read('well1.las').df()
        case1 = well1[['AT10','AT30','AT60','AT90','GR','RV72H_1D_FLT','RH72H_1D_FLT']].dropna() #['RV72H_1D','RH72H_1D']
        case1.columns = column_names
        case1.index.name = index_name
        # well 2
        well2 = lasio.read('well2.LAS').df()
        case2 = well2[['AT10','AT30','AT60','AT90','HCGR','RV72_1DF','RH72_1DF']].dropna()
        case2.columns = column_names
        case2.index.name = index_name
        return case1, case2
    
    def plot_curve(self, ax, df, curve, lb=None, ub=None, color='k', pad=0, s=2, mult=1,
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

    ###############################################################################################################
    
    def resistivity_inversion(self, df, Rvsh=None, Rhsh=None, bounds=[(0,1), (None,None)],
                              method:str='L-BFGS-B', x0=[0.5,1.5], Wd_matrix:bool=True, 
                              lambda_reg=1e-5, tolerance=1e-3, maxiter:int=100):

        if Rvsh is None:
            Rvsh = df['Rv'].iloc[np.argmax(df['GR'])]
        if Rhsh is None:
            Rhsh = df['Rh'].iloc[np.argmax(df['GR'])]

        def objective(variables, *args):
            Csh, Rs = variables
            Rv,  Rh = args[0], args[1]
            eq1 = (Csh*Rvsh + (1-Csh)*Rs) - Rv
            eq2 = (Csh/Rhsh + (1-Csh)/Rs) - (1/Rh)
            eqs = [eq1/Rv, eq2*Rh] if Wd_matrix else [eq1, eq2]
            return linalg.norm(eqs,2) + lambda_reg*linalg.norm(variables,2)
        
        def inversion():
            res_aniso = df[['Rv','Rh']]
            sol, fun, jac, nfev = [], [], [], []
            for _, row in res_aniso.iterrows():
                Rv_value, Rh_value = row['Rv'], row['Rh']
                solution = optimize.minimize(objective,
                                            x0      = x0,
                                            args    = (Rv_value, Rh_value),
                                            bounds  = bounds,
                                            method  = method,
                                            tol     = tolerance,
                                            options = {'maxiter':maxiter})
                fun.append(solution.fun); jac.append(solution.jac); nfev.append(solution.nfev)
                jac1, jac2 = np.array(jac)[:,0], np.array(jac)[:,1]
                sol.append({'Rv':Rv_value, 'Rh':Rh_value, 
                            'Csh':solution.x[0], 'Rs':solution.x[1]})
            sol = pd.DataFrame(sol, index=res_aniso.index)
            sol['fun'],  sol['nfev']                  = fun,  nfev, 
            sol['jac1'], sol['jac2'], sol['norm_jac'] = jac1, jac2, linalg.norm(jac, axis=1)
            return sol
        
        def simulate(sol):
            Csh, Rs = sol['Csh'], sol['Rs']
            Rv_sim  = Csh*Rvsh + (1-Csh)*Rs
            Rh_sim  = Csh/Rhsh + (1-Csh)/Rs
            sim = pd.DataFrame({'Rv_sim':Rv_sim, 'Rh_sim':1/Rh_sim}, index=sol.index)
            return sim
        
        def error(sol, sim):
            Rv_true, Rh_true = sol['Rv'], sol['Rh']
            Rv_pred, Rh_pred = sim['Rv_sim'], sim['Rh_sim']
            Rv_err = np.abs((Rv_pred - Rv_true) / Rv_true) * 100
            Rh_err = np.abs((Rh_pred - Rh_true) / Rh_true) * 100
            res = pd.DataFrame({'Rv_err':Rv_err, 'Rh_err':Rh_err}, index=sol.index)
            return res

        df['Csh_lin'] = (df['GR'] - df['GR'].min()) / (df['GR'].max() - df['GR'].min())
        val = df[['AT10','AT30','AT60','AT90','GR','Csh_lin']]
        sol = inversion()
        sim = simulate(sol)
        err = error(sol, sim)
        res_inv = val.join(sol).join(sim).join(err)
        return res_inv
    
    def quadratic_inversion(self, df, Rvsh=None, Rhsh=None):
        quad_inv = []
        if Rvsh is None:
            Rvsh = df['Rv'].iloc[np.argmax(df['GR'])]
        if Rhsh is None:
            Rhsh = df['Rh'].iloc[np.argmax(df['GR'])]
        for _, row in df.iterrows():
            Rv, Rh = row['Rv'], row['Rh']
            a = Rh*Rvsh - Rh*Rhsh
            b = Rv**2 + Rvsh*Rhsh - 2*Rh*Rhsh
            c = Rv*Rhsh - Rh*Rhsh
            qsol = np.roots([a,b,c])
            if len(qsol) == 1:
                quad_inv.append({'Csh_q':qsol[0], 'Rss_q':np.nan})
            elif len(qsol) == 2:
                quad_inv.append({'Csh_q':qsol[0], 'Rss_q':qsol[1]})
            else:
                quad_inv.append({'Csh_q':np.nan, 'Rss_q':np.nan})
        self.quad_inv = pd.DataFrame(quad_inv, index=df.index)
        return self.quad_inv
    
    def inversion_uq(self, df):
        '''
        Inversion with uncertainty quantification
        '''
        realizations = []
        sigma_v, sigma_h = np.std(df['Rv']), np.std(df['Rh'])
        for _ in range(self.n_ensemble):
            df_noise = df.copy()
            # Add noise to the data
            e = np.random.normal(0, 1, df.shape[0])
            df_noise['Rv'] = df['Rv'] + e*(self.noise_lvl/100)*sigma_v
            df_noise['Rh'] = df['Rh'] + e*(self.noise_lvl/100)*sigma_h
            # Invert the noisy data
            df_inv = self.resistivity_inversion(df_noise)
            realizations.append(df_inv)
        return realizations
    
    def process_and_plot_uq(self, case, case_uq, figsize=(5,10), lw=0.1, alpha=1):
        csh_ensemble = {}
        for i in range(self.n_ensemble):
            csh_ensemble[i] = case_uq[i]['Csh'].values
        csh_ensemble = np.array(list(csh_ensemble.values()))
        print('Ensemble shape: {}'.format(csh_ensemble.shape))
        plt.figure(figsize=figsize)
        for i in range(self.n_ensemble):
            plt.plot(csh_ensemble[i], case.index, 'k', lw=lw, alpha=alpha)
        plt.gca().invert_yaxis()
        plt.tight_layout(); plt.show()
        return None
    
    def plot_inversion_results(self, inv, figsize=(25,12), savefig=True):
        _, axs = plt.subplots(1, 5, figsize=figsize, sharey=True, facecolor='white')
        ax1, ax2, ax3, ax4, ax5 = axs
        ax11, ax12 = ax1.twiny(), ax1.twiny()
        self.plot_curve(ax1,  inv, 'GR',      lb=0, ub=150, color='g',    units='API',  pad=0)
        self.plot_curve(ax11, inv, 'Csh_lin', lb=0, ub=1,   color='gray', units='frac', pad=8)
        self.plot_curve(ax12, inv, 'Csh',     lb=0, ub=1,   color='k',    units='frac', pad=16)
        ax21, ax22 = ax2.twiny(), ax2.twiny()
        self.plot_curve(ax2,  inv, 'AT10', lb=0.2, ub=200, color='r', units='$\Omega.m$', semilog=True, pad=0)
        self.plot_curve(ax21, inv, 'AT90', lb=0.2, ub=200, color='b', units='$\Omega.m$', semilog=True, pad=8)
        self.plot_curve(ax22, inv, 'Rs',   lb=0.2, ub=200, color='k', units='$\Omega.m$', alpha=0.75, semilog=True, pad=16)
        ax31, ax32 = ax3.twiny(), ax3.twiny()
        self.plot_curve(ax3,  inv, 'Rv',     lb=0.2, ub=100, color='darkred',  units='$\Omega.m$',   semilog=True, pad=0)
        self.plot_curve(ax31, inv, 'Rv_err', lb=1e-9, ub=100, color='red', units='%', alpha=0.5, pad=8)
        self.plot_curve(ax32, inv, 'Rv_sim', lb=0.2, ub=100, color='k', units='$\Omega.m$', ls='--', alpha=0.75, semilog=True, pad=16)
        ax41, ax42 = ax4.twiny(), ax4.twiny()
        self.plot_curve(ax4,  inv, 'Rh',     lb=0.2, ub=100, color='darkblue',  units='$\Omega.m$',  semilog=True, pad=0)
        self.plot_curve(ax41, inv, 'Rh_err', lb=1e-9, ub=100, color='blue', units='%', alpha=0.5, pad=8)
        self.plot_curve(ax42, inv, 'Rh_sim', lb=0.2, ub=100, color='k', units='$\Omega.m$', alpha=0.75, ls='--', semilog=True, pad=16)
        ax51, ax52 = ax5.twiny(), ax5.twiny()
        self.plot_curve(ax5,  inv, 'fun',      lb=0, ub=0.6,  color='k', pad=0)
        self.plot_curve(ax51, inv, 'nfev',     lb=50, ub=350, color='g', alpha=0.75, pad=8)
        self.plot_curve(ax52, inv, 'norm_jac', lb=0,  ub=30,  color='m', alpha=0.75, pad=16)
        ax1.set_ylabel('Depth [ft]')
        plt.gca().invert_yaxis()
        plt.savefig('figures/inversion_results_{}.png'.format(str(time.time()).split('.')[0]), dpi=300) if savefig else None
        plt.show()
        return None
    
    def plot_inversion_animation(self, data, objective, R_min=1e-1, R_max=1e2, levels=50, skips=8, figsize=(8,5)):
        xx, yy = np.meshgrid(np.linspace(-0.05,1.05,levels), np.linspace(R_min,R_max,levels))
        fig, axs = plt.subplots(1,3,figsize=figsize, width_ratios=[0.25, 1, 0.05])
        ax1, ax2, ax3 = axs
        def animate(frame):
            k = frame*skips
            constants = data[['Rv','Rh','Rvsh_win','Rhsh_win']].iloc[k].values
            zz = np.array([objective([x,y], *constants) for x,y in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)
            ax1.clear()
            ax1.plot(data['GR'], data.index, 'g', lw=1)
            ax1.hlines(data.index[k], xmin=20, xmax=120, colors='r')
            ax1.xaxis.set_ticks_position('top')
            ax1.set(xlim=(20,120), ylabel='Depth [ft]')
            ax1.set_title('GR [API]', color='g')
            ax1.set_xlabel('z={:.2f} ft'.format(data.index[k]), color='r')
            ax1.grid(True, which='both')
            ax1.invert_yaxis()
            ax2.clear()
            im = ax2.contourf(xx, yy, zz, levels=levels, cmap='jet')
            ax2.contour(xx, yy, zz, levels=levels, colors='k', linewidths=0.5)
            ax2.vlines([[0,1]], ymin=R_min, ymax=R_max, colors='w', linestyles='--', linewidth=1)
            ax2.hlines([R_min, R_max], xmin=0, xmax=1, colors='w', linestyles='--', linewidth=1)
            ax2.set(yscale='log', title='Objective Function | z={:.2f} ft'.format(data.index[k]))
            ax2.set_xlabel('Csh [v/v]', weight='bold')
            ax2.set_ylabel('Rss [$\Omega\cdot m$]', weight='bold')
            cb=fig.colorbar(im, cax=ax3); cb.set_label('Objective Function', rotation=270, labelpad=15)
            plt.tight_layout()
            return fig
        ani = FuncAnimation(fig, animate, frames=len(data)//skips, repeat=False)
        ani.save('animation.gif', writer='pillow', fps=20)