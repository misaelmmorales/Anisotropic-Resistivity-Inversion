import numpy as np
import pandas as pd

import lasio
from scipy import linalg, optimize, stats

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize

class AnisoResInv:
    '''
    Main class for Anisotropic Resistivity Inversion (ARI)
    '''
    def __init__(self):
        self.save_plot   = True
        self.return_data = False
        self.save_data   = False

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
            ax.fill_betweenx(y, x, ub, alpha=alpha, color=color) if rightfill else ax.fill_betweenx(y, lb, x, alpha=alpha, color=color)
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
    
    def plot_well_1(self, df, figsize=(10,10)):
        fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)
        ax1, ax2, ax3 = axs
        ax11 = ax1.twiny()
        self.plot_curve(ax11, df, 'GR', 0, 200, 'g', units='API')
        self.plot_curve(ax1, df, 'CALI', 0, 100, 'navy', alpha=0.25, units='in', fill=True, pad=8)
        ax21, ax22, ax23 = ax2.twiny(), ax2.twiny(), ax2.twiny()
        self.plot_curve(ax2, df, 'AT10', 0.2, 50, 'r', units='$\Omega$m', semilog=True)
        self.plot_curve(ax21, df, 'AT30', 0.2, 50, 'k', units='$\Omega$m', semilog=True, pad=8)
        self.plot_curve(ax22, df, 'AT60', 0.2, 50, 'k', units='$\Omega$m', semilog=True, pad=16)
        self.plot_curve(ax23, df, 'AT90', 0.2, 50, 'b', units='$\Omega$m', semilog=True, pad=24)
        ax31 = ax3.twiny()
        self.plot_curve(ax3, df, 'RV72H_1D', 0.2, 100, 'darkred', units='$\Omega$m', semilog=True)
        self.plot_curve(ax31, df, 'RH72H_1D', 0.2, 100, 'darkblue', units='$\Omega$m', semilog=True, pad=8)
        plt.gca().invert_yaxis()
        plt.show()
        return None
    
    def plot_well_2(self, df, figsize=(10,10)):
        fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)
        ax1, ax2, ax3 = axs
        ax11 = ax1.twiny()
        self.plot_curve(ax11, df, 'HCGR', -5, 200, 'g', units='API')
        self.plot_curve(ax1, df, 'HCAL', 0, 100, 'navy', alpha=0.25, fill=True, units='in', pad=8)
        ax21, ax22, ax23 = ax2.twiny(), ax2.twiny(), ax2.twiny()
        self.plot_curve(ax2, df, 'AT10', 0.2, 200, 'r', units='$\Omega$m', semilog=True)
        self.plot_curve(ax21, df, 'AT30', 0.2, 200, 'k', units='$\Omega$m', semilog=True, pad=8)
        self.plot_curve(ax22, df, 'AT60', 0.2, 200, 'k', units='$\Omega$m', semilog=True, pad=16)
        self.plot_curve(ax23, df, 'AT90', 0.2, 200, 'b', units='$\Omega$m', semilog=True, pad=24)
        ax31 = ax3.twiny()
        self.plot_curve(ax3, df, 'RV72_1DF', 0.2, 200, 'darkred', units='$\Omega$m', semilog=True)
        self.plot_curve(ax31, df, 'RH72_1DF', 0.2, 200, 'darkblue', units='$\Omega$m', semilog=True, pad=8)
        plt.gca().invert_yaxis()
        plt.show()
        return None

############################## MAIN ##############################
if __name__ == '__main__': 
    # Run main routine if main.py called directly
    ari = AnisoResInv()

############################## END ##############################