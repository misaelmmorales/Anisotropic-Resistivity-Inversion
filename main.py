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

    def plot_title(self, fig, df, x=0.5, y=0.05, fs=14):
        fld = df.header['Well']['FLD'].value
        wll = df.header['Well']['WELL'].value
        com = df.header['Well']['COMP'].value
        fig.text(x, y, '{} | {} | {}'.format(fld,com,wll), weight='bold', ha='center', va='center', fontsize=fs)
        return None

    def plot_formations(self, ax, df, lw=1, alpha=0.5, bounds=[0,1], cmap='tab20', align:str='center', triangle:bool=False):
        for i in range(len(df)):
            data = df.iloc[i]
            top, bot, name = data['Top'], data['Bottom'], data['Name']
            my_hatch = data['Hatch'] if 'Hatch' in df.columns else None
            my_edge  = 'gray' if 'Hatch' in df.columns else None
            my_color = data['Color'] if 'Color' in df.columns else mpl.colormaps[cmap](i)
            if triangle:
                ax.fill_betweenx([top,bot], bounds, color=my_color, hatch=my_hatch, edgecolor=my_edge, lw=lw, alpha=alpha)
                ax.text(np.mean(bounds), bot-30, name, ha=align, va='center')
            else:
                ax.fill_betweenx([top,bot], bounds[0], bounds[1], color=my_color, hatch=my_hatch, edgecolor=my_edge, lw=lw, alpha=alpha)
                ax.text(np.mean(bounds), np.mean([top,bot]), name, ha=align, va='center')
            ax.set_xlim(bounds[0], bounds[1])
            my_labels = 'Lithology' if 'Lith' in df.columns else 'Formation'
            ax.set_xlabel(my_labels, weight='bold')
            ax.xaxis.set_label_position('top'); ax.xaxis.set_ticks_position('top')
            ax.spines['top'].set_linewidth(2); ax.set_xticks([])
            return None

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

############################## MAIN ##############################
if __name__ == '__main__': 
    # Run main routine if main.py called directly
    ari = AnisoResInv()

############################## END ##############################