import numpy as np
import pandas as pd

import lasio
from scipy import linalg, optimize
import matplotlib.pyplot as plt

import torch

class ARI:
    def __init__(self):
        self.verbose     = True
        self.save_data   = True
        self.return_data = True
        self.check_torch_gpu()

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
        case1 = well1[['AT10','AT30','AT60','AT90','GR','RV72H_1D', 'RH72H_1D']].dropna()
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
    
