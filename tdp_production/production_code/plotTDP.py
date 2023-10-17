import pandas     as pd
import numpy      as np
import statistics as stt
import warnings   as wrn
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_squared_error

COLORS_LIST = {
  '0': 'cornflowerblue',
  '1': 'palevioletred',
  '2': 'forestgreen',
  '3': 'red',
  '4': 'aqua',
  '5': 'purple',
  '6': 'orange',
  '7': 'lightblue',
  '8': 'gray',
  '9': 'darkgoldenrod',
  '10': 'darkgreen',
  '11': 'lime',
  '12': 'mistyrose',
  '13': 'mediumpurple',
  '14': 'goldenrod',
  '15': 'olive',
  '16': 'dodgerblue',
  '17': 'pink',
  '18': 'limegreen',
  '19': 'indianred',    
}

if wrn.__name__ in sys.modules:
    wrn.filterwarnings('ignore')

def get_bad_head(df): 
    binary = bin(int(str(df['subcode'].values[0]),16))[2:][::-1]
    if int(binary) != 0: 
        bad_head_list = [ind for ind, val in enumerate(binary) if int(val) != 0]
    else:
        return []
    if df['product'].values[0] == 'adq' and any(val >= 9 for val in bad_head_list):
        for ind, head in enumerate(bad_head_list):
            if head >= 9:
                bad_head_list[ind] = head - 9
    return bad_head_list


class TDP:
    def __init__(self, df: pd.DataFrame, bad_head_list: list = []):
        self.df                = df.assign(new_qualifier = lambda x : x['qualifier'].str[:3]).sort_values(by='radius').drop_duplicates()
        self.bad_head_list     = bad_head_list
        self.qualifier_value   = self.filter_qualifier()
    
    def filter_qualifier(self):
        qual_df = self.sorting_qualifier()
        qualifier_value = [qual[:3] for qual in qual_df]
        return  qualifier_value
    
    def sorting_qualifier(self) -> list:
        unique_qualifier_by_process = self.df.groupby('procid')['qualifier'].unique()
        result_list = []
        for value in unique_qualifier_by_process:
            value = self.sorting_key(value)
            result_list.extend(value)
        return result_list
    
    def sorting_key(self, item: list) -> list:
        item = sorted(item, key=lambda x: (x[0], x[1:]))
        return item
    
    def set_plot_param(self, head: str): 
        color = COLORS_LIST[str(head)]
        if not self.bad_head_list:
            color = COLORS_LIST[str(int(head))]
        else:
            if head in self.bad_head_list:
                color = '#fa3030' 
            else:
                color = 'grey'
        lw     = 1 if head in self.bad_head_list or not self.bad_head_list else 0.5
        alpha  = 1 if head in self.bad_head_list or not self.bad_head_list else 0.7
        zorder = 3 if head in self.bad_head_list or not self.bad_head_list else 2
        return color, lw, alpha, zorder
    
    def display(self):
        if self.df.empty:
            return plt.figure()
        fig, ax = None, None
        kwargs = {
            'nrows': 1,
            'ncols': len(self.qualifier_value),
            'figsize': (2.56, 2.56),
            'sharey': True
        }
        fig, ax = plt.subplots(**kwargs);
        if len(self.qualifier_value) == 1: 
            ax = [ax]
        for i, qualifier in enumerate(self.qualifier_value):
            data = self.df.loc[self.df['new_qualifier'] == qualifier[:3]]
            for head, df_head in data.groupby('head'):
                color, lw, alpha, zorder = self.set_plot_param(head)
                ax[i].plot(df_head['radius'], df_head['fittedtdtfcdactc'], color=color, lw=lw, alpha=alpha, label=str(head), zorder=zorder)
            ax[i].tick_params(length=0)
            ax[i].axis(False)
            plt.subplots_adjust(wspace=0,hspace=0,right=0.85)
        return fig

# if __name__ == '__main__':
#     file_path = os.path.join(os.getcwd(), 'CSV','2GHGJYYS.csv')
#     if os.path.exists(file_path):
#         df = pd.read_csv(file_path)
#         failure_head_list = get_bad_head(df=df)
#         tdp = TDP(df=df, bad_head_list=failure_head_list)
#         tdp.display()
#         plt.show()
