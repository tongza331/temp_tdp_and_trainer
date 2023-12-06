import pandas     as pd
import numpy      as np
import statistics as stt
import warnings   as wrn
import sys
import os
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
    print(str(df['subcode'].values[0]))
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
    def __init__(self, df: pd.DataFrame, bad_head_list: list = [], plot_nonfit=False):
        self.df                = df.assign(new_qualifier = lambda x : x['qualifier'].str[:3]).sort_values(by='radius').drop_duplicates()
        self.bad_head_list     = bad_head_list
        self.qualifier_value   = self.filter_qualifier()
        self.plot_nonfit       = plot_nonfit
        self.hddsn             = df['hddsn'].values[0]
    
    def filter_qualifier(self):
        qual_df = self.sorting_qualifier()
        qualifier_value = [qual[:3] for qual in qual_df]
        return  qualifier_value
    
    def sorting_qualifier(self) -> list:
        unique_qualifier_by_process = self.df.groupby('procid')['qualifier'].unique()
        product = self.df['product'].values[0]
        pfcode = self.df['pfcode'].values[0]
        result_list = []
        print(unique_qualifier_by_process)
        for value in unique_qualifier_by_process:
            if product == 'pcm' and pfcode == 'TDD4':    
                value = self.sorting_key_fn_pcm(value)
            else:
                value = self.sorting_key(value)
            result_list.extend(value)
        return result_list
    
    def sorting_key(self, item: list) -> list:
        item = sorted(item, key=lambda x: (x[0], x[1:]))
        return item
    
    def sorting_key_fn_pcm(self, item: list) -> list:
        int_list = []
        str_list = []
        item = self.sorting_key(item)
        
        for i_item in item:
            try:
                _ = int(i_item)
                int_list.append(i_item)
            except:
                str_list.append(i_item)

        sort_item = sorted(str_list, key=lambda x: (x[0], x[1:]))
        first_list = sorted(sort_item[0:2], key=lambda x: (x[0], x[1:]))
        second_list = self._swap_pairs(sort_item[2:])
        print(first_list, second_list)
        if len(int_list) != 0:
            final_item = self._merge_swap_pairs(int_list, second_list)
            final_item = first_list + final_item
        else:
            final_item = first_list + second_list
        return final_item
    
    def _merge_swap_pairs(self, intlist, strlist):
        result_list = []
        ## [31n, 30n, 41n, 40n, 51n, 50n]
        ## [300, 400, 500]
        ## merge to [31n, 30n, 300, 41n, 40n, 400, 51n, 50n, 500]
        idx_temp = 0
        for idx in range(len(strlist)):
            if idx%2 == 0:
                idx_temp = (idx//2) if idx != 0 else idx
                intq = intlist[idx_temp]
                tempq = strlist[idx:idx+2] + [intq]
                result_list.extend(tempq)
        return result_list
    
    def _swap_pairs(self, lst):
        evens = lst[::2]
        odds = lst[1::2]
        swapped = [elem for pair in zip(odds, evens) for elem in pair]
        if len(lst) % 2:
            swapped.append(lst[-1])
        return swapped
    
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
        print("Total qualifier: ", len(self.qualifier_value), f"---> {self.qualifier_value}")
        for i, qualifier in enumerate(self.qualifier_value):
            data = self.df.loc[self.df['new_qualifier'] == qualifier[:3]]
            for head, df_head in data.groupby('head'):
                color, lw, alpha, zorder = self.set_plot_param(head)
                if self.plot_nonfit:
                    col_focus = 'tdtfcdactc'
                else:
                    col_focus = 'fittedtdtfcdactc'
                ax[i].plot(df_head['radius'], df_head[col_focus], color=color, lw=lw, alpha=alpha, label=str(head), zorder=zorder) ## fittedtdtfcdactc, tdtfcdactc
            ax[i].tick_params(length=0)
            ax[i].axis(False)
            plt.subplots_adjust(wspace=0,hspace=0,right=0.85)
        return fig