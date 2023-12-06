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
    """
    This function takes a DataFrame `df` as input and returns a list of bad head values.
    
    Parameters:
        - df: pandas DataFrame
            The DataFrame containing the data from which the bad head values will be extracted.
            
    Returns:
        - bad_head_list: list
            A list of bad head values extracted from the DataFrame.
    """
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
        """
        Initializes an instance of the class.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            bad_head_list (list, optional): A list of bad headers. Defaults to an empty list.
            plot_nonfit (bool, optional): A flag indicating whether to plot non-fitting data. Defaults to False.
        
        Returns:
            None
        """
        self.df                = df.assign(new_qualifier = lambda x : x['qualifier'].str[:3]).sort_values(by='radius').drop_duplicates()
        self.bad_head_list     = bad_head_list
        self.qualifier_value   = self.filter_qualifier()
        self.plot_nonfit       = plot_nonfit
        self.hddsn             = df['hddsn'].values[0]
    
    def filter_qualifier(self):
        """
        Return the first three characters of each qualifier in the sorting qualifier dataframe.

        Parameters:
            None

        Returns:
            List: A list of strings containing the first three characters of each qualifier.
        """
        qual_df = self.sorting_qualifier()
        qualifier_value = [qual[:3] for qual in qual_df]
        return  qualifier_value
    
    def sorting_qualifier(self) -> list:
        """
        Returns a list of qualifiers sorted based on certain conditions.

        This function groups the 'qualifier' column by the 'procid' column in the dataframe and
        retrieves unique qualifiers for each process. It then performs sorting based on certain 
        conditions. If the 'product' is 'pcm' and 'pfcode' is 'TDD4', it calls the 
        'sorting_key_fn_pcm' function to sort the qualifiers. Otherwise, it calls the 
        'sorting_key' function. The sorted qualifiers are then added to the 'result_list' 
        using the 'extend' method. The final 'result_list' is returned.

        Parameters:
        - self: The instance of the class.

        Returns:
        - result_list: A list of sorted qualifiers.

        """
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
        """
        Sorts the given list of items based on a specific key.

        Args:
            item (list): The list of items to be sorted.

        Returns:
            list: The sorted list of items.
        """
        item = sorted(item, key=lambda x: (x[0], x[1:]))
        return item
    
    def sorting_key_fn_pcm(self, item: list) -> list:
        """
        Sorts the given list of items based on specific criteria.

        Parameters:
            item (list): The list of items to be sorted.

        Returns:
            list: The sorted list of items.
        """
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
        """
        Merge swap pairs of integers and strings in two lists into a single list.

        Args:
            intlist (List[int]): A list of integers.
            strlist (List[str]): A list of strings.

        Returns:
            List[Union[int, str]]: A list containing the merged pairs of integers and strings.

        Example:
            >>> obj = ClassName()
            >>> intlist = [31, 30, 41, 40, 51, 50]
            >>> strlist = ['300', '400', '500']
            >>> obj._merge_swap_pairs(intlist, strlist)
            [31, 30, '300', 41, 40, '400', 51, 50, '500']
        """
        result_list = []
        idx_temp = 0
        for idx in range(len(strlist)):
            if idx%2 == 0:
                idx_temp = (idx//2) if idx != 0 else idx
                intq = intlist[idx_temp]
                tempq = strlist[idx:idx+2] + [intq]
                result_list.extend(tempq)
        return result_list
    
    def _swap_pairs(self, lst):
        """
        Swaps pairs of elements in a list.

        Parameters:
            lst (list): The list of elements to swap pairs in.

        Returns:
            list: The list with pairs of elements swapped.
        """
        evens = lst[::2]
        odds = lst[1::2]
        swapped = [elem for pair in zip(odds, evens) for elem in pair]
        if len(lst) % 2:
            swapped.append(lst[-1])
        return swapped
    
    def set_plot_param(self, head: str): 
        """
        Set the plot parameters based on the head value.

        Parameters:
            head (str): The value of the head.

        Returns:
            Tuple[str, float, float, int]: The color, line width, alpha value, and zorder value.
        """
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
        """
        Display the plot of the data.

        This function generates a plot of the data stored in the object. If the data is empty, it returns an empty figure. Otherwise, it creates a figure object and sets up the necessary axes for the number of qualifier values present in the data. If there is only one qualifier value, a single axis is used. Each axis is then populated with the corresponding data for each qualifier value. The plot is generated by plotting the 'radius' values against the 'fittedtdtfcdactc' or 'tdtfcdactc' values for each head in the data. The plot parameters (color, line width, alpha, zorder) are set based on the head value. The plot is then displayed with the tick marks and axes turned off. Finally, the figure is returned.

        Returns:
            fig (Figure): The figure object containing the plot.
        """
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