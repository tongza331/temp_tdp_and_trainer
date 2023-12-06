import pandas     as pd
import numpy      as np
import statistics as stt
import warnings   as wrn
import sys
import os
import yaml

from sklearn.metrics   import mean_squared_error
    
if wrn.__name__ in sys.modules:
    wrn.filterwarnings('ignore')

class EARLY_TDP:
    def __init__(self,df : pd.DataFrame(), bad_head_list : list()):
        """
        Initializes the object with the given DataFrame and bad_head_list.

        Parameters:
            df (pd.DataFrame): The DataFrame to be assigned to the object.
            bad_head_list (list): The list of bad heads.

        Returns:
            None
        """
        self.df               = df.assign(new_qualifier = lambda x : x['qualifier'].str[:3]).sort_values(by = 'radius').drop_duplicates()
        self.bad_head_list    = bad_head_list
        self.GH_df, self.BH_df = self.filter_df_gh_bh()
        self.qualifier_value  = self.filter_qualifier()
        
    def run(self):
        """
        Runs the function and returns a dictionary containing TDP symptoms.
        
        Returns:
            tdp_symptom (dict): A dictionary containing the following keys:
            
                - 'head' (list): A list of TDP head values.
                - 'TDP_level' (list): A list of TDP profile values.
                - 'TDP_change' (list): A list indicating whether there was a change in TDP (either 'yes' or 'no').
                - 'TDP_early' (list): A list of TDP early values.
        """
        tdp_symptom = []
        if self.BH_df.empty:
            return tdp_symptom
        tdp_symptom = {'head':[],'TDP_level':[],'TDP_change':[],'TDP_early' : []}
        
        for head in self.BH_df['head'].unique():
            df  = self.BH_df.loc[self.BH_df['head'] == head]
            tdp = self.analysis(df) 
            tdp_symptom.get('head').append(tdp.get('head'))
            tdp_symptom.get('TDP_level').append(tdp.get('profile'))
            tdp_symptom.get('TDP_change').append('yes' if tdp.get('change') != 0 else 'no')
            tdp_symptom.get('TDP_early').append(tdp.get('early'))
        print(tdp_symptom)
        return tdp_symptom
    
    def filter_df_gh_bh(self):
        """
        Filter the DataFrame based on the 'head' column by checking if each value is in the 'bad_head_list'.
        
        Returns:
            A tuple of two DataFrames. The first DataFrame contains the rows where the 'head' column value is not in the 'bad_head_list'. 
            The second DataFrame contains the rows where the 'head' column value is in the 'bad_head_list'.
        """
        is_bad_head = self.df['head'].isin(self.bad_head_list)
        return self.df.loc[~is_bad_head], self.df.loc[is_bad_head]
    

    def filter_qualifier(self):
        """
        Returns a list of qualifier values derived from the sorting qualifier.

        :param self: The instance of the class.
        :return: A list of qualifier values.
        """
        qual_df = self.sorting_qualifier()
        qualifier_value = [qual[:3] for qual in qual_df]
        return  qualifier_value
    
    def sorting_qualifier(self) -> list:
        """
        Returns a list of qualifiers sorted by process.

        Returns:
            list: A list of qualifiers sorted by process.
        """
        unique_qualifier_by_process = self.df.groupby('procid')['qualifier'].unique()
        result_list = []
        for value in unique_qualifier_by_process:
            value = self.sorting_key(value)
            result_list.extend(value)
        return result_list
    
    def sorting_key(self, item: list) -> list:
        """
        Sorts the given list of items based on the first element of each item and then the remaining elements in ascending order.

        Parameters:
            item (list): The list of items to be sorted.

        Returns:
            list: The sorted list of items.
        """
        item = sorted(item, key=lambda x: (x[0], x[1:]))
        return item
        
    def _verify_change(self,mean_good,mean_bad):
        """
        Verify the change between the mean values of two lists.

        Parameters:
            mean_good (list): A list of mean values representing the "good" data.
            mean_bad (list): A list of mean values representing the "bad" data.

        Returns:
            int: The number of changes that meet the defined thresholds.
        """
        diff_gh_threshold = 0.2
        diff_bh_threshold = 0.1
        count_change   = 0
        for pre_g,pos_g,pre_b,pos_b in zip(mean_good,mean_good[1:],mean_bad,mean_bad[1:]):
            diff_gh = abs((pre_g - pos_g)/pos_g)
            diff_bh = abs((pre_b - pos_b)/pos_b)

            if (diff_gh <= diff_gh_threshold) and (diff_bh >= diff_bh_threshold):
                count_change += 1
        return count_change 
                
    def verify(self,position,threshold):
        """
        Verify if the given position is within the threshold distance.

        Args:
            position (dict): A dictionary containing the positions and distances.
            threshold (float): The maximum distance allowed.

        Returns:
            str: Returns 'normal' if the minimum distance is less than or equal to the threshold. Otherwise, returns the key associated with the minimum distance in the position dictionary.
        """
        min_distance = min(position,key = position.get)
        return 'normal' if position.get(min_distance) <= threshold else min_distance
    
    def verify_TDP_early(self,BH_df):
        """
        Verify if TDP is early.
        
        Args:
            BH_df (pandas.DataFrame): The dataframe containing the TDP data.
        
        Returns:
            bool: True if TDP is early, False otherwise.
        """
        if BH_df.empty:
            return False
        mean_other_zone_tc = np.mean(BH_df['tdtfcdactc'].values[:9])
        id_zone_tc         = BH_df['tdtfcdactc'].values[-1]
        id_zone_fitted     = BH_df['fittedtdtfcdactc'].values[-1]
        diff_tc            = abs((mean_other_zone_tc - id_zone_tc)/id_zone_tc)
        diff_tc_fitted     = abs((id_zone_tc - id_zone_fitted)/id_zone_fitted)
        
        return diff_tc >= 0.23 and diff_tc_fitted >= 0.08
    
    def analysis(self,BH_df):
        """
        Generates a function comment for the given function body.

        Args:
            BH_df (pandas.DataFrame): The input DataFrame.

        Returns:
            dict: A dictionary containing information about the analysis. The dictionary has the following keys:
                - 'head' (int or str): The head value.
                - 'profile' (str): The profile of the bad head distance. Can be 'high', 'normal', or 'low'.
                - 'change' (int): The count of changes between the mean_good and mean_bad values.
                - 'early' (str): Indicates whether an early check was performed. Can be 'yes' or 'no'.
        """
        bad_value,mean_tdp,sigma,position_badhead = list(),list(),list(),list()
        head = BH_df['head'].values[0]
        mean_bad, mean_good, early_check = [], [], []
        for col in ['fittedtdtfcdactc','tdtfcdactc']:
            for qual in self.qualifier_value:
                mean_tdp,sigma,upper_bound,lower_bound = [],[],[],[]
                
                gh_df = self.GH_df.loc[self.GH_df['new_qualifier'] == qual[:3]]
                bh_df = BH_df.loc[BH_df['new_qualifier'] == qual[:3]]

                ## BUG: Check empyty dataframe condition -> comment out for now
                # if gh_df.empty or bh_df.empty or bh_df['head'].unique() == 1:
                #     continue
                
                radius    = gh_df['radius'].unique()
                bad_value = bh_df[col].values
                mean_good.append(gh_df[col].mean())
                mean_bad.append(bh_df[col].mean())

                early_check.append(self.verify_TDP_early(bh_df))
                for rad in radius:
                    tdp  = gh_df[gh_df['radius'] == rad][col] 
                    mean, sig = tdp.mean(), stt.stdev(tdp.values)
                    mean_tdp.append(mean); sigma.append(sig)
                    upper_bound.append(mean + sig)
                    lower_bound.append(mean - sig)
                bad_head_distance = {
                    'high'  : mean_squared_error(upper_bound,bad_value, squared = False),
                    'normal': mean_squared_error(mean_tdp,bad_value   , squared = False),
                    'low'   : mean_squared_error(lower_bound,bad_value, squared = False) 
                }
                position_badhead.append(self.verify(bad_head_distance,threshold = 40))

        early_check  = 'yes' if any(early_check) else 'no' 
        count_change = self._verify_change(mean_good, mean_bad)
        tdp_bad_head = {
            'head' : head,
            'profile' : max(position_badhead,key = position_badhead.count) if len(position_badhead)  != 0 else 'normal',
            'change'  : count_change, 
            'early'   : early_check
        }
        return tdp_bad_head