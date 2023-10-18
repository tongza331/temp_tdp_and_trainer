import pandas     as pd
import numpy      as np
import statistics as stt
import warnings   as wrn
import sys
import os
import yaml

from services        import GET_FORMAT_TYPE
from sklearn.metrics   import mean_squared_error
    
if wrn.__name__ in sys.modules:
    wrn.filterwarnings('ignore')

with open('qualifier.yaml', 'r') as file:
    settings = yaml.load(file, Loader = yaml.FullLoader)

TDP_by_format = settings['TDP_by_format']

class EARLY_TDP:
    def __init__(self,df : pd.DataFrame(), bad_head_list : list()):
        self.df               = df.assign(new_qualifier = lambda x : x['qualifier'].str[:3]).sort_values(by = 'radius').drop_duplicates()
        self.bad_head_list    = bad_head_list
        self.GH_df, self.BH_df = self.filter_df()
        self.qualifier_value  = self.qualifier_sort()
        
    def run(self):
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
        return tdp_symptom
    
    def filter_df(self):
        is_bad_head = self.df['head'].isin(self.bad_head_list)
        return self.df.loc[~is_bad_head], self.df.loc[is_bad_head]
    
    def qualifier_sort(self):
        product_id = self.df['product'].values[0]
        format_type = GET_FORMAT_TYPE(self.df).run()
        qualifiers = TDP_by_format.get(product_id).get(format_type)
        qual_list  = self.df.loc[(self.df['head'].isin(self.bad_head_list))]['qualifier'].unique() 
        qualifier_value = sorted(qual_list, key = lambda x : qualifiers.index(x))
        qual_unique = []
        for qual in qualifier_value:
            if qual[:3] not in qual_unique:
                qual_unique.append(qual)
        
        return qual_unique
        
    def _verify_change(self,mean_good,mean_bad):
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
        min_distance = min(position,key = position.get)
        return 'normal' if position.get(min_distance) <= threshold else min_distance
    
    def verify_TDP_early(self,BH_df):
        if BH_df.empty:
            return False
        mean_other_zone_tc = np.mean(BH_df['tdtfcdactc'].values[:9])
        id_zone_tc         = BH_df['tdtfcdactc'].values[-1]
        id_zone_fitted     = BH_df['fittedtdtfcdactc'].values[-1]
        diff_tc            = abs((mean_other_zone_tc - id_zone_tc)/id_zone_tc)
        diff_tc_fitted     = abs((id_zone_tc - id_zone_fitted)/id_zone_fitted)
        return diff_tc >= 0.23 and diff_tc_fitted >= 0.08
    
    def analysis(self,BH_df):
        bad_value,mean_tdp,sigma,position_badhead = list(),list(),list(),list()
        head = BH_df['head'].values[0]
        mean_bad, mean_good, early_check = [], [], []
        for col in ['fittedtdtfcdactc','tdtfcdactc']:
            for qual in self.qualifier_value:
                mean_tdp,sigma,upper_bound,lower_bound = [],[],[],[]
                
                gh_df = self.GH_df.loc[self.GH_df['new_qualifier'] == qual[:3]]
                bh_df = BH_df.loc[BH_df['new_qualifier'] == qual[:3]]
                
                if gh_df.empty or bh_df.empty or bh_df['head'].nunique() == 1:
                    continue
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

# if __name__ == '__main__':
#     tdp_data  = pd.read_csv(f'2TGKS5SH_TDP.csv')
#     obj = TDP(tdp_data,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
#     symptom = obj.run()
#     print(symptom)