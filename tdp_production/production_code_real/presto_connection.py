import warnings
warnings.filterwarnings('ignore')
from sql_command import SQL_Command
from pyhive import presto
from sqlalchemy import *
from sqlalchemy.engine import create_engine
import pandas as pd
import json
import time
import os

class PrestoConnection:
    def __init__(self, config_path:str, SAVE_CSV:bool=False, SET_PATH:str=None):
        """
        Initializes the class instance with the provided configuration path, save option, and set path.
        
        Parameters:
            config_path (str): The path to the configuration file.
            SAVE_CSV (bool, optional): A flag indicating whether to save CSV files. Defaults to False.
            SET_PATH (str, optional): The path to set as the working directory. Defaults to None.
        """
        self.config_path = config_path
        self.config = self.read_config(self.config_path)
        self.host_name = self.config["host_name"]
        self.engine_name = self.config["engine_name"]
        self.port = self.config["port"]
        self.email = self.config["email"]
        self.cert = self.config["CA_Bundle"]
        self.protocol = self.config["protocol"]
        self.category = self.config["category"]
        self.full_host_name = f"{self.engine_name}://{self.host_name}:{self.port}"
        print("Full name engine: ", self.full_host_name)
        
        ## init query
        self.sqlCmd = SQL_Command()
        self.MFGsql = self.sqlCmd.MFGsql
        self.TDPsql = self.sqlCmd.TDPsql
        self.SERsql = self.sqlCmd.SERsql
        self.HIsql  = self.sqlCmd.HIsql
        self.RWEsql = self.sqlCmd.RWEsql
        self.RRONRROsql = self.sqlCmd.RRONRROsql
        self.CSERsql = self.sqlCmd.CSERsql
        self.CSER_QUALIFIER = self.sqlCmd.CSER_QUALIFIER['CSER']
        
        self.mfg_params_check = ["product", "serial", "procid", "pf_code", "start_enddt", "end_enddt"]
        self.mfg_df = pd.DataFrame()
        
        self.SAVE_CSV = SAVE_CSV
        self.SET_PATH = SET_PATH
        if self.SET_PATH is None:
            self.SET_PATH = os.getcwd()
        else:
            self.SET_PATH = SET_PATH
            
        self.ENDDT_MAP = {
            "6300":"enddt63",
            "6400":"enddt64",
            "6600":"enddt66",
            "6800":"enddt68",
            "9000":"enddt90",
        }

    def read_config(self, config_path):
        """
        Reads a configuration file and returns the parsed JSON object.

        Args:
            config_path (str): The path to the configuration file.

        Returns:
            dict: The parsed JSON object representing the configuration.

        Raises:
            Exception: If there is an error while reading or parsing the file.
        """
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as e:
            print(e)
        return config
    
    def connect(self):
        """
        Connects to the database using the provided parameters and returns the connection object.

        Returns:
            connection: The connection object to the database.

        Raises:
            Exception: If unable to connect to the database.
        """
        try:
            connection = create_engine(self.full_host_name,
                                connect_args={
                                    'protocol': self.protocol,
                                    'catalog': self.category,
                                    'username': self.email,
                                    'requests_kwargs': self.cert
                                }
            )
            print("Connected to the database")
        except Exception as e:
            print("Unable to connect to the database")
            print(e)
        return connection

    def __query(self, sql_command):
        """
        Executes the given SQL command on the connected database and returns the result as a pandas DataFrame.

        Parameters:
            sql_command (str): The SQL command to be executed.

        Returns:
            DataFrame: The result of the SQL query as a pandas DataFrame.

        Raises:
            Exception: If there is an error while executing the SQL command.
        """
        connection = self.connect()
        try:
            print("Querying the database")
            df = pd.read_sql(sql_command, connection)
        except Exception as e:
            print("Unable to query the database")
            print(e)
        return df
    
    ## main query to run
    def query(self, mfg_params:dict(), pfcode_mode:str) -> pd.DataFrame():
        """
            if pfcode_mode is None: only query MFG for specify the parametric
            if pfcode_mode is not None: query MFG and other table for specify the parametric
        """
        
        mfg_params_keys = list(mfg_params.keys())
        assert all(key in self.mfg_params_check for key in mfg_params_keys), "Missing key in mfg_params"
        
        return_dict = {}
        mfgSql_command = self.MFGsql.format(**mfg_params)
        self.mfg_df = self.__query(mfgSql_command)
        print(self.mfg_df)
        mfg_product = self.mfg_df["product"].unique()[0]
        
        map_fail_end = self.ENDDT_MAP[mfg_params["procid"]]
        fail_enddt_value = self.mfg_df[map_fail_end].unique()[0]
        
        if fail_enddt_value is None:
            self.mfg_df[map_fail_end] = self.mfg_df["enddt"].unique()[0]
        
        if pfcode_mode is None:
            return_dict = {"MFG": self.mfg_df}
            return return_dict
            
        if "SER" in pfcode_mode:
            print("Querying pfcode SER mode")
            params = {
                "product": mfg_product,
                "serial": self.mfg_df["hddsn"].unique()[0],
                "procid": self.mfg_df["procid"].unique()[0],
                "pf_code": self.mfg_df["pfcode"].unique()[0],
                "enddt64": self.mfg_df["enddt64"].unique()[0],
                "enddt66": self.mfg_df["enddt66"].unique()[0],
                "enddt68": self.mfg_df["enddt68"].unique()[0],
                "enddt90": self.mfg_df["enddt90"].unique()[0],
            }
            return_dict = self._XSER(**params)
        
        elif "6FP" in pfcode_mode:
            print("Queryinh pfcode 6FP mode")
            params = {
                "product": mfg_product,
                "serial": self.mfg_df["hddsn"].unique()[0],
                "procid": self.mfg_df["procid"].unique()[0],
                "pf_code": self.mfg_df["pfcode"].unique()[0],
                "start_enddt": self.mfg_df["startdate"].unique()[0],
                "enddt": self.mfg_df["enddt"].unique()[0],
                "enddt64": self.mfg_df["enddt64"].unique()[0],
                "enddt66": self.mfg_df["enddt66"].unique()[0],
                "enddt68": self.mfg_df["enddt68"].unique()[0],
                "enddt90": self.mfg_df["enddt90"].unique()[0],
            }
            return_dict = self._6FPx(**params)
        
        elif ('2521' in pfcode_mode) or (pfcode_mode.startswith('4') and pfcode_mode.endswith('1')):
            print("Querying pfcode 2521_4XX1 mode")
            params = {
                "product": mfg_product,
                "serial": self.mfg_df["hddsn"].unique()[0],
                "procid": self.mfg_df["procid"].unique()[0],
                "pf_code": self.mfg_df["pfcode"].unique()[0],
                "start_enddt": self.mfg_df["startdate"].unique()[0],
                "enddt": self.mfg_df["enddt"].unique()[0],
                "enddt64": self.mfg_df["enddt64"].unique()[0],
                "enddt66": self.mfg_df["enddt66"].unique()[0],
                "enddt68": self.mfg_df["enddt68"].unique()[0],
                "enddt90": self.mfg_df["enddt90"].unique()[0],
            }
            return_dict = self._2521_4XX1(**params)
        return return_dict
    
    def TDP_QUERY(self, **TDPparams):
        fail_head_columns = 'phd'
        TDPparams["fail_head_columns"] = fail_head_columns
        print("TDP params: ", TDPparams)
        TDPsql_command = self.TDPsql.format(**TDPparams)
        tdp_df = self.__query(TDPsql_command)
        return tdp_df
    
    def SER_QUERY(self, **SERparams):
        fail_head_columns = 'phd' if SERparams["product"] == 'adq' else 'lhd'
        SERparams["fail_head_columns"] = fail_head_columns
        SERsql_command = self.SERsql.format(**SERparams)
        ser_df = self.__query(SERsql_command)
        return ser_df
    
    def HI_QUERY(self, **HIparams):
        fail_head_columns = 'phd' if HIparams["product"] == 'adq' else 'lhd'
        HIparams["fail_head_columns"] = fail_head_columns
        HIsql_command = self.HIsql.format(**HIparams)
        hi_df = self.__query(HIsql_command)
        return hi_df
    
    def READ_WRITE_ERROR_QUERY(self, **RWEparams):
        RWEsql_command = self.RWEsql.format(**RWEparams)
        rwe_df = self.__query(RWEsql_command)
        return rwe_df
    
    def RRONRRO_QUERY(self, **RROparams):
        rronrro_dict = self.sqlCmd.RRONRRO_SPECIAL_PARAM
        cmd_name = rronrro_dict["CMD"][RROparams["product"]]
        table = rronrro_dict["TABLE"][RROparams["product"]]
        band = rronrro_dict["TEST_POINT"][RROparams["product"]]
        
        RROparams["command_name"] = cmd_name
        RROparams["table"] = table
        RROparams["band"] = band
        
        RRONRRO_command = self.RRONRROsql.format(**RROparams)
        rronrro_df = self.__query(RRONRRO_command)
        return rronrro_df
    
    def CSER_QUERY(self, **CSERparams):
        qual_tuple = self.CSER_QUALIFIER[CSERparams["product"]]
        fail_head_columns = 'phd' if CSERparams["product"] == 'adq' else 'lhd'
        CSERparams["fail_head_columns"] = fail_head_columns
        CSERparams["qualifier"] = qual_tuple
        CSERsql_command = self.CSERsql.format(**CSERparams)
        cser_df = self.__query(CSERsql_command)
        return cser_df
    
    def _2521_4XX1(self, **params):
        return_dict = {
            "SER": pd.DataFrame(),
            "TDP": pd.DataFrame(),
            "HI": pd.DataFrame(),
            "RRONRRO" : pd.DataFrame(),
        }
        ser_df = self.SER_QUERY(**params)
        merge_data = MergeData(mfg_df=self.mfg_df, param_df=ser_df)
        ser_final_df = merge_data.merge_subcode()
        
        tdp_df = self.TDP_QUERY(**params)
        merge_data = MergeData(mfg_df=self.mfg_df, param_df=tdp_df)
        tdp_final_df = merge_data.merge_subcode()
        
        hi_df = self.HI_QUERY(**params)
        merge_data = MergeData(mfg_df=self.mfg_df, param_df=hi_df)
        hi_final_df = merge_data.merge_subcode()
        
        rronrro_df = self.RRONRRO_QUERY(**params)
        merge_data = MergeData(mfg_df=self.mfg_df, param_df=rronrro_df)
        rronrro_final_df = merge_data.merge_subcode()
        
        return_dict["TDP"] = tdp_final_df
        return_dict["SER"] = ser_final_df
        return_dict["HI"] = hi_final_df
        return_dict["RRONRRO"] = rronrro_final_df
        
        if self.SAVE_CSV:
            self._save_csv(tdp_final_df, self.SET_PATH, f"TDP_{params['serial']}.csv")
            self._save_csv(ser_final_df, self.SET_PATH, f"SER_{params['serial']}.csv")
            self._save_csv(hi_final_df, self.SET_PATH, f"HI_{params['serial']}.csv")
            self._save_csv(rronrro_final_df, self.SET_PATH, f"RRONRRO_{params['serial']}.csv")
        
        return return_dict
    
    def _XSER(self, **params):
        return_dict = {
            "SER": pd.DataFrame(),
            "TDP": pd.DataFrame(),
            "HI": pd.DataFrame(),
        }
        ser_df = self.SER_QUERY(**params)
        merge_data = MergeData(mfg_df=self.mfg_df, param_df=ser_df)
        ser_final_df = merge_data.merge_subcode()
        
        tdp_df = self.TDP_QUERY(**params)
        merge_data = MergeData(mfg_df=self.mfg_df, param_df=tdp_df)
        tdp_final_df = merge_data.merge_subcode()
        
        hi_df = self.HI_QUERY(**params)
        merge_data = MergeData(mfg_df=self.mfg_df, param_df=hi_df)
        hi_final_df = merge_data.merge_subcode()
        
        return_dict["TDP"] = tdp_final_df
        return_dict["SER"] = ser_final_df
        return_dict["HI"] = hi_final_df
        
        if self.SAVE_CSV:
            self._save_csv(tdp_final_df, self.SET_PATH, f"TDP_{params['serial']}.csv")
            self._save_csv(ser_final_df, self.SET_PATH, f"SER_{params['serial']}.csv")
            self._save_csv(hi_final_df, self.SET_PATH, f"HI_{params['serial']}.csv")
        return return_dict
    
    def _6FPx(self, **params):
        return_dict = {
            "READ_WRITE_ERROR": pd.DataFrame(),
            "RRONRRO": pd.DataFrame(),
            "TDP": pd.DataFrame(),
            "HI": pd.DataFrame(),
            "CSER": pd.DataFrame(),
        }
        rwe_df = self.READ_WRITE_ERROR_QUERY(**params)
        merge_data = MergeData(mfg_df=self.mfg_df, param_df=rwe_df)
        rwe_final_df = merge_data.merge_subcode()
        
        tdp_df = self.TDP_QUERY(**params)
        merge_data = MergeData(mfg_df=self.mfg_df, param_df=tdp_df)
        tdp_final_df = merge_data.merge_subcode()
        
        rronrro_df = self.RRONRRO_QUERY(**params)
        merge_data = MergeData(mfg_df=self.mfg_df, param_df=rronrro_df)
        rronrro_final_df = merge_data.merge_subcode()
        
        hi_df = self.HI_QUERY(**params)
        merge_data = MergeData(mfg_df=self.mfg_df, param_df=hi_df)
        hi_final_df = merge_data.merge_subcode()
        
        cser_df = self.CSER_QUERY(**params)
        merge_data = MergeData(mfg_df=self.mfg_df, param_df=cser_df)
        cser_final_df = merge_data.merge_subcode()
        
        return_dict["READ_WRITE_ERROR"] = rwe_final_df
        return_dict["TDP"] = tdp_final_df
        return_dict["RRONRRO"] = rronrro_final_df
        return_dict["HI"] = hi_final_df
        return_dict["CSER"] = cser_final_df
        
        if self.SAVE_CSV:
            self._save_csv(rwe_final_df, self.SET_PATH, f"RWE_{params['serial']}.csv")
            self._save_csv(tdp_final_df, self.SET_PATH, f"TDP_{params['serial']}.csv")
            self._save_csv(rronrro_final_df, self.SET_PATH, f"RRONRRO_{params['serial']}.csv")
            self._save_csv(hi_final_df, self.SET_PATH, f"HI_{params['serial']}.csv")
            self._save_csv(cser_final_df, self.SET_PATH, f"CSER_{params['serial']}.csv")
        
        return return_dict
    
    def _save_csv(self, df, path, filename):
        df.to_csv(os.path.join(path, filename), index=False)
    
class MergeData:
    def __init__(self, mfg_df, param_df):
        self.mfg_df = mfg_df
        self.param_df = param_df
        self.product = self.mfg_df["product"].unique()[0] 
        self.subcode = self.mfg_df["subcode"].unique()[0]
        
    def convert_subcode(self, subcode:str) -> list:
        binary = bin(int(subcode,16))[2:][::-1]
        if int(binary) != 0:
            bad_head_list = [ind for ind, val in enumerate(binary) if int(val) != 0]
        else:
            bad_head_list = []

        if self.product == "adq" and any(val >= 9 for val in bad_head_list):
            for ind, head in enumerate(bad_head_list):
                if head >= 9:
                    bad_head_list[ind] = head - 9
        return bad_head_list
    
    def merge_subcode(self):
        temp_df = self.param_df.copy()
        bad_head_list = self.convert_subcode(self.subcode)
        if len(bad_head_list) > 1:
            ## convert list to string
            bad_head_string = ",".join([str(head) for head in bad_head_list])
        else:
            bad_head_string = str(bad_head_list[0])
        temp_df["subcode"] = [self.subcode] * len(temp_df)
        temp_df["bad_head"] = [bad_head_string] * len(temp_df)
        return temp_df


### DONE: - 6FPX
### DONE: - 2521_4XX1
### DONE: - XSER