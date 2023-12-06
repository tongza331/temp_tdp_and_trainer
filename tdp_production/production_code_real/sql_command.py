class SQL_Command:
    def __init__(self):
        self.MFGsql = self.MFGsql()
        self.TDPsql = self.TDPsql()
        self.SERsql = self.SERsql()
        self.HIsql  = self.HIsql()
        self.RWEsql = self.RWEsql()
        self.RRONRROsql = self.RRONRROsql()
        self.CSERsql = self.CSERsql()
        self.RRONRRO_SPECIAL_PARAM = self.RRONRRO_SPECIAL_PARAM()
        self.CSER_QUALIFIER = self.CSER_QUALIFIER()
    
    def MFGsql(self):
        MFGsql = '''
            WITH end_date AS (
            SELECT
                enddt,
                hddsn,
                product,
                procid,
                mfgid,
                pfcode,
                substr(startdate, 1,8) AS startdate,
                pfsubcode AS subcode,
                substr(enddate_6350,1,8) AS enddt63,
                substr(enddate_6400,1,8) AS enddt64,
                substr(enddate_6600,1,8) AS enddt66,
                substr(enddate_6800,1,8) AS enddt68,
                substr(enddate_9000,1,8) AS enddt90,
                RANK() OVER (PARTITION BY hddsn ORDER BY hddsn, enddate DESC) AS hddsn_cycle
            FROM vqaa.fact_hdd_association
            WHERE product = '{product}' 
                AND hddsn IN ('{serial}')
                AND procid = '{procid}'
                AND pfcode = '{pf_code}'
                AND pheadno = 0
                AND enddt BETWEEN '{start_enddt}' AND '{end_enddt}'
            )
            SELECT
            enddt,
            hddsn,
            product,
            procid,
            mfgid,
            pfcode,
            startdate,
            subcode,
            enddt63,
            enddt64,
            enddt66,
            enddt68,
            enddt90
            FROM end_date
            WHERE end_date.hddsn_cycle = 1
        '''
        return MFGsql

    def TDPsql(self):
        TDPsql = '''
            SELECT 
                hddsn, procid, product, pfcode, enddt, mfgid, qualifier, subqualifier, {fail_head_columns} 
                AS head, radius, fittedtdtfcdactc, tdtfcdactc, operationclearance, opeclrpm, enddate
            FROM 
                ghl2.ccb_mi_tdbo 
            WHERE product = '{product}' 
            AND radius <> 0 
            AND pfcode IN ('0000', '{pf_code}') 
            AND procid IN ('6400', '6600', '6800', '9000')
            AND enddt IN ('{enddt64}', '{enddt66}', '{enddt68}', '{enddt90}')
            AND hddsn IN ('{serial}')
        '''
        return TDPsql
        
    def SERsql(self):
        SERsql = '''
            SELECT 
                hddsn, product, procid, pfcode, enddate, mfgid, qualifier, {fail_head_columns} 
                AS head, subqualifier, enddt, band, ser
            FROM 
                ghl2.ccb_ci_ser
            WHERE product = '{product}'
            AND procid IN ('6400', '6600', '6800')
            AND enddt IN ('{enddt64}', '{enddt66}', '{enddt68}')
            AND pfcode IN ('0000', '{pf_code}')
            AND hddsn IN ('{serial}')
            AND subqualifier IN ('AllZnDtR0','AllZnDtR1','AllZnDtR0R1','SovaAllZnR0','SovaAllZnR1','SovaAllZn')
        '''
        return SERsql
    
    def HIsql(self):
        HIsql = '''
            SELECT
                hddsn,product,procid,pfcode,enddt,cmdname,qualifier,subqualifier,{fail_head_columns} 
                AS head,band,cylinder, hi_ser_sig,hi_ser_max,hi_ser_min,hi_idd_sig,hi_idd_max,hi_idd_min
            FROM 
                ghl2.ccb_ci_hi
            WHERE product = '{product}' 
            AND pfcode IN ('0000','{pf_code}')
            AND procid IN ('6400', '6600', '6800')
            AND enddt IN ('{enddt64}', '{enddt66}', '{enddt68}')
            AND hddsn IN ('{serial}')
        '''
        return HIsql 
    
    def RWEsql(self):
        RWEsql = '''
            SELECT
                hddsn,
                startdate,
                enddate,
                pfcode,
                CAST(pfsubcode AS CHAR(8)) 
                AS subcode,
                SUBSTR(TO_HEX(TO_BIG_ENDIAN_64(event002_uniterrorcode)), 13, 4) AS unitErrorCode,
                SUBSTR(TO_HEX(TO_BIG_ENDIAN_64(event002_command)), 15, 2) AS command,
                mfgid,
                event002_drpstep AS drpStep,
                event002_drpbranch AS drpBranch,
                event002_sector AS sector,
                bitwise_and(event002_cylhead, 16777215) AS cyl,
                bitwise_and(event002_cylhead, 4278190080) / 16777216 AS head,
                product,
                procid,
                enddt
            FROM
                ghl2.ccb_testevents
            WHERE
                enddt BETWEEN '{start_enddt}' AND '{enddt}'
                AND procid = '{procid}'
                AND product = '{product}'
                AND pfcode = '{pf_code}'
                AND eventtype = 'ReadWriteErrorEvent'
                AND hddsn IN ('{serial}')
            '''
        return RWEsql
    
    def CSERsql(self):
        CSERsql = '''
            SELECT
                hddsn,product,pfcode,procid,enddt,qualifier,subqualifier,{fail_head_columns} 
                AS head,band,rawmodulationstats0_average,rawmodulationstats0_std
            FROM
                ghl2.ccb_ci_ser
            WHERE product = '{product}' 
            AND pfcode IN ('0000','{pf_code}')
            AND procid IN ('6400', '6600', '6800')
            AND enddt IN ('{enddt64}', '{enddt66}', '{enddt68}')
            AND qualifier IN {qualifier}
            and hddsn IN ('{serial}')
        '''
        return CSERsql
    
    def RRONRROsql(self):
        RRONRROsql = '''
            SELECT DISTINCT
                hddsn, procid, product, enddt, pfcode, mfgid, cmdname, phd 
                AS head, qualifier, subqualifier, {band} AS band, summaxrrodft, summaxnrrofft 
            FROM 
                hive.ghl2.{table}
            WHERE product = '{product}' 
            AND pfcode IN ('0000','{pf_code}')
            AND procid IN ('6400', '6600', '6800', '9000')
            AND cmdname = '{command_name}'
            AND enddt IN ('{enddt64}', '{enddt66}', '{enddt68}', '{enddt90}')
            AND hddsn IN ('{serial}')
        '''
        return RRONRROsql
    
    def RRONRRO_SPECIAL_PARAM(self):
        dict_rronrro = {
            "CMD":{
                "lds":"SVPesRtv",
                "ldv":"SVPesRtv",
                "ldt":"SVPesRtv",
                "adq":"SVPesRtv",
                "pdq":"SVPesRtv",
                "pcm":"SVPES"
            },
            'TABLE': {
                'lee': 'ccb_sv_pes_hddskcntscreen',
                'lhc': 'ccb_sv_pes_hddskcntscreen',
                'lse': 'ccb_sv_pes_hddskcntscreen',
                'pcm': 'ccb_sv_pes_hddskcntscreen',
                'vcc': 'ccb_sv_pes_hddskcntscreen',
                'vcl': 'ccb_sv_pes_hddskcntscreen',
                'vl6': 'ccb_sv_pes_hddskcntscreen',
                'vl8': 'ccb_sv_pes_hddskcntscreen',
                'adq': 'ccb_sv_pesrtv_hdcontact',
                'hdx': 'ccb_sv_pesrtv_hdcontact',
                'lds': 'ccb_sv_pesrtv_hdcontact',
                'ldv': 'ccb_sv_pesrtv_hdcontact',
                'ldt': 'ccb_sv_pesrtv_hdcontact',
                'pdq': 'ccb_sv_pesrtv_hdcontact'
            },
            'TEST_POINT': {
                'lee': 'testpoint',
                'lhc': 'testpoint',
                'lse': 'testpoint',
                'pcm': 'testpoint',
                'vcc': 'testpoint',
                'vcl': 'testpoint',
                'vl6': 'testpoint',
                'vl8': 'testpoint',
                'adq': 'testpointindex',
                'hdx': 'testpointindex',
                'lds': 'testpointindex',
                'ldv': 'testpointindex',
                'ldt': 'testpointindex',
                'pdq': 'testpointindex'
            },
        }
        return dict_rronrro
    
    def CSER_QUALIFIER(self):
        cser_qual_dict = {
            'CSER': {
                'pcm': ('39N', '900', 'B00', '49N', 'K00', 'M00'),
                'pdq': ('39N0', '9000', 'B000', '49N0', 'K000', 'M000'),
                'lds': ('39N0', '9000', 'B000', '49N0', 'K000', 'M000'),
                'ldv': ('39N0', '9000', 'B000', '49N0', 'K000', 'M000'),
                'ldt': ('39N0', '9000', 'B000', '49N0', 'K000', 'M000'),
                'adq': ('39N0', '39N1', '9000', '9001', 'B000', 'B001', '49N0', '49N1', 'K000', 'K001', 'M000', 'M001')
            },
        }
        
        return cser_qual_dict