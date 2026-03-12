"""ML Pattern Scorer configuration."""

import os
import sys

# Paths
DATA_DIR = 'C:/seasonals/data'
CSV_DIR = 'C:/seasonals/data/csv'
US_CSV_DIR = os.path.join(CSV_DIR, 'US')
ETF_CSV_DIR = os.path.join(CSV_DIR, 'ETF')
INDX_CSV_DIR = os.path.join(CSV_DIR, 'INDX')
OPP_BY_SYMBOL_DIR = os.path.join(DATA_DIR, 'sp500', 'opp_by_symbol')
EARNINGS_DIR = 'C:/seasonals/edgar/earnings'
SP500_SYMBOLS = os.path.join(DATA_DIR, 'sp500_symbols.csv')

FEATURE_CACHE_DIR = 'C:/seasonals/ml_scorer/features'
MODEL_DIR = 'C:/seasonals/ml_scorer/models'
RESULTS_DIR = 'C:/seasonals/ml_scorer/results'

# Parallelization: override with env var ML_SCORER_NJOBS or --njobs CLI arg
N_JOBS = int(os.environ.get('ML_SCORER_NJOBS', 24))

# Depth profile
MAX_DEPTH_CAP = 35  # cap depth_utilization denominator at 35 years

# SPX seasonal regime: use 1928-1999 data only (no leakage into 2000+ training data)
SPX_SEASONAL_CUTOFF_YEAR = 1999
SPX_SEASONAL_FORWARD_DAYS = 15  # trading days for forward return calculation

# All year combos (year1_year2) - non-PE
YEAR_COMBOS = []
for y1 in range(5, 41):
    floor = int(y1 * 0.8)
    for y2 in range(floor, y1 + 1):
        YEAR_COMBOS.append(f"{y1}_{y2}")

# PE combos
PE_COMBOS = [f"{y}_{y}_PE2" for y in range(4, 12)]

# Sector ETF mapping (GICS sector -> SPDR ETF)
SECTOR_ETF = {
    'Information Technology': 'XLK',
    'Health Care': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Communication Services': 'XLC',
    'Industrials': 'XLI',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB',
}

# SP500 ticker -> GICS sector (as of early 2026)
# This covers all 475 tickers in opp_by_symbol
TICKER_SECTOR = {
    'AAPL': 'Information Technology', 'ABBV': 'Health Care', 'ABT': 'Health Care',
    'ACN': 'Information Technology', 'ADBE': 'Information Technology', 'ADI': 'Information Technology',
    'ADM': 'Consumer Staples', 'ADP': 'Industrials', 'ADSK': 'Information Technology',
    'AEE': 'Utilities', 'AEP': 'Utilities', 'AES': 'Utilities',
    'AFL': 'Financials', 'AIG': 'Financials', 'AIZ': 'Financials',
    'AJG': 'Financials', 'AKAM': 'Information Technology', 'ALB': 'Materials',
    'ALGN': 'Health Care', 'ALL': 'Financials', 'ALLE': 'Industrials',
    'AMAT': 'Information Technology', 'AMCR': 'Materials', 'AMD': 'Information Technology',
    'AME': 'Industrials', 'AMGN': 'Health Care', 'AMP': 'Financials',
    'AMT': 'Real Estate', 'AMZN': 'Consumer Discretionary', 'ANET': 'Information Technology',
    'ANSS': 'Information Technology', 'AON': 'Financials', 'AOS': 'Industrials',
    'APA': 'Energy', 'APD': 'Materials', 'APH': 'Information Technology',
    'APTV': 'Consumer Discretionary', 'ARE': 'Real Estate', 'ATO': 'Utilities',
    'ATVI': 'Communication Services', 'AVB': 'Real Estate', 'AVGO': 'Information Technology',
    'AVY': 'Materials', 'AWK': 'Utilities', 'AXP': 'Financials',
    'AZO': 'Consumer Discretionary', 'BA': 'Industrials', 'BAC': 'Financials',
    'BAX': 'Health Care', 'BBWI': 'Consumer Discretionary', 'BBY': 'Consumer Discretionary',
    'BDX': 'Health Care', 'BEN': 'Financials', 'BF.B': 'Consumer Staples',
    'BG': 'Consumer Staples', 'BIIB': 'Health Care', 'BIO': 'Health Care',
    'BK': 'Financials', 'BKNG': 'Consumer Discretionary', 'BKR': 'Energy',
    'BLK': 'Financials', 'BMY': 'Health Care', 'BR': 'Industrials',
    'BRK.B': 'Financials', 'BRO': 'Financials', 'BSX': 'Health Care',
    'BWA': 'Consumer Discretionary', 'BXP': 'Real Estate', 'C': 'Financials',
    'CAG': 'Consumer Staples', 'CAH': 'Health Care', 'CARR': 'Industrials',
    'CAT': 'Industrials', 'CB': 'Financials', 'CBOE': 'Financials',
    'CBRE': 'Real Estate', 'CCI': 'Real Estate', 'CCL': 'Consumer Discretionary',
    'CDAY': 'Information Technology', 'CDNS': 'Information Technology', 'CDW': 'Information Technology',
    'CE': 'Materials', 'CEG': 'Utilities', 'CF': 'Materials',
    'CFG': 'Financials', 'CHD': 'Consumer Staples', 'CHRW': 'Industrials',
    'CHTR': 'Communication Services', 'CI': 'Health Care', 'CINF': 'Financials',
    'CL': 'Consumer Staples', 'CLX': 'Consumer Staples', 'CMA': 'Financials',
    'CMCSA': 'Communication Services', 'CME': 'Financials', 'CMG': 'Consumer Discretionary',
    'CMI': 'Industrials', 'CMS': 'Utilities', 'CNC': 'Health Care',
    'CNP': 'Utilities', 'COF': 'Financials', 'COO': 'Health Care',
    'COP': 'Energy', 'COST': 'Consumer Staples', 'CPB': 'Consumer Staples',
    'CPRT': 'Industrials', 'CPT': 'Real Estate', 'CRL': 'Health Care',
    'CRM': 'Information Technology', 'CSCO': 'Information Technology', 'CSGP': 'Real Estate',
    'CSX': 'Industrials', 'CTAS': 'Industrials', 'CTLT': 'Health Care',
    'CTRA': 'Energy', 'CTSH': 'Information Technology', 'CTVA': 'Materials',
    'CVS': 'Health Care', 'CVX': 'Energy', 'CZR': 'Consumer Discretionary',
    'D': 'Utilities', 'DAL': 'Industrials', 'DD': 'Materials',
    'DE': 'Industrials', 'DFS': 'Financials', 'DG': 'Consumer Discretionary',
    'DGX': 'Health Care', 'DHI': 'Consumer Discretionary', 'DHR': 'Health Care',
    'DIS': 'Communication Services', 'DISH': 'Communication Services', 'DLR': 'Real Estate',
    'DLTR': 'Consumer Discretionary', 'DOV': 'Industrials', 'DOW': 'Materials',
    'DPZ': 'Consumer Discretionary', 'DRI': 'Consumer Discretionary', 'DTE': 'Utilities',
    'DUK': 'Utilities', 'DVA': 'Health Care', 'DVN': 'Energy',
    'DXC': 'Information Technology', 'DXCM': 'Health Care', 'EA': 'Communication Services',
    'EBAY': 'Consumer Discretionary', 'ECL': 'Materials', 'ED': 'Utilities',
    'EFX': 'Industrials', 'EIX': 'Utilities', 'EL': 'Consumer Staples',
    'EMN': 'Materials', 'EMR': 'Industrials', 'ENPH': 'Information Technology',
    'EOG': 'Energy', 'EPAM': 'Information Technology', 'EQIX': 'Real Estate',
    'EQR': 'Real Estate', 'EQT': 'Energy', 'ES': 'Utilities',
    'ESS': 'Real Estate', 'ETN': 'Industrials', 'ETR': 'Utilities',
    'ETSY': 'Consumer Discretionary', 'EVRG': 'Utilities', 'EW': 'Health Care',
    'EXC': 'Utilities', 'EXPD': 'Industrials', 'EXPE': 'Consumer Discretionary',
    'EXR': 'Real Estate', 'F': 'Consumer Discretionary', 'FANG': 'Energy',
    'FAST': 'Industrials', 'FBHS': 'Industrials', 'FCX': 'Materials',
    'FDS': 'Financials', 'FDX': 'Industrials', 'FE': 'Utilities',
    'FFIV': 'Information Technology', 'FIS': 'Financials', 'FISV': 'Financials',
    'FITB': 'Financials', 'FLT': 'Financials', 'FMC': 'Materials',
    'FOX': 'Communication Services', 'FOXA': 'Communication Services', 'FRC': 'Financials',
    'FRT': 'Real Estate', 'FTNT': 'Information Technology', 'FTV': 'Industrials',
    'GD': 'Industrials', 'GE': 'Industrials', 'GILD': 'Health Care',
    'GIS': 'Consumer Staples', 'GL': 'Financials', 'GLW': 'Information Technology',
    'GM': 'Consumer Discretionary', 'GNRC': 'Industrials', 'GOOG': 'Communication Services',
    'GOOGL': 'Communication Services', 'GPC': 'Consumer Discretionary', 'GPN': 'Financials',
    'GRMN': 'Consumer Discretionary', 'GS': 'Financials', 'GWW': 'Industrials',
    'HAL': 'Energy', 'HAS': 'Consumer Discretionary', 'HBAN': 'Financials',
    'HCA': 'Health Care', 'HD': 'Consumer Discretionary', 'PEAK': 'Real Estate',
    'HES': 'Energy', 'HIG': 'Financials', 'HII': 'Industrials',
    'HLT': 'Consumer Discretionary', 'HOLX': 'Health Care', 'HON': 'Industrials',
    'HPE': 'Information Technology', 'HPQ': 'Information Technology', 'HRL': 'Consumer Staples',
    'HSIC': 'Health Care', 'HST': 'Real Estate', 'HSY': 'Consumer Staples',
    'HUM': 'Health Care', 'HWM': 'Industrials', 'IBM': 'Information Technology',
    'ICE': 'Financials', 'IDXX': 'Health Care', 'IEX': 'Industrials',
    'IFF': 'Materials', 'ILMN': 'Health Care', 'INCY': 'Health Care',
    'INTC': 'Information Technology', 'INTU': 'Information Technology', 'INVH': 'Real Estate',
    'IP': 'Materials', 'IPG': 'Communication Services', 'IQV': 'Health Care',
    'IR': 'Industrials', 'IRM': 'Real Estate', 'ISRG': 'Health Care',
    'IT': 'Information Technology', 'ITW': 'Industrials', 'IVZ': 'Financials',
    'J': 'Industrials', 'JBHT': 'Industrials', 'JCI': 'Industrials',
    'JKHY': 'Information Technology', 'JNJ': 'Health Care', 'JNPR': 'Information Technology',
    'JPM': 'Financials', 'K': 'Consumer Staples', 'KDP': 'Consumer Staples',
    'KEY': 'Financials', 'KEYS': 'Information Technology', 'KHC': 'Consumer Staples',
    'KIM': 'Real Estate', 'KLAC': 'Information Technology', 'KMB': 'Consumer Staples',
    'KMI': 'Energy', 'KMX': 'Consumer Discretionary', 'KO': 'Consumer Staples',
    'KR': 'Consumer Staples', 'L': 'Financials', 'LDOS': 'Industrials',
    'LEN': 'Consumer Discretionary', 'LH': 'Health Care', 'LHX': 'Industrials',
    'LIN': 'Materials', 'LKQ': 'Consumer Discretionary', 'LLY': 'Health Care',
    'LMT': 'Industrials', 'LNC': 'Financials', 'LNT': 'Utilities',
    'LOW': 'Consumer Discretionary', 'LRCX': 'Information Technology', 'LUMN': 'Communication Services',
    'LUV': 'Industrials', 'LVS': 'Consumer Discretionary', 'LW': 'Consumer Staples',
    'LYB': 'Materials', 'LYV': 'Communication Services', 'MA': 'Financials',
    'MAA': 'Real Estate', 'MAR': 'Consumer Discretionary', 'MAS': 'Industrials',
    'MCD': 'Consumer Discretionary', 'MCHP': 'Information Technology', 'MCK': 'Health Care',
    'MCO': 'Financials', 'MDLZ': 'Consumer Staples', 'MDT': 'Health Care',
    'MET': 'Financials', 'META': 'Communication Services', 'MGM': 'Consumer Discretionary',
    'MHK': 'Consumer Discretionary', 'MKC': 'Consumer Staples', 'MKTX': 'Financials',
    'MLM': 'Materials', 'MMC': 'Financials', 'MMM': 'Industrials',
    'MNST': 'Consumer Staples', 'MO': 'Consumer Staples', 'MOH': 'Health Care',
    'MOS': 'Materials', 'MPC': 'Energy', 'MPWR': 'Information Technology',
    'MRK': 'Health Care', 'MRNA': 'Health Care', 'MRO': 'Energy',
    'MS': 'Financials', 'MSCI': 'Financials', 'MSFT': 'Information Technology',
    'MSI': 'Information Technology', 'MTB': 'Financials', 'MTCH': 'Communication Services',
    'MTD': 'Health Care', 'MU': 'Information Technology', 'NCLH': 'Consumer Discretionary',
    'NDAQ': 'Financials', 'NDSN': 'Industrials', 'NEE': 'Utilities',
    'NEM': 'Materials', 'NFLX': 'Communication Services', 'NI': 'Utilities',
    'NKE': 'Consumer Discretionary', 'NOC': 'Industrials', 'NOW': 'Information Technology',
    'NRG': 'Utilities', 'NSC': 'Industrials', 'NTAP': 'Information Technology',
    'NTRS': 'Financials', 'NUE': 'Materials', 'NVDA': 'Information Technology',
    'NVR': 'Consumer Discretionary', 'NWL': 'Consumer Discretionary', 'NWS': 'Communication Services',
    'NWSA': 'Communication Services', 'NXPI': 'Information Technology', 'O': 'Real Estate',
    'ODFL': 'Industrials', 'OGN': 'Health Care', 'OKE': 'Energy',
    'OMC': 'Communication Services', 'ON': 'Information Technology', 'ORCL': 'Information Technology',
    'ORLY': 'Consumer Discretionary', 'OTIS': 'Industrials', 'OXY': 'Energy',
    'PARA': 'Communication Services', 'PAYC': 'Information Technology', 'PAYX': 'Industrials',
    'PCAR': 'Industrials', 'PCG': 'Utilities', 'PEAK': 'Real Estate',
    'PEG': 'Utilities', 'PEP': 'Consumer Staples', 'PFE': 'Health Care',
    'PFG': 'Financials', 'PG': 'Consumer Staples', 'PGR': 'Financials',
    'PH': 'Industrials', 'PHM': 'Consumer Discretionary', 'PKG': 'Materials',
    'PKI': 'Health Care', 'PLD': 'Real Estate', 'PM': 'Consumer Staples',
    'PNC': 'Financials', 'PNR': 'Industrials', 'PNW': 'Utilities',
    'POOL': 'Consumer Discretionary', 'PPG': 'Materials', 'PPL': 'Utilities',
    'PRU': 'Financials', 'PSA': 'Real Estate', 'PSX': 'Energy',
    'PTC': 'Information Technology', 'PVH': 'Consumer Discretionary', 'PWR': 'Industrials',
    'PXD': 'Energy', 'PYPL': 'Financials', 'QCOM': 'Information Technology',
    'QRVO': 'Information Technology', 'RCL': 'Consumer Discretionary', 'RE': 'Financials',
    'REG': 'Real Estate', 'REGN': 'Health Care', 'RF': 'Financials',
    'RHI': 'Industrials', 'RJF': 'Financials', 'RL': 'Consumer Discretionary',
    'RMD': 'Health Care', 'ROK': 'Industrials', 'ROL': 'Industrials',
    'ROP': 'Industrials', 'ROST': 'Consumer Discretionary', 'RSG': 'Industrials',
    'RTX': 'Industrials', 'SBAC': 'Real Estate', 'SBNY': 'Financials',
    'SBUX': 'Consumer Discretionary', 'SCHW': 'Financials', 'SEE': 'Materials',
    'SHW': 'Materials', 'SIVB': 'Financials', 'SJM': 'Consumer Staples',
    'SLB': 'Energy', 'SNA': 'Industrials', 'SNPS': 'Information Technology',
    'SO': 'Utilities', 'SPG': 'Real Estate', 'SPGI': 'Financials',
    'SRE': 'Utilities', 'STE': 'Health Care', 'STT': 'Financials',
    'STX': 'Information Technology', 'STZ': 'Consumer Staples', 'SWK': 'Industrials',
    'SWKS': 'Information Technology', 'SYF': 'Financials', 'SYK': 'Health Care',
    'SYY': 'Consumer Staples', 'T': 'Communication Services', 'TAP': 'Consumer Staples',
    'TDG': 'Industrials', 'TDY': 'Industrials', 'TECH': 'Health Care',
    'TEL': 'Information Technology', 'TER': 'Information Technology', 'TFC': 'Financials',
    'TFX': 'Health Care', 'TGT': 'Consumer Discretionary', 'TMO': 'Health Care',
    'TMUS': 'Communication Services', 'TPR': 'Consumer Discretionary', 'TRGP': 'Energy',
    'TRMB': 'Information Technology', 'TROW': 'Financials', 'TRV': 'Financials',
    'TSCO': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'TSN': 'Consumer Staples',
    'TT': 'Industrials', 'TTWO': 'Communication Services', 'TXN': 'Information Technology',
    'TXT': 'Industrials', 'TYL': 'Information Technology', 'UAL': 'Industrials',
    'UDR': 'Real Estate', 'UHS': 'Health Care', 'ULTA': 'Consumer Discretionary',
    'UNH': 'Health Care', 'UNP': 'Industrials', 'UPS': 'Industrials',
    'URI': 'Industrials', 'USB': 'Financials', 'V': 'Financials',
    'VFC': 'Consumer Discretionary', 'VICI': 'Real Estate', 'VLO': 'Energy',
    'VMC': 'Materials', 'VNO': 'Real Estate', 'VRSK': 'Industrials',
    'VRSN': 'Information Technology', 'VRTX': 'Health Care', 'VTR': 'Real Estate',
    'VTRS': 'Health Care', 'VZ': 'Communication Services', 'WAB': 'Industrials',
    'WAT': 'Health Care', 'WBA': 'Consumer Staples', 'WBD': 'Communication Services',
    'WDC': 'Information Technology', 'WEC': 'Utilities', 'WELL': 'Real Estate',
    'WFC': 'Financials', 'WHR': 'Consumer Discretionary', 'WM': 'Industrials',
    'WMB': 'Energy', 'WMT': 'Consumer Staples', 'WRB': 'Financials',
    'WRK': 'Materials', 'WST': 'Health Care', 'WTW': 'Financials',
    'WY': 'Real Estate', 'WYNN': 'Consumer Discretionary', 'XEL': 'Utilities',
    'XOM': 'Energy', 'XRAY': 'Health Care', 'XYL': 'Industrials',
    'YUM': 'Consumer Discretionary', 'ZBH': 'Health Care', 'ZBRA': 'Information Technology',
    'ZION': 'Financials', 'ZTS': 'Health Care',
    # Additional tickers that may be in opp_by_symbol but not in current S&P 500
    'ACGL': 'Financials', 'ABNB': 'Consumer Discretionary', 'GEHC': 'Health Care',
    'GEV': 'Utilities', 'KVUE': 'Consumer Staples', 'PANW': 'Information Technology',
    'VLTO': 'Industrials', 'SMCI': 'Information Technology', 'PLTR': 'Information Technology',
    'CRWD': 'Information Technology', 'DECK': 'Consumer Discretionary', 'FICO': 'Information Technology',
    'GDDY': 'Information Technology', 'HUBB': 'Industrials', 'KKR': 'Financials',
    'AXON': 'Industrials', 'VST': 'Utilities', 'TPL': 'Energy',
    'ERIE': 'Financials', 'SW': 'Industrials', 'PODD': 'Health Care',
    'SOLV': 'Health Care', 'BLDR': 'Consumer Discretionary', 'DASH': 'Consumer Discretionary',
}

# Presidential Election cycle: year -> PE year (1=post-election, 2=midterm, 3=pre-election, 4=election)
def get_pe_year(year):
    return ((year - 2001) % 4) + 1  # 2001=post-election(1), 2002=midterm(2), 2003=pre-election(3), 2004=election(4)

# ML thresholds by strategy tier
THRESHOLDS = {
    'small':  {'ml_min': 85, 'max_positions': 4, 'days_min': 10, 'days_max': 30, 'profit_per_day_min': 0.3},
    'medium': {'ml_min': 70, 'max_positions': 10, 'days_min': 7, 'days_max': 60, 'profit_per_day_min': 0.15},
    'prop':   {'ml_min': 65, 'max_positions': 15, 'days_min': 5, 'days_max': 250, 'profit_per_day_min': 0.0},
}
