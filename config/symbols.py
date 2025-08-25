from typing import Dict, List

# Stock symbols organized by sector
SYMBOLS = {
    'technology': [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 
        'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'PYPL', 'UBER',
        'SHOP', 'DOCU', 'ROKU', 'SNAP', 'PINS'
    ],
    'finance': [
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SCHW', 
        'USB', 'PNC', 'TFC', 'COF', 'ALL', 'AIG', 'MET', 'PRU', 'TRV',
        'CB', 'AFL', 'CINF', 'PGR', 'HIG', 'WRB'
    ],
    'healthcare': [
        'UNH', 'JNJ', 'PFE', 'ABT', 'TMO', 'MDT', 'DHR', 'BMY', 'ABBV',
        'MRK', 'CVS', 'CI', 'HUM', 'CNC', 'MOH', 'GILD', 'AMGN',
        'BIIB', 'VRTX', 'REGN', 'ISRG', 'SYK', 'BSX'
    ],
    'energy': [
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY',
        'BKR', 'HAL', 'DVN', 'FANG', 'APA', 'HES', 'NOV', 'RRC', 'MTDR', 
        'SM', 'RIG', 'HP'
    ]
}

COMPANY_NAMES = {
    # Technology
    'AAPL': 'Apple Inc',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc Class A',
    'GOOG': 'Alphabet Inc Class C',
    'AMZN': 'Amazon.com Inc',
    'META': 'Meta Platforms Inc',
    'TSLA': 'Tesla Inc',
    'NVDA': 'NVIDIA Corporation',
    'NFLX': 'Netflix Inc',
    'ADBE': 'Adobe Inc',
    'CRM': 'Salesforce Inc',
    'ORCL': 'Oracle Corporation',
    'INTC': 'Intel Corporation',
    'AMD': 'Advanced Micro Devices Inc',
    'PYPL': 'PayPal Holdings Inc',
    'UBER': 'Uber Technologies Inc',
    'SHOP': 'Shopify Inc',
    'DOCU': 'DocuSign Inc',
    'ROKU': 'Roku Inc',
    'SNAP': 'Snap Inc',
    'PINS': 'Pinterest Inc',

    # Finance
    'JPM': 'JPMorgan Chase & Co',
    'BAC': 'Bank of America Corporation',
    'WFC': 'Wells Fargo & Company',
    'C': 'Citigroup Inc',
    'GS': 'Goldman Sachs Group Inc',
    'MS': 'Morgan Stanley',
    'AXP': 'American Express Company',
    'BLK': 'BlackRock Inc',
    'SCHW': 'Charles Schwab Corporation',
    'USB': 'U.S. Bancorp',
    'PNC': 'PNC Financial Services Group Inc',
    'TFC': 'Truist Financial Corporation',
    'COF': 'Capital One Financial Corporation',
    'ALL': 'Allstate Corporation',
    'AIG': 'American International Group Inc',
    'MET': 'MetLife Inc',
    'PRU': 'Prudential Financial Inc',
    'TRV': 'Travelers Companies Inc',
    'CB': 'Chubb Limited',
    'AFL': 'Aflac Incorporated',
    'CINF': 'Cincinnati Financial Corporation',
    'PGR': 'Progressive Corporation',
    'HIG': 'Hartford Financial Services Group Inc',
    'WRB': 'W. R. Berkley Corporation',

    # Healthcare
    'UNH': 'UnitedHealth Group Incorporated',
    'JNJ': 'Johnson & Johnson',
    'PFE': 'Pfizer Inc',
    'ABT': 'Abbott Laboratories',
    'TMO': 'Thermo Fisher Scientific Inc',
    'MDT': 'Medtronic plc',
    'DHR': 'Danaher Corporation',
    'BMY': 'Bristol-Myers Squibb Company',
    'ABBV': 'AbbVie Inc',
    'MRK': 'Merck & Co Inc',
    'CVS': 'CVS Health Corporation',
    'CI': 'Cigna Group',
    'HUM': 'Humana Inc',
    'CNC': 'Centene Corporation',
    'MOH': 'Molina Healthcare Inc',
    'GILD': 'Gilead Sciences Inc',
    'AMGN': 'Amgen Inc',
    'BIIB': 'Biogen Inc',
    'VRTX': 'Vertex Pharmaceuticals Incorporated',
    'REGN': 'Regeneron Pharmaceuticals Inc',
    'ISRG': 'Intuitive Surgical Inc',
    'SYK': 'Stryker Corporation',
    'BSX': 'Boston Scientific Corporation',

    # Energy
    'XOM': 'Exxon Mobil Corporation',
    'CVX': 'Chevron Corporation',
    'COP': 'ConocoPhillips',
    'EOG': 'EOG Resources Inc',
    'SLB': 'Schlumberger Limited',
    'MPC': 'Marathon Petroleum Corporation',
    'PSX': 'Phillips 66',
    'VLO': 'Valero Energy Corporation',
    'OXY': 'Occidental Petroleum Corporation',
    'BKR': 'Baker Hughes Company',
    'HAL': 'Halliburton Company',
    'DVN': 'Devon Energy Corporation',
    'FANG': 'Diamondback Energy Inc',
    'APA': 'APA Corporation',
    'HES': 'Hess Corporation',
    'NOV': 'NOV Inc',
    'RRC': 'Range Resources Corporation',
    'MTDR': 'Matador Resources Company',
    'SM': 'SM Energy Company',
    'RIG': 'Transocean Ltd',
    'HP': 'Helmerich & Payne Inc'
}

# Company names to cross-check (abbreviate to get most results)
COMPANY_NAMES_ABR = {
    # Technology
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'GOOGL':'Alphabet',
    'GOOG': 'Alphabet',
    'AMZN': 'Amazon',
    'META': 'Meta Platforms',
    'TSLA': 'Tesla',
    'NVDA': 'NVIDIA',
    'NFLX': 'Netflix',
    'ADBE': 'Adobe',
    'CRM': 'Salesforce',
    'ORCL': 'Oracle',
    'INTC': 'Intel',
    'AMD': 'Advanced Micro Devices',
    'PYPL': 'PayPal',
    'UBER': 'Uber',
    'SHOP': 'Shopify',
    'DOCU': 'DocuSign',
    'ROKU': 'Roku',
    'SNAP': 'Snap',
    'PINS': 'Pinterest',

    # Finance
    'JPM': 'JPMorgan',
    'BAC': 'Bank of America',
    'WFC': 'Wells Fargo',
    'C': 'Citigroup',
    'GS': 'Goldman Sachs',
    'MS': 'Morgan Stanley',
    'AXP': 'American Express',
    'BLK': 'BlackRock',
    'SCHW': 'Charles Schwab',
    'USB': 'U.S. Bancorp',
    'PNC': 'PNC Financial',
    'TFC': 'Truist',
    'COF': 'Capital One',
    'ALL': 'Allstate',
    'AIG': 'American International Group',
    'MET': 'MetLife',
    'PRU': 'Prudential',
    'TRV': 'Travelers',
    'CB': 'Chubb',
    'AFL': 'Aflac',
    'CINF': 'Cincinnati',
    'PGR': 'Progressive',
    'HIG': 'Hartford',
    'WRB': 'Berkley',

    # Healthcare
    'UNH': 'UnitedHealth',
    'JNJ': 'Johnson & Johnson',
    'PFE': 'Pfizer',
    'ABT': 'Abbott',
    'TMO': 'Thermo Fisher',
    'MDT': 'Medtronic',
    'DHR': 'Danaher',
    'BMY': 'Bristol-Myers',
    'ABBV': 'AbbVie',
    'MRK': 'Merck & Co',
    'CVS': 'CVS',
    'CI': 'Cigna',
    'HUM': 'Humana',
    'CNC': 'Centene',
    'MOH': 'Molina',
    'GILD': 'Gilead',
    'AMGN': 'Amgen',
    'BIIB': 'Biogen',
    'VRTX': 'Vertex',
    'REGN': 'Regeneron',
    'ISRG': 'Intuitive',
    'SYK': 'Stryker',
    'BSX': 'Boston Scientific',

    # Energy
    'XOM': 'Exxon Mobil',
    'CVX': 'Chevron',
    'COP': 'ConocoPhillips',
    'EOG': 'EOG Resources',
    'SLB': 'Schlumberger',
    'MPC': 'Marathon',
    'PSX': 'Phillips',
    'VLO': 'Valero Energy',
    'OXY': 'Occidental Petroleum',
    'BKR': 'Baker Hughes',
    'HAL': 'Halliburton',
    'DVN': 'Devon Energy',
    'FANG': 'Diamondback',
    'APA': 'APA',
    'HES': 'Hess',
    'NOV': 'NOV',
    'RRC': 'Range Resources',
    'MTDR': 'Matador',
    'SM': 'SM Energy',
    'RIG': 'Transocean',
    'HP': 'Helmerich'
}

# Reverse mapping implementation for sector lookup
SYMBOL_TO_SECTOR = {}
for sector, symbols_list in SYMBOLS.items():
    for symbol in symbols_list:
        SYMBOL_TO_SECTOR[symbol] = sector

def get_all_symbols() -> List[str]:
    """Get all symbols from all sectors"""
    all_symbols = []
    for sector_symbols in SYMBOLS.values():
        all_symbols.extend(sector_symbols)
    return sorted(list(set(all_symbols)))

def get_symbols_by_sector(sector: str) -> List[str]:
    """Get symbols for a specific sector"""
    return SYMBOLS.get(sector.lower(), [])

def get_company_name(symbol: str) -> str:
    """Get company name for symbol"""
    return COMPANY_NAMES.get(symbol.upper, symbol)

def get_company_name_abbr(symbol: str) -> str:
    """Get company name for symbol"""
    return COMPANY_NAMES_ABR.get(symbol.upper, symbol)

def get_symbol_sector(symbol: str) -> str:
    """Get sector for a specific symbol"""
    return SYMBOL_TO_SECTOR.get(symbol.upper())

def get_sector_peers(symbol, include_self=False):
    """Get peer symbols in the same sector"""
    sector = get_symbol_sector(symbol)
    if not sector:
        return []
    
    peers = get_symbols_by_sector(symbol)
    if not include_self and symbol.upper() in peers:
        peers = [s for s in peers if s != symbol.upper()]
    return peers

def validate_symbol(symbol):
    """Check if symbol exists in above tables"""
    return symbol.upper() in SYMBOL_TO_SECTOR

def get_sectors():
    """Get a list of all available sectors"""
    return list(SYMBOLS.keys())

def get_sector_summary():
    """Get summary statistics for each sector"""
    summary = {}
    for sector, symbols in SYMBOLS.items():
        summary[sector] = {
            'symbol_count': len(symbols),
            'symbols': symbols[:5] # Use first 5
        }
    return summary