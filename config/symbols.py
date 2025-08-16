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

# Company names for better news search
COMPANY_NAMES = {
    # Technology
    'AAPL': 'Apple Inc',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc',
    'GOOG': 'Google',
    'AMZN': 'Amazon.com Inc',
    'META': 'Meta Platforms Inc',
    'TSLA': 'Tesla Inc',
    'NVDA': 'NVIDIA Corporation',
    'NFLX': 'Netflix Inc',
    'ADBE': 'Adobe Inc',
    
    # Finance
    'JPM': 'JPMorgan Chase & Co',
    'BAC': 'Bank of America Corporation',
    'WFC': 'Wells Fargo & Company',
    'C': 'Citigroup Inc',
    'GS': 'Goldman Sachs Group Inc',
    'MS': 'Morgan Stanley',
    
    # Healthcare
    'UNH': 'UnitedHealth Group Inc',
    'JNJ': 'Johnson & Johnson',
    'PFE': 'Pfizer Inc',
    'ABT': 'Abbott Laboratories',
    
    # Energy
    'XOM': 'Exxon Mobil Corporation',
    'CVX': 'Chevron Corporation',
    'COP': 'ConocoPhillips',
    'EOG': 'EOG Resources Inc'
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