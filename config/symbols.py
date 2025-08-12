from typing import Dict, List

# Stock symbols organized by sector
STOCK_SYMBOLS = {
    'technology': [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 
        'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL'
    ],
    'finance': [
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'AXP'
    ],
    'healthcare': [
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'MDT', 'ABT'
    ],
    'energy': [
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX'
    ]
}

# Company names for better news search
COMPANY_NAMES = {
    'AAPL': 'Apple Inc',
    'GOOGL': 'Google Alphabet',
    'MSFT': 'Microsoft',
    'AMZN': 'Amazon',
    'TSLA': 'Tesla',
    'NVDA': 'NVIDIA',
    'META': 'Meta Facebook',
    'NFLX': 'Netflix',
    'AMD': 'Advanced Micro Devices',
    'INTC': 'Intel',
    'JPM': 'JPMorgan Chase',
    'BAC': 'Bank of America',
    'JNJ': 'Johnson & Johnson',
    'PFE': 'Pfizer',
    'XOM': 'ExxonMobil'
}

def get_all_symbols() -> List[str]:
    """Get all symbols from all sectors"""
    all_symbols = []
    for sector_symbols in STOCK_SYMBOLS.values():
        all_symbols.extend(sector_symbols)
    return list(set(all_symbols))

def get_symbols_by_sector(sector: str) -> List[str]:
    """Get symbols for a specific sector"""
    return STOCK_SYMBOLS.get(sector, [])

def get_company_name(symbol: str) -> str:
    """Get company name for symbol"""
    return COMPANY_NAMES.get(symbol, symbol)