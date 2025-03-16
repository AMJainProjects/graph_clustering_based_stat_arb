import pandas as pd
import requests
import io
from typing import Dict, Optional, List

def download_ff12_industry_definitions() -> pd.DataFrame:
    """
    Downloads Fama-French 12 Industry definitions.
    
    Returns:
        DataFrame with SIC codes and industry classifications
    """
    # URL for the Fama-French 12 Industry definitions
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes12.zip"
    
    try:
        # Download the zip file
        response = requests.get(url)
        response.raise_for_status()
        
        # Read the zip file
        with io.BytesIO(response.content) as zip_file:
            # Read the CSV file from the zip
            industry_def = pd.read_csv(
                zip_file, 
                skiprows=9,  # Skip header rows
                sep='\s+',   # Whitespace separator
                header=None,
                names=['SIC_start', 'SIC_end', 'industry_id', 'industry_name']
            )
        
        return industry_def
    
    except Exception as e:
        print(f"Error downloading Fama-French industry definitions: {e}")
        return pd.DataFrame()

def map_sic_to_industry(
    sic_code: int,
    industry_def: pd.DataFrame
) -> Optional[int]:
    """
    Maps a SIC code to a Fama-French 12 Industry ID.
    
    Args:
        sic_code: SIC code
        industry_def: DataFrame with industry definitions
        
    Returns:
        Industry ID or None if not found
    """
    # Find rows where SIC code is in range
    matches = industry_def[
        (industry_def['SIC_start'] <= sic_code) & 
        (industry_def['SIC_end'] >= sic_code)
    ]
    
    if not matches.empty:
        return matches.iloc[0]['industry_id']
    else:
        return None

def get_stock_sic_codes(tickers: List[str]) -> Dict[str, int]:
    """
    Gets SIC codes for a list of stock tickers using yfinance.
    
    Args:
        tickers: List of stock ticker symbols
        
    Returns:
        Dictionary mapping tickers to SIC codes
    """
    import yfinance as yf
    
    sic_codes = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            profile = stock.get_info()
            
            # YFinance doesn't directly provide SIC codes, so we'll use the sector/industry
            # for a simplified mapping
            sector = profile.get('sector', None)
            industry = profile.get('industry', None)
            
            # Map sector/industry to a mock SIC code for demonstration
            if sector == 'Technology':
                sic_codes[ticker] = 7370  # Computer Programming, Data Processing, etc.
            elif sector == 'Financial Services':
                sic_codes[ticker] = 6000  # Depository Institutions
            elif sector == 'Healthcare':
                sic_codes[ticker] = 8000  # Health Services
            elif sector == 'Consumer Cyclical':
                sic_codes[ticker] = 5200  # Retail
            elif sector == 'Communication Services':
                sic_codes[ticker] = 4800  # Communications
            elif sector == 'Consumer Defensive':
                sic_codes[ticker] = 2000  # Food and Kindred Products
            elif sector == 'Energy':
                sic_codes[ticker] = 1300  # Oil and Gas Extraction
            elif sector == 'Industrials':
                sic_codes[ticker] = 3500  # Industrial and Commercial Machinery
            elif sector == 'Basic Materials':
                sic_codes[ticker] = 2800  # Chemicals and Allied Products
            elif sector == 'Utilities':
                sic_codes[ticker] = 4900  # Electric, Gas, and Sanitary Services
            elif sector == 'Real Estate':
                sic_codes[ticker] = 6500  # Real Estate
            else:
                sic_codes[ticker] = 9900  # Nonclassifiable Establishments
        
        except Exception as e:
            print(f"Error getting SIC code for {ticker}: {e}")
            sic_codes[ticker] = 9900  # Default to nonclassifiable
    
    return sic_codes

def get_fama_french_industries(tickers: List[str]) -> Dict[str, int]:
    """
    Gets Fama-French 12 Industry classifications for a list of stock tickers.
    
    Args:
        tickers: List of stock ticker symbols
        
    Returns:
        Dictionary mapping tickers to industry IDs
    """
    # Get industry definitions
    industry_def = download_ff12_industry_definitions()
    
    if industry_def.empty:
        # If we couldn't download the definitions, use a predefined mapping
        return get_predefined_industry_mapping(tickers)
    
    # Get SIC codes for tickers
    sic_codes = get_stock_sic_codes(tickers)
    
    # Map SIC codes to industries
    industry_mapping = {}
    
    for ticker, sic_code in sic_codes.items():
        industry_id = map_sic_to_industry(sic_code, industry_def)
        if industry_id is not None:
            industry_mapping[ticker] = industry_id
        else:
            # Default to "Other" if no mapping is found
            industry_mapping[ticker] = 12
    
    return industry_mapping

def get_predefined_industry_mapping(tickers: List[str]) -> Dict[str, int]:
    """
    Gets a predefined Fama-French 12 Industry mapping for common stock tickers.
    
    Args:
        tickers: List of stock ticker symbols
        
    Returns:
        Dictionary mapping tickers to industry IDs
    """
    # Predefined mappings for common tickers
    # 1: NoDur (Consumer Non-Durables)
    # 2: Durbl (Consumer Durables)
    # 3: Manuf (Manufacturing)
    # 4: Enrgy (Oil, Gas, and Coal Extraction and Products)
    # 5: Chems (Chemicals and Allied Products)
    # 6: BusEq (Business Equipment)
    # 7: Telcm (Telephone and Television Transmission)
    # 8: Utils (Utilities)
    # 9: Shops (Wholesale, Retail, and Some Services)
    # 10: Hlth (Healthcare, Medical Equipment, and Drugs)
    # 11: Money (Finance)
    # 12: Other (Other)
    
    common_mappings = {
        # Consumer Non-Durables (Food, Tobacco, Textiles, Apparel, Leather, Toys)
        'PG': 1,    # Procter & Gamble
        'KO': 1,    # Coca-Cola
        'PEP': 1,   # PepsiCo
        'MO': 1,    # Altria
        'CL': 1,    # Colgate-Palmolive
        'GIS': 1,   # General Mills
        'K': 1,     # Kellogg
        'JNJ': 1,   # Johnson & Johnson (also in Healthcare)
        
        # Consumer Durables (Cars, TVs, Furniture, Household Appliances)
        'F': 2,     # Ford
        'GM': 2,    # General Motors
        'HOG': 2,   # Harley-Davidson
        'NKE': 2,   # Nike
        'WHR': 2,   # Whirlpool
        'TSLA': 2,  # Tesla
        
        # Manufacturing (Machinery, Trucks, Planes, Office Furniture, Paper, Printing)
        'GE': 3,    # General Electric
        'BA': 3,    # Boeing
        'CAT': 3,   # Caterpillar
        'DE': 3,    # Deere
        'MMM': 3,   # 3M
        'CMI': 3,   # Cummins
        
        # Energy (Oil, Gas, and Coal Extraction and Products)
        'XOM': 4,   # Exxon Mobil
        'CVX': 4,   # Chevron
        'COP': 4,   # ConocoPhillips
        'SLB': 4,   # Schlumberger
        'OXY': 4,   # Occidental Petroleum
        'HAL': 4,   # Halliburton
        
        # Chemicals and Allied Products
        'DOW': 5,   # Dow Chemical
        'DD': 5,    # DuPont
        'LYB': 5,   # LyondellBasell
        'PPG': 5,   # PPG Industries
        'APD': 5,   # Air Products
        
        # Business Equipment (Computers, Software, Electronic Equipment)
        'AAPL': 6,  # Apple
        'MSFT': 6,  # Microsoft
        'INTC': 6,  # Intel
        'IBM': 6,   # IBM
        'ORCL': 6,  # Oracle
        'HPQ': 6,   # HP
        'DELL': 6,  # Dell
        'CSCO': 6,  # Cisco
        'NVDA': 6,  # NVIDIA
        'AMD': 6,   # AMD
        'ADBE': 6,  # Adobe
        'CRM': 6,   # Salesforce
        'GOOGL': 6, # Alphabet (Google)
        'META': 6,  # Meta (Facebook)
        
        # Telephone and Television Transmission
        'T': 7,     # AT&T
        'VZ': 7,    # Verizon
        'TMUS': 7,  # T-Mobile
        'CMCSA': 7, # Comcast
        'NFLX': 7,  # Netflix
        'DIS': 7,   # Disney
        
        # Utilities
        'NEE': 8,   # NextEra Energy
        'DUK': 8,   # Duke Energy
        'SO': 8,    # Southern Company
        'D': 8,     # Dominion Energy
        'AEP': 8,   # American Electric Power
        
        # Wholesale, Retail, and Some Services
        'WMT': 9,   # Walmart
        'AMZN': 9,  # Amazon
        'HD': 9,    # Home Depot
        'TGT': 9,   # Target
        'LOW': 9,   # Lowe's
        'COST': 9,  # Costco
        'MCD': 9,   # McDonald's
        'SBUX': 9,  # Starbucks
        
        # Healthcare, Medical Equipment, and Drugs
        'PFE': 10,  # Pfizer
        'MRK': 10,  # Merck
        'ABT': 10,  # Abbott Laboratories
        'ABBV': 10, # AbbVie
        'LLY': 10,  # Eli Lilly
        'BMY': 10,  # Bristol Myers Squibb
        'UNH': 10,  # UnitedHealth
        'CVS': 10,  # CVS Health
        
        # Finance
        'JPM': 11,  # JPMorgan Chase
        'BAC': 11,  # Bank of America
        'WFC': 11,  # Wells Fargo
        'C': 11,    # Citigroup
        'GS': 11,   # Goldman Sachs
        'MS': 11,   # Morgan Stanley
        'V': 11,    # Visa
        'MA': 11,   # Mastercard
        'AXP': 11,  # American Express
        'BLK': 11,  # BlackRock
        'PYPL': 11, # PayPal
        
        # Other (Mines, Construction, BldMt, Trans, Hotels, Bus Serv, Entertainment)
        'AA': 12,   # Alcoa
        'FCX': 12,  # Freeport-McMoRan
        'NEM': 12,  # Newmont
        'LEN': 12,  # Lennar
        'DHI': 12,  # D.R. Horton
        'UPS': 12,  # UPS
        'FDX': 12,  # FedEx
        'MAR': 12,  # Marriott
        'HLT': 12,  # Hilton
        'LVS': 12,  # Las Vegas Sands
    }
    
    # Create mapping for the requested tickers
    mapping = {}
    
    for ticker in tickers:
        if ticker in common_mappings:
            mapping[ticker] = common_mappings[ticker]
        else:
            # Default to "Other" if no mapping is found
            mapping[ticker] = 12
    
    return mapping

def get_industry_names() -> Dict[int, str]:
    """
    Gets the names of the Fama-French 12 Industries.
    
    Returns:
        Dictionary mapping industry IDs to names
    """
    return {
        1: "Consumer Non-Durables",
        2: "Consumer Durables",
        3: "Manufacturing",
        4: "Energy",
        5: "Chemicals",
        6: "Business Equipment",
        7: "Telecommunications",
        8: "Utilities",
        9: "Shops",
        10: "Healthcare",
        11: "Finance",
        12: "Other"
    }

if __name__ == "__main__":
    # Example usage
    tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG',
        'JNJ', 'WMT', 'MA', 'DIS', 'NFLX', 'ADBE', 'CRM', 'INTC', 'AMD', 'PYPL'
    ]
    
    # Get industry mapping
    industry_mapping = get_fama_french_industries(tickers)
    
    # Get industry names
    industry_names = get_industry_names()
    
    # Print results
    print("Fama-French 12 Industry Classifications:")
    for ticker, industry_id in sorted(industry_mapping.items()):
        industry_name = industry_names.get(industry_id, "Unknown")
        print(f"{ticker}: {industry_id} - {industry_name}")
