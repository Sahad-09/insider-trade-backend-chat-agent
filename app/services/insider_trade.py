"""Service for interacting with Financial Modeling Prep API for insider trading data.

Uses the new stable API format: https://financialmodelingprep.com/stable/{endpoint}
Documentation: https://site.financialmodelingprep.com/developer/docs

NOTE: Insider trading endpoints require paid subscription. We use mock data for those.
Free endpoints (profile, quote) call the real API.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import logger


# =============================================================================
# MOCK DATA FOR PREMIUM ENDPOINTS
# These match the FMP API response structure exactly
# Top 30+ US companies by market cap with realistic insider data
# =============================================================================

# Top 30+ US companies insider data
MOCK_INSIDERS = {
    # ==========================================================================
    # MEGA CAP TECH ($1T+)
    # ==========================================================================
    "AAPL": [
        {"name": "COOK TIMOTHY D", "title": "Chief Executive Officer", "cik": "0001214128", "shares": 3280557},
        {"name": "MAESTRI LUCA", "title": "Chief Financial Officer", "cik": "0001513362", "shares": 110938},
        {"name": "WILLIAMS JEFFREY E", "title": "Chief Operating Officer", "cik": "0001496686", "shares": 489944},
        {"name": "O'BRIEN DEIRDRE", "title": "SVP, Retail", "cik": "0001767094", "shares": 136135},
        {"name": "SROUJI JOHNY", "title": "SVP, Hardware Technologies", "cik": "0001631902", "shares": 43587},
    ],
    "MSFT": [
        {"name": "NADELLA SATYA", "title": "Chairman and CEO", "cik": "0001513157", "shares": 812563},
        {"name": "HOOD AMY E", "title": "EVP and CFO", "cik": "0001456522", "shares": 267891},
        {"name": "SMITH BRADFORD L", "title": "Vice Chair and President", "cik": "0001221493", "shares": 456123},
        {"name": "JORGENSEN KATHLEEN T", "title": "EVP and Chief HR Officer", "cik": "0001789456", "shares": 45678},
    ],
    "GOOGL": [
        {"name": "PICHAI SUNDAR", "title": "CEO", "cik": "0001560467", "shares": 2456789},
        {"name": "PORAT RUTH", "title": "President and CIO", "cik": "0001513158", "shares": 123456},
        {"name": "WALKER KENT", "title": "President, Global Affairs", "cik": "0001456789", "shares": 87654},
        {"name": "KURIAN THOMAS", "title": "CEO, Google Cloud", "cik": "0001567890", "shares": 234567},
    ],
    "GOOG": [
        {"name": "PICHAI SUNDAR", "title": "CEO", "cik": "0001560467", "shares": 2456789},
        {"name": "PORAT RUTH", "title": "President and CIO", "cik": "0001513158", "shares": 123456},
        {"name": "WALKER KENT", "title": "President, Global Affairs", "cik": "0001456789", "shares": 87654},
    ],
    "AMZN": [
        {"name": "JASSY ANDREW R", "title": "President and CEO", "cik": "0001370514", "shares": 94729},
        {"name": "OLSAVSKY BRIAN T", "title": "SVP and CFO", "cik": "0001423893", "shares": 12456},
        {"name": "SELIPSKY ADAM", "title": "CEO, AWS", "cik": "0001678901", "shares": 34567},
        {"name": "CLARK DAVID H", "title": "CEO, Worldwide Consumer", "cik": "0001789012", "shares": 23456},
    ],
    "NVDA": [
        {"name": "HUANG JEN-HSUN", "title": "President and CEO", "cik": "0001045810", "shares": 86259656},
        {"name": "KRESS COLETTE M", "title": "EVP and CFO", "cik": "0001443854", "shares": 456321},
        {"name": "PURI AJAY K", "title": "EVP, Worldwide Field Operations", "cik": "0001213971", "shares": 234567},
        {"name": "SHOQUIST DEBORA", "title": "EVP, Operations", "cik": "0001324567", "shares": 123456},
    ],
    "META": [
        {"name": "ZUCKERBERG MARK", "title": "Chairman and CEO", "cik": "0001548760", "shares": 350000000},
        {"name": "WEHNER DAVID M", "title": "Chief Strategy Officer", "cik": "0001567234", "shares": 45678},
        {"name": "SANDBERG SHERYL", "title": "Director", "cik": "0001456123", "shares": 1234567},
        {"name": "OLIVAN JAVIER", "title": "Chief Operating Officer", "cik": "0001678234", "shares": 34567},
    ],
    "TSLA": [
        {"name": "MUSK ELON R", "title": "CEO", "cik": "0001494730", "shares": 715022222},
        {"name": "TANEJA VAIBHAV", "title": "CFO", "cik": "0001891234", "shares": 45678},
        {"name": "BAGLINO ANDREW D", "title": "SVP, Powertrain and Energy", "cik": "0001651759", "shares": 65432},
        {"name": "GUILLEN JEROME M", "title": "President, Heavy Trucking", "cik": "0001756789", "shares": 23456},
    ],
    # ==========================================================================
    # LARGE CAP TECH
    # ==========================================================================
    "AVGO": [
        {"name": "HOCK TAN", "title": "President and CEO", "cik": "0001234890", "shares": 2345678},
        {"name": "KRAUSE KIRSTEN M", "title": "CFO", "cik": "0001345901", "shares": 123456},
        {"name": "BATRA CHARLIE B", "title": "President, Semiconductor Solutions", "cik": "0001456012", "shares": 87654},
    ],
    "ORCL": [
        {"name": "CATZ SAFRA A", "title": "CEO", "cik": "0001108524", "shares": 23456789},
        {"name": "ELLISON LAWRENCE J", "title": "Chairman and CTO", "cik": "0000930420", "shares": 1234567890},
        {"name": "SCREVEN EDWARD", "title": "EVP, Corporate Architecture", "cik": "0001567123", "shares": 456789},
    ],
    "CRM": [
        {"name": "BENIOFF MARC", "title": "Chair and CEO", "cik": "0001108524", "shares": 15678901},
        {"name": "WEAVER AMY E", "title": "President and CFO", "cik": "0001678234", "shares": 234567},
        {"name": "TAYLOR BRET", "title": "Vice Chair", "cik": "0001789345", "shares": 123456},
    ],
    "AMD": [
        {"name": "SU LISA T", "title": "President and CEO", "cik": "0001461240", "shares": 1234567},
        {"name": "PAPERMASTER MARK D", "title": "EVP and CTO", "cik": "0001572351", "shares": 345678},
        {"name": "KUMAR DEVINDER", "title": "EVP and CFO", "cik": "0001683462", "shares": 234567},
    ],
    "INTC": [
        {"name": "GELSINGER PATRICK P", "title": "CEO", "cik": "0001108527", "shares": 567890},
        {"name": "ZINSNER DAVID A", "title": "EVP and CFO", "cik": "0001794573", "shares": 123456},
        {"name": "HOLTHAUS SANDRA L", "title": "EVP and GM, Client Computing", "cik": "0001805684", "shares": 87654},
    ],
    "QCOM": [
        {"name": "AMON CRISTIANO R", "title": "President and CEO", "cik": "0001567234", "shares": 456789},
        {"name": "PALKHIWALA AKASH", "title": "CFO", "cik": "0001678345", "shares": 234567},
        {"name": "ROSENBERG JAMES H", "title": "EVP and GM", "cik": "0001789456", "shares": 123456},
    ],
    "ADBE": [
        {"name": "NARAYEN SHANTANU", "title": "Chairman and CEO", "cik": "0001106380", "shares": 567890},
        {"name": "WADHWANI DAN", "title": "President, Digital Media", "cik": "0001789567", "shares": 234567},
        {"name": "RENCHER BRAD", "title": "EVP and GM", "cik": "0001890678", "shares": 123456},
    ],
    # ==========================================================================
    # FINANCIALS
    # ==========================================================================
    "BRK.A": [
        {"name": "BUFFETT WARREN E", "title": "Chairman and CEO", "cik": "0000315090", "shares": 229016},
        {"name": "ABEL GREGORY E", "title": "Vice Chairman", "cik": "0001456234", "shares": 567},
        {"name": "JAIN AJIT", "title": "Vice Chairman", "cik": "0001567345", "shares": 234},
    ],
    "BRK.B": [
        {"name": "BUFFETT WARREN E", "title": "Chairman and CEO", "cik": "0000315090", "shares": 229016},
        {"name": "ABEL GREGORY E", "title": "Vice Chairman", "cik": "0001456234", "shares": 567},
        {"name": "JAIN AJIT", "title": "Vice Chairman", "cik": "0001567345", "shares": 234},
    ],
    "JPM": [
        {"name": "DIMON JAMES", "title": "Chairman and CEO", "cik": "0001005463", "shares": 8567890},
        {"name": "BARNUM JEREMY", "title": "CFO", "cik": "0001678456", "shares": 234567},
        {"name": "PINTO DANIEL E", "title": "President and COO", "cik": "0001789567", "shares": 456789},
        {"name": "LAKE MARIANNE", "title": "CEO, Consumer and Community Banking", "cik": "0001890678", "shares": 345678},
    ],
    "V": [
        {"name": "MCINERNEY RYAN M", "title": "CEO", "cik": "0001456567", "shares": 234567},
        {"name": "PRABHU VASANT M", "title": "Vice Chairman and CFO", "cik": "0001234567", "shares": 456789},
        {"name": "KELLY ALFRED F JR", "title": "Executive Chairman", "cik": "0001345678", "shares": 567890},
    ],
    "MA": [
        {"name": "MIEBACH MICHAEL", "title": "CEO", "cik": "0001567678", "shares": 123456},
        {"name": "MEHRA SACHIN", "title": "CFO", "cik": "0001678789", "shares": 87654},
        {"name": "BANGA AJAY", "title": "Executive Chairman", "cik": "0001789890", "shares": 345678},
    ],
    "BAC": [
        {"name": "MOYNIHAN BRIAN T", "title": "Chairman and CEO", "cik": "0001207641", "shares": 2345678},
        {"name": "BORTHWICK ALASTAIR M", "title": "CFO", "cik": "0001789901", "shares": 234567},
        {"name": "BESSANT CATHY", "title": "Vice Chair", "cik": "0001890012", "shares": 123456},
    ],
    "WFC": [
        {"name": "SCHARF CHARLES W", "title": "CEO", "cik": "0001108529", "shares": 1234567},
        {"name": "SANTOMASSIMO MICHAEL P", "title": "CFO", "cik": "0001901123", "shares": 234567},
        {"name": "SHREWSBERRY JOHN R", "title": "EVP", "cik": "0001012234", "shares": 123456},
    ],
    "GS": [
        {"name": "SOLOMON DAVID M", "title": "Chairman and CEO", "cik": "0001179858", "shares": 567890},
        {"name": "SCHERR DENIS P", "title": "CFO", "cik": "0001123345", "shares": 234567},
        {"name": "WALDRON JOHN E", "title": "President and COO", "cik": "0001234456", "shares": 345678},
    ],
    # ==========================================================================
    # HEALTHCARE
    # ==========================================================================
    "UNH": [
        {"name": "WITTY ANDREW", "title": "CEO", "cik": "0001234567", "shares": 345678},
        {"name": "WICHMANN JOHN F", "title": "President", "cik": "0001345678", "shares": 234567},
        {"name": "REX JOHN", "title": "CFO", "cik": "0001456789", "shares": 123456},
    ],
    "JNJ": [
        {"name": "DUATO JOAQUIN", "title": "Chairman and CEO", "cik": "0001567890", "shares": 234567},
        {"name": "WOLK JOSEPH J", "title": "EVP and CFO", "cik": "0001678901", "shares": 123456},
        {"name": "STOFFELS PAUL", "title": "Vice Chairman", "cik": "0001789012", "shares": 87654},
    ],
    "LLY": [
        {"name": "RICKS DAVID A", "title": "Chairman and CEO", "cik": "0001890123", "shares": 456789},
        {"name": "SMILEY ANAT ASHKENAZI", "title": "EVP and CFO", "cik": "0001901234", "shares": 234567},
        {"name": "LUNDBERG ILYA", "title": "EVP and President, Lilly International", "cik": "0001012345", "shares": 123456},
    ],
    "PFE": [
        {"name": "BOURLA ALBERT", "title": "Chairman and CEO", "cik": "0001123456", "shares": 567890},
        {"name": "DENTON DAVID M", "title": "EVP and CFO", "cik": "0001234567", "shares": 234567},
        {"name": "DOLSTEN MIKAEL", "title": "Chief Scientific Officer", "cik": "0001345678", "shares": 123456},
    ],
    "ABBV": [
        {"name": "GONZALEZ RICHARD A", "title": "Chairman and CEO", "cik": "0001456789", "shares": 2345678},
        {"name": "CHASE ROBERT A", "title": "EVP and CFO", "cik": "0001567890", "shares": 234567},
        {"name": "SEVERINO MICHAEL E", "title": "Vice Chairman", "cik": "0001678901", "shares": 345678},
    ],
    "MRK": [
        {"name": "DAVIS ROBERT M", "title": "Chairman and CEO", "cik": "0001789012", "shares": 456789},
        {"name": "SCHLICHTING CAROLINE LITCHFIELD", "title": "EVP and CFO", "cik": "0001890123", "shares": 234567},
        {"name": "FRAZIER KENNETH C", "title": "Executive Chairman", "cik": "0001901234", "shares": 567890},
    ],
    # ==========================================================================
    # CONSUMER & RETAIL
    # ==========================================================================
    "WMT": [
        {"name": "MCMILLON C DOUGLAS", "title": "President and CEO", "cik": "0001012345", "shares": 1234567},
        {"name": "RAINEY JOHN DAVID", "title": "EVP and CFO", "cik": "0001123456", "shares": 234567},
        {"name": "FURNER JOHN R", "title": "CEO, Walmart US", "cik": "0001234567", "shares": 123456},
    ],
    "COST": [
        {"name": "VACHRIS RON", "title": "President and CEO", "cik": "0001345678", "shares": 345678},
        {"name": "GALANTI RICHARD A", "title": "EVP and CFO", "cik": "0001456789", "shares": 456789},
        {"name": "JELINEK W CRAIG", "title": "Director", "cik": "0001567890", "shares": 234567},
    ],
    "HD": [
        {"name": "MENEAR CRAIG A", "title": "Chairman", "cik": "0001678901", "shares": 567890},
        {"name": "DECKER TED", "title": "CEO and President", "cik": "0001789012", "shares": 345678},
        {"name": "MCPHAIL RICHARD V", "title": "EVP and CFO", "cik": "0001890123", "shares": 234567},
    ],
    "PG": [
        {"name": "MOELLER JON R", "title": "Chairman and CEO", "cik": "0001901234", "shares": 234567},
        {"name": "SCHULTEN ANDRE", "title": "CFO", "cik": "0001012345", "shares": 123456},
        {"name": "TAYLOR DAVID S", "title": "Executive Chairman", "cik": "0001123456", "shares": 456789},
    ],
    "KO": [
        {"name": "QUINCEY JAMES R", "title": "Chairman and CEO", "cik": "0001234567", "shares": 345678},
        {"name": "MURPHY JOHN", "title": "President and CFO", "cik": "0001345678", "shares": 234567},
        {"name": "WALLER NANCY QUAN", "title": "EVP and Chief Technical Officer", "cik": "0001456789", "shares": 123456},
    ],
    "PEP": [
        {"name": "LAGUARTA RAMON L", "title": "Chairman and CEO", "cik": "0001567890", "shares": 456789},
        {"name": "JOHNSTON HUGH F", "title": "Vice Chairman and CFO", "cik": "0001678901", "shares": 234567},
        {"name": "KHAN STEVEN", "title": "CEO, PepsiCo Foods North America", "cik": "0001789012", "shares": 123456},
    ],
    "MCD": [
        {"name": "KEMPCZINSKI CHRIS", "title": "Chairman and CEO", "cik": "0001890123", "shares": 234567},
        {"name": "OZAN KEVIN M", "title": "Senior EVP", "cik": "0001901234", "shares": 123456},
        {"name": "ERLINGER IAN", "title": "President, McDonald's USA", "cik": "0001012345", "shares": 87654},
    ],
    # ==========================================================================
    # ENERGY
    # ==========================================================================
    "XOM": [
        {"name": "WOODS DARREN W", "title": "Chairman and CEO", "cik": "0001123456", "shares": 567890},
        {"name": "MIKELLS KATHRYN A", "title": "SVP and CFO", "cik": "0001234567", "shares": 234567},
        {"name": "CHAPMAN NEIL A", "title": "SVP", "cik": "0001345678", "shares": 123456},
    ],
    "CVX": [
        {"name": "WIRTH MICHAEL K", "title": "Chairman and CEO", "cik": "0001456789", "shares": 456789},
        {"name": "BREBER PIERRE R", "title": "VP and CFO", "cik": "0001567890", "shares": 234567},
        {"name": "JOHNSON MARK A", "title": "EVP, Upstream", "cik": "0001678901", "shares": 123456},
    ],
    # ==========================================================================
    # INDUSTRIALS & OTHER
    # ==========================================================================
    "CAT": [
        {"name": "UMPLEBY JIM", "title": "Chairman and CEO", "cik": "0001789012", "shares": 234567},
        {"name": "FAULKNER ANDREW R J", "title": "CFO", "cik": "0001890123", "shares": 123456},
        {"name": "BONFIELD JOSEPH E", "title": "Group President", "cik": "0001901234", "shares": 87654},
    ],
    "BA": [
        {"name": "CALHOUN DAVID L", "title": "President and CEO", "cik": "0001012345", "shares": 345678},
        {"name": "WEST BRIAN J", "title": "EVP and CFO", "cik": "0001123456", "shares": 234567},
        {"name": "DEAL STEPHANIE F", "title": "EVP, Enterprise Performance", "cik": "0001234567", "shares": 123456},
    ],
    "HON": [
        {"name": "ADAMCZYK DARIUS", "title": "Executive Chairman", "cik": "0001345678", "shares": 456789},
        {"name": "KAPUR VIMAL", "title": "CEO", "cik": "0001456789", "shares": 234567},
        {"name": "LEWIS GREGORY P", "title": "SVP and CFO", "cik": "0001567890", "shares": 123456},
    ],
    "UPS": [
        {"name": "TOME CAROL B", "title": "CEO", "cik": "0001678901", "shares": 234567},
        {"name": "NEWMAN BRIAN O", "title": "EVP and CFO", "cik": "0001789012", "shares": 123456},
        {"name": "BARBER NANDO", "title": "EVP and President, US Operations", "cik": "0001890123", "shares": 87654},
    ],
    "DIS": [
        {"name": "IGER ROBERT A", "title": "CEO", "cik": "0001015744", "shares": 1234567},
        {"name": "MCCARTHY HUGH F", "title": "SVP and CFO", "cik": "0001901234", "shares": 234567},
        {"name": "D'AMARO JOSH", "title": "Chairman, Disney Parks", "cik": "0001012345", "shares": 123456},
    ],
    "NFLX": [
        {"name": "SARANDOS TED", "title": "Co-CEO", "cik": "0001123456", "shares": 567890},
        {"name": "PETERS GREG", "title": "Co-CEO", "cik": "0001234567", "shares": 456789},
        {"name": "NEUMANN SPENCER ADAM", "title": "CFO", "cik": "0001345678", "shares": 234567},
    ],
    "PYPL": [
        {"name": "CHRISS ALEX", "title": "President and CEO", "cik": "0001456789", "shares": 234567},
        {"name": "RAINEY JAMIE", "title": "EVP and CFO", "cik": "0001567890", "shares": 123456},
        {"name": "SCHULMAN DAN", "title": "Director", "cik": "0001678901", "shares": 567890},
    ],
    "NOW": [
        {"name": "MCDERMOTT BILL", "title": "Chairman and CEO", "cik": "0001789012", "shares": 345678},
        {"name": "MASTERTON GINA", "title": "CFO", "cik": "0001890123", "shares": 123456},
        {"name": "DONAHOE CHIRANTAN J", "title": "President", "cik": "0001901234", "shares": 234567},
    ],
}

# Default insiders for unknown symbols
DEFAULT_INSIDERS = [
    {"name": "SMITH JOHN A", "title": "Chief Executive Officer", "cik": "0001234567", "shares": 500000},
    {"name": "JOHNSON MARY B", "title": "Chief Financial Officer", "cik": "0001234568", "shares": 150000},
    {"name": "WILLIAMS ROBERT C", "title": "Chief Operating Officer", "cik": "0001234569", "shares": 200000},
    {"name": "BROWN SARAH D", "title": "General Counsel", "cik": "0001234570", "shares": 75000},
]


def _generate_mock_insider_trades(symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Generate realistic mock insider trading data for a symbol."""
    symbol = symbol.upper()
    
    insiders = MOCK_INSIDERS.get(symbol, DEFAULT_INSIDERS)
    trades = []
    base_date = datetime.now()
    
    # Generate trades for each insider
    for i, insider in enumerate(insiders):
        for j in range(min(limit // len(insiders), 10)):
            days_ago = (i * 10) + (j * 7) + 1
            transaction_date = base_date - timedelta(days=days_ago)
            filing_date = transaction_date + timedelta(days=2)
            
            # Alternate between purchases and sales
            is_purchase = (i + j) % 3 != 0
            transaction_type = "P-Purchase" if is_purchase else "S-Sale"
            acq_disp = "A" if is_purchase else "D"
            
            # Realistic transaction sizes
            shares_transacted = (1000 + (i * 500) + (j * 250)) * (1 if is_purchase else 2)
            price = 150.0 + (i * 10) - (j * 2)  # Varying prices
            
            trades.append({
                "symbol": symbol,
                "filingDate": filing_date.strftime("%Y-%m-%d"),
                "transactionDate": transaction_date.strftime("%Y-%m-%d"),
                "reportingCik": insider["cik"],
                "reportingName": insider["name"],
                "typeOfOwner": f"officer: {insider['title']}",
                "acquistionOrDisposition": acq_disp,
                "transactionType": transaction_type,
                "securitiesOwned": insider["shares"],
                "securitiesTransacted": shares_transacted,
                "securityName": "Common Stock",
                "price": round(price, 2),
                "formType": "4",
                "link": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={insider['cik']}&type=4"
            })
    
    # Sort by filing date descending
    trades.sort(key=lambda x: x["filingDate"], reverse=True)
    return trades[:limit]


def _generate_mock_market_activity(limit: int = 50) -> List[Dict[str, Any]]:
    """Generate mock market-wide insider activity from top 30 US companies."""
    # Top 30 US company CEOs for market-wide activity
    insiders = [
        # Mega Cap Tech
        ("COOK TIMOTHY D", "AAPL", "CEO"),
        ("NADELLA SATYA", "MSFT", "Chairman and CEO"),
        ("PICHAI SUNDAR", "GOOGL", "CEO"),
        ("JASSY ANDREW R", "AMZN", "President and CEO"),
        ("HUANG JEN-HSUN", "NVDA", "President and CEO"),
        ("ZUCKERBERG MARK", "META", "Chairman and CEO"),
        ("MUSK ELON R", "TSLA", "CEO"),
        # Large Cap Tech
        ("HOCK TAN", "AVGO", "President and CEO"),
        ("CATZ SAFRA A", "ORCL", "CEO"),
        ("BENIOFF MARC", "CRM", "Chair and CEO"),
        ("SU LISA T", "AMD", "President and CEO"),
        ("GELSINGER PATRICK P", "INTC", "CEO"),
        ("NARAYEN SHANTANU", "ADBE", "Chairman and CEO"),
        # Financials
        ("DIMON JAMES", "JPM", "Chairman and CEO"),
        ("MCINERNEY RYAN M", "V", "CEO"),
        ("MIEBACH MICHAEL", "MA", "CEO"),
        ("MOYNIHAN BRIAN T", "BAC", "Chairman and CEO"),
        ("SCHARF CHARLES W", "WFC", "CEO"),
        ("SOLOMON DAVID M", "GS", "Chairman and CEO"),
        # Healthcare
        ("WITTY ANDREW", "UNH", "CEO"),
        ("DUATO JOAQUIN", "JNJ", "Chairman and CEO"),
        ("RICKS DAVID A", "LLY", "Chairman and CEO"),
        ("BOURLA ALBERT", "PFE", "Chairman and CEO"),
        # Consumer & Retail
        ("MCMILLON C DOUGLAS", "WMT", "President and CEO"),
        ("DECKER TED", "HD", "CEO and President"),
        ("QUINCEY JAMES R", "KO", "Chairman and CEO"),
        # Energy & Industrial
        ("WOODS DARREN W", "XOM", "Chairman and CEO"),
        ("WIRTH MICHAEL K", "CVX", "Chairman and CEO"),
        ("IGER ROBERT A", "DIS", "CEO"),
        ("SARANDOS TED", "NFLX", "Co-CEO"),
    ]
    
    trades = []
    base_date = datetime.now()
    
    for i, (name, symbol, title) in enumerate(insiders):
        for j in range(limit // len(insiders)):
            days_ago = j * 2 + 1
            transaction_date = base_date - timedelta(days=days_ago)
            filing_date = transaction_date + timedelta(days=1)
            
            is_purchase = (i + j) % 4 != 0
            transaction_type = "P-Purchase" if is_purchase else "S-Sale"
            
            trades.append({
                "symbol": symbol,
                "filingDate": filing_date.strftime("%Y-%m-%d %H:%M:%S"),
                "transactionDate": transaction_date.strftime("%Y-%m-%d"),
                "reportingCik": f"000{1000000 + i}",
                "reportingName": name,
                "typeOfOwner": f"officer: {title}",
                "acquistionOrDisposition": "A" if is_purchase else "D",
                "transactionType": transaction_type,
                "securitiesTransacted": 5000 + (i * 1000) + (j * 500),
                "price": round(100 + (i * 15) + (j * 2), 2),
                "formType": "4",
                "link": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=000{1000000 + i}&type=4"
            })
    
    trades.sort(key=lambda x: x["filingDate"], reverse=True)
    return trades[:limit]


def _generate_mock_insider_roster(symbol: str) -> List[Dict[str, Any]]:
    """Generate mock insider roster for a company."""
    symbol = symbol.upper()
    
    # Use the MOCK_INSIDERS data to generate roster
    if symbol in MOCK_INSIDERS:
        roster_data = []
        for insider in MOCK_INSIDERS[symbol]:
            # Generate realistic bought/sold based on shares
            shares = insider["shares"]
            bought = int(shares * 0.01) if shares > 100000 else int(shares * 0.05)
            sold = int(shares * 0.02) if shares > 1000000 else int(shares * 0.01)
            roster_data.append({
                "name": insider["name"],
                "title": insider["title"],
                "shares": shares,
                "bought": bought,
                "sold": sold,
                "cik": insider["cik"],
            })
    else:
        roster_data = [
            {"name": "SMITH JOHN A", "title": "CEO", "shares": 500000, "bought": 10000, "sold": 5000, "cik": "0001234567"},
            {"name": "JOHNSON MARY B", "title": "CFO", "shares": 150000, "bought": 5000, "sold": 10000, "cik": "0001234568"},
            {"name": "WILLIAMS ROBERT C", "title": "COO", "shares": 200000, "bought": 0, "sold": 15000, "cik": "0001234569"},
            {"name": "BROWN SARAH D", "title": "General Counsel", "shares": 75000, "bought": 2000, "sold": 3000, "cik": "0001234570"},
        ]
    base_date = datetime.now() - timedelta(days=5)
    
    return [
        {
            "symbol": symbol,
            "reportingCik": f"000{1234567 + i}",
            "reportingName": r["name"],
            "typeOfOwner": f"officer: {r['title']}",
            "lastTransactionDate": (base_date - timedelta(days=i * 7)).strftime("%Y-%m-%d"),
            "securitiesOwned": r["shares"],
            "totalBought": r["bought"],
            "totalSold": r["sold"]
        }
        for i, r in enumerate(roster_data)
    ]


def _generate_mock_insider_statistics(symbol: str) -> Dict[str, Any]:
    """Generate mock insider statistics for a company."""
    symbol = symbol.upper()
    
    # Statistics for top 30 US companies
    stats = {
        # Mega Cap Tech - Generally more selling due to high valuations
        "AAPL": {"purchases": 8, "sales": 15, "totalBought": 45000, "totalSold": 125000},
        "MSFT": {"purchases": 5, "sales": 12, "totalBought": 30000, "totalSold": 95000},
        "GOOGL": {"purchases": 3, "sales": 18, "totalBought": 15000, "totalSold": 200000},
        "GOOG": {"purchases": 3, "sales": 18, "totalBought": 15000, "totalSold": 200000},
        "AMZN": {"purchases": 4, "sales": 20, "totalBought": 25000, "totalSold": 180000},
        "NVDA": {"purchases": 2, "sales": 25, "totalBought": 10000, "totalSold": 650000},
        "META": {"purchases": 1, "sales": 22, "totalBought": 5000, "totalSold": 500000},
        "TSLA": {"purchases": 1, "sales": 30, "totalBought": 5000, "totalSold": 850000},
        # Large Cap Tech
        "AVGO": {"purchases": 3, "sales": 15, "totalBought": 20000, "totalSold": 150000},
        "ORCL": {"purchases": 2, "sales": 18, "totalBought": 15000, "totalSold": 200000},
        "CRM": {"purchases": 6, "sales": 10, "totalBought": 35000, "totalSold": 80000},
        "AMD": {"purchases": 4, "sales": 14, "totalBought": 25000, "totalSold": 120000},
        "INTC": {"purchases": 8, "sales": 8, "totalBought": 50000, "totalSold": 60000},
        "QCOM": {"purchases": 5, "sales": 12, "totalBought": 30000, "totalSold": 90000},
        "ADBE": {"purchases": 3, "sales": 16, "totalBought": 18000, "totalSold": 140000},
        # Financials - Generally more balanced
        "BRK.A": {"purchases": 10, "sales": 2, "totalBought": 100, "totalSold": 10},
        "BRK.B": {"purchases": 10, "sales": 2, "totalBought": 100, "totalSold": 10},
        "JPM": {"purchases": 12, "sales": 8, "totalBought": 75000, "totalSold": 50000},
        "V": {"purchases": 6, "sales": 10, "totalBought": 40000, "totalSold": 70000},
        "MA": {"purchases": 5, "sales": 12, "totalBought": 35000, "totalSold": 85000},
        "BAC": {"purchases": 10, "sales": 6, "totalBought": 60000, "totalSold": 40000},
        "WFC": {"purchases": 8, "sales": 10, "totalBought": 50000, "totalSold": 65000},
        "GS": {"purchases": 7, "sales": 9, "totalBought": 45000, "totalSold": 55000},
        # Healthcare - Mixed signals
        "UNH": {"purchases": 6, "sales": 14, "totalBought": 38000, "totalSold": 110000},
        "JNJ": {"purchases": 9, "sales": 7, "totalBought": 55000, "totalSold": 45000},
        "LLY": {"purchases": 3, "sales": 18, "totalBought": 20000, "totalSold": 160000},
        "PFE": {"purchases": 7, "sales": 11, "totalBought": 42000, "totalSold": 75000},
        "ABBV": {"purchases": 5, "sales": 13, "totalBought": 32000, "totalSold": 95000},
        "MRK": {"purchases": 6, "sales": 10, "totalBought": 38000, "totalSold": 70000},
        # Consumer & Retail
        "WMT": {"purchases": 8, "sales": 6, "totalBought": 48000, "totalSold": 38000},
        "COST": {"purchases": 4, "sales": 12, "totalBought": 25000, "totalSold": 90000},
        "HD": {"purchases": 7, "sales": 9, "totalBought": 42000, "totalSold": 60000},
        "PG": {"purchases": 6, "sales": 8, "totalBought": 35000, "totalSold": 50000},
        "KO": {"purchases": 5, "sales": 7, "totalBought": 30000, "totalSold": 45000},
        "PEP": {"purchases": 6, "sales": 8, "totalBought": 36000, "totalSold": 52000},
        "MCD": {"purchases": 4, "sales": 10, "totalBought": 25000, "totalSold": 70000},
        # Energy
        "XOM": {"purchases": 9, "sales": 5, "totalBought": 55000, "totalSold": 32000},
        "CVX": {"purchases": 8, "sales": 6, "totalBought": 50000, "totalSold": 38000},
        # Industrial & Other
        "CAT": {"purchases": 7, "sales": 7, "totalBought": 42000, "totalSold": 45000},
        "BA": {"purchases": 5, "sales": 12, "totalBought": 30000, "totalSold": 85000},
        "HON": {"purchases": 6, "sales": 9, "totalBought": 36000, "totalSold": 58000},
        "UPS": {"purchases": 5, "sales": 8, "totalBought": 32000, "totalSold": 52000},
        "DIS": {"purchases": 4, "sales": 14, "totalBought": 25000, "totalSold": 100000},
        "NFLX": {"purchases": 3, "sales": 16, "totalBought": 18000, "totalSold": 130000},
        "PYPL": {"purchases": 5, "sales": 11, "totalBought": 30000, "totalSold": 75000},
        "NOW": {"purchases": 4, "sales": 13, "totalBought": 24000, "totalSold": 95000},
    }
    
    default_stats = {"purchases": 4, "sales": 10, "totalBought": 20000, "totalSold": 75000}
    s = stats.get(symbol, default_stats)
    
    return {
        "symbol": symbol,
        "year": datetime.now().year,
        "quarter": (datetime.now().month - 1) // 3 + 1,
        "purchases": s["purchases"],
        "sales": s["sales"],
        "buySellRatio": round(s["purchases"] / max(s["sales"], 1), 2),
        "totalBought": s["totalBought"],
        "totalSold": s["totalSold"],
        "averageBought": s["totalBought"] // max(s["purchases"], 1),
        "averageSold": s["totalSold"] // max(s["sales"], 1),
        "pPurchases": s["purchases"] - 2,  # P-Purchase type
        "sPurchases": 2,  # S-Purchase type (stock option exercise)
        "pSales": s["sales"] - 3,  # P-Sale type
        "sSales": 3  # S-Sale type
    }


# =============================================================================
# MAIN SERVICE CLASS
# =============================================================================

class InsiderTradeService:
    """Service to fetch insider trading data from Financial Modeling Prep.
    
    Uses mock data for premium endpoints (insider trading).
    Uses real API for free endpoints (profile, quote).
    """

    BASE_URL = "https://financialmodelingprep.com/stable/"

    def __init__(self):
        """Initialize the service with API key."""
        self.api_key = settings.FINANCIAL_MODELING_PREP_API_KEY
        if not self.api_key:
            logger.warning("financial_modeling_prep_api_key_missing")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for requests."""
        return {"Content-Type": "application/json"}

    def _get_params(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get query parameters including API key."""
        p = params or {}
        p["apikey"] = self.api_key
        return p

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        """Make an async request to the stable API (for FREE endpoints only)."""
        if not self.api_key:
            raise ValueError("Financial Modeling Prep API key is not set")

        url = f"{self.BASE_URL}{endpoint}"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    params=self._get_params(params),
                    headers=self._get_headers(),
                    timeout=15.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(
                    "fmp_api_error",
                    status_code=e.response.status_code,
                    endpoint=endpoint,
                    url=url,
                    error=str(e),
                )
                raise
            except Exception as e:
                logger.error("fmp_request_failed", endpoint=endpoint, error=str(e))
                raise

    # =========================================================================
    # PREMIUM ENDPOINTS - Using Mock Data
    # =========================================================================

    async def get_insider_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent insider trades for a specific company.
        
        NOTE: Uses mock data (premium endpoint).
        
        Args:
            symbol: The stock ticker symbol (e.g., AAPL)
            limit: Number of records to return
            
        Returns:
            List of insider trading records (mock data)
        """
        logger.info("get_insider_trades_mock", symbol=symbol, limit=limit)
        return _generate_mock_insider_trades(symbol, limit)

    async def get_recent_insider_activity(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent insider trading activity across the market.
        
        NOTE: Uses mock data (premium endpoint).
        
        Args:
            limit: Number of records to return
            
        Returns:
            List of recent insider trading records (mock data)
        """
        logger.info("get_recent_insider_activity_mock", limit=limit)
        return _generate_mock_market_activity(limit)

    async def get_insider_roster(self, symbol: str) -> List[Dict[str, Any]]:
        """Get list of insiders for a company.
        
        NOTE: Uses mock data (premium endpoint).
        
        Args:
            symbol: The stock ticker symbol
            
        Returns:
            List of company insiders (mock data)
        """
        logger.info("get_insider_roster_mock", symbol=symbol)
        return _generate_mock_insider_roster(symbol)

    async def get_insider_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get insider trading statistics for a company.
        
        NOTE: Uses mock data (premium endpoint).
        
        Args:
            symbol: The stock ticker symbol
            
        Returns:
            Statistics on insider buying/selling (mock data)
        """
        logger.info("get_insider_statistics_mock", symbol=symbol)
        return _generate_mock_insider_statistics(symbol)

    # =========================================================================
    # FREE ENDPOINTS - Using Real API
    # =========================================================================

    async def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time stock quote.
        
        Uses REAL API (free endpoint).
        
        Args:
            symbol: The stock ticker symbol
            
        Returns:
            Stock quote data with price, change, volume, etc.
        """
        data = await self._make_request("quote", params={"symbol": symbol.upper()})
        # Handle both list and dict responses
        if isinstance(data, dict) and "value" in data:
            data = data["value"]
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data if isinstance(data, dict) else {}

    async def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Get company profile data.
        
        Uses REAL API (free endpoint).
        
        Args:
            symbol: The stock ticker symbol
            
        Returns:
            Company profile with name, sector, industry, market cap, etc.
        """
        data = await self._make_request("profile", params={"symbol": symbol.upper()})
        # Handle both list and dict responses
        if isinstance(data, dict) and "value" in data:
            data = data["value"]
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data if isinstance(data, dict) else {}


insider_trade_service = InsiderTradeService()

