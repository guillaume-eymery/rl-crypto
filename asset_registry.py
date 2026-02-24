"""
asset_registry.py — Registre des assets supportés pour l'évaluation cross-market.

Chaque entrée définit :
  - start_date   : depuis quand récupérer les données yfinance
  - annual_days  : 365 crypto, 252 equity/FX/commodity
  - fee          : coût aller-retour (fraction du notional)
  - funding_cost : coût de portage short par jour (0 = pas de short coûteux)
  - allow_short  : bool
  - max_dd_kill  : seuil de drawdown pour kill-switch épisode
  - min_ep_steps : durée min d'un épisode
  - is_crypto    : active les features crypto-spécifiques dans make_features()
  - cross_assets : symboles yfinance à télécharger comme cross-assets
  - norm_mode    : "target" (normalise avec stats de l'asset) ou "source" (stats BTC train)

Usage :
  from asset_registry import get_asset_config, REGISTRY
  cfg = get_asset_config("SPY")
"""

REGISTRY: dict[str, dict] = {

    # ── Crypto ────────────────────────────────────────────────────────────────
    "BTC-USD": {
        "name": "Bitcoin",
        "asset_class": "crypto",
        "start_date": "2015-01-01",
        "annual_days": 365,
        "fee": 0.001,            # 10bps spot/perp
        "funding_cost": 0.0002,  # 2bps/jour portage short
        "allow_short": True,
        "max_dd_kill": 0.40,
        "min_ep_steps": 365,
        "is_crypto": True,
        "cross_assets": ["ETH-USD", "BNB-USD", "ADA-USD"],
        "norm_mode": "target",
    },
    "ETH-USD": {
        "name": "Ethereum",
        "asset_class": "crypto",
        "start_date": "2017-01-01",
        "annual_days": 365,
        "fee": 0.001,
        "funding_cost": 0.0002,
        "allow_short": True,
        "max_dd_kill": 0.45,
        "min_ep_steps": 365,
        "is_crypto": True,
        "cross_assets": ["BTC-USD", "BNB-USD", "ADA-USD"],
        "norm_mode": "target",
    },
    "SOL-USD": {
        "name": "Solana",
        "asset_class": "crypto",
        "start_date": "2020-08-01",
        "annual_days": 365,
        "fee": 0.001,
        "funding_cost": 0.0003,
        "allow_short": True,
        "max_dd_kill": 0.50,
        "min_ep_steps": 200,
        "is_crypto": True,
        "cross_assets": ["BTC-USD", "ETH-USD"],
        "norm_mode": "target",
    },
    "AVAX-USD": {
        "name": "Avalanche",
        "asset_class": "crypto",
        "start_date": "2020-09-01",
        "annual_days": 365,
        "fee": 0.001,
        "funding_cost": 0.0003,
        "allow_short": True,
        "max_dd_kill": 0.50,
        "min_ep_steps": 200,
        "is_crypto": True,
        "cross_assets": ["BTC-USD", "ETH-USD"],
        "norm_mode": "target",
    },
    "LINK-USD": {
        "name": "Chainlink",
        "asset_class": "crypto",
        "start_date": "2019-01-01",
        "annual_days": 365,
        "fee": 0.001,
        "funding_cost": 0.0003,
        "allow_short": True,
        "max_dd_kill": 0.50,
        "min_ep_steps": 252,
        "is_crypto": True,
        "cross_assets": ["BTC-USD", "ETH-USD"],
        "norm_mode": "target",
    },

    # ── Indices actions ───────────────────────────────────────────────────────
    "SPY": {
        "name": "S&P 500 ETF",
        "asset_class": "equity",
        "start_date": "2005-01-01",
        "annual_days": 252,
        "fee": 0.0002,           # 2bps — très liquide
        "funding_cost": 0.0001,  # coût emprunt CFD
        "allow_short": True,
        "max_dd_kill": 0.25,
        "min_ep_steps": 252,
        "is_crypto": False,
        "cross_assets": ["QQQ", "GLD", "TLT"],
        "norm_mode": "target",   # normalise avec ses propres stats
    },
    "QQQ": {
        "name": "Nasdaq 100 ETF",
        "asset_class": "equity",
        "start_date": "2005-01-01",
        "annual_days": 252,
        "fee": 0.0002,
        "funding_cost": 0.0001,
        "allow_short": True,
        "max_dd_kill": 0.30,
        "min_ep_steps": 252,
        "is_crypto": False,
        "cross_assets": ["SPY", "GLD", "TLT"],
        "norm_mode": "target",
    },
    "^GDAXI": {
        "name": "DAX 40",
        "asset_class": "equity",
        "start_date": "2005-01-01",
        "annual_days": 252,
        "fee": 0.0003,
        "funding_cost": 0.0001,
        "allow_short": True,
        "max_dd_kill": 0.30,
        "min_ep_steps": 252,
        "is_crypto": False,
        "cross_assets": ["SPY", "GLD"],
        "norm_mode": "target",
    },

    # ── Commodités ────────────────────────────────────────────────────────────
    "GC=F": {
        "name": "Gold Futures",
        "asset_class": "commodity",
        "start_date": "2005-01-01",
        "annual_days": 252,
        "fee": 0.0003,
        "funding_cost": 0.0001,
        "allow_short": True,
        "max_dd_kill": 0.20,
        "min_ep_steps": 252,
        "is_crypto": False,
        "cross_assets": ["SPY", "DX-Y.NYB"],
        "norm_mode": "target",
    },
    "CL=F": {
        "name": "Crude Oil Futures",
        "asset_class": "commodity",
        "start_date": "2005-01-01",
        "annual_days": 252,
        "fee": 0.0005,
        "funding_cost": 0.0001,
        "allow_short": True,
        "max_dd_kill": 0.40,
        "min_ep_steps": 252,
        "is_crypto": False,
        "cross_assets": ["GC=F", "SPY"],
        "norm_mode": "target",
    },

    # ── FX ────────────────────────────────────────────────────────────────────
    "EURUSD=X": {
        "name": "EUR/USD",
        "asset_class": "fx",
        "start_date": "2005-01-01",
        "annual_days": 252,
        "fee": 0.00005,          # 0.5bps — très liquide
        "funding_cost": 0.00002,
        "allow_short": True,
        "max_dd_kill": 0.10,     # FX peu volatil en absolu
        "min_ep_steps": 252,
        "is_crypto": False,
        "cross_assets": ["GBPUSD=X", "DX-Y.NYB"],
        "norm_mode": "target",
    },
    "GBPUSD=X": {
        "name": "GBP/USD",
        "asset_class": "fx",
        "start_date": "2005-01-01",
        "annual_days": 252,
        "fee": 0.00005,
        "funding_cost": 0.00002,
        "allow_short": True,
        "max_dd_kill": 0.12,
        "min_ep_steps": 252,
        "is_crypto": False,
        "cross_assets": ["EURUSD=X", "DX-Y.NYB"],
        "norm_mode": "target",
    },
}


def get_asset_config(symbol: str) -> dict:
    """Retourne la config d'un asset. Raise ValueError si inconnu."""
    if symbol not in REGISTRY:
        known = list(REGISTRY.keys())
        raise ValueError(
            f"Asset '{symbol}' non trouvé dans le registry.\n"
            f"Assets disponibles : {known}"
        )
    return REGISTRY[symbol]


def list_by_class(asset_class: str) -> list[str]:
    return [s for s, c in REGISTRY.items() if c["asset_class"] == asset_class]


def print_registry():
    print(f"\n{'Symbol':<14} {'Name':<22} {'Class':<12} {'Fee':<8} {'Short':<7} {'Start'}")
    print("─" * 78)
    for sym, cfg in REGISTRY.items():
        print(f"{sym:<14} {cfg['name']:<22} {cfg['asset_class']:<12} "
              f"{cfg['fee']*100:.3f}%  {'✓' if cfg['allow_short'] else '✗':<7} "
              f"{cfg['start_date']}")


if __name__ == "__main__":
    print_registry()
    print(f"\nCrypto : {list_by_class('crypto')}")
    print(f"Equity : {list_by_class('equity')}")
    print(f"FX     : {list_by_class('fx')}")
    print(f"Commod : {list_by_class('commodity')}")