"""
make_features.py — Feature engineering adapté crypto daily.

Features crypto-spécifiques :
  - BTC dominance proxy (ratio volume BTC / alts)
  - Funding rate proxy : basis EMA court vs prix spot
  - Fear & Greed proxy : skewness rolling des returns
  - Drawdown depuis ATH (proxy fear/capitulation)
  - Volume velocity (accélération = signal de move imminent)
  - Cycle halving BTC (~4 ans) encodage sin/cos
  - Weekend effect (crypto 7j/7)

Compatibilité équités conservée (is_crypto=False).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf


def load_data(symbols, start="2017-01-01", end=None):
    if isinstance(symbols, str):
        symbols = [symbols]
    result = {}
    for sym in symbols:
        df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
        df = df.dropna()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        result[sym] = df
    return result


def _safe_series(s):
    return s.squeeze()

def _log_returns(close):
    return np.log(close).diff()

def _rolling_vol(log_ret, w, annual_days=365):
    return log_ret.rolling(w).std() * np.sqrt(annual_days)

def _rsi(close, w=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = gain.ewm(span=w, min_periods=w).mean()
    al = loss.ewm(span=w, min_periods=w).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _macd_components(close):
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    m = e12 - e26
    s = m.ewm(span=9, adjust=False).mean()
    return m, s, m - s

def _zscore(series, w):
    mu = series.rolling(w).mean()
    sigma = series.rolling(w).std() + 1e-9
    return (series - mu) / sigma

def _sma_dist(close, w):
    return (close / close.rolling(w).mean()) - 1.0

def _atr(high, low, close, w=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(w).mean()

def _amihud(log_ret, volume, w=21):
    ratio = log_ret.abs() / (volume * 1e-6).replace(0, np.nan)
    return np.log1p(ratio.rolling(w).mean())

def _vwap_dist(close, volume, w=14):
    vwap = (close * volume).rolling(w).sum() / volume.rolling(w).sum()
    return (close / vwap) - 1.0

def _hurst(log_ret, w=60):
    def _rs(x):
        n = len(x)
        if n < 10: return 0.5
        d = np.cumsum(x - x.mean())
        R = d.max() - d.min()
        S = x.std() + 1e-12
        return 0.5 if R <= 0 else float(np.log(R/S) / np.log(n))
    return log_ret.rolling(w).apply(_rs, raw=True)

def _momentum(close, lb, skip=1):
    return close.shift(skip).pct_change(lb)

# ── Crypto-specific ──────────────────────────────────────────────────────────

def _dd_from_ath(close):
    """Drawdown depuis ATH rolling. ∈ [-1, 0]. Proxy fear."""
    running_max = close.expanding().max()
    return (close - running_max) / running_max

def _rolling_skew(log_ret, w):
    """Skewness rolling. Négatif = queues gauches = bear."""
    return log_ret.rolling(w).skew()

def _volume_velocity(volume, w_fast=5, w_slow=20):
    """Ratio SMA rapide / lente du volume. Spike → move imminent."""
    return (volume.rolling(w_fast).mean() / (volume.rolling(w_slow).mean() + 1e-9)) - 1.0

def _funding_proxy(close):
    """Basis EMA3 et EMA7 vs spot. Proxy surchauffe longs (funding élevé)."""
    basis3 = (close - close.ewm(span=3, adjust=False).mean()) / close
    basis7 = (close - close.ewm(span=7, adjust=False).mean()) / close
    return basis3, basis7

def _halving_encoding(index):
    """Encodage sin/cos du cycle halving BTC (~1461j). Ref: mai 2020."""
    days = (index - pd.Timestamp("2020-05-11")).days.astype(float)
    phase = (days % 1461.0) / 1461.0
    return pd.DataFrame({
        "halving_sin": np.sin(2 * np.pi * phase),
        "halving_cos": np.cos(2 * np.pi * phase),
    }, index=index)

def _crypto_calendar(index):
    dow = index.dayofweek / 6.0
    month = (index.month - 1) / 11.0
    return pd.DataFrame({
        "dow_sin": np.sin(2 * np.pi * dow),
        "dow_cos": np.cos(2 * np.pi * dow),
        "month_sin": np.sin(2 * np.pi * month),
        "month_cos": np.cos(2 * np.pi * month),
        "is_monday": (index.dayofweek == 0).astype(float),
        "is_weekend": (index.dayofweek >= 5).astype(float),
    }, index=index)

def _btc_dominance_proxy(primary_vol, cross_vols, index, w=21):
    if not cross_vols:
        return pd.Series(0.5, index=index, name="btc_dom")
    total = primary_vol.copy()
    for v in cross_vols.values():
        total = total + v.reindex(index).ffill().fillna(0)
    dom = primary_vol / (total + 1e-9)
    return dom.rolling(w).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def make_features(
    df,
    cross_asset_closes=None,
    cross_asset_volumes=None,
    include_calendar=True,
    is_crypto=True,
):
    close = _safe_series(df["Close"])
    high = _safe_series(df["High"])
    low = _safe_series(df["Low"])
    volume = _safe_series(df["Volume"]).replace(0, np.nan)
    log_ret = _log_returns(close)
    ann = 365 if is_crypto else 252

    feat = pd.DataFrame(index=df.index)

    # Returns multi-échelle
    for lb in ([1, 3, 7, 14, 30, 90] if is_crypto else [1, 5, 21, 63]):
        feat[f"ret_{lb}d"] = log_ret.rolling(lb).sum()

    # Volatilité
    windows_vol = [7, 21, 60] if is_crypto else [5, 21, 63]
    for w in windows_vol:
        feat[f"vol_{w}d"] = _rolling_vol(log_ret, w, ann)
    vk = [f"vol_{w}d" for w in windows_vol]
    feat["vol_ratio_s_m"] = feat[vk[0]] / (feat[vk[1]] + 1e-9)
    feat["vol_ratio_m_l"] = feat[vk[1]] / (feat[vk[2]] + 1e-9)
    feat["vol_regime_z"] = _zscore(feat[vk[1]], 180 if is_crypto else 252)

    # RSI
    feat["rsi_14"] = _rsi(close, 14)
    feat["rsi_7"] = _rsi(close, 7)

    # MACD
    macd, _, macd_hist = _macd_components(close)
    feat["macd_hist"] = macd_hist
    feat["macd_zscore"] = _zscore(macd, 180)

    # Z-score / SMA dist
    windows_z = [14, 30, 90] if is_crypto else [20, 60, 120]
    for w in windows_z:
        feat[f"zscore_{w}d"] = _zscore(log_ret, w)
        feat[f"sma_dist_{w}d"] = _sma_dist(close, w)

    # Momentum
    mom_lbs = [7, 30, 90, 180] if is_crypto else [21, 63, 126, 252]
    mom_names = ["mom_1w", "mom_1m", "mom_3m", "mom_6m"] if is_crypto else ["mom_1m", "mom_3m", "mom_6m", "mom_12m"]
    for lb, name in zip(mom_lbs, mom_names):
        feat[name] = _momentum(close, lb, skip=1)

    # Microstructure
    feat["atr_14_norm"] = _atr(high, low, close, 14) / (close + 1e-9)
    feat["amihud_21"] = _amihud(log_ret, volume, 21)
    feat["vwap_dist"] = _vwap_dist(close, volume, 14)
    feat["hl_range"] = (high - low) / (close + 1e-9)
    feat["hl_range_z"] = _zscore(feat["hl_range"], 21)

    # Régime
    feat["hurst_60"] = _hurst(log_ret, 60)

    # ── Crypto-specific ───────────────────────────────────────────────────
    if is_crypto:
        feat["dd_from_ath"] = _dd_from_ath(close)
        feat["skew_14d"] = _rolling_skew(log_ret, 14)
        feat["skew_30d"] = _rolling_skew(log_ret, 30)
        feat["vol_velocity"] = _volume_velocity(volume, 5, 20)
        feat["vol_velocity_z"] = _zscore(feat["vol_velocity"], 60)
        b3, b7 = _funding_proxy(close)
        feat["basis_3d"] = b3
        feat["basis_7d"] = b7
        feat = pd.concat([feat, _halving_encoding(df.index)], axis=1)

    # Cross-asset
    # ffill + bfill : évite de tronquer le dataset si un cross-asset commence tard
    # (ex: SOL depuis 2020) — les NaN du début sont remplis avec la 1ère valeur dispo.
    if cross_asset_closes:
        ca_vols = cross_asset_volumes or {}
        for name, ca_c in cross_asset_closes.items():
            ca_c = _safe_series(ca_c).reindex(df.index).ffill().bfill()
            ca_lr = np.log(ca_c).diff()
            # Les features cross-asset sur les périodes bfill seront moins fiables
            # mais au moins on ne perd pas 3 ans de données du primary asset.
            feat[f"corr_21d_{name}"] = log_ret.rolling(21).corr(ca_lr)
            feat[f"rel_mom_{name}"] = log_ret.rolling(21).sum() - ca_lr.rolling(21).sum()

        if ca_vols and is_crypto:
            ca_v_aligned = {k: _safe_series(v).reindex(df.index).ffill().bfill()
                            for k, v in ca_vols.items()}
            feat["btc_dom_proxy"] = _btc_dominance_proxy(volume, ca_v_aligned, df.index)

    # Calendar
    if include_calendar:
        feat = pd.concat([feat, _crypto_calendar(df.index)], axis=1)

    # Nettoyage
    # On remplace inf par nan, puis on dropna uniquement sur les features core
    # (pas sur les cross-assets qui peuvent avoir des NaN légitimes au début).
    # Cela évite de tronquer le dataset à cause d'un cross-asset récent (ex: SOL).
    feat = feat.replace([np.inf, -np.inf], np.nan)
    # Colonnes core = tout sauf les cross-asset features
    cross_cols = [c for c in feat.columns if c.startswith("corr_21d_") or
                  c.startswith("rel_mom_") or c == "btc_dom_proxy"]
    core_cols = [c for c in feat.columns if c not in cross_cols]
    # Drop les lignes où les features core sont NaN (warm-up indicators)
    feat = feat.dropna(subset=core_cols)
    # Pour les cross-assets, ffill les NaN résiduels puis fillna(0)
    if cross_cols:
        feat[cross_cols] = feat[cross_cols].ffill().bfill().fillna(0.0)
    prices = close.loc[feat.index].values.astype(np.float32)
    X = feat.values.astype(np.float32)
    make_features._last_feature_names = list(feat.columns)
    return X, prices, feat.index


make_features._last_feature_names = []


def standardize_train_only(X_train, X_test, eps=1e-8):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + eps
    clip = 5.0
    return (np.clip((X_train - mean) / std, -clip, clip),
            np.clip((X_test - mean) / std, -clip, clip),
            mean, std)

def apply_scaler(X, mean, std, clip=5.0):
    return np.clip((X - mean) / std, -clip, clip)

def save_feature_config(path, n_features, feature_names,
                        cross_asset_symbols, include_calendar, is_crypto=True):
    import json, os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"n_features": n_features, "feature_names": feature_names,
                   "cross_asset_symbols": cross_asset_symbols,
                   "include_calendar": include_calendar, "is_crypto": is_crypto}, f, indent=2)

def load_feature_config(path):
    import json
    with open(path) as f:
        return json.load(f)

def walk_forward_splits(n, n_splits=5, test_frac=0.10, gap=0):
    test_size = max(int(n * test_frac), 30)
    min_train = n - test_size * n_splits - gap * n_splits
    splits = []
    for i in range(n_splits):
        train_end = min_train + i * test_size
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > n: break
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation cross-market
# ─────────────────────────────────────────────────────────────────────────────

def normalize_for_target(X_target_train: np.ndarray,
                          X_target_test: np.ndarray,
                          X_source_train: np.ndarray | None = None,
                          mode: str = "target",
                          eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalise les features d'un asset cible pour l'évaluation cross-market.

    Deux modes :
      "target"  : normalise avec les stats de X_target_train (80% passé de l'asset cible)
                  → recommandé pour l'éval cross-market, les features sont dans la bonne plage
      "source"  : normalise avec les stats de X_source_train (train set BTC)
                  → ce qu'on faisait avant, peut créer un décalage si les distributions diffèrent

    Retourne : (X_train_norm, X_test_norm, mean, std)
    """
    assert mode in ("target", "source")

    if mode == "source":
        if X_source_train is None:
            raise ValueError("mode='source' requiert X_source_train")
        ref = X_source_train
    else:
        ref = X_target_train

    mean = ref.mean(axis=0, keepdims=True)
    std = ref.std(axis=0, keepdims=True) + eps
    clip = 5.0
    return (
        np.clip((X_target_train - mean) / std, -clip, clip),
        np.clip((X_target_test - mean) / std, -clip, clip),
        mean,
        std,
    )


def pad_features_to_expected(X: np.ndarray,
                              current_names: list[str],
                              expected_names: list[str]) -> np.ndarray:
    """
    Aligne les features d'un asset sur le schéma attendu par le modèle.
    Features manquantes → colonne de zéros.
    Features en trop → ignorées.
    Retourne X avec exactement len(expected_names) colonnes dans le bon ordre.
    """
    n = len(expected_names)
    X_out = np.zeros((X.shape[0], n), dtype=np.float32)
    for j, fname in enumerate(expected_names):
        if fname in current_names:
            X_out[:, j] = X[:, current_names.index(fname)]
        # sinon reste à 0 (feature crypto manquante sur un asset non-crypto)
    return X_out