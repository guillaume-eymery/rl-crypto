from features.make_features import load_data, make_features

all_data = load_data(["BTC-USD","ETH-USD","SOL-USD","BNB-USD","ADA-USD"], start="2017-01-01")
primary_df = all_data["BTC-USD"]
cross_closes = {s: df["Close"].squeeze() for s, df in all_data.items() if s != "BTC-USD"}
cross_volumes = {s: df["Volume"].squeeze() for s, df in all_data.items() if s != "BTC-USD"}

X, prices, idx = make_features(primary_df, cross_asset_closes=cross_closes,
                                 cross_asset_volumes=cross_volumes, is_crypto=True)
print(f"Début: {idx[0].date()}")  
print(f"Shape: {X.shape}")     
print(f"NaN: {(X != X).sum()}")   