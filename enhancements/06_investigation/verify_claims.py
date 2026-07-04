import pandas as pd, numpy as np

RAW='data/dukascopy_raw'
# --- 1) 2023 measured round-trip spread as fraction of ATR ---
bid=pd.read_csv(f'{RAW}/xauusd-m1-bid-2023.csv')
ask=pd.read_csv(f'{RAW}/xauusd-m1-ask-2023.csv')
bid['ts']=pd.to_datetime(bid['timestamp'],unit='ms',utc=True)
ask['ts']=pd.to_datetime(ask['timestamp'],unit='ms',utc=True)
m=pd.merge(bid[['ts','close']],ask[['ts','close']],on='ts',suffixes=('_bid','_ask'))
m['spread']=m['close_ask']-m['close_bid']
spread_med=m['spread'].median()
print(f"[2023] n aligned M1 rows: {len(m)}")
print(f"[2023] median M1 ask-bid spread (price): {spread_med:.4f}")
print(f"[2023] mean spread: {m['spread'].mean():.4f}  p25/p75: {m['spread'].quantile(.25):.3f}/{m['spread'].quantile(.75):.3f}")

# H1 OHLC from M1 bid, ATR(14)
b=bid.set_index('ts')['close']
o=bid.set_index('ts').resample('1h').agg(open=('close','first'),high=('close','max'),low=('close','min'),close=('close','last')).dropna()
prev=o['close'].shift()
tr=pd.concat([o['high']-o['low'],(o['high']-prev).abs(),(o['low']-prev).abs()],axis=1).max(axis=1)
atr_simple=tr.rolling(14).mean()
atr_wilder=tr.ewm(alpha=1/14,adjust=False).mean()
print(f"[2023] median H1 ATR(14) simple: {atr_simple.median():.4f}   Wilder: {atr_wilder.median():.4f}")
for name,atr in [('simple',atr_simple.median()),('Wilder',atr_wilder.median())]:
    print(f"[2023] spread/ATR frac ({name}): {spread_med/atr:.4f}   vs config 0.0623  -> {'HIGHER (model lenient)' if spread_med/atr>0.0623 else 'lower'}")

# --- 2) gold buy-and-hold over fold windows (market fact) ---
print("\n=== gold buy-and-hold (backbone close ratio) ===")
bb=pd.read_csv('data/XAUUSD_M1_Bid_Dukascopy_2003.05.05_2026.07.02.csv',parse_dates=['DateTime']).set_index('DateTime')
windows={'fold9 (best win +34.6%)':('2015-07-29','2016-01-15'),
         'fold17 (worst -18.0%)':('2019-07-29','2020-01-16'),
         'fold25 (bull)':('2023-07-27','2024-01-16')}
for name,(s,e) in windows.items():
    seg=bb.loc[s:e,'Close']
    bh=seg.iloc[-1]/seg.iloc[0]-1
    print(f"  {name:28s} {s}->{e}: B&H {bh*100:+.2f}%  (first {seg.iloc[0]:.1f} last {seg.iloc[-1]:.1f}, n={len(seg)})")
print(f"\n[data] backbone spans {bb.index.min()} -> {bb.index.max()}")
