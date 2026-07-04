import pandas as pd
bb=pd.read_csv('data/XAUUSD_M1_Bid_Dukascopy_2003.05.05_2026.07.02.csv',parse_dates=['DateTime']).set_index('DateTime')
# Old test window per Agent A (from notebook cell 1e4b075a)
seg=bb.loc['2024-02-06':'2026-05-30','Close']
print(f"OLD test window 2024-02-06 -> 2026-05-30: first={seg.iloc[0]:.2f} last={seg.iloc[-1]:.2f} n={len(seg)}")
print(f"  gold buy-and-hold (M1 close/close): {(seg.iloc[-1]/seg.iloc[0]-1)*100:+.2f}%   (Agent A claim: +124.05% M1 / +123.88% H1)")
# ATR over the window (H1)
o=bb.loc['2024-02-06':'2026-05-30','Close'].resample('1h').agg(['first','max','min','last']).dropna()
o.columns=['open','high','low','close']; prev=o['close'].shift()
tr=pd.concat([o['high']-o['low'],(o['high']-prev).abs(),(o['low']-prev).abs()],axis=1).max(axis=1)
atr=tr.rolling(14).mean()
print(f"  H1 ATR(14) over window: mean={atr.mean():.2f} median={atr.median():.2f}   (Agent A: mean 12.38 / median 8.33)")
