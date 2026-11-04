"""Two follow-ups of studies/brti_proxy.py (04.11.2026). Run from the project root."""
import sys; sys.argv=["x","data/raw/usd_minutes"]; sys.path.insert(0,".")
import pandas as pd
from studies import brti_proxy as bp
from studies import wings_ladder as wl
bp.wl.LAD=wl.LAD=__import__("pathlib").Path("data/raw/kalshi_kxbtcd_60m")
E=bp.settlements(); cb=bp.candles("cb"); bs=bp.candles("bs")
sp={"binance×basis":bp.binance(),"coinbase×basis":cb.c,"mean cb+bs×basis":((cb.c+bs.c)/2).dropna()}
rows=[]
for nm,S in sp.items():
    c=bp.wings_frame(S,E)
    for lab,p in {"W03 frozen (normal ≥1¢)":(0.10,0.20,5,15,"normal",0.01,1),"W03 t4 ≥1¢":(0.10,0.20,5,15,"t4",0.01,1),
                  "W03 t4 ≥0":(0.10,0.20,5,15,"t4",0.0,1),"W04 frozen (t4 ≥0)":(0.06,0.10,15,30,"t4",0.0,1),
                  "W04 t4 ≥1¢":(0.06,0.10,15,30,"t4",0.01,1)}.items():
        g=wl.run(c,*p)
        for half,gg in (("Jul–Aug",g[g["T"]<wl.SPLIT]),("Sep–Oct",g[g["T"]>=wl.SPLIT])):
            s=wl.stats(gg); rows.append({"spot":nm,"rule":lab,"half":half,"trades":s.get("trades"),"¢/trade":s.get("¢/trade"),"t":s.get("t (hours)")})
R=pd.DataFrame(rows).pivot_table(index=["rule","spot"],columns="half",values=["trades","¢/trade","t"]).round(2)
print(R.to_string())
