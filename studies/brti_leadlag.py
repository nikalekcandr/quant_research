"""Two follow-ups of studies/brti_proxy.py (04.11.2026). Run from the project root."""
import sys; sys.argv=["x","data/raw/usd_minutes"]
sys.path.insert(0,".")
import numpy as np, pandas as pd, statsmodels.api as sm
from studies import brti_proxy as bp
from studies import wings_ladder as wl
E=bp.settlements(); bn=bp.binance(); cb=bp.candles("cb"); bs=bp.candles("bs")
usd=((cb.c+bs.c)/2).dropna()
bb=bp.lagged_basis(E,bn); bu=bp.lagged_basis(E,usd)
print("does the Binance–USD gap at h−τ predict the settlement beyond the USD spot? y=log(E/(USD·b)), x=log(BN·b_bn/(USD·b_usd)); HAC")
for tau in (1,2,5,10,15,30):
    t0=E.index-pd.Timedelta(minutes=tau)
    U=usd.reindex(t0).values*bu.values; B=bn.reindex(t0).values*bb.values
    y=np.log(E.values/U); x=np.log(B/U)
    d=pd.DataFrame({"y":y,"x":x},index=E.index).dropna()
    for half,dd in (("Dec–Aug",d[d.index<wl.SPLIT]),("Sep–Nov",d[d.index>=wl.SPLIT])):
        m=sm.OLS(dd.y,sm.add_constant(dd.x)).fit(cov_type="HAC",cov_kwds={"maxlags":5})
        print(f"τ={tau:>2} {half}: beta={m.params.x:.2f} t={m.tvalues.x:.1f} R2={m.rsquared:.3f} sd(x)={1e4*dd.x.std():.2f}bp n={len(dd)}")
# lead-lag at minute level: corr of USD return at t+1 with BN-USD gap at t
j=pd.DataFrame({"bn":bn,"u":usd}).dropna()
g=np.log(j.bn/j.u); g=g-g.rolling(1440,min_periods=100).mean()
r1=np.log(j.u).diff().shift(-1)
for k in (1,2,5):
    rk=np.log(j.u).shift(-k)-np.log(j.u)
    m=sm.OLS(rk,sm.add_constant(g),missing="drop").fit(cov_type="HAC",cov_kwds={"maxlags":k+2})
    print(f"USD return next {k} min on de-meaned BN/USD gap: beta={m.params.iloc[1]:.2f} t={m.tvalues.iloc[1]:.1f} R2={m.rsquared:.4f}")
