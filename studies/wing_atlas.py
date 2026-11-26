"""Standalone, offline interactive charts for the November wing research cohort."""
import html
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core import RESULTS, atomic_write, kalshi_fee
from lab.history import Minutes
from lab.wing_regime import regimes
from studies.wing_ten import CACHE,archive,enrich,books,deepen,replay
from studies.wing_sleep_replay import next_minute
from studies.wings_two_sided import offers,picks
from studies.wing_sizing import RULES,simulate

TZ="Europe/Moscow"
charts=[]
def dates(x):
    return pd.to_datetime(x,utc=True).dt.tz_convert(TZ)

def clock(frame,times):
    frame["Час МСК"]=times.dt.hour
    frame["Дата МСК"]=times.dt.strftime("%Y-%m-%d")
    frame["День"]=np.where(times.dt.day_of_week<5,"Будни","Выходные")

def graph(section,title,frame,x,y,unit,*,by=None,agg="median",kind="scatter",note=""):
    traces=[]
    for label,g in (frame.groupby(by,sort=False) if by else [(title,frame)]):
        a=g.dropna(subset=[x,y]).groupby(x)[y].agg([agg,"count"])
        if x=="Час МСК":
            a=a.reindex(range(24))
        traces.append(dict(name=str(label),type=kind,x=a.index.tolist(),
                           y=[round(float(v),5) if pd.notna(v) else None for v in a.iloc[:,0]],
                           customdata=a.iloc[:,1].fillna(0).astype(int).tolist(),
                           hovertemplate="%{x} · %{y:.3f} "+unit+"<br>Наблюдений: %{customdata}<extra>%{fullData.name}</extra>"))
    charts.append(dict(section=section,title=title,x=x,unit=unit,data=traces,note=note))

def main():
    history=pd.read_parquet(ROOT/"data/interim/minutes.parquet",columns=["open_time","close"])
    close=pd.concat([history.set_index(pd.to_datetime(history.open_time,utc=True)).close,Minutes().close])
    close=close[~close.index.duplicated(keep="last")].sort_index().asfreq("min")
    def period(times):
        return f"{times.min():%d.%m.%Y} – {times.max():%d.%m.%Y}. "
    r=np.log(close.asfreq("min")).diff()
    v=pd.DataFrame(index=r.index)
    for minutes in [5,30,60]:
        v[f"RV {minutes} мин"]=100*np.sqrt(r.pow(2).rolling(minutes).sum())
    v["Диапазон за час"]=100*(close.rolling(61).max()/close.rolling(61).min()-1)
    v["Модуль доходности за 5 мин"]=100*r.rolling(5).sum().abs()
    v["Доля вариации вниз"]=100*r.clip(upper=0).pow(2).rolling(30).sum()/r.pow(2).rolling(30).sum()
    v.index+=pd.Timedelta(minutes=1)
    v=v.iloc[::5].copy()
    clock(v,pd.Series(v.index.tz_convert(TZ),index=v.index))
    for metric in v.columns[:6]:
        for axis,split in [("Час МСК","День"),("Дата МСК",None)]:
            graph("Волатильность",metric+(" по времени суток" if split else " по датам"),v,axis,metric,"%",by=split,
                  note=period(v.index.tz_convert(TZ))+"Медиана пятиминутных наблюдений BTC. RV — корень суммы квадратов минутных лог-доходностей за окно.")
    dvol=pd.read_csv(ROOT/"data/raw/deribit/dvol.csv.gz",usecols=["ts","close"])
    dvol=dvol.drop_duplicates("ts").dropna()
    clock(dvol,dates(dvol.ts))
    for axis,split in [("Дата МСК",None),("Час МСК","День")]:
        graph("Волатильность","Индекс Deribit DVOL · "+axis,dvol,axis,"close","% годовых",by=split,
              note=period(dates(dvol.ts))+"Медиана часовых закрытий индекса ожидаемой волатильности BTC.")
    cb=pd.concat([pd.read_csv(path,usecols=["t","close"]) for path in sorted((ROOT/"data/raw/coinbase_1m").glob("*.csv"))])
    cb=cb.drop_duplicates("t").set_index(pd.to_datetime(cb.drop_duplicates("t").t,unit="s",utc=True)).close.sort_index().asfreq("min")
    match=pd.concat([cb.rename("coinbase"),close.rename("binance")],axis=1,join="inner")
    cross=(100*(match.coinbase/match.binance-1)).dropna().rename("Базис").reset_index()
    times=pd.to_datetime(cross.iloc[:,0],utc=True).dt.tz_convert(TZ)+pd.Timedelta(minutes=1)
    clock(cross,times)
    for axis,split in [("Дата МСК",None),("Час МСК","День")]:
        graph("Спот разных бирж","BTC/USD Coinbase к BTC/USDT Binance · "+axis,cross,axis,"Базис","%",by=split,
              note=period(times)+"Медиана разницы синхронных минутных закрытий в процентах от цены Binance; включает разницу USD и USDT.")
    files=[*(ROOT/"data/raw/kalshi_kxbtcd_60m").glob("*.csv.gz"),*CACHE.glob("*.csv.gz")]
    q,arch_hours=archive(enrich,files=files,minute_close=close)
    live,_=books(close.index.max().floor("h"),deepen,since=close.index.min())
    live=live[live.left.ge(16)].copy()
    val=live.event.map({k:r["value"] for k,r in json.loads((CACHE.parent/"wing_quant_inputs/settlements.json").read_text()).items()})
    yes=val.gt(live.strike).astype(float)
    live["y"]=live.y.fillna(yes.where(live.cheap_yes,1-yes).where(val.notna()))
    q=q[q.left.ge(16)].copy()
    q["candle"]=True
    supplement=live[~live.end.isin(arch_hours)].assign(candle=False)
    q=pd.concat([q,supplement],ignore_index=True)
    near=q.curve.where(q.cheap_yes,1-q.curve)
    wing=q.bid.where(q.cheap_yes,1-q.ask)
    fee=(1-wing).map(kalshi_fee)
    q["Edge соседей"]=100*(wing-(near.fillna(q.t4)+q.t4)/2-fee)
    q["Edge t₄"]=100*(wing-q.t4-fee)
    q["Расхождение моделей"]=100*(near-q.t4).abs()
    q["Спред"]=100*(q.ask-q.bid)
    q["Выше порога"]=100*q["Edge соседей"].ge(1-1e-9)
    q["Сторона"]=np.where(q.cheap_yes,"YES","NO")
    q["Режим"]=np.where(q.ts.map(regimes(close).blocked).eq(False),"Вход разрешён","Пауза")
    clock(q,dates(q.ts))
    q["Время до расчёта, мин"]=q.left.round().astype(int)
    q["RV за 30 мин, %"]=pd.cut(q.ts.map((100*np.sqrt(r.pow(2).rolling(30).sum())).shift(freq="min")),
                               [-np.inf,.1,.2,.3,.5,1,np.inf],labels=["1 · <0,1","2 · 0,1–0,2","3 · 0,2–0,3","4 · 0,3–0,5","5 · 0,5–1","6 · >1"]).astype("string")
    for metric,unit in [("Edge соседей","¢"),("Edge t₄","¢"),("Расхождение моделей","¢"),("Спред","¢"),("Выше порога","%")]:
        for axis,split in [("Час МСК","День"),("Время до расчёта, мин","Сторона"),("RV за 30 мин, %","Режим")]:
            # Give each expiry one vote within each bucket.
            method="mean" if metric=="Выше порога" else "median"
            grouped=q.groupby(["end",axis,split],observed=True)[metric].agg(method).reset_index()
            graph("Edge и котировки",metric+" · "+axis.lower(),grouped,axis,metric,unit,by=split,
                  kind="bar" if axis.startswith("RV") else "scatter",agg=method,
                  note=period(dates(q.ts))+f"{q.end.nunique():,} экспираций с котировками крыла. Для пропусков архива добавлены непересекающиеся часы локальной записи. Один голос каждой экспирации внутри группы; порог после комиссии 1¢.")
    q["base"]=(near.fillna(q.t4)+q.t4)/2
    allowed=q[q.candle & q["Режим"].eq("Вход разрешён")]
    t=next_minute(picks(offers(allowed.assign(t4=allowed.base),False))).reset_index().dropna(subset=["pnl"])
    audit=pd.read_csv(RESULTS/"wing_sleep_hours.csv")
    t["group"]=t.end.map(dict(zip(pd.to_datetime(audit.end,utc=True),audit.group))).fillna("Нет классификации сна")
    clock(t,dates(t.ts))
    t["fill"]=t.filled*100
    for axis,split in [("Час МСК","День"),("Дата МСК",None),("group",None)]:
        title=axis if axis!="group" else "режим записи"
        graph("Исполнение", "Исполнение · "+title,t,axis,"fill","%",by=split,agg="mean",kind="bar",
              note=period(dates(t.ts))+f"{len(t)} сигналов по свечам. Доля сигналов с подходящей ценой минутой позже. Один контракт.")
        graph("Исполнение","Нетто · "+title,t,axis,"pnl","$",by=split,agg="sum",kind="bar",
              note=period(dates(t.ts))+"Сумма результатов сигналов на архивных свечах с комиссией; без исполнения прибыль равна нулю.")
    live=live[live.ts.dt.floor("min").map(regimes(close).blocked).eq(False)].copy()
    live["base"]=(live.t4+live.curve.where(live.cheap_yes,1-live.curve).fillna(live.t4))/2
    signals=replay(live,"base","вся история",include_sizing=True)
    known=[s for s in signals if pd.notna(s["pay"])]
    size=pd.DataFrame([row for rule in [0,9,5] for row in simulate(known,rule)]).sort_values("end")
    size=size[size.rule.isin(["Полная глубина","По прошлым исполнениям","Штраф за расхождение"])].copy()
    size["Экспирация МСК"]=dates(size.end).dt.strftime("%m-%d %H:%M")
    size["Кумулятивное нетто"]=size.groupby("rule").net.cumsum()
    size["Доля исполнения"]=100*size.contracts/size.requested.replace(0,np.nan)
    for metric,title,unit in [("Кумулятивное нетто","Накопленная прибыль","$"),("drawdown","Просадка от максимума","$"),
                              ("net","Нетто отдельной экспирации","$"),("contracts","Исполненный объём","контр."),
                              ("Доля исполнения","Доля запрошенного объёма","%")]:
        graph("Размер позиции",title,size,"Экспирация МСК",metric,unit,by="rule",
              note=period(dates(size.end))+f"{len(known)} сигналов; исключено без исхода: {len(signals)-len(known)}. Локальные стаканы, начальный капитал $10 000.")
    ledger=pd.read_csv(RESULTS/"wing_ledger_capacity_entries.csv")
    ledger=ledger[ledger.pnl.notna()].copy()
    for col,title,unit in [("pnl","Нетто по стратегиям W","$"),("fees","Комиссии по стратегиям W","$"),("n","Объём по стратегиям W","контр.")]:
        graph("Стратегии W",title,ledger,"rule",col,unit,agg="sum",kind="bar",
              note=period(dates(ledger.ts))+"Сопоставленные записи с известной прибылью, объём по наблюдаемой лучшей цене. Пустые стратегии опущены.")
    short=pd.read_parquet(ROOT/"data/interim/kalshi.parquet",columns=["settle_ts","candle_end","yes_bid","yes_ask","strike_type"])
    short=short[short.strike_type.eq("greater") & short.yes_bid.gt(0) & short.yes_ask.lt(1) & short.yes_bid.le(short.yes_ask)].copy()
    cheap=short.yes_bid.where(short.yes_bid<.5,1-short.yes_ask)
    short=short[cheap.round(6).between(.06,.10)].copy()
    short["Спред"]=100*(short.yes_ask-short.yes_bid)
    short["Месяц МСК"]=dates(short.candle_end).dt.strftime("%Y-%m")
    short["сторона"]=np.where(short.yes_bid<.5,"YES","NO")
    early=short.groupby(["settle_ts","Месяц МСК","сторона"]).Спред.median().reset_index()
    graph("Ранняя история Kalshi","Спред крыла в коротком архиве",early,"Месяц МСК","Спред","¢",by="сторона",
          note=period(dates(short.candle_end))+"Короткий архив котировок: медиана по экспирациям, bid крыла 6–10¢. Время сбора отличается от окна входа основной стратегии.")
    prints=[]
    for path in sorted((ROOT/"_export/compact/prints").glob("*.parquet")):
        tape=pd.read_parquet(path)
        tape=tape[tape.yes_c.between(6,10)|tape.yes_c.between(90,94)].copy()
        tape["Поток"]=np.where(tape.taker_yes.eq(tape.yes_c.le(10)),"Покупка крыла","Продажа крыла")
        tape["Дата МСК"]=dates(tape.sec).dt.strftime("%Y-%m-%d")
        prints.append(tape.groupby(["Дата МСК","Поток"])[["count","prints"]].sum().reset_index())
    if prints:
        daily=pd.concat(prints).groupby(["Дата МСК","Поток"],as_index=False)[["count","prints"]].sum()
        for col,unit,title in [("count","контр.","Объём сделок с крылом"),("prints","сделок","Число сделок с крылом")]:
            graph("Лента сделок",title,daily,"Дата МСК",col,unit,by="Поток",agg="sum",
                  note=f"{daily['Дата МСК'].min()} – {daily['Дата МСК'].max()}. Биржевые сделки в архиве выбранных страйков, цена крыла 6–10¢, направление по стороне агрессора.")
    print("BTC",period(close.index),"wing",period(dates(q.ts)),q.end.nunique(),"signals",len(t),"depth",len(known),"ledger",len(ledger))
    write()

def write():
    sections=list(dict.fromkeys(c["section"] for c in charts))
    nav="".join(f'<a href="#s{i}">{s}</a>' for i,s in enumerate(sections))
    body=""
    for i,s in enumerate(sections):
        body+=f'<section id="s{i}"><h2>{s}</h2><div class="grid">'
        for n,c in enumerate(charts):
            if c["section"]==s:
                body+=f'<article><h3>{html.escape(c["title"])}</h3><div class="chart" id="c{n}"></div><p>{c["note"]}</p></article>'
        body+="</div></section>"
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Атлас крыльев · Alpha Lab</title>
<style>*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:auto;max-width:1600px;padding:24px;color:#182f38;background:#f4f5ef;font:16px system-ui}a{color:#116752}h1{font-size:clamp(32px,5vw,64px);margin:18px 0 4px}h2{margin-top:42px}h3{margin:4px 0;font-size:17px}nav{display:flex;gap:18px;flex-wrap:wrap}article{padding:16px;background:white;border:1px solid #d7dfda;border-radius:10px;min-width:0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,470px),1fr));gap:14px}.chart{height:320px;max-width:100%;overflow:hidden}p{color:#485d62;line-height:1.5}article p{font-size:13px}</style><a href="lab.html">← Alpha Lab</a><h1>Атлас крыльев</h1>'''
    page+=f"<p>Графиков: {len(charts)} · время московское. Период источника указан под каждым графиком. Наведите на точку, чтобы увидеть значение и число наблюдений. Нажмите название ряда в легенде, чтобы скрыть его.</p><nav>{nav}</nav>{body}"
    page+="<script>"+get_plotlyjs()+"</script><script>const charts="+json.dumps(charts,ensure_ascii=False,allow_nan=False)+""";
const observer=new IntersectionObserver(es=>es.forEach(e=>{if(!e.isIntersecting)return;observer.unobserve(e.target);const c=charts[Number(e.target.id.slice(1))];Plotly.newPlot(e.target,c.data,{margin:{l:58,r:14,t:12,b:80},colorway:['#067866','#be6927','#525caa'],showlegend:c.data.length>1,legend:{orientation:'h',y:-0.33},xaxis:{title:{text:({'rule':'Стратегия','group':'Режим записи'})[c.x]||c.x},automargin:true},yaxis:{title:{text:c.unit},gridcolor:'#e2e9e6'},hovermode:'closest',barmode:'group'},{responsive:true,displaylogo:false})}),{rootMargin:'400px'});document.querySelectorAll('.chart').forEach(el=>observer.observe(el));</script></html>"""
    atomic_write(ROOT/"reports/wing_atlas.html",page)
    print(len(charts),"charts")

if __name__=="__main__":
    main()
