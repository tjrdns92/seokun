"""
종목 확률 분석기 — 웹 앱 (Streamlit)

실행:  streamlit run stock_tool/app.py
배포:  Streamlit Community Cloud (README 참고)

데이터: 야후파이낸스(yfinance) 가격·시세, 한국 종목 이름 검색은 FinanceDataReader(KRX 상장목록).
환경변수 STOCKTOOL_OFFLINE=1 이면 인터넷 없이 가상 데이터로 동작한다(테스트용).
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analyze as az

OFFLINE = os.environ.get("STOCKTOOL_OFFLINE") == "1"
KST = timezone(timedelta(hours=9))
HORIZONS = [6, 12, 24, 36, 60, 120]
SIMS = 8000

ACCENT, GAIN, LOSS = "#2E6B57", "#D2334A", "#2B67D1"

# 투자계획의 병목 종목 (바로가기)
WATCHLIST = [
    ("SK하이닉스", "000660.KS"), ("삼성전자", "005930.KS"), ("HD현대일렉트릭", "267260.KS"),
    ("효성중공업", "298040.KS"), ("LS ELECTRIC", "010120.KS"), ("두산에너빌리티", "034020.KS"),
    ("GE Vernova", "GEV"), ("NVIDIA", "NVDA"), ("TSMC", "TSM"), ("Vertiv", "VRT"),
]

# KRX 목록을 못 받을 때 쓰는 최소 목록
FALLBACK_KR = [
    ("005930", "삼성전자", "KOSPI"), ("000660", "SK하이닉스", "KOSPI"), ("267260", "HD현대일렉트릭", "KOSPI"),
    ("298040", "효성중공업", "KOSPI"), ("010120", "LS ELECTRIC", "KOSPI"), ("034020", "두산에너빌리티", "KOSPI"),
    ("006260", "LS", "KOSPI"), ("009150", "삼성전기", "KOSPI"), ("042700", "한미반도체", "KOSPI"),
    ("005380", "현대차", "KOSPI"), ("035420", "NAVER", "KOSPI"), ("035720", "카카오", "KOSPI"),
    ("373220", "LG에너지솔루션", "KOSPI"), ("207940", "삼성바이오로직스", "KOSPI"), ("012450", "한화에어로스페이스", "KOSPI"),
    ("329180", "HD현대중공업", "KOSPI"), ("009540", "HD한국조선해양", "KOSPI"), ("105560", "KB금융", "KOSPI"),
    ("055550", "신한지주", "KOSPI"), ("000270", "기아", "KOSPI"), ("068270", "셀트리온", "KOSPI"),
    ("015760", "한국전력", "KOSPI"), ("001440", "대한전선", "KOSPI"), ("007660", "이수페타시스", "KOSPI"),
    ("247540", "에코프로비엠", "KOSDAQ"), ("086520", "에코프로", "KOSDAQ"), ("196170", "알테오젠", "KOSDAQ"),
    ("403870", "HPSP", "KOSDAQ"), ("058470", "리노공업", "KOSDAQ"), ("240810", "원익IPS", "KOSDAQ"),
]


# ─────────────────────────────── 데이터 ───────────────────────────────

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def kr_listing() -> pd.DataFrame:
    if not OFFLINE:
        try:
            import FinanceDataReader as fdr
            df = fdr.StockListing("KRX")
            code_col = "Code" if "Code" in df.columns else "Symbol"
            df = df.rename(columns={code_col: "Code"})[["Code", "Name", "Market"]]
            df = df[df["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])]
            if len(df) > 100:
                return df.reset_index(drop=True)
        except Exception:
            pass
    return pd.DataFrame(FALLBACK_KR, columns=["Code", "Name", "Market"])


def kr_symbol(code: str, market: str) -> str:
    return f"{code}.KQ" if market.startswith("KOSDAQ") else f"{code}.KS"


@st.cache_data(ttl=600, show_spinner=False)
def search(q: str) -> list[tuple[str, str]]:
    """(표시 이름, 야후 심볼) 목록."""
    q = q.strip()
    if not q:
        return []
    out: list[tuple[str, str]] = []
    kr = kr_listing()
    if re.fullmatch(r"\d{6}", q):
        hit = kr[kr["Code"] == q]
        if len(hit):
            r = hit.iloc[0]
            out.append((f"{r.Name} · {r.Code} · {r.Market}", kr_symbol(r.Code, r.Market)))
        else:
            out.append((f"{q} (코스피로 조회)", f"{q}.KS"))
    else:
        hit = kr[kr["Name"].str.contains(q, case=False, regex=False)]
        hit = hit.assign(_exact=hit["Name"].str.lower() != q.lower()).sort_values(["_exact", "Name"]).head(8)
        out += [(f"{r.Name} · {r.Code} · {r.Market}", kr_symbol(r.Code, r.Market)) for r in hit.itertuples()]
    if re.search(r"[A-Za-z]", q) and not OFFLINE:
        try:
            import yfinance as yf
            for x in yf.Search(q, max_results=8).quotes:
                if x.get("quoteType") not in ("EQUITY", "ETF", "INDEX"):
                    continue
                name = x.get("shortname") or x.get("longname") or x["symbol"]
                out.append((f"{name} · {x['symbol']} · {x.get('exchDisp', x.get('exchange', ''))}", x["symbol"]))
        except Exception:
            pass
    seen, uniq = set(), []
    for label, sym in out:
        if sym not in seen:
            seen.add(sym)
            uniq.append((label, sym))
    return uniq


@st.cache_data(ttl=3600, show_spinner=False)
def history(sym: str) -> pd.Series:
    if OFFLINE:
        return az._demo_prices(seed=sum(map(ord, sym)) % 13 + 1)
    import yfinance as yf
    df = yf.download(sym, period="max", auto_adjust=True, progress=False)
    if df is not None and not df.empty:
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna()
    m = re.fullmatch(r"(\d{6})\.K[SQ]", sym)
    if m:  # 한국 종목은 FinanceDataReader(네이버/KRX)로 재시도
        import FinanceDataReader as fdr
        df = fdr.DataReader(m.group(1), "2000-01-01")
        if not df.empty:
            return df["Close"].astype(float).dropna()
    return pd.Series(dtype=float)


@st.cache_data(ttl=30, show_spinner=False)
def quote(sym: str) -> dict | None:
    """최근 시세. 30초 캐시."""
    if OFFLINE:
        p = history(sym)
        return {"price": float(p.iloc[-1]), "prev": float(p.iloc[-2]), "currency": "KRW", "time": datetime.now(KST)}
    try:
        import yfinance as yf
        fi = yf.Ticker(sym).fast_info
        return {"price": float(fi["last_price"]), "prev": float(fi["previous_close"]),
                "currency": fi["currency"] or "", "time": datetime.now(KST)}
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fundamentals(sym: str) -> dict:
    if OFFLINE:
        return {}
    try:
        import yfinance as yf
        raw = yf.Ticker(sym).info or {}
        return {k: raw.get(k) for k in ("shortName", "longName", "trailingPE", "forwardPE", "priceToBook",
                                         "marketCap", "currency", "sector")}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def run_analysis(sym: str, lump: float, monthly: float, keep: float, target: float):
    p = history(sym)
    if len(p) < az.TRADING_DAYS * 2:
        return None
    stats = az.basic_stats(p)
    values, invested = az.simulate_values(p, max(HORIZONS), monthly, lump, keep, SIMS)
    mc = az.summarize(values, invested, HORIZONS)
    fan = pd.DataFrame({
        "m": np.arange(1, max(HORIZONS) + 1), "inv": invested,
        **{f"p{q}": np.percentile(values, q, axis=0) for q in (10, 25, 50, 75, 90)},
    })
    cond = az.conditional_now(p)
    return {
        "p": p, "stats": stats, "mc": mc, "fan": fan, "cond": cond,
        "lump_hist": az.rolling_lump(p, [0.25, 0.5, 1, 2, 3, 5, 10]),
        "dca": az.dca_backtest(p, HORIZONS, monthly, lump),
        "verdict": az.make_verdict(stats, mc, cond, target),
    }


# ─────────────────────────────── 표시 헬퍼 ───────────────────────────────

def won_html(x: float) -> str:
    cls = "gain" if x > 0 else "loss" if x < 0 else ""
    return f'<span class="{cls}">{az.won(x)}</span>'


def pct_html(x: float, d: int = 0) -> str:
    cls = "gain" if x > 0 else "loss" if x < 0 else ""
    return f'<span class="{cls}">{x:+.{d}%}</span>'


def fmt_price(v: float, cur: str) -> str:
    if cur in ("KRW", "JPY") or v >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def axis_won(v: float) -> str:
    return f"{v / 1e8:.1f}억" if v >= 1e8 else f"{v / 1e4:,.0f}만"


CSS = f"""
<style>
.block-container{{padding-top:2rem;max-width:1180px}}
.gain{{color:{GAIN}}} .loss{{color:{LOSS}}}
.card{{border:1px solid rgba(128,128,128,.25);border-radius:10px;padding:16px 18px;margin-bottom:12px}}
.headline{{font-size:1.45rem;font-weight:650;line-height:1.4;margin:.2rem 0 .6rem}}
.headline b{{color:{ACCENT}}}
.grade{{display:inline-block;font-weight:700;font-size:.85rem;padding:3px 10px;border-radius:999px;margin-left:6px}}
.grade.A{{background:rgba(46,107,87,.15);color:{ACCENT}}}
.grade.B{{background:rgba(168,102,15,.15);color:#A8660F}}
.grade.C{{background:rgba(210,51,74,.13);color:{GAIN}}}
.kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}}
@media (max-width:640px){{.kpis{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
.kpi{{border:1px solid rgba(128,128,128,.2);border-radius:8px;padding:8px 10px}}
.kpi .k{{font-size:.72rem;opacity:.7}} .kpi .v{{font-size:1.1rem;font-weight:650;font-variant-numeric:tabular-nums}}
.quote{{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 14px}}
.quote .px{{font-size:2rem;font-weight:700;font-variant-numeric:tabular-nums}}
.quote .chg{{font-size:1.05rem;font-weight:600}}
.quote .t{{font-size:.8rem;opacity:.65}}
.note{{background:rgba(168,102,15,.12);border-radius:7px;padding:9px 12px;margin-top:8px;font-size:.92rem}}
table.t{{width:100%;border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums}}
table.t th,table.t td{{padding:7px 8px;text-align:right;border-bottom:1px solid rgba(128,128,128,.18);white-space:nowrap}}
table.t th:first-child,table.t td:first-child{{text-align:left}}
table.t th{{font-size:.75rem;opacity:.7}}
table.t tr.pick td{{background:rgba(46,107,87,.12)}}
.tw{{overflow-x:auto}}
</style>
"""


def table(headers: list[str], rows: list[list[str]], pick: int | None = None) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join(f'<tr class="{"pick" if i == pick else ""}">' + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                  for i, r in enumerate(rows))
    return f'<div class="tw"><table class="t"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def fan_figure(fan: pd.DataFrame) -> go.Figure:
    x = np.concatenate([[0], fan["m"]]) / 12
    first = fan["inv"].iloc[0] - (fan["inv"].iloc[1] - fan["inv"].iloc[0])

    def col(c):
        return np.concatenate([[first], fan[c]])

    fig = go.Figure()
    for lo, hi, name, alpha in (("p10", "p90", "80% 확률 범위", .15), ("p25", "p75", "50% 확률 범위", .3)):
        fig.add_trace(go.Scatter(x=x, y=col(hi), line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=col(lo), fill="tonexty", line=dict(width=0),
                                 fillcolor=f"rgba(46,107,87,{alpha})", name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=col("inv"), name="넣은 원금", line=dict(color="#7C8794", dash="dash", width=1.5),
                             hovertemplate="%{x:.1f}년 원금 %{y:,.0f}원<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=col("p50"), name="중앙값", line=dict(color=ACCENT, width=2.6),
                             hovertemplate="%{x:.1f}년 중앙값 %{y:,.0f}원<extra></extra>"))
    top = float(fan["p90"].max())
    ticks = np.linspace(0, top, 5)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15), xaxis=dict(title="보유 기간(년)", tickvals=[1, 2, 3, 5, 10]),
                      yaxis=dict(tickvals=ticks, ticktext=[axis_won(v) for v in ticks], rangemode="tozero"))
    return fig


def price_figure(p: pd.Series) -> go.Figure:
    fig = go.Figure(go.Scatter(x=p.index, y=p.values, line=dict(color="#4A5563", width=1.4),
                               hovertemplate="%{x|%Y-%m-%d} %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=p.index, y=p.rolling(200).mean(), line=dict(color=ACCENT, width=1, dash="dot"),
                             name="200일선", hoverinfo="skip"))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, yaxis_type="log")
    return fig


# ─────────────────────────────── 화면 ───────────────────────────────

st.set_page_config(page_title="종목 확률 분석기", page_icon="📈", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.title("종목 확률 분석기")
st.caption("몇 년 동안 넣고 들고 가면, 몇 % 확률로, 얼마가 되는지 계산합니다." + (" · 오프라인 테스트 모드(가상 데이터)" if OFFLINE else ""))

with st.sidebar:
    st.header("내 투자 방식")
    lump = st.number_input("처음 일시금 (만원)", min_value=0, value=1000, step=100, key="lump") * 1e4
    monthly = st.number_input("매월 적립 (만원)", min_value=0, value=200, step=10, key="monthly") * 1e4
    target = st.select_slider("권장 보유기간 기준 이익확률", options=[0.7, 0.8, 0.9], value=0.8,
                              format_func=lambda v: f"{v:.0%}", key="target")
    keep = st.slider("과거 수익률을 얼마나 믿을까", 0, 100, 50, 10, format="%d%%", key="keep",
                     help="0%는 매우 보수적, 100%는 과거 수익률이 그대로 반복된다고 가정") / 100
    live = st.toggle("시세 자동 새로고침 (60초)", value=True, key="live")
    st.caption("시세는 야후파이낸스 기준이며 거래소에 따라 최대 20분 지연될 수 있습니다.")

if "sym" not in st.session_state:
    st.session_state.sym, st.session_state.label = WATCHLIST[0][1], WATCHLIST[0][0]

q = st.text_input("종목 검색", placeholder="예: 삼성전자, 하이닉스, 005930, NVDA, Vertiv", key="q")
if q:
    hits = search(q)
    if hits:
        choice = st.selectbox("검색 결과", hits, format_func=lambda h: h[0], key="pick")
        if choice and choice[1] != st.session_state.sym:
            st.session_state.sym, st.session_state.label = choice[1], choice[0].split(" · ")[0]
            st.session_state.watch = None
    else:
        st.info("검색 결과가 없습니다. 한국 종목은 이름이나 6자리 코드로, 해외 종목은 영문 이름이나 티커로 찾아 보세요.")



def on_watch() -> None:
    name = st.session_state.watch
    if name:
        st.session_state.sym, st.session_state.label = dict(WATCHLIST)[name], name
        st.session_state.q = ""


st.pills("병목 종목 바로가기", [n for n, _ in WATCHLIST], key="watch", on_change=on_watch)

sym, label = st.session_state.sym, st.session_state.label


@st.fragment(run_every=60 if live else None)
def quote_card():
    qt = quote(sym)
    if not qt:
        st.warning("실시간 시세를 불러오지 못했습니다. 잠시 뒤 다시 시도합니다.")
        return
    chg = qt["price"] / qt["prev"] - 1
    cls = "gain" if chg > 0 else "loss" if chg < 0 else ""
    arrow = "▲" if chg > 0 else "▼" if chg < 0 else ""
    st.markdown(f"""<div class="card"><div style="font-weight:600">{label} <span style="opacity:.6">{sym}</span></div>
      <div class="quote"><span class="px {cls}">{fmt_price(qt['price'], qt['currency'])}</span>
      <span class="chg {cls}">{arrow} {fmt_price(abs(qt['price'] - qt['prev']), qt['currency'])} ({chg:+.2%})</span>
      <span class="t">{qt['currency']} · {qt['time']:%H:%M:%S} 기준{' · 60초마다 갱신' if live else ''}</span></div></div>""",
                unsafe_allow_html=True)


quote_card()

with st.spinner(f"{label} 가격 기록을 불러와 1만 가지 미래를 계산하는 중..."):
    try:
        res = run_analysis(sym, lump, monthly, keep, target)
    except Exception as e:
        res = None
        st.error(f"데이터를 불러오지 못했습니다: {e}")
if res is None:
    st.warning("가격 기록이 2년 미만이거나 불러올 수 없어 분석할 수 없습니다. 다른 종목을 골라 보세요.")
    st.stop()

stats, mc, v, cond = res["stats"], res["mc"], res["verdict"], res["cond"]
pick = next((r for r in mc if r["months"] == v.hold_months), None) or max(mc, key=lambda r: r["p_win"])
grade = v.grade[0]
plan = f"일시금 {az.plain_won(lump)} + 매월 {az.plain_won(monthly)}"
past5 = next((r for r in res["lump_hist"] if r["h"] == 5), None)
fut5 = next((r for r in mc if r["months"] == 60), None)

notes = "".join(f'<div class="note"><b>주의</b> {n}</div>' for n in v.notes)
if past5 and fut5 and past5["p_win"] - fut5["p_win"] >= 0.15:
    notes = (f'<div class="note"><b>과거 승률을 그대로 믿지 마세요.</b> 과거엔 5년 보유 시 {past5["p_win"]:.0%} 이겼지만 '
             f'그건 한 번 일어난 역사입니다. 미래를 보수적으로 돌리면 {fut5["p_win"]:.0%}입니다.</div>') + notes
hold_txt = (f"최소 <b>{az.horizon_label(v.hold_months)}</b> (이익 확률 {target:.0%} 이상이 되는 가장 짧은 기간)"
            if v.hold_months else f"10년 안에 이익 확률 {target:.0%}에 도달하지 못함")
st.markdown(f"""<div class="card">
  <div><b>{label}</b><span class="grade {grade}">등급 {v.grade}</span></div>
  <div class="headline">{az.horizon_label(pick['months'])} 동안 넣고 보유하면 이익 확률 <b>{pick['p_win']:.0%}</b></div>
  <div class="kpis">
    <div class="kpi"><div class="k">넣는 원금</div><div class="v">{az.plain_won(pick['invested'])}</div></div>
    <div class="kpi"><div class="k">예상 손익 (중앙값)</div><div class="v">{won_html(pick['median'])}</div></div>
    <div class="kpi"><div class="k">나쁜 경우 (하위 10%)</div><div class="v">{won_html(pick['p10'])}</div></div>
    <div class="kpi"><div class="k">좋은 경우 (상위 10%)</div><div class="v">{won_html(pick['p90'])}</div></div>
  </div>
  <p style="margin:.8rem 0 0">보유기간: {hold_txt}<br>진입 방법: {v.entry}<br>투자 방식: {plan}</p>
  {notes}
</div>""", unsafe_allow_html=True)

st.subheader("앞으로 내 계좌는 어떻게 될까")
st.caption(f"{plan} · 시뮬레이션 {SIMS:,}회 · 과거 평균수익률의 {keep:.0%}만 인정")
st.plotly_chart(fan_figure(res["fan"]), width="stretch", config={"displayModeBar": False})

st.subheader("보유기간별 결과")
st.caption("빨강은 이익, 파랑은 손실 (국내 증시 표기)")
rows = [[az.horizon_label(r["months"]), az.plain_won(r["invested"]), f"{r['p_win']:.0%}", won_html(r["median"]),
         won_html(r["p10"]), won_html(r["p90"]), f"{r['p_loss20']:.0%}"] for r in mc]
st.markdown(table(["보유기간", "원금", "이익 확률", "예상 손익", "나쁜 경우", "좋은 경우", "-20% 넘게 잃을 확률"], rows,
                  pick=next(i for i, r in enumerate(mc) if r["months"] == pick["months"])), unsafe_allow_html=True)

st.subheader("가격 흐름과 현재 위치")
st.caption(f"{stats['start']} ~ {stats['end']} · {stats['years']:.1f}년 · 로그 스케일, 점선은 200일선")
st.plotly_chart(price_figure(res["p"]), width="stretch", config={"displayModeBar": False})
info = fundamentals(sym)
items = [("연평균 수익률", pct_html(stats["cagr"], 1)), ("연 변동성", f"{stats['vol']:.1%}"),
         ("역사적 최대낙폭", f'<span class="loss">{stats["mdd"]:.1%}</span>'), ("고점 대비 현재", pct_html(stats["dd_now"], 1)),
         ("52주 고점 대비", pct_html(stats["off_52w_high"], 1)), ("200일선 대비", pct_html(stats["ma200_gap"], 1)),
         ("RSI(14)", f"{stats['rsi']:.0f}")]
for k, name in (("trailingPE", "PER(후행)"), ("forwardPE", "PER(선행)"), ("priceToBook", "PBR")):
    if info.get(k):
        items.append((name, f"{info[k]:.1f}"))
st.markdown('<div class="kpis">' + "".join(f'<div class="kpi"><div class="k">{k}</div><div class="v">{val}</div></div>'
                                          for k, val in items) + "</div>", unsafe_allow_html=True)

if cond:
    st.subheader("지금과 비슷했던 때 샀다면, 1년 뒤")
    st.caption(f"고점 대비 {cond['dd_now']:+.0%}(±7%p) · 200일선 {'위' if cond['above'] else '아래'}였던 과거 {cond['n']:,}일")
    st.markdown(table(["", "이 조건", "전체 평균"], [
        ["이익 확률", f"{cond['p_win']:.0%}", f"{cond['base_p_win']:.0%}"],
        ["중앙 수익률", pct_html(cond["median"]), pct_html(cond["base_median"])]]), unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("과거: 아무 날에 한 번에 샀다면")
    st.markdown(table(["보유", "이익 확률", "중앙", "하위10%", "상위10%"], [
        [f"{round(r['h'] * 12)}개월" if r["h"] < 1 else f"{r['h']:g}년", f"{r['p_win']:.0%}", pct_html(r["median"]),
         pct_html(r["p10"]), pct_html(r["p90"])] for r in res["lump_hist"]]), unsafe_allow_html=True)
with c2:
    st.subheader("과거: 내 방식으로 시작했다면")
    st.markdown(table(["기간", "이익 확률", "손익 중앙", "하위10%", "상위10%"], [
        [az.horizon_label(r["months"]), f"{r['p_win']:.0%}", won_html(r["median"]), won_html(r["p10"]),
         won_html(r["p90"])] for r in res["dca"]]), unsafe_allow_html=True)

with st.expander("이 숫자를 읽는 법", expanded=False):
    st.markdown("""
- 무조건 이기는 분석은 없습니다. 모든 확률은 과거 가격의 움직임이 앞으로도 비슷하다는 가정 위의 추정치입니다.
- 같은 성격의 가상 종목도 운에 따라 15년 성과가 연 -4%에서 +29%까지 갈립니다. 과거 표보다 미래 시뮬레이션을 기준으로 판단하세요.
- 지금 검색되는 종목은 살아남은 종목입니다. 상장폐지되거나 장기 부진한 종목은 통계에 없습니다(생존편향).
- 실적, 산업 사이클, 금리, 전쟁 같은 펀더멘털은 계산에 없습니다. 환율, 세금, 수수료도 빠져 있습니다.
- 이 도구는 "얼마나 오래, 어떤 방식으로" 넣을지 정하는 용도입니다. "무엇을" 살지는 따로 판단하세요.
""")
