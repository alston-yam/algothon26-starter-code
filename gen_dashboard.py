#!/usr/bin/env python
"""Backtests teamName.py, writes dashboard_data.json, serves dashboard.html."""
import numpy as np, pandas as pd, json, webbrowser, http.server, socketserver, os
from teamName import getMyPosition as getPosition

PORT = 8000
NUM_TEST_DAYS = 250
DEF_COMM, INST0_COMM = 0.0001, 0.00002
DEF_LIMIT, INST0_LIMIT = 10_000, 100_000

def loadPrices(fn):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.values.T, list(df.columns)

def scoreFn(mu, sigma, param=1.0):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    return mu * sr**2 / (sr**2 + param**2)

def run():
    prcAll, names = loadPrices("prices.txt")
    nInst, nt = prcAll.shape
    commRate = np.full(nInst, DEF_COMM); commRate[0] = INST0_COMM
    dlrLimit = np.full(nInst, DEF_LIMIT); dlrLimit[0] = INST0_LIMIT
    startDay = nt - NUM_TEST_DAYS

    cash = value = comm = totVol = 0.0
    curPos = np.zeros(nInst)
    days, vals, dpl, dvols, cumpl = [], [], [], [], []
    instPnl = np.zeros(nInst); instVol = np.zeros(nInst)
    instDays = np.zeros(nInst); instMaxExp = np.zeros(nInst)
    prevPx, running = None, 0.0

    for t in range(startDay, nt + 1):
        px = prcAll[:, :t][:, -1]
        if t < nt:
            newPosOrig = getPosition(prcAll[:, :t])
            lim = (dlrLimit / px).astype(int)
            newPos = np.clip(newPosOrig, -lim, lim).astype(int)
        else:
            newPos = np.array(curPos)

        deltaPos = newPos - curPos
        cash -= px.dot(deltaPos) + comm
        dvol_i = px * np.abs(deltaPos)
        dvol = dvol_i.sum()
        totVol += dvol
        comm = (dvol_i * commRate).sum()

        if prevPx is not None:
            instPnl += curPos * (px - prevPx) - dvol_i * commRate
        instVol += dvol_i
        instDays += (newPos != 0).astype(int)
        instMaxExp = np.maximum(instMaxExp, np.abs(newPos) * px)

        curPos = np.array(newPos)
        posVal = curPos.dot(px)
        todayPL = cash + posVal - value
        value = cash + posVal
        prevPx = px

        if t > startDay:
            running += todayPL
            days.append(t); vals.append(value); dpl.append(todayPL)
            dvols.append(dvol); cumpl.append(running)

    dpl_np, cumpl_np = np.array(dpl), np.array(cumpl)
    dd = cumpl_np - np.maximum.accumulate(cumpl_np)
    W = 20
    rollSharpe = []
    for i in range(len(dpl_np)):
        w = dpl_np[max(0, i - W + 1):i + 1]
        rollSharpe.append(0.0 if len(w) < 5 or w.std() < 1e-9 else float(np.sqrt(250) * w.mean() / w.std()))

    plmu, plstd = float(dpl_np.mean()), float(dpl_np.std())
    annSharpe = float(np.sqrt(250) * plmu / plstd) if plstd > 0 else 0.0
    scoreVal = float(scoreFn(plmu, plstd))
    finalRet = float(value / totVol) if totVol > 0 else 0.0

    finalPx, startPx = prcAll[:, -1], prcAll[:, startDay]
    instruments = [{
        "idx": i, "name": names[i], "finalPos": int(curPos[i]),
        "finalDollarExposure": round(float(curPos[i] * finalPx[i]), 2),
        "maxDollarExposure": round(float(instMaxExp[i]), 2),
        "pnl": round(float(instPnl[i]), 2),
        "volume": round(float(instVol[i]), 2),
        "volumePct": round(float(instVol[i] / totVol * 100), 3) if totVol > 0 else 0.0,
        "daysActive": int(instDays[i]),
        "priceChangePct": round(float((finalPx[i] / startPx[i] - 1) * 100), 2),
    } for i in range(nInst)]

    data = {
        "summary": {
            "meanPL": round(plmu, 2), "stdPL": round(plstd, 2), "annSharpe": round(annSharpe, 3),
            "totDvolume": round(float(totVol), 2), "score": round(scoreVal, 2),
            "returnVal": round(finalRet, 5), "numTestDays": NUM_TEST_DAYS, "nInst": nInst,
            "finalValue": round(float(value), 2),
        },
        "days": days, "portfolioValue": vals, "dailyPL": dpl, "cumPL": cumpl,
        "drawdown": dd.tolist(), "dollarVolume": dvols, "rollingSharpe": rollSharpe,
        "instruments": instruments,
    }
    with open("dashboard_data.json", "w") as f:
        json.dump(data, f)
    print(f"score={scoreVal:.2f} sharpe={annSharpe:.2f} meanPL={plmu:.1f}")

def serve():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        url = f"http://localhost:{PORT}/dashboard.html"
        print(f"serving {url}  (ctrl+c to stop)")
        webbrowser.open(url)
        httpd.serve_forever()

if __name__ == "__main__":
    run()
    serve()