import numpy as np
import pandas as pd

nInst = 51
currentPos = np.zeros(nInst, dtype=int)

FAST, SLOW, VOL_LB = 3, 20, 20
DOLLAR_TARGET = np.full(nInst, 10000.0)
DOLLAR_TARGET[0] = 100000.0
THRESH = 0.3
DECAY = 0.8  # shrink toward flat when signal weakens


def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape
    if nt < SLOW + 2:
        return np.zeros(nins, dtype=int)

    logp = np.log(prcSoFar.T)  # nt x nins
    df = pd.DataFrame(logp)

    fastE = df.ewm(span=FAST, adjust=False).mean().iloc[-1].values
    slowE = df.ewm(span=SLOW, adjust=False).mean().iloc[-1].values

    logret = df.diff().values
    vol = np.nanstd(logret[-VOL_LB:], axis=0) + 1e-8

    # mean reversion: fade extended EMA gaps
    signal = (fastE - slowE) / vol
    z = np.clip(-signal / 3.0, -1, 1)

    curPrices = prcSoFar[:, -1]
    target = currentPos.copy().astype(float)
    active = np.abs(signal) > THRESH

    target[active] = z[active] * DOLLAR_TARGET[active] / curPrices[active]
    target[~active] *= DECAY  # signal weak/gone: unwind toward flat

    currentPos = target.astype(int)
    return currentPos