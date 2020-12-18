import codecs
import numpy as np
import matplotlib.pyplot as plt
import logging

from scipy.stats.stats import pearsonr
from scipy import signal
from file_operate import *
from utils import *

LOG_FORMAT = LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class CPA:
    traces = ''
    traceNum = ''
    signalLength = 0
    plainTexts = []
    LeakModel = None

    def __init__(self, traces, plain_texts, leakModel=HW):
        self.traces = traces
        self.traceNum = np.shape(traces)[0]
        self.signalLength = np.shape(traces)[1]
        self.plainTexts = plain_texts
        self.LeakModel = np.array(leakModel)
        self.LeakModel = self.LeakModel.reshape(-1)

    def getKey(self, numBytes=1):
        keys = np.zeros(numBytes)
        hws = np.zeros(self.traceNum)
        pccs = np.zeros(self.signalLength)
        cpa = np.zeros(256)
        transTraces = np.transpose(self.traces)
        # pge = np.zeros(numBytes)
        # for i in range(numBytes):

        for guess in range(256):
            for j in range(self.traceNum):
                input = self.plainTexts[j] ^ guess
                input = SBOX[input]
                hws[j] = self.LeakModel[input]
            for j in range(self.signalLength):
                pccs[j] = self.getPCC(transTraces[j], hws)
            cpa[guess] = np.max(abs(pccs))
            # logging.info("Guess %d" % guess)
        keys[0] = np.argmax(cpa)
        logging.info("Done %d byte 1")
        return keys, cpa

    def getPCC(self, X, Y):
        # return np.corrcoef(X,Y)[0,1]
        return pearsonr(X, Y)[0]


def align(trace0, tracei):
    correlation = signal.correlate(tracei ** 2, trace0 ** 2)
    shift = np.argmax(correlation) - (len(trace0) - 1)
    return shift


if __name__ == '__main__':
    # Setting for data operation, the REAL KEY is 66
    filename = r'mega128a5V4M_origin'
    path = r'./data'
    trace_num = 100000

    # Transfer trs to npz
    trs2Npz(path, filename, filename, trace_num)
    target = np.load(path + '\\' + filename + '.npz')

    # Trace set
    raw_traces = target["trace"]
    raw_plaintexts = target["crypto_data"]

    # Process the raw_trace with a naive alignment
    traces = raw_traces
    plaintexts = raw_plaintexts

    if False:  # change this if you need alignment
        traces, plaintexts = cor_align(raw_traces, raw_plaintexts)

    plot_sample(raw_traces, 'raw')
    plot_sample(traces, 'aligned')
    # exit()

    logging.info("Toal traces: " + str(len(traces)))

    attacker = CPA(traces=traces, plain_texts=plaintexts)
    keys, cpa = attacker.getKey(numBytes=1)

    plt.plot(cpa)
    # plt.axvline(real_keys[0][0],color='r', linestyle='--',alpha=0.3)
    plt.title('Correlation distribution of 1st Byte.')
    plt.show()

    print("Recovered key: " + str(keys))
    # print("Real key: " + str(real_keys[0]))
    # print(pge)
