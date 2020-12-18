import numpy
import matplotlib.pyplot as plt

from pca import *
from file_operate import *
from utils import *


class TA:
    '''
    A implementation of TA on 1 Byte of AES, the leak model is Hamming Weight by default.
    '''
    leak_model = None
    leak_range = None
    pois = None
    mean_matrix = None
    cov_matrix = None

    def __init__(self, traces, plain_texts, real_key, num_pois, leak_model=HW, poi_spacing=5):
        [trace_num, trace_point] = traces.shape
        self.leak_range = max(leak_model) + 1
        self.leak_model = leak_model
        self.mean_matrix = np.zeros((self.leak_range, num_pois))
        self.cov_matrix = np.zeros((self.leak_range, num_pois, num_pois))
        temp_SBOX = [SBOX[plain_texts[i] ^ real_key] for i in range(trace_num)]
        temp_lm = [leak_model[s] for s in temp_SBOX]
        # Sort traces by HW
        # Make self.leak_range blank lists - one for each Hamming weight
        temp_traces_lm = [[] for _ in range(self.leak_range)]
        # Fill them up
        for i, trace in enumerate(traces):
            temp_traces_lm[temp_lm[i]].append(trace)
        for mid in range(self.leak_range):
            assert len(temp_traces_lm[
                           mid]) != 0, "No trace with leak model value = %d, try increasing the number of traces" % mid
        # Switch to numpy arrays
        temp_traces_lm = [np.array(temp_traces_lm[_]) for _ in range(self.leak_range)]
        # Find averages
        tempMeans = np.zeros((self.leak_range, trace_point))
        for mid in range(self.leak_range):
            tempMeans[mid] = np.average(temp_traces_lm[mid], 0)
        # Find sum of differences
        tempSumDiff = np.zeros(trace_point)
        for i in range(self.leak_range):
            for j in range(i):
                tempSumDiff += np.abs(tempMeans[i] - tempMeans[j])
        # Find POIs
        self.pois = []
        for i in range(num_pois):
            # Find the max
            nextPOI = tempSumDiff.argmax()
            self.pois.append(nextPOI)
            # Make sure we don't pick a nearby value

            poiMin = max(0, nextPOI - poi_spacing)
            poiMax = min(nextPOI + poi_spacing, len(tempSumDiff))
            for j in range(poiMin, poiMax):
                tempSumDiff[j] = 0
        # Fill up mean and covariance matrix for each HW
        self.mean_matrix = np.zeros((self.leak_range, num_pois))
        self.cov_matrix = np.zeros((self.leak_range, num_pois, num_pois))
        for mid in range(self.leak_range):
            for i in range(num_pois):
                # Fill in mean
                self.mean_matrix[mid][i] = tempMeans[mid][self.pois[i]]
                for j in range(num_pois):
                    x = temp_traces_lm[mid][:, self.pois[i]]
                    y = temp_traces_lm[mid][:, self.pois[j]]
                    self.cov_matrix[mid, i, j] = cov(x, y)
        print("The template has been created.")
        return

    def attack(self, traces, plaintext):
        rank_key = np.zeros(256)
        for j, trace in enumerate(traces):
            # Grab key points and put them in a small matrix
            a = [trace[poi] for poi in self.pois]

            # Test each key
            for k in range(256):
                # Find leak model coming out of sbox
                mid = self.leak_model[SBOX[plaintext[j] ^ k]]

                # Find p_{k,j}
                # print(np.linalg.det(self.cov_matrix[mid]))
                rv = multivariate_normal(self.mean_matrix[mid], self.cov_matrix[mid],allow_singular=True)
                p_kj = PRE[mid]*rv.pdf(a)

                # Add it to running total
                rank_key[k] += np.log(p_kj)

        guessed = rank_key.argsort()[-1]
        print("Key found: %d" % guessed)
        return self.mean_matrix, self.cov_matrix, guessed


if __name__ == '__main__':
    # Setting for data operation
    filename = r'mega128a5V4M_origin'
    path = r'./data'
    trace_num = 10000
    train_key = 66

    # Transfer trs to npz
    trs2Npz(path, filename, filename, trace_num)
    target = np.load(path + '\\' + filename + '.npz')
    raw_traces=target["trace"]
    plaintexts=target["crypto_data"]

    # Normalization on raw data traces
    traces=standardize(raw_traces)

    # If you need PCA, uncomment this
    # pca=PCA(traces,explain_ratio=0.8)
    # traces=pca.proj(traces)

    # Train set
    num_train = 9800
    train_tr = traces[:num_train, :]
    train_pt = plaintexts[:num_train]
    # Attack set
    attack_tr = traces[num_train:, :]
    attack_pt = plaintexts[num_train:]

    # Get a TA attacker
    ta = TA(traces=train_tr, plain_texts=train_pt, real_key=train_key, num_pois=5)
    mean_matrix, cov_matrix, guessed = ta.attack(attack_tr, attack_pt)
