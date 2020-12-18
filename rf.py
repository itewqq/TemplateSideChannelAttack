import numpy
from numpy.linalg import inv,pinv,eig
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from pca import *
from file_operate import *
from utils import *

class RF_Attacker:
    '''
    An implementation of Random Forest Attack on 1 Byte of AES, the leak model is Hamming Weight by default.
    '''
    clf = None
    leak_model = None
    leak_range = None

    def __init__(self, traces, plain_texts, real_key, leak_model=HW):
        self.leak_model = leak_model
        self.leak_range = max(leak_model) + 1
        labels = [leak_model[SBOX[pt ^ real_key]] for pt in plain_texts]
        labels = np.asarray(labels)
        self.clf = RandomForestClassifier(n_estimators=100)
        self.clf.fit(traces, labels)
        print("The random forest template has been created.")

    def attack(self, traces, plaintexts):
        probs=self.clf.predict_proba(traces)
        score = np.zeros(256)
        for prob, plaintext in zip(probs, plaintexts):
            for k in range(256):
                mid=self.leak_model[SBOX[plaintext ^ k]]
                score[k] += PRE[mid]*prob[mid]


        print("Key found: %d" % score.argsort()[-1])

if __name__ == '__main__':
    # Setting for data operation, the REAL KEY is 66
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
    pca=PCA(traces,explain_ratio=0.95)
    traces=pca.proj(traces)

    # Train set
    num_train = 9800
    train_tr = traces[:num_train, :]
    train_pt = plaintexts[:num_train]
    # Attack set
    attack_tr = traces[num_train:, :]
    attack_pt = plaintexts[num_train:]

    # Attack
    svm_ta = RF_Attacker(traces=train_tr, plain_texts=train_pt, real_key=train_key)
    svm_ta.attack(attack_tr, attack_pt)
