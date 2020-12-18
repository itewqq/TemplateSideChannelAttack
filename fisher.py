import numpy
from numpy.linalg import inv,pinv,eig
import matplotlib.pyplot as plt

from pca import *
from file_operate import *
from utils import *


class LDAClassifier:

    def __init__(self, projection_dim):
        self.projection_dim = projection_dim
        self.W = None  # weights
        self.g_means, self.g_covariance, self.priors = None, None, None

    def fit(self, X):
        means_k = self.__compute_means(X)

        Sks = []
        for class_i, m in means_k.items():
            sub = np.subtract(X[class_i], m)
            Sks.append(np.dot(np.transpose(sub), sub))

        Sks = np.asarray(Sks)
        Sw = np.sum(Sks, axis=0)  # shape = (D,D)

        Nk = {}
        sum_ = 0
        for class_id, data in X.items():
            Nk[class_id] = data.shape[0]
            sum_ += np.sum(data, axis=0)

        self.N = sum(list(Nk.values()))

        # m is the mean of the total data set
        m = sum_ / self.N

        SB = []
        for class_id, mean_class_i in means_k.items():
            sub_ = mean_class_i - m
            SB.append(np.multiply(Nk[class_id], np.outer(sub_, sub_.T)))

        # between class covariance matrix shape = (D,D). D = input vector dimensions
        SB = np.sum(SB, axis=0)  # sum of K (# of classes) matrices

        matrix = np.dot(pinv(Sw), SB)
        # find eigen values and eigen-vectors pairs for np.dot(pinv(SW),SB)
        eigen_values, eigen_vectors = eig(matrix)

        eiglist = [(eigen_values[i], eigen_vectors[:, i]) for i in range(len(eigen_values))]

        # sort the eigvals in decreasing order
        eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)

        # take the first num_dims eigvectors
        self.W = np.array([eiglist[i][1] for i in range(self.projection_dim)])
        self.W = np.asarray(self.W).T

        # get parameter of the Gaussian distribution
        self.g_means, self.g_covariance, self.priors = self.gaussian(X)

    # Returns the parameters of the Gaussian distributions
    def gaussian(self, X):
        means = {}
        covariance = {}
        priors = {}  # p(Ck)
        for class_id, values in X.items():
            proj = np.dot(values, self.W)
            means[class_id] = np.mean(proj, axis=0)
            covariance[class_id] = np.cov(proj, rowvar=False)
            priors[class_id] = values.shape[0] / self.N
        return means, covariance, priors

    # model a multi-variate Gaussian distribution for each class’ likelihood distribution P(x|Ck)
    def gaussian_distribution(self, x, u, cov):
        scalar = (1. / ((2 * np.pi) ** (x.shape[0] / 2.))) * (1 / np.sqrt(np.linalg.det(cov)))
        x_sub_u = np.subtract(x, u)
        return scalar * np.exp(-np.dot(np.dot(x_sub_u, inv(cov)), x_sub_u.T) / 2.)


    def prob(self,x,y):
        return  self.priors[y] * self.gaussian_distribution(self.project(x), self.g_means[y], self.g_covariance[y])

    def score(self, X, y):
        proj = self.project(X)
        gaussian_likelihoods = []
        classes = sorted(list(self.g_means.keys()))
        for x in proj:
            row = []
            for c in classes:  # number of classes
                res = self.priors[c] * self.gaussian_distribution(x, self.g_means[c], self.g_covariance[
                    c])  # Compute the posterios P(Ck|x) prob of a class k given a point x
                row.append(res)

            gaussian_likelihoods.append(row)

        gaussian_likelihoods = np.asarray(gaussian_likelihoods)

        # assign x to the class with the largest posterior probability
        predictions = np.argmax(gaussian_likelihoods, axis=1)
        return np.sum(predictions == y) / len(y), predictions, proj

    def project(self, X):
        return np.dot(X, self.W)

    def __compute_means(self, X):
        # Compute the means for each class k=1,2,3...K
        # If the dataset has K classes, then, self.means_k.shape = [# of records, K]
        means_k = {}
        for class_i, input_vectors in X.items():
            means_k[class_i] = np.mean(input_vectors, axis=0)
        return means_k

class LDAAttacker:
    '''
    A implementation of Fisher-LDA Attack on 1 Byte of AES, the leak model is Hamming Weight by default.
    '''
    clf=None
    leak_model=None
    leak_range = None

    def __init__(self, traces, plain_texts, real_key, leak_model=HW):
        self.leak_model = leak_model
        self.leak_range = max(leak_model)+1
        data_dict = {}
        for x, pt in zip(traces, plain_texts):
            # Get Hamming Weight
            y=leak_model[SBOX[pt^real_key]]
            if y not in data_dict:
                data_dict[y] = [x.flatten()]
            else:
                data_dict[y].append(x.flatten())

        for i in range(self.leak_range):
            data_dict[i] = np.asarray(data_dict[i])

        self.clf=LDAClassifier(self.leak_range)
        self.clf.fit(data_dict)
        print("The Fisher-LDA template has been created.")

    def attack(self,traces,plaintexts):
        score=np.ones(256)
        for trace,plaintext in zip(traces,plaintexts):
            for k in range(256):
                score[k]+=np.real(self.clf.prob(trace,self.leak_model[SBOX[plaintext^k]]))
        print("Key found: %d" % score.argsort()[-1])


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

    # Attack
    fish=LDAAttacker(traces=train_tr,plain_texts=train_pt,real_key=train_key)
    fish.attack(attack_tr,attack_pt)


