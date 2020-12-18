from utils import *

class PCA:
    '''
    A general PCA class
    '''
    proj_matrix=None
    def __init__(self,X,explain_ratio=0.95):
        cov_matrix=np.cov(X.T)
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
        esum=np.sum(eigen_values)
        variance_explained = np.zeros(eigen_values.shape[0])
        for i,v in enumerate(eigen_values):
            variance_explained[i]=v / sum(eigen_values)
        cumulative_variance_explained = np.cumsum(variance_explained)
        self.proj_matrix=eigen_vectors[:,:max(1,np.where(explain_ratio<=cumulative_variance_explained)[0][0])]
        print("Dimension reduced from %d to %d"%(len(eigen_values),self.proj_matrix.shape[1]))
        return

    def proj(self,x):
        return np.real(np.dot(x,self.proj_matrix))


if __name__=="__main__":
    # Test
    X=np.random.random((5,10))
    print(X)
    pca=PCA(X,0.85)
    X_pca=pca.proj(X)
    print(X_pca)

