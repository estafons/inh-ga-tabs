"""

"""
import numpy as np
class LinearLeastSqaureModel:
    def fit(self, A, Y):
        A_transpose = A.transpose()
        ATA = A_transpose.dot(A)
        ATY = A_transpose.dot(Y)
        model = (np.linalg.inv(ATA)).dot(ATY) ## For a linear eq. AP = Y to solve a least sqaure problem,  P = (inverse(A'A))(A'Y) 
        return model