# Copyright (c) 2015, Thomas Hornung
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .kern import Kern
from ...core.parameterization import Param

class ThinPlate(Kern):
    """
    This kernel function is created following this tutorial: https://gpy.readthedocs.io/en/latest/tuto_creating_new_kernels.html# 
    Thin-plate Spline regression.
    https://www.microsoft.com/en-us/research/wp-content/uploads/2007/04/gpc1.pdf
    """

    def __init__(self, input_dim, variance=1., R=50, epsilon=1e-3+1, active_dims=None, name='thin_plate'):
        """Definition of the kernel, children of the Kern class.

        Args:
            input_dim (int): dimension of the input space
            variance (float, optional): variance for the noisy observation. Defaults to 1..
            R (int, optional): the radius of the region within which the TP model will minimize the second-order gradient. Defaults to 50.
            epsilon (float, optional): _description_. Defaults to 1e-3+1.
            active_dims (int, optional): All kernels will get sliced Xes as inputs, if _all_dims_active is not None. Defaults to None.
            name (str, optional): name of the kernal. Defaults to 'thin_plate'.
        """
        super(ThinPlate, self).__init__(input_dim, active_dims, name)
        self.variance = Param('variance', variance)
        self.R = Param('R', R)
        self.epsilon = epsilon


    def K(self, X, X2=None):
        """This function is used to compute the covariance matrix associated with the inputs X, X2 with self.input_dim columns.

        Args:
            X (array): np.arrays with arbitrary number of lines, n1, corresponding to the number of samples over which to calculate covariance
            X2 (array, optional): np.arrays with arbitrary number of lines, n2, corresponding to the number of samples over which to calculate covariance

        Returns:
            float: kernal covariance value
        """
        if X2 is None: X2=X
        # l2-distance between the two set of samples, X is of shamep (n1, 3), 
        # in which n1 number of samples, each sample is contact position of 3 numbers
        pdist = np.linalg.norm(np.expand_dims(X, axis=-1)-np.expand_dims(X2.T, axis=0), axis=1)
        # calculate the three terms in TP model
        term1 = 2*np.abs(pdist)**3
        term2 = -3*self.R*pdist**2
        term3 = self.R**3
        # if X and X2 has same number of samples (on diagnal), we add noise
        # otherwise, we don't since no need for noise for the off-diagnal elements in the covariance matrix?
        # TODO: is it because the observation is noisy? (X2 = X means we are calculating the covariance of observed points)
        if term1.shape[0] == term1.shape[1]:
            noise = self.variance**2*np.eye(X.shape[0])
            return noise + (term1 + term2 + term3)/12
        else:
            return (term1 + term2 + term3)/12
        
    def to_dict(self):
        input_dict = {}
        input_dict["name"] = self.name
        input_dict["class"] = "GPy.kern.ThinPlate"
        input_dict["variance"] = self.variance.values.tolist()
        input_dict["R"] = self.R.values.tolist()
        input_dict["epsilon"] = self.epsilon
        return input_dict

    def Kdiag(self, X):
        """This function is similar to K but it computes only the values of the kernel on the diagonal.

        Args:
            X (array): 1-dimensional np.array of length nx1

        Returns:
            float: convariance value on the diagnal
        """
        pdist = np.linalg.norm(np.expand_dims(X, axis=-1)-np.expand_dims(X.T, axis=0), axis=1)
        term1 = 2*np.abs(pdist)**3
        term2 = -3*self.R*pdist**2
        term3 = self.R**3
        # since we are calculating the values on the diagonal
        # which is observation data, we need to add noise
        noise = self.variance**2*np.eye(X.shape[0])
        return noise + (term1 + term2 + term3)/12

    def update_gradients_full(self, dL_dK, X, X2=None):
        """This function is required for the optimization of the parameters.
        Computes the gradients and sets them on the parameters of this model using chain rule
        eg. suppose K is modeled using parameter theta
        dL/dTheta =  dL/dK * dK/dTheta

        Args:
            dL_dK (float): _description_
            X (array): array of samples
            X2 (_type_, optional): array of samples. Defaults to None.
        """
        if X2 is None: X2=X
        pdist = np.linalg.norm(np.expand_dims(X, axis=-1)-np.expand_dims(X2.T, axis=0), axis=1)

        # gradient of R can be calculated by dL/dK * dK/dR
        self.R.gradient = np.sum(dL_dK * ((-3*pdist**2 + 3*self.R**2)/12))
        # gradient of variance can be calculated by dL/dK * dK/dSigma
        self.variance.gradient = np.sum(dL_dK * (2*self.variance*np.eye(X.shape[0])))

    def update_gradients_diag(self, dL_dKdiag, X):
        raise NotImplementedError

    def gradients_X(self, dL_dK, X, X2=None):
        raise NotImplementedError

    def gradients_X_diag(self, dL_dKdiag, X):
        raise NotImplementedError
