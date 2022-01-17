import numpy as np
import matplotlib.pyplot as plt



def pegasos(K,y,N,C,seed=41,full_returns=False):
    """
    Pegasos algorithm for
    K.. Gram matrix of kernel
    y.. ground truth of the training data
    N.. Number of iterations
    C.. Regularization

    Returns:
    y_pred.. final prediction of the training set
    
    if full_returns is set to True:

        y_pred.. prediction of training set at every step
        a.. dual coefficients at every step
        sums.. the sum in the condition at every step
        evals.. total kernel evaluations at every step

    """

    np.random.seed(seed)
    # Size of the data set
    M = K.shape[0]
    # Weights of the support vectors
    a = np.zeros((N+1,M),dtype=int)
    # Number of kernel evaluations
    evals = np.zeros(N+1,dtype=int)
    # Sums evaluated for the if condition
    sums = np.zeros(N)
    # Soft-max predictions
    y_preds = np.zeros((N,M))

    
    for t in range(1,N+1):
        # Choose random index
        i = np.random.randint(M)
        
        # Calculate weighted kernel sum
        s = C/t*np.sum(y*a[t-1]*K[i,:])
        sums[t-1] = s
        a[t,:] = a[t-1,:]
        # Update alphas if condition is fulfilled
        if y[i]*s < 1:
            a[t,i] = a[t-1,i] + 1
        # Predict labels for the training set
        y_preds[t-1] = np.sum(y*a[t]*K,axis=1)

        # only count evaluations for non-zero alphas
        evals[t] = evals[t-1] + np.sum(a[t-1] > 0)

    if full_returns:
        # Multiply the alphas at step 't' by 'C/t' to normalize
        a_new = a[1:] * C/np.arange(1,N+1).reshape(-1,1)
        y_preds = y_preds * C/np.arange(1,N+1).reshape(-1,1)
        return y_preds, a_new, sums, evals[1:]
    else:
        return np.sign(y_preds[-1])


