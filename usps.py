import numpy as np
import h5py

def get_data(number_of_points=1000):
    # Load digits
    with h5py.File('usps.h5', 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]
    
    # Squash data together
    X = np.vstack((X_tr, X_te))
    y = np.hstack((y_tr, y_te))

    # Sort labels numerically and sort the data the same way
    order = np.argsort(y)
    X = X[order]
    y = y[order]

    # Get cutoff
    i_breakpoint = get_breakpoint(y, 5)

    # Partition into two sets
    x_0_4 = X[:i_breakpoint]
    y_0_4 = -np.ones(x_0_4.shape[0])
    x_5_9 = X[i_breakpoint:]
    y_5_9 = np.ones(x_5_9.shape[0])

    return x_0_4[:number_of_points], x_5_9[:number_of_points], y_0_4[:number_of_points], y_5_9[:number_of_points]

def get_breakpoint(l, val):
    for i, x in enumerate(l):
        if x == val:
            return i

    return -1




get_data()
