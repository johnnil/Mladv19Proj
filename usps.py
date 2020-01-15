import numpy as np
import h5py

def get_data():
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
    y_0_4 = y[:i_breakpoint]
    x_5_9 = X[i_breakpoint:]
    y_5_9 = y[i_breakpoint:]

    return x_0_4, x_5_9, y_0_4, y_5_9

def get_breakpoint(l, val):
    for i, x in enumerate(l):
        if x == val:
            return i

    return -1




get_data()
