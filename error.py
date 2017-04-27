import numpy as np
import numdifftools as nd
import math

digits = 5

def propogate(f, x, s):
    x = np.array(x)
    s = np.array(s)
    return np.round([f(x), math.sqrt(np.dot(nd.Jacobian(f)(x)**2, s**2))], digits)