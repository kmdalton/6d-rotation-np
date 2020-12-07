import numpy as np


def normalize_vector(x):
    return x / np.linalg.norm(x, axis=-1)[:,None]

def original(ortho6d):
    """
    This implementataion is based on Yi Zhou's torch implementation
    https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/sanity_test/code/tools.py#L47

    It was not obvious to me that this was equivalent to the implementation in the paper. 
    So, I wrote this little script to check. 
    """
    x_raw = ortho6d[...,0:3]
    y_raw = ortho6d[...,3:6]
        
    x = normalize_vector(x_raw)
    z = np.cross(x,y_raw) 
    z = normalize_vector(z)
    y = np.cross(z,x)
        
    matrix = np.dstack((x, y, z))
    return matrix

def paper(ortho6d):
    """
    This implementataion is verbatim the formalae from Yi Zhou's paper
    `On the Continuity of Rotation Representations in Neural Networks`
    """
    x_raw = ortho6d[...,0:3]
    y_raw = ortho6d[...,3:6]
        
    x = normalize_vector(x_raw)
    y = y_raw - (x*y_raw).sum(-1)[:,None]*x
    y = normalize_vector(y)
    z = np.cross(x,y)
        
    matrix = np.dstack((x, y, z))
    return matrix



data = np.random.random((1000, 6))
og = original(data)
pp = paper(data)


#Test that the implementations are equivalent
assert np.all(np.isclose(og, pp))

#Test for |R| == 1
assert np.all(np.isclose(np.linalg.det(og), 1.))
assert np.all(np.isclose(np.linalg.det(pp), 1.))

#Test for orthogonality (R^T == R^{-1})
assert np.all(np.isclose(og.swapaxes(-1, -2), np.linalg.inv(og)))
assert np.all(np.isclose(pp.swapaxes(-1, -2), np.linalg.inv(pp)))


