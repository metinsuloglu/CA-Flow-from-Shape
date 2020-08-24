"""
Collection of various useful functions
"""

import math
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import minimize


def icp3d(src, trgt, abs_tol=1e-8, max_iter=500, verbose=True):
    """

    Parameters
    ----------
    src : numpy array
        Source object. Each row should be a point with (X, Y, Z) columns.
    trgt : numpy array
        Target object. Each row should be a point with (X, Y, Z) columns.
    abs_tol : float, optional
        Absolute tolerance for convergence. The default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations. The default is 500.
    verbose : boolean, optional
        Whether to enable output. The default is True.

    Returns
    -------
    t : numpy array
        Translation matrix.
    R: numpy array
        Rotation matrix.
    src_trans : numpy array
        Transformed source points.

    """

    def least_squares(p, q, verbose=1):
        P = np.mean(p, axis=0)
        Q = np.mean(q, axis=0)
        pp = p-P
        qq = q-Q
        H = np.dot(pp.T, qq)
        U, S, Vt = np.linalg.svd(H)
        X = np.dot(Vt.T, U.T)
        if np.linalg.det(X) < 0:
            S_zero = np.nonzero(S == 0)[0]
            if S_zero.size > 0:
                Vt[S_zero, :] *= -1
                X = np.dot(Vt.T, U.T)
            else:
                if verbose:
                    print('None of the singular values are 0. ' +
                          'Conventional least-squares is probably not appropriate.')
                return np.zeros((3, 1)), np.eye(3)
        return Q.T - np.dot(X, P.T), X

    p = np.copy(src)
    q = np.copy(trgt)

    kd = KDTree(q)

    prev_mse = 0.0
    for i in range(max_iter):
        nns = kd.query(p)
        mse = (nns[0] ** 2).mean()
        if abs(mse - prev_mse) < abs_tol:
            if verbose:
                print('Converged in {} iterations.'.format(i))
            break
        prev_mse = mse
        t, R = least_squares(p, q[nns[1]], verbose=verbose)
        p = np.dot(p, R.T) + t.T
    if verbose and i == max_iter-1:
        print('Maximum number of iterations reached.')
    t, R = least_squares(src, p)
    return t, R, np.dot(src, R.T) + t.T, nns
    
    
def affine_no_rotation(src, trgt):
    """
    Function to transform a source point cloud to a target point cloud
    using translation, scaling and shear but no rotation  - not used in the project.
    
    Returns
    -------
    A, b matrices for the affine transformation s.t. A * src + b ~= trgt
    res: results of SciPy minimisation function
    """

    def objective(x, p, q):
        W = x[:9].reshape(3,3)
        S = np.diag(x[9:12])
        b = x[12:].reshape(3,1)
        return np.sum(np.linalg.norm(W @ S @ W.T @ p.T + b - q.T, axis=0))

    x0 = np.array([1,0,0,0,1,0,0,0,1,1,1,1,0,0,0])
    constraints = [
        {'type':'eq','fun': lambda x: x[0]**2+x[1]**2+x[2]**2-1},
        {'type':'eq','fun': lambda x: x[3]**2+x[4]**2+x[5]**2-1},
        {'type':'eq','fun': lambda x: x[6]**2+x[7]**2+x[8]**2-1},
        {'type':'eq','fun': lambda x: x[0]*x[3]+x[1]*x[4]+x[2]*x[5]},
        {'type':'eq','fun': lambda x: x[0]*x[6]+x[1]*x[7]+x[2]*x[8]},
        {'type':'eq','fun': lambda x: x[3]*x[6]+x[4]*x[7]+x[5]*x[8]},
        {'type':'ineq','fun': lambda x: x[9]},
        {'type':'ineq','fun': lambda x: x[10]},
        {'type':'ineq','fun': lambda x: x[11]}
    ]
    res = minimize(fun=objective, x0=x0, args=(src, trgt), jac=False,
                   constraints=constraints, method='SLSQP', tol=1e-12,
                   options={'disp':True})
    W = res.x[:9].reshape(3,3)
    S = np.diag(res.x[9:12])
    b = res.x[12:].reshape(3,1)
    A = W @ S @ W.T
    return A, b, res


def erosion(sphere, grayscale_map):
    """
    Morphological erosion over an icosphere
    
    sphere: object of type SphereIcosahedron
    grayscale_map: binary values for each vertex of the sphere (n_sphere,)
    """
    sources, targets, _ = sphere.get_edge_list()
    inverted_map = np.logical_not(grayscale_map)
    edges = np.hstack((sources.reshape(-1,1), targets.reshape(-1,1)))
    edges = np.vstack((edges, edges[:, [1,0]])) # target -> source
    # every vertex connected to itself:
    edges = np.vstack((edges, np.repeat(np.arange(len(grayscale_map)).reshape(-1,1), 2, axis=1)))
    outputs = np.zeros_like(grayscale_map)
    for i in range(outputs.shape[0]):
        outputs[i] = np.logical_not(np.any(inverted_map[edges[edges[:,0]==i, 1]]))
    return outputs


def dilation(sphere, grayscale_map):
    """
    Morphological dilation over an icosphere
    
    sphere: object of type SphereIcosahedron
    grayscale_map: binary values for each vertex of the sphere (n_sphere,)
    """
    sources, targets, _ = sphere.get_edge_list()
    edges = np.hstack((sources.reshape(-1,1), targets.reshape(-1,1)))
    edges = np.vstack((edges, edges[:, [1,0]])) # target -> source
    # every vertex connected to itself:
    edges = np.vstack((edges, np.repeat(np.arange(len(grayscale_map)).reshape(-1,1), 2, axis=1)))
    outputs = np.ones_like(grayscale_map)
    for i in range(outputs.shape[0]):
        outputs[i] = np.any(grayscale_map[edges[edges[:,0]==i, 1]])
    return outputs


def opening(sphere, grayscale_map):
    """
    Morphological opening over an icosphere
    Applies erosion followed by dilation
    """
    e = erosion(sphere, grayscale_map)
    return dilation(sphere, e)


def closing(sphere, grayscale_map):
    """
    Morphological closing over an icosphere
    Applies dilation followed by erosion
    """
    d = dilation(sphere, grayscale_map)
    return erosion(sphere, d)
    
    
def count_holes(sphere, grayscale_map):
    """
    Count number of connected regions over an icosphere where
    the grayscale_map is 0.
    """

    def floodfill(curr_map, index):
        if curr_map[index] == 1: return
        curr_map[index] = 1
        for ind in edges[edges[:,0]==index, 1]:
            floodfill(curr_map, ind)
        
    curr_map = grayscale_map.copy()

    sources, targets, _ = sphere.get_edge_list()
    edges = np.hstack((sources.reshape(-1,1), targets.reshape(-1,1)))
    edges = np.vstack((edges, edges[:, [1,0]])) # target -> source

    count = 0
    for i in range(len(sphere.coords)):
        if curr_map[i] == 1: continue
        floodfill(curr_map, i)
        count += 1
        
    return count


def reconstruct(sphere, dists):
    """
    Given the distances to a sphere with a radius of 1 centred at the origin,
    this function constructs a point cloud.
    """
    if dists.ndim == 1: dists = dists[:,None]
    return np.multiply(sphere, (1-dists))


def fibonacci_sphere(samples=1000, semi=False):
    """
    Creates a point cloud of a Fibonacci sphere - not used in the project.
    """
    points = np.array([[0, 1, 0]])
    phi = math.pi * (3. - math.sqrt(5.))
    num_points = samples
    if semi: samples = samples * 2
    for i in range(1, num_points):
        y = 1 - (i / float(samples - 1)) * 2
        radius = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points = np.vstack((np.array([[x, y, z]]), points))
    return points

