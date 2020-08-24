import numpy as np
import os
import pickle
from collections import defaultdict
from scipy.interpolate import Rbf
from utils.sphereicosahedron import SphereIcosahedron
from utils.samplings import icosahedron_order_calculator
from utils.funcs import closing

def interpolate3d(points, indices, func_vals, **kwargs):
    """
    Interpolate values on 3D point clouds.
    """
    rbfi = Rbf(points[indices,0], points[indices,1], points[indices,2],
               func_vals, **kwargs)
    oth_ind = list(set(range(len(points))) - set(indices))
    di = rbfi(points[oth_ind,0], points[oth_ind,1], points[oth_ind,2])
    new_vals = np.zeros(len(points))
    new_vals[indices] = func_vals
    new_vals[oth_ind] = di
    return new_vals

def calc_corresp(p1, p2, theta=None, verbose=False):
    """

    Parameters
    ----------
    p1 : numpy array
        The points which will be assigned to the points in p2.
    p2 : numpy array
        The points for which correspondences will be found.
        Should have less rows than p1.
    theta : float, optional
        Half the aperture angle for the conic search area. The default is None.

    Returns
    -------
    corresp_ind : numpy array (dtype = np.int32)
        The 1st column contains the indices of the points in p2
        and the 2nd column contains the indices of the
        correspondences in p1.

    """
    if len(p1) < len(p2):
        print('p1 has less points than p2.')
    if theta is None: theta = (15 * np.pi)/p1.shape[0]
    theta = np.cos(theta)
    corr_list = defaultdict(list)
    O = np.copy(p1)
    P = np.copy(p2)
    O_norm = O / np.linalg.norm(O, axis=1)[:, None]
    P_norm = P / np.linalg.norm(P, axis=1)[:, None]
    Q = np.argwhere(P_norm.dot(O_norm.T) >= theta)
    for p, q in zip(Q[:,0], Q[:,1]):
        corr_list[p].append(q)
    assigned = defaultdict(int)
    collisions = 0
    for k, v in corr_list.items():
        v.sort(key=lambda x: np.linalg.norm(O[x,:]-P[k,:]))
        for p in v:
            if p not in set(assigned.values()):
                assigned[k] = p
                break
            else:
                collisions += 1
    if verbose:
        print('Collisions: {}'.format(collisions))
    return np.array(list(assigned.items()))

def normalise(cases, scale=None):
    """
    Normalises the cases such that centroids are at the origin and
    the max distance of a point on the surface to the origin is 'scale'.
    If scale is None then no scaling is performed.
    """
    new_points = []
    for p in cases:
        pp = p - p.mean(axis=0)
        if scale is not None:
            max_norm = max(np.linalg.norm(pp, axis=1))
            new_points.append(pp / max_norm * scale)
        else: new_points.append(pp)
    return new_points

def _get_corrs(sphere, scaled_cases, theta):
    corrs = []
    for i, c in enumerate(scaled_cases):
        if len(c) < len(sphere):
            corr_list = calc_corresp(sphere, scaled_cases[i],
                                           theta=theta)
            corrs.append(corr_list[:, [1,0]])
        else:
            corr_list = calc_corresp(scaled_cases[i], sphere,
                                           theta=theta)
            corrs.append(corr_list)
        if not len(corr_list):
            raise RuntimeError('No corresponding points found ' +
                               'for at least one point cloud.')
    return corrs

def _get_dists(sphere, norm_cases, corrs):
    dists, idxs = [], []
    for i, c in enumerate(corrs):
        dists.append(np.apply_along_axis(lambda x:
                                         np.linalg.norm(sphere[x[0]])
                                         - np.linalg.norm(norm_cases[i][x[1]])
                                         , 1, c))
        idxs.append(c[:,0])
    return dists, idxs

def _get_tawss_map(sphere, tawss, corrs):
    wsss, idxs = [], []
    for i, c in enumerate(corrs):
        wsss.append(tawss[i][c[:,1]])
        idxs.append(c[:,0])
    return wsss, idxs


def map_to_sphere(shapes, wss_vals, theta=0.08, interp_type='linear', epsilon=None, npix=None):
    """
    shapes: list of (n_points, 3) matrices. Assumed to be at the origin.
    wss_vals: list of (npoints, ) arrays corresponding to shapes. -1 for missing values
    """
    
    sphere = SphereIcosahedron(level=int(icosahedron_order_calculator(npix)))
    sphere_xyz = normalise([sphere.coords], scale=1.0)[0]
    scaled_cases = normalise(shapes, scale=1.0)
    
    try:
        corrs = _get_corrs(sphere_xyz, scaled_cases, theta)
    except RuntimeError as e: sys.exit(str(e))
    
    dists, idxs = _get_dists(sphere_xyz, shapes, corrs)
    wss_map, wss_idxs = _get_tawss_map(sphere_xyz, wss_vals, corrs)
    
    binary_map = []
    for i in range(len(idxs)):
        bm = np.zeros(sphere_xyz.shape[0])
        bm[idxs[i]] = 1
        # k=7 works well for closing:
        binary_map.append(closing(SphereIcosahedron(level=int(icosahedron_order_calculator(npix)), k=7), bm))
    binary_map = np.vstack(binary_map)
    del bm
    
    try:
        for i in range(len(dists)):
            dists[i] = interpolate3d(sphere_xyz, idxs[i],
                                     dists[i], function=interp_type, epsilon=epsilon)
            dists[i][binary_map[i] == 0] = 1.0
            wss_map[i] = interpolate3d(sphere_xyz, idxs[i][wss_map[i]!=-1],
                                       wss_map[i][wss_map[i]!=-1], function=interp_type, epsilon=epsilon)
    except Exception as e: sys.exit(str(e))
    
    dists = np.vstack(dists)
    wss_map = np.vstack(wss_map)
    
    return dists, wss_map, sphere, binary_map, corrs
