# post/stress.py
import numpy as np

def stress_from_bonds(cell, bonds):
    """
    bonds: iterable of (n, t, f_n, f_t, edge_length)
    returns sigma_xx, sigma_yy, sigma_xy
    """
    A = []
    b = []
    for (n, t, fn, ft, w) in bonds:
        tau = fn*n + ft*t
        nx, ny = n
        # tau_x = s_xx nx + s_xy ny
        A.append([nx, 0.0, ny]); b.append(tau[0])
        # tau_y = s_xy nx + s_yy ny
        A.append([0.0, ny, nx]); b.append(tau[1])
    A = np.asarray(A); b = np.asarray(b)
    # Weighted least squares
    # (You can repeat rows proportional to w or scale them)
    try:
        s_xx, s_yy, s_xy = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        s_xx = s_yy = s_xy = 0.0
    return s_xx, s_yy, s_xy

def von_mises_plane_stress(s_xx, s_yy, s_xy):
    return np.sqrt(s_xx**2 - s_xx*s_yy + s_yy**2 + 3.0*s_xy**2)
