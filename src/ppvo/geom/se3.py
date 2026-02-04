import numpy as np

def Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3)
    return T

def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti
