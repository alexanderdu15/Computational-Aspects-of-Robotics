from numba import njit, prange
import numpy as np

def transform_is_valid(t, tolerance=1e-3):
    """ Check if array is a valid SE(3) transform.
    You can refer to the lecture notes to 
    see how to check if a matrix is a valid
    transform. 

    Args:
        t (numpy.array [4, 4]): Transform candidate.
        tolerance (float, optional): maximum absolute difference
            for two numbers to be considered close enough to each
            other. Defaults to 1e-3.

    Returns:
        bool: True if array is a valid transform else False.
    """
    #check if the matrix is 4x4 and real valued
    if t.shape != (4, 4) or not np.isrealobj(t):
        return False
    #check if the bottom row is [0, 0, 0, 1]
    if not np.allclose(t[3,:], np.array([0, 0, 0, 1]), atol=tolerance):
        return False
    #check if the top left 3x3 has determinant 1
    if not np.isclose(np.linalg.det(t[:3,:3]), 1, atol=tolerance):
        return False
    #check if the top left 3x3 is orthonormal
    if not np.allclose(t[:3,:3]@t[:3,:3].T, np.eye(3), atol=tolerance) or not np.allclose(t[:3,:3].T@t[:3,:3], np.eye(3), atol=tolerance):
        return False
    #otherwise is a valid transform
    return True

def transform_concat(t1, t2):
    """ Concatenate two transforms. Hint: 
        use numpy matrix multiplication. 

    Args:
        t1 (numpy.array [4, 4]): SE3 transform.
        t2 (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: t1 is invalid.
        ValueError: t2 is invalid.

    Returns:
        numpy.array [4, 4]: t1 * t2.
    """
    if not transform_is_valid(t1):
        raise ValueError("t1 is not a valid transform")

    if not transform_is_valid(t2):
        raise ValueError("t2 is not a valid transform")

    #return the concatenated transform
    return np.matmul(t1, t2)


def transform_point3s(t, ps):
    """ Transfrom a list of 3D points
    from one coordinate frame to another.

    Args:
        t (numpy.array [4, 4]): SE3 transform.
        ps (numpy.array [n, 3]): Array of n 3D points (x, y, z).

    Raises:
        ValueError: If t is not a valid transform.
        ValueError: If ps does not have correct shape.

    Returns:
        numpy.array [n, 3]: Transformed 3D points.
    """
    if not transform_is_valid(t):
        raise ValueError("t is not a valid transform")
        
    if ps.shape[1] != 3:
        raise ValueError("ps does not have correct shape")

    #add padding of 1 to the end of each point
    ps = np.hstack((ps, np.ones((ps.shape[0],1))))
    
    #apply the transform
    return np.matmul(t, ps.T).T[:,:3]

def transform_inverse(t):
    """Find the inverse of the transfom. Hint:
        use Numpy's linear algebra native methods. 

    Args:
        t (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: If t is not a valid transform.

    Returns:
        numpy.array [4, 4]: Inverse of the input transform.
    """
    if not transform_is_valid(t):
        raise ValueError("t is not a valid transform")

    #return the inverse of the transform
    return np.linalg.inv(t)

    
    

