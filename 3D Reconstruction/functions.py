import numpy as np

def minimal_triangulation(Ps, us):
    """
    Ps: list of two (3x4) camera projection matrices as numpy arrays
    us: (2x2) numpy array of image coordinates, us.shape=2xn where n=#points
    Returns: 3D point (3,) as a numpy array
    """

    # Your code here

    P1, P2 = Ps
    x1, y1 = us[:, 0]
    x2, y2 = us[:, 1]

    A = np.array([P1[2, :]*x1 - P1[0, :],
                  P1[2, :]*y1 - P1[1, :],
                  P2[2, :]*x2 - P2[0, :],
                  P2[2, :]*y2 - P2[1, :]
                 ])

    # Solve AX = 0
    _, _, Vh = np.linalg.svd(A)
    U_hom = Vh[-1]               # Homogeneous solution
    U_dehom = U_hom / U_hom[-1]  # Dehomogenize
    U = U_dehom[:3]              # Dehomogeneous solution

    return U 

def check_depths(Ps, U):
    """
    Ps: list of camera matrices (each 3x4)
    U: 3D point as a NumPy array of shape (3,)
    
    Returns: NumPy array of 0s and 1s indicating positive depth for each camera
    """

    # Your code here
    
    U_hom = np.append(U, 1)

    positive = []
    for P in Ps:
        p = P @ U_hom
        positive.append(1 if p[2] > 0 else 0)
    positive = np.array(positive)

    return positive

def reprojection_errors(Ps, us, U):
    """
    Ps: list of camera matrices (each 3x4 numpy array)
    us: 2xN numpy array of image coordinates (each column is a point)
    U: 3D point as a numpy array of shape (3,)
    
    Returns: Numpy array of reprojection errors (N,)
    """

    # Your code here
    
    N = len(Ps)
    U_hom = np.append(U, 1)
    positive_depth = check_depths(Ps, U)
    errors = np.zeros(N)
    
    for i in range(N):
        if positive_depth[i]:
            p = Ps[i] @ U_hom
            u_projected = p[:2] / p[2]
            u_measured = us[:, i]
            errors[i] = np.linalg.norm(u_projected - u_measured)            
        else:
            errors[i] = np.inf

    return errors

def ransac_triangulation(Ps, us, threshold, num_iter=100):
    """
    Ps: list of camera matrices (each 3x4)
    us: 2xN numpy array of image coordinates
    threshold: reprojection error threshold for inlier selection
    
    Returns:
    - U: best estimated 3D point (3,)
    - nbr_inliers: number of inliers for best estimate
    """

    # Your code here
    
    N = len(Ps)
    best_U = None
    nbr_inliers = -1

    for _ in range(num_iter):
        # Triangulate 3D point
        idx = np.random.choice(N, 2, replace=False)
        P1 = Ps[idx[0]]
        P2 = Ps[idx[1]]
        us_sample = us[:, idx]
        U_current = minimal_triangulation([P1, P2], us_sample)

        # Compute depth
        positive_depth = check_depths(Ps, U_current)
        
        # Compute reprojection errors
        errors = reprojection_errors(Ps, us, U_current)

        # Maximize the number of inliers
        in_current = np.sum((positive_depth == 1) & (errors < threshold))
        if in_current > nbr_inliers:
            best_U = U_current
            nbr_inliers = in_current

    return best_U, nbr_inliers

def compute_residuals(Ps, us, U):
    """
    Ps: list of camera matrices (each 3x4 numpy array)
    us: 2xN numpy array of observed image coordinates
    U: 3D point as a numpy array of shape (3,)
    
    Returns:
    - all_residuals: (2N,) numpy array of residuals (x and y reprojection errors)
    """

    # Your code here
    
    all_residuals = []
    U_hom = np.append(U, 1)
    
    for i, P in enumerate(Ps):
        # Projected 3D point
        p = P @ U_hom
        
        # Projected 2D point
        u_projected = p / p[2]
        u_projected = u_projected[:2]
        
        # Residual
        residual = us[:, i] - u_projected
        all_residuals.append(residual)
    
    all_residuals = np.concatenate(all_residuals).flatten()
    
    return all_residuals

def compute_jacobian(Ps, U):
    """
    Ps: list of camera matrices (each 3x4 numpy array)
    U: 3D point as a numpy array of shape (3,)
    
    Returns:
    - jacobian: (2N x 3) numpy array, Jacobian matrix of reprojection errors w.r.t. U
    """

    # Your code here

    N = len(Ps)
    U_hom = np.append(U, 1)
    jacobian = np.zeros((2 * N, 3))

    for i, P in enumerate(Ps):

        p = P @ U_hom
        p_x, p_y, p_z = p[0], p[1], p[2]

        # Derivatives of reprojection error
        jacobian[2 * i, :] = - (P[0,0:3] * p[2] - p[0] * P[2,0:3]) / (p[2] ** 2)
        jacobian[2 * i + 1, :] = - (P[1,0:3] * p[2] - p[1] * P[2,0:3]) / (p[2] ** 2)

    return jacobian

def refine_triangulation(Ps, us, Uhat, iterations=5):
    """
    Refines a 3D point estimate using Gauss-Newton optimization.

    Parameters:
    - Ps: list of camera matrices (3x4 numpy arrays)
    - us: 2xN numpy array of 2D image points
    - Uhat: initial estimate of the 3D point (3,)
    - iterations: number of Gauss-Newton iterations (default: 5)

    Returns:
    - U: refined 3D point (3,)
    """

    # Your code here

    U_temp = Uhat.copy()

    for i in range(iterations):
        all_residuals = compute_residuals(Ps, us, U_temp)
        jacobian = compute_jacobian(Ps, U_temp)
        gradient = jacobian.T @ all_residuals
        step = jacobian.T @ jacobian
        delta = - np.linalg.solve(step, gradient)
        U_temp = U_temp + delta

        # Print current sum of squared residuals
        sum_residuals = np.sum(all_residuals ** 2)
        # print(f"Sum of squared residuals = {sum_residuals:.6f}")

    return U_temp