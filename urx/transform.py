import quaternion
import numpy as np

# Taken from covariant/robots/common/kinematics/utils.py
def get_pose_6d(pose_mat: np.ndarray) -> np.ndarray:
    """ Compute the 6D pose vector(s) (3D position + 3D rotation vector) from the (set of) 4x4 pose matrix(es).

    Parameters
    ----------
    pose_mat : np.ndarray, shape=(..., 4, 4)
        A (stack of) 4x4 pose matrix(es) for which to compute the 6D pose vector(s).

    Returns
    -------
    np.ndarray, shape=(..., 6)
        The pose vector(s) corresponding to the given pose matrix(es), where the first 3 elements of each pose vector
        is the 3D position and the last 3 elements of each pose vector is the 3D rotation vector.
    """
    pose_6d = np.zeros((*pose_mat.shape[:-2], 6))
    pose_6d[..., :3] = pose_mat[..., :3, -1]
    pose_6d[..., 3:] = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(pose_mat[..., :3, :3]))
    return pose_6d


# Taken from covariant/robots/common/kinematics/utils.py
def get_pose_mat(pose_6d_or_7d: np.ndarray) -> np.ndarray:
    """ Computes 4x4 pose matrices from either 6D pose vectors (interpreted as 3D position followed by 3D rotation
    vector) or 7D pose vectors (interpreted as 3D position followed by 4D quaternion.

    Parameters
    ----------
    pose_6d_or_7d : np.ndarray, shape=(..., Union[6, 7])
        A (stack of) pose vector(s) for which to compute 4x4 pose matrices.

    Returns
    -------
    np.ndarray, shape=(..., 4, 4)
        The (stack of) pose matrix(es) corresponding to the given pose vector(s).

    Notes
    -----
    This method supports vectorization.
    """
    if pose_6d_or_7d.ndim >= 2 and pose_6d_or_7d.shape[-2:] == (4, 4):
        return pose_6d_or_7d
    pose_6d_or_7d = np.asarray(pose_6d_or_7d)
    extra_dims = tuple(pose_6d_or_7d.shape[:-1])
    N = int(np.prod(extra_dims))
    mat = np.tile(np.eye(4).reshape((1, 4, 4)), (N, 1, 1))
    pose_6d_or_7d = pose_6d_or_7d.reshape((N, -1))
    if pose_6d_or_7d.shape[-1] == 6:
        mat[:, :3, :3] = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(np.ascontiguousarray(pose_6d_or_7d[:, 3:]))
        )
        mat[:, :3, -1] = pose_6d_or_7d[:, :3]
    elif pose_6d_or_7d.shape[-1] == 7:
        mat[:, :3, :3] = quaternion.as_rotation_matrix(
            quaternion.as_quat_array(np.ascontiguousarray(pose_6d_or_7d[:, 3:]))
        )
        mat[:, :3, -1] = pose_6d_or_7d[:, :3]
    else:
        raise NotImplementedError
    return mat.reshape(extra_dims + (4, 4))


class Transform:
    """Helper transform wrapper"""
    def __init__(self, pose_or_pose_vector: np.array = np.eye(4)):
        """Init a 3D transform"""
        if pose_or_pose_vector.shape[-1] == 6:
            self.pose = get_pose_mat(pose_or_pose_vector)
        elif pose_or_pose_vector.shape == (4, 4):
            self.pose = pose_or_pose_vector

    @property
    def pose_vector(self) -> np.array:
        """pose as a position and rotation vector"""

        return get_pose_6d(self.pose)

    @property
    def pos(self) -> np.array:
        """position"""
        T = self.pose[:3, 3]
        return T

    @property
    def ori(self) -> np.array:
        """orintation as a (3, 3) rotation matrix"""

        R = self.pose[:3, :3]
        return R

    @property
    def inverse(self) -> "Transform3D":
        """inverse homogenous transform"""

        H = np.eye(4)
        T = self.pos
        R = self.ori
        H[:3, :3] = R.T
        H[:3, 3] = -R.T.dot(T)
        return Transform3D(H)

    def dist(self, trans: "Transform3D") -> float:
        """distance to the other transform"""
        mat_rotation_difference = self.ori.T.dot(trans.ori)
        rotation_vector = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(mat_rotation_difference))
        angle = np.linalg.norm(rotation_vector)
        return np.sum((self.pos - trans.pos) ** 2) + angle ** 2

    def __mul__(self, other):
        if type(other) == Transform3D:
            return Transform3D(self.pose.dot(other.pose))
        else:
            raise NotImplementedError
