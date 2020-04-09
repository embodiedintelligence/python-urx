"""
Python library to control an UR robot through its TCP/IP interface
DOC LINK
http://support.universal-robots.com/URRobot/RemoteAccess
"""

import numpy as np
import quaternion
import attr
import copy

from urx.urrobot import URRobot

__author__ = "Olivier Roulet-Dubonnet"
__copyright__ = "Copyright 2011-2016, Sintef Raufoss Manufacturing"
__license__ = "LGPLv3"


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


class Transform3D:
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


class Robot(URRobot):

    """
    Generic Python interface to an industrial robot.
    Compared to the URRobot class, this class adds the possibilty to work directly with matrices
    and includes support for setting a reference coordinate system
    """

    def __init__(self, host, use_rt=False):
        URRobot.__init__(self, host, use_rt)
        self.csys = Transform3D()

    def _get_lin_dist(self, target):
        pose = URRobot.getl(self, wait=True)
        target = Transform3D(target)
        pose = Transform3D(pose)
        return pose.dist(target)

    def set_tcp(self, tcp):
        """
        set robot flange to tool tip transformation
        """
        if isinstance(tcp, Transform3D):
            tcp = tcp.pose_vector
        URRobot.set_tcp(self, tcp)

    def set_csys(self, transform):
        """
        Set reference coordinate system to use
        """
        if isinstance(tcp, Transform3D):
            self.csys = transform
        else:
            self.csys = Transform3D(transform)

    def set_orientation(self, orient, acc=0.01, vel=0.01, wait=True, threshold=None):
        """
        set tool orientation using a (3, 3) rotation matrix
        """
        assert orient.shape == (3, 3)
        trans = self.get_pose()
        trans[:3, :3] = orient
        self.set_pose(trans, acc, vel, wait=wait, threshold=threshold)

    def translate_tool(self, vect, acc=0.01, vel=0.01, wait=True, threshold=None):
        """
        move tool in tool coordinate, keeping orientation
        """
        t = Transform3D()
        assert len(vect) == 3
        t.pose[:3, 3] += vect
        return self.add_pose_tool(t, acc, vel, wait=wait, threshold=threshold)

    def back(self, z=0.05, acc=0.01, vel=0.01):
        """
        move in z tool
        """
        self.translate_tool((0, 0, -z), acc=acc, vel=vel)

    def set_pos(self, vect, acc=0.01, vel=0.01, wait=True, threshold=None):
        """
        set tool to given pos, keeping constant orientation
        """
        assert len(vect) == 3
        trans = copy.deepcopy(self)
        trans.pose[:3, 3] = vect
        return self.set_pose(trans, acc, vel, wait=wait, threshold=threshold)

    def movec(self, pose_via, pose_to, acc=0.01, vel=0.01, wait=True, threshold=None):
        """
        Move Circular: Move to position (circular in tool-space)
        see UR documentation
        """
        pose_via = self.csys * Transform3D(pose_via)
        pose_to = self.csys * Transform3D(pose_to)
        pose = URRobot.movec(self, pose_via.pose_vector, pose_to.pose_vector, acc=acc, vel=vel, wait=wait, threshold=threshold)
        if pose is not None:
            return (self.csys.inverse * Transform3D(pose)).pose

    def set_pose(self, trans: Transform3D, acc=0.01, vel=0.01, wait=True, command="movel", threshold=None):
        """
        move tcp to point and orientation defined by a transformation
        UR robots have several move commands, by default movel is used but it can be changed
        using the command argument
        """
        self.logger.debug("Setting pose to %s", trans.pose_vector)
        t = self.csys * trans
        pose = URRobot.movex(self, command, t.pose_vector, acc=acc, vel=vel, wait=wait, threshold=threshold)
        if pose is not None:
            return (self.csys.inverse * Transform3D(pose)).pose

    def add_pose_base(self, trans, acc=0.01, vel=0.01, wait=True, command="movel", threshold=None):
        """
        Add transform expressed in base coordinate
        """
        if not isinstance(trans, Transform3D):
            trans = Transform3D(trans)

        pose = Transform3D(self.get_pose())
        return self.set_pose(trans * pose, acc, vel, wait=wait, command=command, threshold=threshold)

    def add_pose_tool(self, trans, acc=0.01, vel=0.01, wait=True, command="movel", threshold=None):
        """
        Add transform expressed in tool coordinate
        """
        if not isinstance(trans, Transform3D):
            trans = Transform3D(trans)
        pose = Transform3D(self.get_pose())
        return self.set_pose(pose * trans, acc, vel, wait=wait, command=command, threshold=threshold)

    def get_pose(self, wait=False, _log=True):
        """
        get current transform from base to to tcp. Return (4, 4) homogenerous transform
        """
        pose = URRobot.getl(self, wait, _log)
        trans = self.csys.inverse * Transform3D(pose)
        if _log:
            self.logger.debug("Returning pose to user: %s", trans.pose_vector)
        return trans.pose

    def get_orientation(self, wait=False):
        """
        get tool orientation in base coordinate system
        """
        trans = self.get_pose(wait)
        return trans[:3, :3]

    def get_pos(self, wait=False):
        """
        get tool tip pos(x, y, z) in base coordinate system
        """
        trans = self.get_pose(wait)
        return trans[:3, 3]

    def speedl(self, velocities, acc, min_time):
        """
        move at given velocities until minimum min_time seconds
        """
        v = self.csys.ori.dot(velocities[:3])
        w = self.csys.ori.dot(velocities[3:])
        vels = np.concatenate((v.array, w.array))
        return self.speedx("speedl", vels, acc, min_time)

    def speedj(self, velocities, acc, min_time):
        """
        move at given joint velocities until minimum min_time seconds
        """
        return self.speedx("speedj", velocities, acc, min_time)

    def speedl_tool(self, velocities, acc, min_time):
        """
        move at given velocities in tool csys until minimum min_time seconds
        """
        pose = self.get_pose()
        v = pose.ori.dot(velocities[:3])
        w = pose.ori.dot(velocities[3:])
        self.speedl(np.concatenate((v.array, w.array)), acc, min_time)

    def movex(self, command, pose, acc=0.01, vel=0.01, wait=True, relative=False, threshold=None):
        """
        Send a move command to the robot. since UR robotene have several methods this one
        sends whatever is defined in 'command' string
        """
        t = Transform3D(pose)
        if relative:
            return self.add_pose_base(t, acc, vel, wait=wait, command=command, threshold=threshold)
        else:
            return self.set_pose(t, acc, vel, wait=wait, command=command, threshold=threshold)

    def movexs(self, command, pose_list, acc=0.01, vel=0.01, radius=0.01, wait=True, threshold=None):
        """
        Concatenate several movex commands and applies a blending radius
        pose_list is a list of pose.
        This method is usefull since any new command from python
        to robot make the robot stop
        """
        new_poses = []
        for pose in pose_list:
            t = self.csys * Transform3D(pose)
            pose = t.pose_vector
            new_poses.append(pose)
        return URRobot.movexs(self, command, new_poses, acc, vel, radius, wait=wait, threshold=threshold)

    def movel_tool(self, pose, acc=0.01, vel=0.01, wait=True, threshold=None):
        """
        move linear to given pose in tool coordinate
        """
        return self.movex_tool("movel", pose, acc=acc, vel=vel, wait=wait, threshold=threshold)

    def movex_tool(self, command, pose, acc=0.01, vel=0.01, wait=True, threshold=None):
        t = Transform3D(pose)
        self.add_pose_tool(t, acc, vel, wait=wait, command=command, threshold=threshold)

    def getl(self, wait=False, _log=True):
        """
        return current transformation from tcp to current csys
        """
        t = Transform3D(self.get_pose(wait, _log))
        return t.pose_vector

    def set_gravity(self, vector):
        assert len(vector) == 3
        return URRobot.set_gravity(self, vector)
