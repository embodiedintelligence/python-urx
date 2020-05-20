"""
Python library to control an UR robot through its TCP/IP interface
DOC LINK
http://support.universal-robots.com/URRobot/RemoteAccess
"""

import numpy as np

from urx.urrobot import URRobot
from urx.transform import Transform

__author__ = "Olivier Roulet-Dubonnet"
__copyright__ = "Copyright 2011-2016, Sintef Raufoss Manufacturing"
__license__ = "LGPLv3"


class Robot(URRobot):

    """
    Generic Python interface to an industrial robot.
    Compared to the URRobot class, this class adds the possibilty to work directly with matrices
    and includes support for setting a reference coordinate system
    """

    def __init__(self, host, use_rt=False):
        URRobot.__init__(self, host, use_rt)
        self.csys = Transform()

    def _get_lin_dist(self, target):
        pose = URRobot.getl(self, wait=True)
        target = Transform(target)
        pose = Transform(pose)
        return pose.dist(target)

    def set_tcp(self, tcp):
        """
        Set robot flange to tool tip transformation.
        """
        if not isinstance(tcp, Transform):
            tcp = Transform(tcp)
        tcp = tcp.pose_vector
        URRobot.set_tcp(self, tcp)

    def set_csys(self, transform):
        """
        Set reference coordinate system to use.
        """
        if isinstance(transform, Transform):
            self.csys = transform
        else:
            self.csys = Transform(transform)

    def set_orientation(self, orient, acc=0.01, vel=0.01, wait=True, threshold=None):
        """
        Set tool orientation using a (3, 3) rotation matrix.
        """
        assert orient.shape == (3, 3)
        trans = Transform(self.get_pose())
        trans.pose[:3, :3] = orient
        self.set_pose(trans, acc, vel, wait=wait, threshold=threshold)

    def translate_tool(self, vect, acc=0.01, vel=0.01, wait=True, threshold=None):
        """
        Move tool in tool coordinate, keeping orientation
        """
        t = Transform()
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
        trans = Transform(self.get_pose())
        trans.pose[:3, 3] = vect
        return self.set_pose(trans, acc, vel, wait=wait, threshold=threshold)

    def movec(self, pose_via, pose_to, acc=0.01, vel=0.01, wait=True, threshold=None):
        """
        Move Circular: Move to position (circular in tool-space)
        see UR documentation
        """
        pose_via = self.csys * Transform(pose_via)
        pose_to = self.csys * Transform(pose_to)
        pose = URRobot.movec(self, pose_via.pose_vector, pose_to.pose_vector, acc=acc, vel=vel, wait=wait, threshold=threshold)
        if pose is not None:
            return (self.csys.inverse * Transform(pose)).pose

    def set_pose(self, trans, acc=0.01, vel=0.01, wait=True, command="movel", threshold=None):
        """
        move tcp to point and orientation defined by a transformation
        UR robots have several move commands, by default movel is used but it can be changed
        using the command argument
        """
        if not isinstance(trans, Transform):
            trans = Transform(trans)

        self.logger.debug("Setting pose to %s", trans.pose_vector)
        t = self.csys * trans
        pose = URRobot.movex(self, command, t.pose_vector, acc=acc, vel=vel, wait=wait, threshold=threshold)
        if pose is not None:
            return (self.csys.inverse * Transform(pose)).pose

    def add_pose_base(self, trans, acc=0.01, vel=0.01, wait=True, command="movel", threshold=None):
        """
        Add transform expressed in base coordinate
        """
        if not isinstance(trans, Transform):
            trans = Transform(trans)

        pose = Transform(self.get_pose())
        return self.set_pose(trans * pose, acc, vel, wait=wait, command=command, threshold=threshold)

    def add_pose_tool(self, trans, acc=0.01, vel=0.01, wait=True, command="movel", threshold=None):
        """
        Add transform expressed in tool coordinate
        """
        if not isinstance(trans, Transform):
            trans = Transform(trans)
        pose = Transform(self.get_pose())
        return self.set_pose(pose * trans, acc, vel, wait=wait, command=command, threshold=threshold)

    def get_pose(self, wait=False, _log=True):
        """
        Get current transform from base to to tcp. Return (4, 4) homogenerous transform.
        """
        pose = URRobot.getl(self, wait, _log)
        trans = self.csys.inverse * Transform(pose)
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
        vels = np.concatenate((v, w))
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
        self.speedl(np.concatenate((v, w)), acc, min_time)

    def movex(self, command, pose, acc=0.01, vel=0.01, wait=True, relative=False, threshold=None):
        """
        Send a move command to the robot. since UR robotene have several methods this one
        sends whatever is defined in 'command' string
        """
        t = Transform(pose)
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
            t = self.csys * Transform(pose)
            pose = t.pose_vector
            new_poses.append(pose)
        return URRobot.movexs(self, command, new_poses, acc, vel, radius, wait=wait, threshold=threshold)

    def movel_tool(self, pose, acc=0.01, vel=0.01, wait=True, threshold=None):
        """
        move linear to given pose in tool coordinate
        """
        return self.movex_tool("movel", pose, acc=acc, vel=vel, wait=wait, threshold=threshold)

    def movex_tool(self, command, pose, acc=0.01, vel=0.01, wait=True, threshold=None):
        t = Transform(pose)
        self.add_pose_tool(t, acc, vel, wait=wait, command=command, threshold=threshold)

    def getl(self, wait=False, _log=True):
        """
        return current transformation from tcp to current csys
        """
        t = Transform(self.get_pose(wait, _log))
        return t.pose_vector.tolist()

    def set_gravity(self, vector):
        assert len(vector) == 3
        return URRobot.set_gravity(self, vector)
