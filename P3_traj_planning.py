import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
from scipy.interpolate import *
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        ########## Code starts here ##########
        # if we're near the end, switch to the pose_controller
        if t > (self.traj_controller.traj_times[-1] - self.t_before_switch):
            V, om = self.pose_controller.compute_control(x, y, th, t)
        # otherwise keep using the trajectory controller
        else:
            V, om = self.traj_controller.compute_control(x, y, th, t)

        return V, om
        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    # make path a numpy array (for some reason it is coming in as a list of tuples)
    path = np.asarray(path)

    # create a time vector for our current path based on Vdes
    t_old = np.zeros(np.shape(path)[0])
    for i in range(1,np.shape(path)[0]):
        t_old[i] = np.linalg.norm(path[i] - path[i-1]) / V_des + t_old[i-1]

    # create our smoothed time vector
    t_smoothed = np.arange(0, t_old[-1], dt)

    # generate a spline for our path in x and y
    xspline = splrep(t_old, path[:,0], s=alpha)
    yspline = splrep(t_old, path[:,1], s=alpha)

    # compute the new x and y values of the path using t_smoothed
    x_smoothed = splev(t_smoothed, xspline, der=0)
    y_smoothed = splev(t_smoothed, yspline, der=0)
    theta = np.arctan2(y_smoothed, x_smoothed)
    xd_smoothed = splev(t_smoothed, xspline, der=1)
    yd_smoothed = splev(t_smoothed, yspline, der=1)
    xdd_smoothed = splev(t_smoothed, xspline, der=2)
    ydd_smoothed = splev(t_smoothed, yspline, der=2)

    ########## Code ends here ##########
    traj_smoothed = np.stack((x_smoothed, y_smoothed, theta, xd_smoothed, yd_smoothed, xdd_smoothed, ydd_smoothed), axis=1)
    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajectory
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    """
    ########## Code starts here ##########
    # compute the controls for our desired trajectory
    V, om = compute_controls(traj=traj)

    # rescale as needed
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)

    # get our final state
    s_f = State(x=traj[-1,0], y=traj[-1,1], V=V_tilde[-1], th=traj[-1,2])

    # interpolate the values to get our scaled trajectory
    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)
    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
