import math
import typing as T

import numpy as np
from numpy import linalg
from scipy.integrate import cumtrapz  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from utils import save_dict, maybe_makedirs

class State:
    def __init__(self, x: float, y: float, V: float, th: float) -> None:
        self.x = x
        self.y = y
        self.V = V
        self.th = th

    @property
    def xd(self) -> float:
        return self.V*np.cos(self.th)

    @property
    def yd(self) -> float:
        return self.V*np.sin(self.th)


def compute_traj_coeffs(initial_state: State, final_state: State, tf: float) -> np.ndarray:
    """
    Inputs:
        initial_state (State)
        final_state (State)
        tf (float) final time
    Output:
        coeffs (np.array shape [8]), coefficients on the basis functions

    Hint: Use the np.linalg.solve function.
    """
    ########## Code starts here ##########
    
    # Define xdot and ydot
    init_xdot = initial_state.V * np.cos(initial_state.th)
    final_xdot = final_state.V * np.cos(final_state.th)
    init_ydot = initial_state.V * np.sin(initial_state.th)
    final_ydot = final_state.V * np.sin(final_state.th)

    # Store the B matrix
    B = np.array([initial_state.x,final_state.x,init_xdot,final_xdot,
                 initial_state.y,final_state.y,init_ydot,final_ydot])
    # Store the A matrix
    A = np.array([[1,0,0,0,0,0,0,0],
                 [1,tf,tf**2,tf**3,0,0,0,0],
                 [0,1,0,0,0,0,0,0],
                 [0,1,2*tf,3*tf**2,0,0,0,0],
                 [0,0,0,0,1,0,0,0],
                 [0,0,0,0,1,tf,tf**2,tf**3],
                 [0,0,0,0,0,1,0,0],
                 [0,0,0,0,0,1,2*tf,3*tf**2]])
    
    # Solve for the coeffs array
    coeffs = np.linalg.solve(A, B)
    
    ########## Code ends here ##########
    return coeffs

def compute_traj(coeffs: np.ndarray, tf: float, N: int) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
        coeffs (np.array shape [8]), coefficients on the basis functions
        tf (float) final_time
        N (int) number of points
    Output:
        t (np.array shape [N]) evenly spaced time points from 0 to tf
        traj (np.array shape [N,7]), N points along the trajectory, from t=0
            to t=tf, evenly spaced in time
    """
    t = np.linspace(0, tf, N) # generate evenly spaced points from 0 to tf
    traj = np.zeros((N, 7))
    ########## Code starts here ##########
    
    state_xy = np.zeros((6,N))
    state_th = np.zeros((1,N))
    
    for i in range(0,N):
        A_xy = np.array([[1,t[i],t[i]**2,t[i]**3,0,0,0,0],
                         [0,0,0,0,1,t[i],t[i]**2,t[i]**3],
                         [0,1,2*t[i],3*t[i]**2,0,0,0,0],
                         [0,0,0,0,0,1,2*t[i],3*t[i]**2],
                         [0,0,2,6*t[i],0,0,0,0],
                         [0,0,0,0,0,0,2,6*t[i]]])
        state_xy[:,i] = np.matmul(A_xy,coeffs)
        state_th[0,i] = np.arctan(state_xy[3,i]/state_xy[2,i])

    traj[:,0] = state_xy[0,:]
    traj[:,1] = state_xy[1,:]
    traj[:,2] = state_th
    traj[:,3] = state_xy[2,:]
    traj[:,4] = state_xy[3,:]
    traj[:,5] = state_xy[4,:]
    traj[:,6] = state_xy[5,:]
    
    ########## Code ends here ##########

    return t, traj

def compute_controls(traj: np.ndarray) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        traj (np.array shape [N,7])
    Outputs:
        V (np.array shape [N]) V at each point of traj
        om (np.array shape [N]) om at each point of traj
    """
    ########## Code starts here ##########
    
    N = traj.shape[0]
    V = np.zeros(N)
    om = np.zeros(N)
    
    for i in range(0,N):
        V[i] = np.sqrt(traj[i,3]**2 + traj[i,4]**2)
        A = np.array([[np.cos(traj[i,2]), -V[i] * np.sin(traj[i,2])],
                      [np.sin(traj[i,2]), V[i] * np.cos(traj[i,2])]])
        xy_dot = np.array([traj[i,5],traj[i,6]])
        a_om = np.matmul(np.linalg.inv(A),xy_dot)
        om[i] = a_om[1]
    
    ########## Code ends here ##########

    return V, om

def interpolate_traj(
    traj: np.ndarray,
    tau: np.ndarray,
    V_tilde: np.ndarray,
    om_tilde: np.ndarray,
    dt: float,
    s_f: State
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inputs:
        traj (np.array [N,7]) original unscaled trajectory
        tau (np.array [N]) rescaled time at orignal traj points
        V_tilde (np.array [N]) new velocities to use
        om_tilde (np.array [N]) new rotational velocities to use
        dt (float) timestep for interpolation
        s_f (State) final state

    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    """
    # Get new final time
    tf_new = tau[-1]

    # Generate new uniform time grid
    N_new = int(tf_new/dt)
    t_new = dt*np.array(range(N_new+1))

    # Interpolate for state trajectory
    traj_scaled = np.zeros((N_new+1,7))
    traj_scaled[:,0] = np.interp(t_new,tau,traj[:,0])   # x
    traj_scaled[:,1] = np.interp(t_new,tau,traj[:,1])   # y
    traj_scaled[:,2] = np.interp(t_new,tau,traj[:,2])   # th
    # Interpolate for scaled velocities
    V_scaled = np.interp(t_new, tau, V_tilde)           # V
    om_scaled = np.interp(t_new, tau, om_tilde)         # om
    # Compute xy velocities
    traj_scaled[:,3] = V_scaled*np.cos(traj_scaled[:,2])    # xd
    traj_scaled[:,4] = V_scaled*np.sin(traj_scaled[:,2])    # yd
    # Compute xy acclerations
    traj_scaled[:,5] = np.append(np.diff(traj_scaled[:,3])/dt,-s_f.V*om_scaled[-1]*np.sin(s_f.th)) # xdd
    traj_scaled[:,6] = np.append(np.diff(traj_scaled[:,4])/dt, s_f.V*om_scaled[-1]*np.cos(s_f.th)) # ydd

    return t_new, V_scaled, om_scaled, traj_scaled

if __name__ == "__main__":
    # Constants
    tf = 25.

    # time
    dt = 0.005
    N = int(tf/dt)+1
    t = dt*np.array(range(N))

    # Initial conditions
    s_0 = State(x=0, y=0, V=0.5, th=-np.pi/2)

    # Final conditions
    s_f = State(x=5, y=5, V=0.5, th=-np.pi/2)

    coeffs = compute_traj_coeffs(initial_state=s_0, final_state=s_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)
    V,om = compute_controls(traj=traj)

    maybe_makedirs('plots')

    # Plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(traj[:,0], traj[:,1], 'k-',linewidth=2)
    plt.grid(True)
    plt.plot(s_0.x, s_0.y, 'go', markerfacecolor='green', markersize=15)
    plt.plot(s_f.x, s_f.y, 'ro', markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title("Path (position)")
    plt.axis([-1, 6, -1, 6])

    ax = plt.subplot(1, 2, 2)
    plt.plot(t, V, linewidth=2)
    plt.plot(t, om, linewidth=2)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
    plt.title('Original Control Input')
    plt.tight_layout()

    plt.savefig("plots/differential_flatness.png")
    plt.show()
