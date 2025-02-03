import numpy as np
import matplotlib.pyplot as plt

# Point Mass Dynamics
class PointMass:
    def __init__(self, mass=1.0, dt=0.1):
        self.mass = mass
        self.dt = dt
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])

    def update(self, acceleration):
        self.acceleration = acceleration
        self.velocity += self.acceleration * self.dt
        self.position += self.velocity * self.dt

    def get_state(self):
        return self.position, self.velocity

# Abstract Controller Interface
class Controller:
    def compute_control(self, current_state, target_state):
        raise NotImplementedError

# PID Controller
class PIDController(Controller):
    def __init__(self, kp=1.0, ki=0.0, kd=0.1, dt=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = np.array([0.0, 0.0])
        self.previous_error = np.array([0.0, 0.0])
        self.dt = dt

    def compute_control(self, current_state, target_state):
        position, velocity = current_state
        target_position, _ = target_state

        error = target_position - position
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt

        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        return control

# MPC Controller (Simplified)
class MPCController(Controller):
    def __init__(self, horizon=5, dt=0.1):
        self.horizon = horizon
        self.dt = dt

    def compute_control(self, current_state, target_state):
        # Simplified MPC: Just move towards the target
        position, _ = current_state
        target_position, _ = target_state
        control = (target_position - position) / (self.horizon * self.dt)
        return control

# LQR Controller (Simplified)
class LQRController(Controller):
    def __init__(self, Q=np.eye(2), R=np.eye(2)):
        self.Q = Q
        self.R = R

    def compute_control(self, current_state, target_state):
        position, velocity = current_state
        target_position, _ = target_state

        # Simplified LQR: Just use a linear feedback control law
        error = target_position - position
        control = np.dot(self.Q, error) - np.dot(self.R, velocity)
        return control

# Simulation
def simulate(controller, waypoints, dt=0.1, total_time=10.0, tolerance=0.1):
    point_mass = PointMass(dt=dt)
    time_steps = int(total_time / dt)
    trajectory = []
    current_waypoint_idx = 0

    for t in range(time_steps):
        current_state = point_mass.get_state()
        target_state = (waypoints[current_waypoint_idx], np.array([0.0, 0.0]))

        acceleration = controller.compute_control(current_state, target_state)
        point_mass.update(acceleration)

        trajectory.append(point_mass.position.copy())

        # Check if the current waypoint is reached
        if np.linalg.norm(point_mass.position - waypoints[current_waypoint_idx]) < tolerance:
            current_waypoint_idx += 1
            if current_waypoint_idx >= len(waypoints):
                break

    return np.array(trajectory)

# Plotting
def plot_trajectory(trajectory, waypoints):
    plt.figure()
    plt.scatter(trajectory[:, 0], trajectory[:, 1], c=np.linspace(0,1,len(trajectory)), label="Trajectory", marker='x', s=1)
    plt.scatter(waypoints[:, 0], waypoints[:, 1], c=np.linspace(0,1, len(waypoints)), label="Waypoints")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Point Mass Trajectory")
    plt.grid(True)
    plt.show()

# Main
if __name__ == "__main__":
    waypoints = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])

    # Choose controller
    # controller = PIDController(kp=1.0, ki=0.0, kd=0.1)
    # controller = MPCController(horizon=5, dt=0.1)
    # controller = LQRController(Q=np.eye(2), R=np.eye(2))

    trajectory = simulate(controller, waypoints, dt=0.01, total_time=10.0)
    plot_trajectory(trajectory, waypoints)