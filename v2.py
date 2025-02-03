import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Point Mass Dynamics
class PointMass:
    def __init__(self, mass=1.0, dt=0.1):
        self.mass = mass
        self.dt = dt
        self.position = np.array([0.0, 0.0])  # [x, y]
        self.velocity = np.array([0.0, 0.0])  # [vx, vy]
        self.acceleration = np.array([0.0, 0.0])  # [ax, ay]

    def update(self, acceleration):
        self.acceleration = acceleration
        self.velocity += self.acceleration * self.dt
        self.position += self.velocity * self.dt

    def get_state(self):
        return np.concatenate((self.position, self.velocity, self.acceleration))

# Trajectory Optimizer (Generates smooth setpoints from waypoints)
class TrajectoryOptimizer:
    def __init__(self, waypoints, total_time=10.0):
        self.waypoints = waypoints
        self.total_time = total_time
        self.time_points = np.linspace(0, total_time, len(waypoints))

        # Fit cubic splines for x and y trajectories
        self.x_spline = CubicSpline(self.time_points, waypoints[:, 0])
        self.y_spline = CubicSpline(self.time_points, waypoints[:, 1])

    def generate_setpoints(self, t):
        # Position
        position = np.array([self.x_spline(t), self.y_spline(t)])

        # Velocity (first derivative)
        velocity = np.array([self.x_spline(t, 1), self.y_spline(t, 1)])

        # Acceleration (second derivative)
        acceleration = np.array([self.x_spline(t, 2), self.y_spline(t, 2)])

        return np.array([position, velocity, acceleration])

# Abstract Controller Interface
class Controller:
    def compute_control(self, current_state, setpoint_state):
        raise NotImplementedError

# Open-Loop Controller (Follows Acceleration Setpoints Directly)
class OpenLoopController(Controller):
    def compute_control(self, current_state, setpoint_state):
        _, _, acceleration_setpoint = setpoint_state
        return acceleration_setpoint

# PID Controller (Feedback Control)
class PIDController(Controller):
    def __init__(self, kp=1.0, ki=0.0, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = np.array([0.0, 0.0])
        self.previous_error = np.array([0.0, 0.0])

    def compute_control(self, current_state, setpoint_state):
        position, velocity, _ = current_state[:2], current_state[2:4], current_state[4:6]
        position_setpoint, velocity_setpoint, _ = setpoint_state

        error = position_setpoint - position
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt

        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        return control

# Simulation
def simulate(trajectory_optimizer: TrajectoryOptimizer, controller: Controller, dt=0.1, total_time=10.0):
    point_mass = PointMass(dt=dt)
    time_steps = int(total_time / dt)
    trajectory = []

    for t in range(time_steps):
        current_time = t * dt
        current_state = point_mass.get_state()

        # Generate setpoints from the trajectory optimizer
        setpoint_state = trajectory_optimizer.generate_setpoints(current_time)

        # Compute control input
        acceleration = controller.compute_control(current_state, setpoint_state)

        # Update point mass dynamics
        point_mass.update(acceleration)

        # Save trajectory
        trajectory.append(point_mass.position.copy())

    return np.array(trajectory)

# Plotting with viridis colormap
def plot_trajectory(target_trajectory, unrolled_trajectory, waypoints):
    plt.figure(figsize=(8, 6))

    # Plot trajectory with viridis colormap
    plt.scatter(target_trajectory[0,0,:], target_trajectory[0,1,:], c=np.linspace(0,1,target_trajectory.shape[2]), label="Target Trajectory")
    plt.scatter(unrolled_trajectory[:,0], unrolled_trajectory[:,1], c=np.linspace(0,1,len(unrolled_trajectory)), marker="x", label="Followed Trajectory")

    # Plot waypoints
    plt.scatter(waypoints[:, 0], waypoints[:, 1], color="red", s=100, label="Waypoints")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Point Mass Trajectory with Waypoints")
    plt.legend()
    plt.grid(True)
    # plt.colorbar(plt.cm.ScalarMappable(cmap=viridis), label="Time")
    plt.show()

# Main
if __name__ == "__main__":
    # Define waypoints
    waypoints = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 3.0],
        [5.0, 1.0],
        [6.0, 0.0]
    ])

    # Create trajectory optimizer
    trajectory_optimizer = TrajectoryOptimizer(waypoints, total_time=10.0)

    # Choose controller
    controller = OpenLoopController()  # Open-loop control
    # controller = PIDController(kp=1.0, ki=0.0, kd=0.1)  # Feedback control

    # Run simulation
    followed_trajectory = simulate(trajectory_optimizer, controller, dt=0.1, total_time=10.0)
    generated_trajectory = trajectory_optimizer.generate_setpoints(np.linspace(0, 10, len(followed_trajectory)))
    plot_trajectory(generated_trajectory, followed_trajectory, waypoints)