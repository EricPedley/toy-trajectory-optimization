import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

# Point Mass Dynamics
class PointMass:
    def __init__(self, mass=1.0, dt=0.5):
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
    def __init__(self, waypoints, total_time=10.0, dt=0.5):
        self.waypoints = waypoints
        self.total_time = total_time
        self.dt = dt
        self.time_points = np.arange(0, total_time / dt)
        self.num_steps = len(self.time_points)

    def optimize_trajectory(self):
        # Optimization variables: [x0, y0, vx0, vy0, ax0, ay0, x1, y1, vx1, vy1, ax1, ay1, ...]
        num_vars = 6 * self.num_steps  # 6 variables per step (x, y, vx, vy, ax, ay)
        x0 = np.zeros(num_vars)  # Initial guess (all zeros)

        # Bounds for optimization variables
        bounds = [(None, None)] * num_vars  # No bounds (can be customized)

        # Constraints (waypoints and dynamics)
        constraints = []
        for i, wp in enumerate(self.waypoints):
            t_index = int(i * (self.num_steps - 1) / (len(self.waypoints) - 1))  # Spread waypoints evenly
            constraints.append({
                'type': 'eq',
                'fun': lambda x, idx=t_index, target=wp: np.array([x[6 * idx] - target[0], x[6 * idx + 1] - target[1]])
            })

        # Cost function: Minimize jerk (third derivative of position)
        def cost_function(x):
            jerk = 0.0
            for t in range(1, self.num_steps - 1):
                ax_prev, ay_prev = x[6 * (t - 1) + 4], x[6 * (t - 1) + 5]
                ax_curr, ay_curr = x[6 * t + 4], x[6 * t + 5]
                ax_next, ay_next = x[6 * (t + 1) + 4], x[6 * (t + 1) + 5]
                jerk += (ax_next - 2 * ax_curr + ax_prev)**2 + (ay_next - 2 * ay_curr + ay_prev)**2
            return jerk

        # Dynamics constraints
        def dynamics_constraints(x):
            constraints = []
            for t in range(self.num_steps - 1):
                x_curr, y_curr = x[6 * t], x[6 * t + 1]
                vx_curr, vy_curr = x[6 * t + 2], x[6 * t + 3]
                ax_curr, ay_curr = x[6 * t + 4], x[6 * t + 5]
                x_next, y_next = x[6 * (t + 1)], x[6 * (t + 1) + 1]
                vx_next, vy_next = x[6 * (t + 1) + 2], x[6 * (t + 1) + 3]
                ax_next, ay_next = x[6 * (t + 1) + 4], x[6 * (t + 1) + 5]

                # Position update
                constraints.append(x_next - (x_curr + vx_curr * self.dt))
                constraints.append(y_next - (y_curr + vy_curr * self.dt))

                # Velocity update
                constraints.append(vx_next - (vx_curr + ax_curr * self.dt))
                constraints.append(vy_next - (vy_curr + ay_curr * self.dt))

            return np.array(constraints)
        

        constraints.append({'type': 'eq', 'fun': dynamics_constraints})

        # Initial conditions constraints. Initial position is already covered by waypoints constraints, and adding it here makes the solver fail because of the redundancy
        def initial_constraints(x):
            constraints = []
            constraints.append(x[2] - 0)
            constraints.append(x[3] - 0)
        
            return np.array(constraints)
        
        constraints.append({'type': 'eq', 'fun': initial_constraints})

        # Solve the optimization problem
        result = minimize(cost_function, x0, bounds=bounds, constraints=constraints, method='SLSQP')
        if not result.success:
            raise ValueError("Optimization failed!")

        # Extract the optimized trajectory
        optimized_trajectory = result.x.reshape(-1, 6)  # Reshape to [num_steps, 6]
        return optimized_trajectory

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
def simulate(optimized_trajectory: np.ndarray, controller: Controller, dt=0.5, total_time=10.0):
    point_mass = PointMass(dt=dt)
    time_steps = int(total_time / dt)
    trajectory = []

    # Generate optimized trajectory

    for t in range(time_steps):
        current_time = t * dt
        current_state = point_mass.get_state()

        # Get setpoints from the optimized trajectory
        setpoint_state = optimized_trajectory[t, :2], optimized_trajectory[t, 2:4], optimized_trajectory[t, 4:6]

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
    plt.scatter(target_trajectory[:,0], target_trajectory[:,1], c=np.linspace(0,1,target_trajectory.shape[0]), label="Target Trajectory")
    plt.scatter(unrolled_trajectory[:,0], unrolled_trajectory[:,1], c=np.linspace(0,1,len(unrolled_trajectory)), marker="x", label="Followed Trajectory", s=80)

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

    optimized_trajectory = trajectory_optimizer.optimize_trajectory()
    followed_trajectory = simulate(optimized_trajectory, controller, dt=0.5, total_time=10.0)
    plot_trajectory(optimized_trajectory, followed_trajectory, waypoints)