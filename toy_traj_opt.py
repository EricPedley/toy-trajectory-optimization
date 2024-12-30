import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline

waypoints = np.array([
    [1, 0],
    [6, 5],
    [11, 0],
    [16, 5],
])

curr_time = 0
DELTA_T = 1/30
MAX_VEL = 5
MAX_ACC = 5
WP_TOLERANCE = 0.2

def advance_state(current_state, input, dt = DELTA_T):
    pos = current_state[:2]
    vel = current_state[2:]
    acc = np.array(input)
    vel_mag = np.linalg.norm(vel)
    if vel_mag > MAX_VEL:
        vel = vel / vel_mag * MAX_VEL
    return np.concatenate([
        pos + vel * dt + 0.5 * acc * dt**2,
        vel + acc * dt,
    ])

def generate_spline_trajectory(waypoints):
    # just return more waypoints that make a spline
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    t = np.linspace(0, 1, len(x))
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    t2 = np.linspace(0, 1, 4*len(x))
    return np.array([cs_x(t2), cs_y(t2)]).T

class Controller:
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        self.input_x, self.input_y = waypoints[0]
        self.last_sln = None
    
    def update(self, current_state: np.ndarray):
        current_pos = current_state[:2]
        if np.linalg.norm(current_pos - self.waypoints[self.current_waypoint_idx]) < WP_TOLERANCE:
            self.current_waypoint_idx += 1
            self.last_sln = None
            if self.current_waypoint_idx >= len(self.waypoints):
                self.current_waypoint_idx = 0
        INPUT_SCHEDULE_LENGTH = 10
        def residuals(input_schedule):
            simulated_state = current_state.copy()
            simulated_waypoint_idx = self.current_waypoint_idx
            for i in range(INPUT_SCHEDULE_LENGTH):
                acc = input_schedule[i*2:i*2+2]
                simulated_state = advance_state(simulated_state, acc, 2*DELTA_T)
                if np.linalg.norm(simulated_state[:2] - self.waypoints[simulated_waypoint_idx]) < WP_TOLERANCE:
                    simulated_waypoint_idx = min(simulated_waypoint_idx + 1, len(self.waypoints) - 1)
            return simulated_state[:2] - self.waypoints[simulated_waypoint_idx]

        initial_guess = np.zeros(INPUT_SCHEDULE_LENGTH*2) if self.last_sln is None else self.last_sln 
        best_input_schedule = least_squares(
            residuals, 
            initial_guess,
            bounds = (
                [-MAX_ACC] * INPUT_SCHEDULE_LENGTH * 2,
                [MAX_ACC] * INPUT_SCHEDULE_LENGTH * 2,
            )
        ).x

        self.input_x, self.input_y = best_input_schedule[:2]

    
    def get_input(self):
        return self.input_x, self.input_y

current_state = np.zeros(4, dtype=np.float32)
pixel_size = 1000
actual_bounds_xyw = [-1, -10, 21]
def actual_to_pixel(x, y):
    x = (x - actual_bounds_xyw[0]) / actual_bounds_xyw[2] * pixel_size
    y = (y - actual_bounds_xyw[1]) / actual_bounds_xyw[2] * pixel_size
    return int(x), int(y)

# Draw spline
spline_points = generate_spline_trajectory(waypoints)
preview_img = np.zeros((pixel_size, pixel_size, 3), np.uint8)
for i in range(len(spline_points)):
    x, y = spline_points[i]
    x, y = actual_to_pixel(x, y)
    cv2.circle(preview_img, (x, y), 5, (255, 255, 0), -1)

cv2.imshow('spline preview', preview_img)
cv2.waitKey(0)

controller = Controller(spline_points)

while True:
    # Create a blank image
    img = np.zeros((pixel_size, pixel_size, 3), np.uint8)

    
    # Draw waypoints
    for i in range(len(waypoints)):
        x, y = waypoints[i]
        x, y = actual_to_pixel(x, y)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw current position
    controller.update(current_state)

    input_x, input_y = controller.get_input()

    acc = np.array([input_x, input_y])
    current_state = advance_state(current_state, acc)

    pixel_x, pixel_y = actual_to_pixel(current_state[0], current_state[1])
    cv2.circle(img, (pixel_x, pixel_y), 5, (0, 0, 255), -1)

    curr_time += DELTA_T
    cv2.putText(img, f"Time: {curr_time:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"Vel: {np.linalg.norm(current_state[2:]):.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('image', img)
    cv2.waitKey(1)


