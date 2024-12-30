import cv2
import numpy as np

waypoints = np.array([
    [1, 0],
    [6, 5],
    [11, 0],
    [16, 5],
])


class Controller:
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.current_waypoint = 0
        self.input_x, self.input_y = waypoints[0]
    
    def update(self, current_pos):
        if np.linalg.norm(current_pos - self.waypoints[self.current_waypoint]) < 1:
            self.current_waypoint += 1
            if self.current_waypoint >= len(self.waypoints):
                self.current_waypoint = 0
        temp_x, temp_y = self.waypoints[self.current_waypoint] - current_pos
        mag = np.linalg.norm([temp_x, temp_y])
        mag_capped = max(1, mag)
        self.input_x = temp_x / mag * mag_capped
        self.input_y = temp_y / mag * mag_capped

    
    def get_input(self):
        return self.input_x, self.input_y

controller = Controller(waypoints)
current_pos = np.array([0,0], dtype=np.float32)
current_vel = np.array([0,0], dtype=np.float32)

dt = 1/30
MAX_VEL = 5

while True:
    # Create a blank image
    pixel_size = 1000
    actual_bounds_xyw = [-1, -10, 21]
    img = np.zeros((pixel_size, pixel_size, 3), np.uint8)

    def actual_to_pixel(x, y):
        x = (x - actual_bounds_xyw[0]) / actual_bounds_xyw[2] * pixel_size
        y = (y - actual_bounds_xyw[1]) / actual_bounds_xyw[2] * pixel_size
        return int(x), int(y)
    
    # Draw waypoints
    for i in range(len(waypoints)):
        x, y = waypoints[i]
        x, y = actual_to_pixel(x, y)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw current position
    controller.update(current_pos)

    input_x, input_y = controller.get_input()

    current_vel += np.array([input_x, input_y]) * dt
    vel_mag = np.linalg.norm(current_vel)
    if vel_mag > MAX_VEL:
        current_vel = current_vel / vel_mag * MAX_VEL
    current_pos += current_vel * dt

    pixel_x, pixel_y = actual_to_pixel(current_pos[0], current_pos[1])
    cv2.circle(img, (pixel_x, pixel_y), 5, (0, 0, 255), -1)

    cv2.imshow('image', img)
    cv2.waitKey(1)


