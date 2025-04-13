#!/usr/bin/env python3
"""
hybrid_a_star.py

A concise Hybrid A* planner for Ackermann vehicles.
Construct an instance with the desired grid, vehicle, and cost parameters.
Then call set_occupancy_grid(occ_msg) to feed in obstacles from a ROS2 OccupancyGrid message.
Plan a path by calling plan_path(start, goal), where start and goal are [x, y, yaw]
(with yaw in radians).

This version has been updated with a matplotlib visualization helper.
"""

import math
import numpy as np
from heapdict import heapdict
from scipy.spatial import KDTree

# For matplotlib visualization (only used when running the script interactively)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# Helper: wrap an angle to [-pi, pi]
def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


# Simple Node used in planning.
class Node:
    def __init__(self, grid_idx, traj, steer, direction, cost, parent_idx):
        self.grid_idx = grid_idx      # [ix, iy, iyaw]
        self.traj = traj              # list of [x, y, yaw]
        self.steer = steer            # steering angle used (None for RS shortcut)
        self.direction = direction    # +1 for forward, -1 for reverse (None for RS shortcut)
        self.cost = cost              # cumulative cost
        self.parent_idx = parent_idx  # parent node’s index (tuple)


def _node_index(node):
    return (node.grid_idx[0], node.grid_idx[1], node.grid_idx[2])


class HybridAStarPlanner:
    def __init__(self, *, 
                 xy_resolution, 
                 yaw_resolution, 
                 occ_threshold=50,
                 car_max_steer=0.2, 
                 steer_precision=2,
                 wheel_base=0.325,
                 axle_to_front=0.1,
                 axle_to_back=0.1,
                 car_width=0.3,
                 cost_reverse=100,
                 cost_direction_change=150,
                 cost_steer_angle=0,
                 cost_steer_angle_change=50,
                 cost_hybrid=50):
        """
        Instantiate the planner.
          xy_resolution: grid cell size in meters.
          yaw_resolution: discretization for yaw in radians.
          occ_threshold: occupancy value threshold.
          (The remaining parameters configure the vehicle and cost settings.)
        """
        self.xy_resolution = xy_resolution
        self.yaw_resolution = yaw_resolution
        self.occ_threshold = occ_threshold

        # Vehicle parameters.
        self.car_max_steer = car_max_steer
        self.steer_precision = steer_precision
        self.wheel_base = wheel_base
        self.axle_to_front = axle_to_front
        self.axle_to_back  = axle_to_back
        self.car_width     = car_width

        # Cost parameters.
        self.cost_reverse = cost_reverse
        self.cost_direction_change = cost_direction_change
        self.cost_steer_angle = cost_steer_angle
        self.cost_steer_angle_change = cost_steer_angle_change
        self.cost_hybrid = cost_hybrid

        # These will be set from the occupancy grid.
        self.obstacle_x = None
        self.obstacle_y = None
        self.obstacle_tree = None
        # Instead of basing boundaries on extracted obstacles (which may be dummy),
        # we now set them based on the map’s bounds.
        self.map_min_x = None
        self.map_min_y = None
        self.map_max_x = None
        self.map_max_y = None

        # Kinematic simulation step – smaller step produces a denser check.
        self.kinematic_step = 0.5

        # Precompute motion commands (list of [steer, direction]).
        self.motion_set = self._motion_commands()

    def _motion_commands(self):
        cmds = []
        # We use a symmetric range of steering commands.
        for ang in np.arange(self.car_max_steer,
                             -self.car_max_steer - self.car_max_steer/self.steer_precision,
                             -self.car_max_steer/self.steer_precision):
            cmds.append([ang, 1])
            #cmds.append([ang, -1])
        return cmds

    def set_occupancy_grid(self, occ_msg):
        """
        Processes a ROS2 OccupancyGrid message.
        Cells with occupancy value >= self.occ_threshold are considered occupied.
        Changes made:
         • Only interpret cells as obstacles if their value is >= occ_threshold.
         • Set map boundaries from the occupancy grid info instead of using obstacles.
        """
        ox, oy = [], []
        info = occ_msg.info
        res = info.resolution
        width = info.width
        height = info.height
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        data = occ_msg.data

        # OccupancyGrid data is row-major.
        for j in range(height):
            for i in range(width):
                idx = j * width + i
                # Do not treat unknown (typically -1) as occupied.
                if data[idx] >= self.occ_threshold:
                    # Use cell center as obstacle coordinate.
                    x = origin_x - (i) * res
                    y = origin_y - (j) * res
                    ox.append(x)
                    oy.append(y)
        if len(ox) == 0:
            # Optionally, you can warn that no obstacles were found.
            # For ROS2 logging, you might use get_logger() if inside a node;
            # here we simply print.
            print("No obstacles found in occupancy grid. Using map boundaries only.")
        else:
            print("Found %d obstacles in occupancy grid." % len(ox))
        self.obstacle_x = ox
        self.obstacle_y = oy
        if ox and oy:
            self.obstacle_tree = KDTree(list(zip(ox, oy)))
        else:
            # If no obstacles are found, create an empty tree that will report no collisions.
            self.obstacle_tree = KDTree([(1e10, 1e10)])
        # Set map boundaries from occupancy grid info.
        self.map_min_x = round(origin_x - width * res)
        self.map_min_y = round(origin_y - height * res)
        self.map_max_x = round((origin_x))
        self.map_max_y = round((origin_y))

    def _collision(self, traj):
        """
        Checks if any point along traj (list of [x,y,yaw]) collides with an obstacle.
        A circular safety envelope is used.
        """
        car_radius = ((self.axle_to_front + self.axle_to_back + self.wheel_base))
        dl = (self.axle_to_front - self.axle_to_back) / 2.0
        for pose in traj:
            cx = pose[0]
            cy = pose[1]
            pts = self.obstacle_tree.query_ball_point([cx, cy], car_radius)
            if not pts:
                continue
            return True
        return False

    def _is_valid(self, traj, grid_idx):
        x, y, _ = traj[-1]
        if (x < self.map_min_x or x > self.map_max_x or
             y < self.map_min_y or y > self.map_max_y):
             return False
        return not self._collision(traj)

    def _simulated_path_cost(self, current_node, motion_cmd, sim_length, goal_node):
        cost = current_node.cost
        if motion_cmd[1] == 1:
            cost += sim_length
        else:
            cost += sim_length * self.cost_reverse
        if current_node.direction != motion_cmd[1]:
            cost += self.cost_direction_change
        cost += abs(motion_cmd[0]) * self.cost_steer_angle
        cost += abs(motion_cmd[0] - current_node.steer) * self.cost_steer_angle_change
        cost += self.cost_hybrid
        cost += math.sqrt((current_node.grid_idx[0] - goal_node.grid_idx[0]) ** 2 + (current_node.grid_idx[1] - goal_node.grid_idx[1]) ** 2)
        return cost

    def _kinematic_simulation_node(self, current_node, motion_cmd, goal_node, sim_length=0.0, step=None):
        """
        Simulates the vehicle’s kinematics for a given motion command.
        A finer simulation step is used for more accurate collision checking.
        """
        if step is None:
            step = self.kinematic_step
        traj = []
        x0, y0, yaw0 = current_node.traj[-1]
        beta = math.tan(motion_cmd[0])
        # First step:
        angle = pi_2_pi(yaw0 + motion_cmd[1] * step / self.wheel_base * beta)
        x = x0 + motion_cmd[1] * step * math.cos(angle)
        y = y0 + motion_cmd[1] * step * math.sin(angle)
        yaw = pi_2_pi(angle + motion_cmd[1] * step / self.wheel_base * beta)
        traj.append([x, y, yaw])
        n_steps = int(sim_length / step)
        for _ in range(n_steps - 1):
            last = traj[-1]
            x = last[0] + motion_cmd[1] * step * math.cos(last[2])
            y = last[1] + motion_cmd[1] * step * math.sin(last[2])
            yaw = pi_2_pi(last[2] + motion_cmd[1] * step / self.wheel_base * beta)
            traj.append([x, y, yaw])
        grid_idx = [round(traj[-1][0] / self.xy_resolution),
                    round(traj[-1][1] / self.xy_resolution),
                    round(traj[-1][2] / self.yaw_resolution)]
        if not self._is_valid(traj, grid_idx):
           return None
        cost = self._simulated_path_cost(current_node, motion_cmd, sim_length, goal_node)
        return Node(grid_idx, traj, motion_cmd[0], motion_cmd[1], cost, _node_index(current_node))

    def _backtrack(self, start_node, goal_node, closed_set):
        path_x, path_y, path_yaw = [], [], []
        current = goal_node  
        while _node_index(current) != _node_index(start_node):
            xs, ys, yaws = zip(*current.traj)
            path_x = list(xs) + path_x
            path_y = list(ys) + path_y
            path_yaw = list(yaws) + path_yaw
            current = closed_set.get(current.parent_idx)
            if current is None:
                print("Backtracking failed because a parent node is missing.")
                return [], [], []
        xs, ys, yaws = zip(*start_node.traj)
        path_x = list(xs) + path_x
        path_y = list(ys) + path_y
        path_yaw = list(yaws) + path_yaw
        return path_x, path_y, path_yaw

    def plan_path(self, start, goal):
        """
        Plans a path from start to goal.
        - start and goal: [x, y, yaw] (yaw in radians).
        Returns: (x_path, y_path, yaw_path) lists if a solution is found; otherwise None.
        Note: set_occupancy_grid(occ_msg) must have been called beforehand.
        """
        if self.obstacle_tree is None:
            raise ValueError("Occupancy grid not set. Call set_occupancy_grid() first.")

        s_idx = [round(start[0] / self.xy_resolution),
                 round(start[1] / self.xy_resolution),
                 round(start[2] / self.yaw_resolution)]
        g_idx = [round(goal[0] / self.xy_resolution),
                 round(goal[1] / self.xy_resolution),
                 round(goal[2] / self.yaw_resolution)]
        print(g_idx, s_idx)
        start_node = Node(s_idx, [start], 0, 1, 0, tuple(s_idx))
        goal_node = Node(g_idx, [goal], 0, 1, 0, tuple(g_idx))
        open_set = { _node_index(start_node): start_node }
        closed_set = {}
        cost_queue = heapdict()
        cost_queue[_node_index(start_node)] = start_node.cost

        while open_set:
            current_key, _ = cost_queue.popitem()
            if current_key not in open_set:
                continue
            current_node = open_set.pop(current_key)
            closed_set[current_key] = current_node

            # If we reached the goal cell, terminate.
            if abs(current_node.grid_idx[0] - goal_node.grid_idx[0]) < 2 and abs(current_node.grid_idx[1] - goal_node.grid_idx[1]) < 2:
                print(current_node.grid_idx, goal_node.grid_idx)
                return self._backtrack(start_node, current_node, closed_set)
            
            for cmd in self.motion_set:
                new_node = self._kinematic_simulation_node(current_node, cmd, goal_node)
                if new_node is None:
                    continue

                nkey = _node_index(new_node)
                if nkey in closed_set:
                    continue
                if (nkey not in open_set) or (new_node.cost < open_set[nkey].cost):
                    open_set[nkey] = new_node
                    cost_queue[nkey] = new_node.cost
        return None

    def plot_plan(self, x_path, y_path, title="Hybrid A* Path"):
        """
        Displays a matplotlib plot of the planned path.
         • Plots obstacles (if any) as red dots.
         • Plots the path as a blue line.
         • Marks the start (green) and goal (magenta) positions.
        Note: This function requires matplotlib.
        """
        if plt is None:
            print("matplotlib is not installed. Skipping visualization.")
            return

        plt.figure(figsize=(8, 8))
        # Plot obstacles if available.
        if self.obstacle_x is not None and len(self.obstacle_x) > 0:
            plt.scatter(self.obstacle_x, self.obstacle_y, c="red", marker=".", label="Obstacles")
        # Plot planned path.
        plt.plot(x_path, y_path, 'b-', linewidth=2, label="Planned Path")
        # Mark start and goal.
        plt.plot(x_path[0], y_path[0], 'go', markersize=10, label="Start")
        plt.plot(x_path[-1], y_path[-1], 'mo', markersize=10, label="Goal")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(title)
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()


# For simple testing without ROS2:
if __name__ == '__main__':
    import math

    # Dummy occupancy grid definitions.
    class DummyOrigin:
        def __init__(self, x=0.0, y=0.0):
            self.position = type("pos", (), {"x": x, "y": y})
    class DummyInfo:
        def __init__(self, res, width, height, origin):
            self.resolution = res
            self.width = width
            self.height = height
            self.origin = origin
    class DummyOccMsg:
        def __init__(self, res, width, height, origin, data):
            self.info = DummyInfo(res, width, height, origin)
            self.data = data

    # Create a dummy occupancy grid of 50x50 cells with all cells free.
    grid_w, grid_h = 50, 50
    data = [0] * (grid_w * grid_h)

    # In this test, you can simulate obstacles by manually setting some cells:
    # For example, mark a vertical wall in the middle:
    for j in range(10, 40):
        i = 25
        data[j * grid_w + i] = 100
        data[i * grid_w + j ] = 100

    occ_msg = DummyOccMsg(res=0.2, width=grid_w, height=grid_h, origin=DummyOrigin(0, 0), data=data)
    
    planner = HybridAStarPlanner(xy_resolution=0.2, yaw_resolution=math.radians(15))
    planner.set_occupancy_grid(occ_msg)
    start = [2.0, 8.0, math.radians(90)]
    goal  = [3.0, 7.0, math.radians(270)]
    plan = planner.plan_path(start, goal)
    if plan:
        x_path, y_path, yaw_path = plan
        print("Path found with {} poses.".format(len(x_path)))
        print("X:", x_path)
        print("Y:", y_path)
        print("Yaw (deg):", [math.degrees(a) for a in yaw_path])
        planner.plot_plan(x_path, y_path, title="Hybrid A* Planned Path")
    else:
        print("No valid path found.")


