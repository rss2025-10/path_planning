#!/usr/bin/env python3
"""
rtt.py

An implementation of RRT (Rapidly-exploring Random Tree) for path planning.
This implementation supports Ackermann vehicle constraints and handles
obstacle avoidance using the same occupancy grid structure as the hybrid A* planner.

Usage:
1. Construct an instance with the desired parameters
2. Call set_occupancy_grid(occ_msg) to feed in obstacles from a ROS2 OccupancyGrid message
3. Call plan_path(start, goal) where start and goal are [x, y, yaw] (with yaw in radians)
"""

import math
import random
import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import KDTree

# For matplotlib visualization (only used when running the script interactively)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# Helper: wrap an angle to [-pi, pi]
def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Node:
    """Node class for RRT tree representation"""
    def __init__(self, x, y, yaw=None, parent=None):
        self.x = x
        self.y = y
        self.yaw = yaw  # Only used for kinodynamic planning
        self.parent = parent
        self.path_x = []
        self.path_y = []
        self.path_yaw = []
        self.cost = 0.0


class RRTPlanner:
    def __init__(self, *,
                 max_iter=3000,
                 expand_dist=0.5,
                 goal_sample_rate=10,
                 path_resolution=0.1,
                 xy_resolution=0.2,
                 occ_threshold=50,
                 wheel_base=0.325,
                 max_steer=0.2,
                 car_width=0.3,
                 axle_to_front=0.1,
                 axle_to_back=0.1):
        """
        Initialize the RRT planner.
        
        Args:
            max_iter: Maximum number of iterations/nodes to add
            expand_dist: Distance to expand the tree in each step
            goal_sample_rate: Percentage chance of sampling the goal
            path_resolution: Resolution for path interpolation
            xy_resolution: Grid cell size in meters
            occ_threshold: Occupancy grid threshold for obstacles
            wheel_base: Distance between front and rear axles
            max_steer: Maximum steering angle
            car_width: Width of the car for collision checking
            axle_to_front: Distance from rear axle to front of car
            axle_to_back: Distance from rear axle to back of car
        """
        self.max_iter = max_iter
        self.expand_dist = expand_dist
        self.goal_sample_rate = goal_sample_rate
        self.path_resolution = path_resolution
        self.xy_resolution = xy_resolution
        self.occ_threshold = occ_threshold
        
        # Vehicle parameters 
        self.wheel_base = wheel_base
        self.max_steer = max_steer
        self.car_width = car_width
        self.axle_to_front = axle_to_front
        self.axle_to_back = axle_to_back
        
        # These will be set from the occupancy grid
        self.obstacle_x = None
        self.obstacle_y = None
        self.obstacle_tree = None
        self.map_min_x = None
        self.map_min_y = None
        self.map_max_x = None
        self.map_max_y = None
        
        # Simulation parameters
        self.kinematic_step = 0.2

    def set_occupancy_grid(self, occ_msg):
        """
        Process a ROS2 OccupancyGrid message.
        Cells with occupancy value >= self.occ_threshold are considered occupied.
        """
        ox, oy = [], []
        info = occ_msg.info
        res = info.resolution
        width = info.width
        height = info.height
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        data = occ_msg.data
        
        print(f"Map info - origin: ({origin_x}, {origin_y}), width: {width}, height: {height}, res: {res}")

        # OccupancyGrid data is row-major
        for j in range(height):
            for i in range(width):
                idx = j * width + i
                if data[idx] >= self.occ_threshold:
                    # Use cell center as obstacle coordinate
                    x = origin_x - (i) * res
                    y = origin_y - (j) * res
                    ox.append(x)
                    oy.append(y)
                    
        if len(ox) == 0:
            print("No obstacles found in occupancy grid. Using map boundaries only.")
        else:
            print(f"Found {len(ox)} obstacles in occupancy grid.")
            
        self.obstacle_x = ox
        self.obstacle_y = oy
        if ox and oy:
            self.obstacle_tree = KDTree(list(zip(ox, oy)))
        else:
            # If no obstacles are found, create an empty tree that will report no collisions
            self.obstacle_tree = KDTree([(1e10, 1e10)])
            
        # Set map boundaries from occupancy grid info
        self.map_min_x = origin_x - width * res
        self.map_min_y = origin_y - height * res
        self.map_max_x = origin_x
        self.map_max_y = origin_y
        
        # Print out map boundaries for debugging
        print(f"Map boundaries: X [{self.map_min_x}, {self.map_max_x}], Y [{self.map_min_y}, {self.map_max_y}]")
        
    def _collision(self, node_x, node_y, yaw=None):
        """
        Check if a configuration is in collision with obstacles.
        
        Args:
            node_x: x-coordinate
            node_y: y-coordinate
            yaw: orientation (optional, only used for kinodynamic planning)
            
        Returns:
            True if in collision, False otherwise
        """
        if self.obstacle_tree is None:
            return False
            
        # Basic collision check using car radius
        car_radius = ((self.axle_to_front + self.axle_to_back + self.wheel_base))
        
        # Check if point is within collision distance from any obstacle
        if self.obstacle_tree.query_ball_point([node_x, node_y], car_radius):
            return True
            
        return False
        
    def _steer(self, from_node, to_point):
        """
        Steer from from_node towards to_point, respecting motion constraints.
        
        Args:
            from_node: Node to steer from
            to_point: (x, y) coordinates to steer towards
            
        Returns:
            New node after steering
        """
        # Calculate direction and distance
        dist = math.sqrt((from_node.x - to_point[0])**2 + (from_node.y - to_point[1])**2)
        
        # If the point is close enough, just return it
        if dist < self.expand_dist:
            new_node = Node(to_point[0], to_point[1])
            new_node.path_x = [from_node.x, to_point[0]]
            new_node.path_y = [from_node.y, to_point[1]]
            new_node.parent = from_node
            new_node.cost = from_node.cost + dist
            return new_node
            
        # Otherwise, expand in the direction of the point
        theta = math.atan2(to_point[1] - from_node.y, to_point[0] - from_node.x)
        new_x = from_node.x + self.expand_dist * math.cos(theta)
        new_y = from_node.y + self.expand_dist * math.sin(theta)
        
        # Create new node
        new_node = Node(new_x, new_y)
        new_node.path_x = [from_node.x, new_x]
        new_node.path_y = [from_node.y, new_y]
        new_node.parent = from_node
        new_node.cost = from_node.cost + self.expand_dist
        
        return new_node
        
    def _kinematic_steer(self, from_node, to_point):
        """
        Steer from from_node towards to_point using vehicle kinematics.
        This function considers the Ackermann steering constraints.
        
        Args:
            from_node: Node to steer from
            to_point: (x, y) coordinates to steer towards
            
        Returns:
            New node after steering or None if not reachable
        """
        # Distance to target
        dist = math.sqrt((from_node.x - to_point[0])**2 + (from_node.y - to_point[1])**2)
        
        # Target direction
        target_angle = math.atan2(to_point[1] - from_node.y, to_point[0] - from_node.x)
        
        # Current heading
        if from_node.yaw is None:
            from_node.yaw = target_angle  # Assume facing target if no orientation specified
            
        # How many steps to simulate
        n_steps = min(int(dist / self.kinematic_step), 20)  # Limit to prevent excessive computation
        if n_steps <= 0:
            n_steps = 1
            
        # Try different steering angles
        best_node = None
        min_dist = float('inf')
        
        # Sample several steering angles
        candidate_steers = np.linspace(-self.max_steer, self.max_steer, 5)
        
        # Add boundary safety margin (half the car's length)
        safety_margin = (self.axle_to_front + self.axle_to_back) / 2
        
        for steer in candidate_steers:
            # Simulate forward
            path_x, path_y, path_yaw = [], [], []
            x, y, yaw = from_node.x, from_node.y, from_node.yaw
            
            valid_path = True
            
            for _ in range(n_steps):
                x, y, yaw = self._update_state(x, y, yaw, steer, self.kinematic_step)
                
                # Check collision
                if self._collision(x, y, yaw):
                    valid_path = False
                    break
                    
                # Check map bounds with safety margin
                if (x < self.map_min_x + safety_margin or 
                    x > self.map_max_x - safety_margin or
                    y < self.map_min_y + safety_margin or 
                    y > self.map_max_y - safety_margin):
                    valid_path = False
                    break
                
                # Only add this point if it's valid
                path_x.append(x)
                path_y.append(y)
                path_yaw.append(yaw)
            
            # Skip evaluating this path if it's invalid
            if not valid_path or len(path_x) == 0:
                continue
                
            # Evaluate this steering option
            final_dist = math.sqrt((path_x[-1] - to_point[0])**2 + (path_y[-1] - to_point[1])**2)
            
            # Only consider this path if it's better than what we have
            if final_dist < min_dist:
                # Create new node
                new_node = Node(path_x[-1], path_y[-1], path_yaw[-1], from_node)
                new_node.path_x = path_x
                new_node.path_y = path_y
                new_node.path_yaw = path_yaw
                new_node.cost = from_node.cost + len(path_x) * self.kinematic_step
                
                best_node = new_node
                min_dist = final_dist
        
        return best_node
        
    def _update_state(self, x, y, yaw, steer, distance):
        """
        Update the state of the vehicle based on bicycle model.
        
        Args:
            x, y: Current position
            yaw: Current heading
            steer: Steering angle
            distance: Distance to move
            
        Returns:
            New state (x, y, yaw)
        """
        # Bicycle model
        yaw = pi_2_pi(yaw)
        x = x + distance * math.cos(yaw)
        y = y + distance * math.sin(yaw)
        yaw = pi_2_pi(yaw + distance * math.tan(steer) / self.wheel_base)
        
        return x, y, yaw
        
    def _get_random_point(self, goal):
        """
        Get a random point in the map, with some bias toward the goal.
        
        Args:
            goal: Goal position (x, y)
            
        Returns:
            Random point as (x, y)
        """
        if random.randint(0, 100) > self.goal_sample_rate:
            # Random point in map bounds
            return (
                random.uniform(self.map_min_x, self.map_max_x),
                random.uniform(self.map_min_y, self.map_max_y)
            )
        else:
            # Goal point
            return (goal[0], goal[1])
            
    def _get_nearest_node_index(self, node_list, point):
        """
        Find the nearest node in the tree to the given point.
        
        Args:
            node_list: List of nodes
            point: Target point (x, y)
            
        Returns:
            Index of the nearest node
        """
        distances = [(node.x - point[0])**2 + (node.y - point[1])**2 for node in node_list]
        return distances.index(min(distances))
        
    def _check_segment_collision(self, x1, y1, x2, y2):
        """
        Check if a line segment between two points collides with obstacles.
        
        Args:
            x1, y1: First point
            x2, y2: Second point
            
        Returns:
            True if in collision, False otherwise
        """
        # Sample points along the segment
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist == 0:
            return self._collision(x1, y1)
            
        n_steps = max(2, int(dist / self.path_resolution))
        for i in range(n_steps + 1):
            t = i / n_steps
            x = x1 + t * dx
            y = y1 + t * dy
            if self._collision(x, y):
                return True
                
        return False
        
    def _extract_path(self, final_node):
        """
        Extract the path from the start to the final_node.
        
        Args:
            final_node: Final node in the path
            
        Returns:
            Lists of x, y, and yaw coordinates of the path
        """
        path_x = []
        path_y = []
        path_yaw = []
        node = final_node
        
        while node is not None:
            # Add path segments in reverse
            for i in range(len(node.path_x) - 1, -1, -1):
                path_x.insert(0, node.path_x[i])
                path_y.insert(0, node.path_y[i])
                if hasattr(node, 'path_yaw') and node.path_yaw:
                    if i < len(node.path_yaw):
                        path_yaw.insert(0, node.path_yaw[i])
            node = node.parent
            
        return path_x, path_y, path_yaw
        
    def plan_path(self, start, goal, use_kinodynamic=True):
        """
        Plan a path from start to goal using RRT.
        
        Args:
            start: Start position [x, y, yaw] (yaw in radians)
            goal: Goal position [x, y, yaw] (yaw in radians)
            use_kinodynamic: Whether to use kinodynamic constraints
            
        Returns:
            (x_path, y_path, yaw_path) if path found, None otherwise
        """
        if self.obstacle_tree is None:
            raise ValueError("Occupancy grid not set. Call set_occupancy_grid() first.")
            
        # Initialize tree with start node
        node_list = [Node(start[0], start[1], start[2])]
        
        # Goal point
        goal_point = (goal[0], goal[1])
        
        # First check if start and goal are within bounds
        safety_margin = (self.axle_to_front + self.axle_to_back) / 2
        
        if (start[0] < self.map_min_x + safety_margin or 
            start[0] > self.map_max_x - safety_margin or
            start[1] < self.map_min_y + safety_margin or 
            start[1] > self.map_max_y - safety_margin):
            print(f"Start position ({start[0]}, {start[1]}) is outside map boundaries or too close to edge")
            return None
            
        if (goal[0] < self.map_min_x + safety_margin or 
            goal[0] > self.map_max_x - safety_margin or
            goal[1] < self.map_min_y + safety_margin or 
            goal[1] > self.map_max_y - safety_margin):
            print(f"Goal position ({goal[0]}, {goal[1]}) is outside map boundaries or too close to edge")
            return None
            
        # Check for obstacles at start and goal
        if self._collision(start[0], start[1], start[2]):
            print("Start position is in collision with obstacles")
            return None
            
        if self._collision(goal[0], goal[1], goal[2]):
            print("Goal position is in collision with obstacles")
            return None
            
        # Print planner parameters
        print(f"Planning with {'kinodynamic' if use_kinodynamic else 'geometric'} constraints")
        print(f"Max iterations: {self.max_iter}, Expand distance: {self.expand_dist}")
        
        # Main loop
        for i in range(self.max_iter):
            # Get random point
            rnd_point = self._get_random_point(goal_point)
            
            # Find nearest node
            nearest_ind = self._get_nearest_node_index(node_list, rnd_point)
            nearest_node = node_list[nearest_ind]
            
            # Steer towards the random point
            if use_kinodynamic:
                new_node = self._kinematic_steer(nearest_node, rnd_point)
            else:
                new_node = self._steer(nearest_node, rnd_point)
                
            if new_node is None:
                continue
            
            # Check if the new path is collision-free
            if not use_kinodynamic:
                if self._check_segment_collision(nearest_node.x, nearest_node.y, new_node.x, new_node.y):
                    continue
            
            # Add node to the tree
            node_list.append(new_node)
            
            # Check if goal reached
            dist_to_goal = math.sqrt((new_node.x - goal[0])**2 + (new_node.y - goal[1])**2)
            if dist_to_goal <= self.expand_dist:
                # Try connecting to goal
                final_node = None
                
                if use_kinodynamic:
                    final_node = self._kinematic_steer(new_node, goal_point)
                else:
                    if not self._check_segment_collision(new_node.x, new_node.y, goal[0], goal[1]):
                        final_node = Node(goal[0], goal[1], goal[2])
                        final_node.parent = new_node
                        final_node.path_x = [new_node.x, goal[0]]
                        final_node.path_y = [new_node.y, goal[1]]
                
                if final_node:
                    print(f"Found path after {i+1} iterations")
                    return self._extract_path(final_node)
                    
            if i % 100 == 0:
                print(f"Iteration {i}, tree size: {len(node_list)}")
                
        print(f"Failed to find path after {self.max_iter} iterations")
        return None
        
    def plot_plan(self, x_path, y_path, title="RRT Path"):
        """
        Displays a matplotlib plot of the planned path.
         • Plots obstacles (if any) as red dots
         • Plots the path as a blue line
         • Marks the start (green) and goal (magenta) positions
        
        Note: This function requires matplotlib.
        """
        if plt is None:
            print("matplotlib is not installed. Skipping visualization.")
            return

        plt.figure(figsize=(8, 8))
        # Plot obstacles if available
        if self.obstacle_x is not None and len(self.obstacle_x) > 0:
            plt.scatter(self.obstacle_x, self.obstacle_y, c="red", marker=".", label="Obstacles")
            
        # Plot planned path
        plt.plot(x_path, y_path, 'b-', linewidth=2, label="Planned Path")
        
        # Mark start and goal
        plt.plot(x_path[0], y_path[0], 'go', markersize=10, label="Start")
        plt.plot(x_path[-1], y_path[-1], 'mo', markersize=10, label="Goal")
        
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(title)
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()


# For simple testing without ROS2
if __name__ == '__main__':
    import math

    # Dummy occupancy grid definitions
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

    # Create a dummy occupancy grid of 50x50 cells with all cells free
    grid_w, grid_h = 50, 50
    data = [0] * (grid_w * grid_h)

    # Simulate obstacles by manually setting some cells
    # Create a challenging environment with a U-shaped obstacle
    for j in range(10, 40):
        for i in range(23, 27):
            data[j * grid_w + i] = 100  # Vertical wall
    
    for i in range(10, 27):
        for j in range(38, 42):
            data[j * grid_w + i] = 100  # Horizontal wall at top
    
    for i in range(10, 27):
        for j in range(8, 12):
            data[j * grid_w + i] = 100  # Horizontal wall at bottom

    occ_msg = DummyOccMsg(res=0.2, width=grid_w, height=grid_h, origin=DummyOrigin(0, 0), data=data)
    
    # Create planner
    planner = RRTPlanner(xy_resolution=0.2, max_iter=2000, expand_dist=1.0)
    planner.set_occupancy_grid(occ_msg)
    
    # Plan path
    start = [2.0, 8.0, math.radians(90)]
    goal  = [8.0, 25.0, math.radians(0)]
    
    # Try with kinodynamic constraints first
    print("Planning with kinodynamic constraints...")
    plan = planner.plan_path(start, goal, use_kinodynamic=True)
    if plan:
        x_path, y_path, yaw_path = plan
        print(f"Path found with {len(x_path)} poses.")
        planner.plot_plan(x_path, y_path, title="RRT Path (Kinodynamic)")
    else:
        print("No valid path found with kinodynamic constraints. Trying without...")
        # Try without kinodynamic constraints as fallback
        plan = planner.plan_path(start, goal, use_kinodynamic=False)
        if plan:
            x_path, y_path, yaw_path = plan
            print(f"Path found with {len(x_path)} poses.")
            planner.plot_plan(x_path, y_path, title="RRT Path (Geometric)")
        else:
            print("No valid path found.")
