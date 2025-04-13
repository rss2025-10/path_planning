import rclpy
from rclpy.node import Node
import numpy as np
from queue import PriorityQueue

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
    
    def map_cb(self, msg):
        """Callback for map messages"""
        self.get_logger().info("Received map")
        self.get_logger().info(f"Map dimensions: {msg.info.width}x{msg.info.height}")
        self.get_logger().info(f"Map resolution: {msg.info.resolution}")
        self.get_logger().info(f"Map origin: ({msg.info.origin.position.x}, {msg.info.origin.position.y})")
        
        # Count occupied and free cells
        occupied = sum(1 for cell in msg.data if cell > 50)
        free = sum(1 for cell in msg.data if cell >= 0 and cell <= 50)
        unknown = sum(1 for cell in msg.data if cell < 0)
        
        self.get_logger().info(f"Map statistics: {occupied} occupied cells, {free} free cells, {unknown} unknown cells")
        
        self.map = msg
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin
        
        # Convert 1D map data to 2D grid for easier processing
        self.grid = np.array(msg.data, dtype=np.int8).reshape((self.map_height, self.map_width))
    
    # def map_cb(self, msg):
    #     """Callback for map messages"""
    #     self.get_logger().info("Received map")
    #     self.map = msg
    #     self.map_width = msg.info.width
    #     self.map_height = msg.info.height
    #     self.map_resolution = msg.info.resolution
    #     self.map_origin = msg.info.origin
        
    #     # Convert 1D map data to 2D grid for easier processing
    #     self.grid = np.array(msg.data, dtype=np.int8).reshape((self.map_height, self.map_width))

    def pose_cb(self, pose):
        """Callback for pose messages"""
        self.get_logger().info("Received pose")
        self.current_pose = pose.pose.pose
        self.has_pose = True

    def goal_cb(self, msg):
        """Callback for goal messages"""
        self.get_logger().info("Received goal")
        if not hasattr(self, 'map') or not hasattr(self, 'has_pose') or not self.has_pose:
            self.get_logger().warning("No map or pose available yet!")
            return
            
        start_point = (self.current_pose.position.x, self.current_pose.position.y)
        end_point = (msg.pose.position.x, msg.pose.position.y)
        
        self.get_logger().info(f"Planning path from {start_point} to {end_point}")
        self.plan_path(start_point, end_point, self.map)

    def world_to_grid(self, world_point):
        """Convert world coordinates to grid coordinates"""
        # Apply translation and rotation from map origin
        x = (world_point[0] - self.map_origin.position.x) / self.map_resolution
        y = (world_point[1] - self.map_origin.position.y) / self.map_resolution
        
        # Convert to integer grid coordinates
        grid_x = int(round(x))
        grid_y = int(round(y))
        
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_point):
        """Convert grid coordinates to world coordinates"""
        world_x = grid_point[0] * self.map_resolution + self.map_origin.position.x
        world_y = grid_point[1] * self.map_resolution + self.map_origin.position.y
        
        return (world_x, world_y)
    
    def is_valid(self, point):
        """Check if a grid point is valid (within bounds and not an obstacle)"""
        x, y = point
        
        # Check if within grid bounds
        if x < 0 or y < 0 or x >= self.map_width or y >= self.map_height:
            return False
        
        # Check if cell is free (not an obstacle)
        # In occupancy grid, 0 is free, 100 is occupied, -1 is unknown
        return self.grid[y, x] < 50
    
    def get_neighbors(self, point):
        """Get valid neighboring grid cells"""
        x, y = point
        neighbors = [
            (x+1, y), (x-1, y), (x, y+1), (x, y-1),  # 4-connected neighbors
            (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)  # Diagonal neighbors
        ]
        
        return [n for n in neighbors if self.is_valid(n)]
    
    def heuristic(self, a, b):
        """Euclidean distance heuristic"""
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    
    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from map"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        return path

    def plan_path(self, start_point, end_point, map):
        """A* path planning algorithm"""
        # Clear previous trajectory
        self.trajectory.clear()
        
        # Convert world coordinates to grid coordinates
        start_grid = self.world_to_grid(start_point)
        goal_grid = self.world_to_grid(end_point)
        
        self.get_logger().info(f"Grid start: {start_grid}, goal: {goal_grid}")
        
        # A* algorithm
        open_set = PriorityQueue()
        open_set.put((0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        open_set_hash = {start_grid}
        
        while not open_set.empty():
            current = open_set.get()[1]
            open_set_hash.remove(current)
            
            if current == goal_grid:
                # Path found, reconstruct and convert to world coordinates
                grid_path = self.reconstruct_path(came_from, current)
                for grid_point in grid_path:
                    world_point = self.grid_to_world(grid_point)
                    self.trajectory.addPoint(world_point)
                break
            
            for neighbor in self.get_neighbors(current):
                # Tentative g score
                tentative_g = g_score[current] + self.heuristic(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path to neighbor is better than any previous one, record it
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    
                    if neighbor not in open_set_hash:
                        open_set.put((f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        # Publish the trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
