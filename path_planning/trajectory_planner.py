#!/usr/bin/env python3
"""
trajectory_planner.py

A ROS2 node that uses path planning algorithms (Hybrid A* or RRT) to plan a path (as a PoseArray)
from the current car pose (from /initial_pose_topic or /odom if using ground truth)
to a goal pose set via RViz (using the "2D Nav Goal" button). 
The planned path, along with start and goal markers, is published for visualization.
"""

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose, PoseArray
from nav_msgs.msg import OccupancyGrid

# Import our planning libraries and utilities.
from .hybrid_a_star import HybridAStarPlanner  # Hybrid A* implementation
from .rtt import RRTPlanner  # RRT implementation
from .utils import LineTrajectory  # Utility for trajectory visualization (publishes markers)

class PathPlan(Node):
    """Plan a collision free trajectory using path planning algorithms from the current pose to a goal.
    
    The current pose is updated from the /initial_pose_topic (or /odom) and
    the goal pose is set via RViz (publishing to /goal_pose).
    The planned trajectory is output as a geometry_msgs/PoseArray.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        # Declare parameters to get proper topic names (if needed)
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")
        self.declare_parameter('planner_type', "hybrid_a_star")  # Options: "hybrid_a_star" or "rrt"
        self.declare_parameter('rrt_max_iter', 25000)
        self.declare_parameter('rrt_goal_sample_rate', 10)
        self.declare_parameter('rrt_expand_dist', 0.5)
        self.declare_parameter('use_kinodynamic', True)  # For RRT: whether to use kinodynamic constraints

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value
        self.planner_type = self.get_parameter('planner_type').get_parameter_value().string_value
        self.use_kinodynamic = self.get_parameter('use_kinodynamic').get_parameter_value().bool_value

        self.get_logger().info(f"Using planner type: {self.planner_type}")
        self.get_logger().info(f"Map topic: {self.map_topic}")

        # Subscribe to the map.
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        # Subscribe to goal pose from RViz (2D Nav Goal).
        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        # Subscribe to the current pose (initial estimate / odometry).
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        # Publisher for the planned trajectory.
        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        # Create a visualization helper instance which publishes markers for
        # start (sphere), trajectory (line strip) and end (sphere) in RViz.
        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        # Instantiate planners
        
        # Hybrid A* planner
        self.a_star_planner = HybridAStarPlanner(
            xy_resolution=0.5, 
            yaw_resolution=math.radians(30)
        )
        
        # RRT planner
        self.rrt_planner = RRTPlanner(
            xy_resolution=0.5,
            max_iter=self.get_parameter('rrt_max_iter').get_parameter_value().integer_value,
            expand_dist=self.get_parameter('rrt_expand_dist').get_parameter_value().double_value,
            goal_sample_rate=self.get_parameter('rrt_goal_sample_rate').get_parameter_value().integer_value
        )

        self.current_pose = None  # This will be set from pose_cb

    def map_cb(self, msg: OccupancyGrid):
        """
        Process the OccupancyGrid message from the map.
         • Convert the occupancy grid into lists of obstacles.
         • Compute map bounds.
         • Pass the map to both planners.
        """
        self.get_logger().info("Map received. Setting occupancy grid for planners.")
        self.get_logger().info(f"{msg.info.origin}")
        self.a_star_planner.set_occupancy_grid(msg)
        self.rrt_planner.set_occupancy_grid(msg)
        
    def pose_cb(self, msg: PoseWithCovarianceStamped):
        """
        Process the current pose from the odometry (or initial pose),
        and store it for planning. This callback is triggered when the
        car's estimated (or ground-truth) pose is updated.
        """
        self.current_pose = msg.pose.pose
        self.get_logger().debug("Updated current pose (x=%.2f, y=%.2f)" %
                                  (self.current_pose.position.x, self.current_pose.position.y))

    def get_yaw_from_pose(self, pose: Pose) -> float:
        """
        Extract yaw (in radians) from a geometry_msgs/Pose message.
        Since we assume the roll and pitch are zero, the yaw calculation is simplified.
        """
        # For a quaternion (x,y,z,w) representing a yaw-only rotation:
        #   yaw = atan2(2*(w*z), 1 - 2*(z*z))
        q = pose.orientation
        siny = 2.0 * (q.w * q.z)
        cosy = 1.0 - 2.0 * (q.z * q.z)
        return math.atan2(siny, cosy)

    def goal_cb(self, msg: PoseStamped):
        """
        Process the goal pose published by RViz.
        This callback uses the current car pose (from /initial_pose_topic) as the start,
        then calls the selected planner to compute a collision-free path to the new goal.
        The result is then published as a PoseArray for visualization.
        """
        if self.current_pose is None:
            self.get_logger().warn("Current pose not received yet. Ignoring goal.")
            return

        # Construct start [x, y, yaw] from the current pose.
        start = [self.current_pose.position.x,
                 self.current_pose.position.y,
                 self.get_yaw_from_pose(self.current_pose)]
        # Construct goal [x, y, yaw] from the incoming goal message.
        goal = [msg.pose.position.x,
                msg.pose.position.y,
                self.get_yaw_from_pose(msg.pose)]
        self.get_logger().info("Planning from (%.2f, %.2f) to (%.2f, %.2f)" %
                               (start[0], start[1], goal[0], goal[1]))

        # Run the selected planner
        plan = None
        
        if self.planner_type == "hybrid_a_star":
            self.get_logger().info("Using Hybrid A* planner")
            plan = self.a_star_planner.plan_path(start, goal)
        elif self.planner_type == "rrt":
            self.get_logger().info(f"Using RRT planner (kinodynamic: {self.use_kinodynamic})")
            plan = self.rrt_planner.plan_path(start, goal, use_kinodynamic=self.use_kinodynamic)
        else:
            self.get_logger().error(f"Unknown planner type: {self.planner_type}")
            return
            
        if plan:
            x_path, y_path, yaw_path = plan

            # Convert path lists into a PoseArray.
            pose_array = PoseArray()
            pose_array.header.stamp = self.get_clock().now().to_msg()
            pose_array.header.frame_id = "map"  # Adjust frame id as needed

            for x, y, yaw in zip(x_path, y_path, yaw_path):
                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.position.z = 0.0
                # For a yaw-only rotation, the quaternion is:
                #   q = [x=0, y=0, z=sin(yaw/2), w=cos(yaw/2)]
                qz = math.sin(yaw/2.0)
                qw = math.cos(yaw/2.0)
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = qz
                pose.orientation.w = qw
                pose_array.poses.append(pose)

            # Publish the planned path.
            self.traj_pub.publish(pose_array)
            self.get_logger().info("Published planned path with %d poses." % len(pose_array.poses))

            # Update visualization markers for start, trajectory, and goal.
            # Instead of using a non-existent set_path() method, we clear the current trajectory
            # and add each new point.
            self.trajectory.clear()
            for x, y in zip(x_path, y_path):
                self.trajectory.addPoint((x, y))
            self.trajectory.publish_viz()
        else:
            self.get_logger().warn("No valid path found!")

    def plan_path(self, start_point, end_point, map_msg, planner_type=None):
        """
        This is a helper if you want to drive the planning externally.
        Allows specifying which planner to use.
        
        Args:
            start_point: Start position [x, y, yaw]
            end_point: Goal position [x, y, yaw]
            map_msg: OccupancyGrid message
            planner_type: Optional override of which planner to use
        
        Returns:
            Tuple of (x_path, y_path, yaw_path) if successful, None otherwise
        """
        # Use provided planner type or default to the node's setting
        planner = planner_type or self.planner_type
        
        # Set the occupancy grid if provided
        if map_msg:
            if planner == "hybrid_a_star":
                self.a_star_planner.set_occupancy_grid(map_msg)
            elif planner == "rrt":
                self.rrt_planner.set_occupancy_grid(map_msg)
            
        # Plan using selected planner
        plan = None
        if planner == "hybrid_a_star":
            plan = self.a_star_planner.plan_path(start_point, end_point)
        elif planner == "rrt":
            plan = self.rrt_planner.plan_path(start_point, end_point, use_kinodynamic=self.use_kinodynamic)
        
        if plan:
            x_path, y_path, yaw_path = plan
            
            # Update visualization
            self.trajectory.clear()
            for x, y in zip(x_path, y_path):
                self.trajectory.addPoint((x, y))
            self.trajectory.publish_viz()
            
            # Publish as PoseArray
            self.traj_pub.publish(self.trajectory.toPoseArray())
            
        return plan


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
