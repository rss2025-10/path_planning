#!/usr/bin/env python
import math
import numpy as np

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from nav_msgs.msg import Odometry

from .utils import LineTrajectory  # Adjust the import according to your package structure

# --- Module-level constants ---
FINISH_RADIUS = 1.0       # meters; when within this radius of the final waypoint, finish (stop)
LOOKAHEAD_DISTANCE = 0.5     # meters; the lookahead distance for pure pursuit
CAR_LENGTH = 0.325         # meters; your vehicle’s wheelbase length

def quaternion_to_yaw(q: Quaternion):
    # Convert quaternion to yaw angle.
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class PurePursuit(Node):
    """
    Pure Pursuit trajectory tracking node.
    It subscribes to odometry and trajectory (as a PoseArray) then uses the
    pure pursuit algorithm with fixed lookahead, constant speed, and a wheelbase length
    to generate Ackermann steering commands.
    """
    def __init__(self):
        super().__init__("trajectory_follower")
        
        # Declare parameters (with defaults)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('drive_topic', '/drive')
        
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        # Parameters that you may tune (using our module-level constants):
        self.lookahead = LOOKAHEAD_DISTANCE     # meters (from constant)
        self.speed = 3.0                        # meters/second (constant speed)
        self.wheelbase_length = CAR_LENGTH      # meters
        
        self.initialized_traj = False
        self.stopped = True

        # Create an instance of the trajectory helper.
        self.trajectory = LineTrajectory(self, "/followed_trajectory")

        # Subscribe to trajectory updates.
        self.traj_sub = self.create_subscription(
            PoseArray,
            "/trajectory/current",
            self.trajectory_callback,
            1)

        # Subscribe to odometry updates.
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.pose_callback,
            1)

        # Create a publisher for drive commands.
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 1)

        self.get_logger().info("Pure Pursuit node initialized")

    def trajectory_callback(self, msg: PoseArray):
        self.get_logger().info(f"Receiving new trajectory with {len(msg.poses)} points")
        # Reset and update the trajectory.
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        self.initialized_traj = True
        self.stopped = False

    def pose_callback(self, odom_msg: Odometry):
        if not self.initialized_traj or self.stopped:
            return  # Do nothing if trajectory isn’t initialized.

        # Get the vehicle pose from odometry.
        vehicle_position = odom_msg.pose.pose.position
        vehicle_orientation = odom_msg.pose.pose.orientation
        vehicle_yaw = quaternion_to_yaw(vehicle_orientation)
        
        # --- Check if we are close enough to the end of the trajectory ---
        if self.trajectory.points:
            end_point = self.trajectory.points[-1]  # (x,y) tuple of the final waypoint
            distance_to_end = math.hypot(end_point[0] - vehicle_position.x,
                                         end_point[1] - vehicle_position.y)
            if distance_to_end < FINISH_RADIUS:
                #self.get_logger().info(f"Reached end of trajectory (within {FINISH_RADIUS} m). Stopping.")
                self.publish_drive_command(0.0, 0.0)
                self.stopped = True
                return

        # Get the target point (as a Pose) that is at least lookahead distance ahead.
        target_point = self.find_target_point(vehicle_position)
        if target_point is None:
            self.get_logger().warn("No valid target point found. Stopping.")
            self.publish_drive_command(0.0, 0.0)
            return

        # Compute the vector from vehicle to target point in world frame.
        dx = target_point.position.x - vehicle_position.x
        dy = target_point.position.y - vehicle_position.y

        # Transform the target point into the vehicle coordinate frame.
        local_x = math.cos(-vehicle_yaw) * dx - math.sin(-vehicle_yaw) * dy
        local_y = math.sin(-vehicle_yaw) * dx + math.cos(-vehicle_yaw) * dy

        # Compute the angle to the target point relative to the vehicle’s x-axis.
        alpha = math.atan2(local_y, local_x)
        
        if self.lookahead == 0:
            self.get_logger().error("Lookahead is 0; cannot compute steering.")
            return

        # Pure pursuit curvature formula.
        steering_angle = math.atan2(self.wheelbase_length * math.sin(alpha), self.lookahead)
        
        #self.get_logger().info(f"Target in vehicle frame: ({local_x:.2f}, {local_y:.2f}), "
        #                       f"alpha: {alpha:.2f}, steering: {steering_angle:.2f}")

        self.publish_drive_command(self.speed, steering_angle)

    def find_target_point(self, current_position):
        """
        Use the stored trajectory points (which are 2-tuples), and return
        the first point (converted to a Pose) at least lookahead distance ahead.
        If all points are closer, return the last point (as a Pose).
        """
        if not self.trajectory.points:
            return None

        # Find the index of the trajectory point closest to the current position.
        closest_index = 0
        min_dist = float("inf")
        for i, pt in enumerate(self.trajectory.points):
            distance = math.hypot(pt[0] - current_position.x, pt[1] - current_position.y)
            if distance < min_dist:
                min_dist = distance
                closest_index = i

        selected = None
        for pt in self.trajectory.points[closest_index:]:
            distance = math.hypot(pt[0] - current_position.x, pt[1] - current_position.y)
            if distance >= self.lookahead:
                selected = pt
                break

        # If no point meets lookahead, select the last one.
        if selected is None:
            selected = self.trajectory.points[-1]

        # Convert the (x,y) tuple into a Pose message.
        pose = Pose()
        pose.position.x = selected[0]
        pose.position.y = selected[1]
        return pose

    def publish_drive_command(self, speed, steering_angle):
        """
        Helper function that publishes an AckermannDriveStamped command.
        """
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.drive.speed = float(speed)
        cmd.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
