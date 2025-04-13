import rclpy
import time

import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseArray, Point
from std_msgs.msg import Header
import os
from typing import List, Tuple
import json

EPSILON = 0.00000000001

''' These data structures can be used in the search function
'''


class LineTrajectory:
    """ A class to wrap and work with piecewise linear trajectories. """

    def __init__(self, node, viz_namespace=None):
        self.points: List[Tuple[float, float]] = []
        self.distances = []
        self.has_acceleration = False
        self.visualize = False
        self.viz_namespace = viz_namespace
        self.node = node

        if viz_namespace:
            self.visualize = True
            self.start_pub = self.node.create_publisher(Marker, viz_namespace + "/start_point", 1)
            self.traj_pub = self.node.create_publisher(Marker, viz_namespace + "/path", 1)
            self.end_pub = self.node.create_publisher(Marker, viz_namespace + "/end_pose", 1)

    # compute the distances along the path for all path segments beyond those already computed
    def update_distances(self):
        num_distances = len(self.distances)
        num_points = len(self.points)

        for i in range(num_distances, num_points):
            if i == 0:
                self.distances.append(0)
            else:
                p0 = self.points[i - 1]
                p1 = self.points[i]
                delta = np.array([p0[0] - p1[0], p0[1] - p1[1]])
                self.distances.append(self.distances[i - 1] + np.linalg.norm(delta))

    def distance_to_end(self, t):
        if not len(self.points) == len(self.distances):
            print(
                "WARNING: Different number of distances and points, this should never happen! Expect incorrect results. See LineTrajectory class.")
        dat = self.distance_along_trajectory(t)
        if dat == None:
            return None
        else:
            return self.distances[-1] - dat

    def distance_along_trajectory(self, t):
        # compute distance along path
        # ensure path boundaries are respected
        if t < 0 or t > len(self.points) - 1.0:
            return None
        i = int(t)  # which segment
        t = t % 1.0  # how far along segment
        if t < EPSILON:
            return self.distances[i]
        else:
            return (1.0 - t) * self.distances[i] + t * self.distances[i + 1]

    def addPoint(self, point: Tuple[float, float]) -> None:
        print("adding point to trajectory:", point)
        self.points.append(point)
        self.update_distances()
        self.mark_dirty()

    def clear(self):
        self.points = []
        self.distances = []
        self.mark_dirty()

    def empty(self):
        return len(self.points) == 0

    def save(self, path):
        print("Saving trajectory to:", path)
        data = {}
        data["points"] = []
        for p in self.points:
            data["points"].append({"x": p[0], "y": p[1]})
        with open(path, 'w') as outfile:
            json.dump(data, outfile)

    def mark_dirty(self):
        self.has_acceleration = False

    def dirty(self):
        return not self.has_acceleration

    def load(self, path):
        print("Loading trajectory:", path)

        # resolve all env variables in path
        path = os.path.expandvars(path)

        with open(path) as json_file:
            json_data = json.load(json_file)
            for p in json_data["points"]:
                self.points.append((p["x"], p["y"]))
        self.update_distances()
        print("Loaded:", len(self.points), "points")
        self.mark_dirty()

    # build a trajectory class instance from a trajectory message
    def fromPoseArray(self, trajMsg):
        for p in trajMsg.poses:
            self.points.append((p.position.x, p.position.y))
        self.update_distances()
        self.mark_dirty()
        print("Loaded new trajectory with:", len(self.points), "points")

    def toPoseArray(self):
        traj = PoseArray()
        traj.header = self.make_header("/map")
        for i in range(len(self.points)):
            p = self.points[i]
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            traj.poses.append(pose)
        return traj

    def publish_start_point(self, duration=0.0, scale=0.1):
        """ Publishes the start point of the trajectory. """
        self.node.get_logger().info("Before Publishing start point")
        # Wait a fraction of a second for subscribers to connect
        attempts = 0
        while self.start_pub.get_subscription_count() == 0 and attempts < 5:
            time.sleep(0.1) # Wait 100ms
            attempts += 1
            if attempts > 1:
                 self.node.get_logger().debug(f"Waiting for start point subscribers (attempt {attempts})...")

        if self.start_pub.get_subscription_count() > 0:
            self.node.get_logger().info("Publishing start point")
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            if len(self.points) > 0:
                marker.pose.position.x = self.points[0][0]
                marker.pose.position.y = self.points[0][1]
                marker.pose.position.z = 0.0  # Assuming 2D
                marker.pose.orientation.w = 1.0

                marker.scale.x = scale
                marker.scale.y = scale
                marker.scale.z = scale

                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                 # Optionally handle the case where there are no points (e.g., delete marker)
                 marker.action = Marker.DELETE

            if duration > 0:
                marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()

            self.start_pub.publish(marker)
        else:
            self.node.get_logger().warn("Not publishing start point, no subscribers after waiting.")

    def publish_end_point(self, duration=0.0, scale=0.1):
        """ Publishes the end point of the trajectory. """
        self.node.get_logger().info("Before Publishing end point")
        # Wait a fraction of a second for subscribers to connect
        attempts = 0
        while self.end_pub.get_subscription_count() == 0 and attempts < 5:
            time.sleep(0.1) # Wait 100ms
            attempts += 1
            if attempts > 1:
                 self.node.get_logger().debug(f"Waiting for end point subscribers (attempt {attempts})...")

        if self.end_pub.get_subscription_count() > 0:
            self.node.get_logger().info("Publishing end point")
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace
            marker.id = 1 # Different ID from start point
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            if len(self.points) > 1: # Need at least start and end
                marker.pose.position.x = self.points[-1][0]
                marker.pose.position.y = self.points[-1][1]
                marker.pose.position.z = 0.0  # Assuming 2D
                marker.pose.orientation.w = 1.0

                marker.scale.x = scale
                marker.scale.y = scale
                marker.scale.z = scale

                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                # Optionally handle the case where there are not enough points
                marker.action = Marker.DELETE

            if duration > 0:
                marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()

            self.end_pub.publish(marker)
        else:
            self.node.get_logger().warn("Not publishing end point, no subscribers after waiting.")

    def publish_trajectory(self, duration=0.0):
        should_publish = len(self.points) > 1
        if self.visualize and self.traj_pub.get_subscription_count() > 0:
            self.node.get_logger().info("Publishing trajectory")
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace + "/trajectory"
            marker.id = 2
            marker.type = marker.LINE_STRIP  # line strip
            marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()
            if should_publish:
                marker.action = marker.ADD
                marker.scale.x = 0.3
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
                marker.color.a = 1.0
                for p in self.points:
                    pt = Point()
                    pt.x = p[0]
                    pt.y = p[1]
                    pt.z = 0.0
                    marker.points.append(pt)
            else:
                # delete
                marker.action = marker.DELETE
            self.traj_pub.publish(marker)
            print('publishing traj')
        elif self.traj_pub.get_subscription_count() == 0:
            print("Not publishing trajectory, no subscribers")

    def publish_path(self, duration=0.0, scale=0.03):
        """ Publishes the path of the trajectory. """
        self.node.get_logger().info("Before Publishing path")
        # Wait a fraction of a second for subscribers to connect
        attempts = 0
        # Use self.viz_path_pub here, matching the publisher name in __init__
        while self.viz_path_pub.get_subscription_count() == 0 and attempts < 5:
            time.sleep(0.1) # Wait 100ms
            attempts += 1
            if attempts > 1:
                 self.node.get_logger().debug(f"Waiting for path subscribers (attempt {attempts})...")

        # Use self.viz_path_pub here
        if self.viz_path_pub.get_subscription_count() > 0:
            self.node.get_logger().info("Publishing path")
            marker = Marker()
            # ... (rest of the marker setup code should be correct from original) ...
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.node.get_clock().now().to_msg()
            marker.ns = self.viz_namespace
            marker.id = 2 # Different ID from start/end points
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            marker.scale.x = scale # Line width

            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0 # Blue path

            marker.points = []
            if len(self.points) > 1:
                for p in self.points:
                    pt_msg = Point()
                    pt_msg.x = p[0]
                    pt_msg.y = p[1]
                    pt_msg.z = 0.0
                    marker.points.append(pt_msg)
            else:
                 # Optionally handle the case where there are not enough points
                 marker.action = Marker.DELETE

            if duration > 0:
                 marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()

            # Use self.viz_path_pub here
            self.viz_path_pub.publish(marker)
        else:
            # Use self.viz_path_pub here
            self.node.get_logger().warn("Not publishing path, no subscribers after waiting.")

    def publish_viz(self, duration=0.0):
        """ Publishes the trajectory visualization markers. """
        if not self.visualize:
            print("Cannot visualize path, not initialized with visualization enabled")
            return
        self.publish_start_point(duration=duration)
        self.publish_trajectory(duration=duration)
        self.publish_end_point(duration=duration)

    def make_header(self, frame_id, stamp=None):
        if stamp == None:
            stamp = self.node.get_clock().now().to_msg()
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return header
