import math
from geometry_msgs.msg import Pose, PoseArray, PointStamped
from ackermann_msgs.msg import AckermannDriveStamped
from .utils import LineTrajectory

class StraightLiner:
    def __init__(self, *, node):
        self.node = node
        self.clicked_points = []
        self.start_pose = None
        self.end_pose = None

        self.node.declare_parameter('drive_topic', '/drive')
        self.drive_topic = self.node.get_parameter('drive_topic').get_parameter_value().string_value
        self.drive_pub = self.node.create_publisher(AckermannDriveStamped, '/drive', 1)

        self.pose_array_pub = self.node.create_publisher(PoseArray, "/straight_line_path", 10)
        self.visualizer = LineTrajectory(node=self.node, viz_namespace="/straight_line_trajectory")


        self.point_sub = self.node.create_subscription(PointStamped, "/clicked_point", self.point_cb, 10)



    def set_start_pose(self, pose: Pose):
        self.start_pose = pose
        self.clicked_points = []
        self.visualizer.clear()
        self.end_pose = None
        self.node.get_logger().info("New start pose received. Resetting trajectory.")

        # cmd = AckermannDriveStamped()
        # cmd.header.stamp = self.get_clock().now().to_msg()
        # cmd.header.frame_id = "base_link"
        # cmd.drive.speed = float(0)
        # cmd.drive.steering_angle = float(0)
        # self.drive_pub.publish(cmd)

        
        return self.try_publish()


    def set_end_pose(self, pose: Pose):
        self.end_pose = pose
        self.node.get_logger().info("Updated end pose.")
        return self.try_publish()


    def point_cb(self, msg: PointStamped):
        """Callback for intermediate clicked points in RViz."""
        point = (msg.point.x, msg.point.y)
        self.node.get_logger().info(f"Clicked point received: {point}")
        self.clicked_points.append(point)
        self.try_publish()

    def try_publish(self):
        """publish the full path if start and end are available"""
        if self.start_pose is None or self.end_pose is None:
            return None

        poses = [self.start_pose] + self._clicked_to_poses() + [self.end_pose]

        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.header.stamp = self.node.get_clock().now().to_msg()
        pose_array.poses = poses
        self.pose_array_pub.publish(pose_array)

        self.visualizer.clear()
        for pose in poses:
            self.visualizer.addPoint((pose.position.x, pose.position.y))
        self.visualizer.publish_viz()

        self.node.get_logger().info(f"Published straight path with {len(poses)} poses.")

        x_path = [pose.position.x for pose in poses]
        y_path = [pose.position.y for pose in poses]
        yaw_path = [0.0] * len(poses) # assuming straight-line segments, yaw=0
        return x_path, y_path, yaw_path

    def _clicked_to_poses(self):
        poses = []
        for x, y in self.clicked_points:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0  # yaw = 0
            poses.append(pose)
        return poses