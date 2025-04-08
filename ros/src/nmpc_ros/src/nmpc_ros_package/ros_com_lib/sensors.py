"""
  Definition of topics to use
"""
from sensor_msgs.msg import *
from std_msgs.msg import *
from geometry_msgs.msg import *
from cv_bridge import CvBridge
import numpy as np
from nmpc_ros.srv import GetTreesPoses
import tf
import tf2_ros
from std_msgs.msg import Float32MultiArray
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import tf.transformations
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import rospy

bridge = CvBridge()
# Define the lambda function to create PoseArray from list of [x, y, yaw]

create_pose_array = lambda data: PoseArray(
    poses=[
        Pose(
            position=Point(x=data[0, i, 0], y=data[1, i, 0], z=0),
            orientation=Quaternion(*tf.transformations.quaternion_from_euler(0, 0, data[2, i, 0]))
        ) for i in range(data.shape[1])
    ]
)

def create_path_from_mpc_prediction(mpc_prediction):
    path = Path()
    path.header.frame_id = "map"  # Adjust as needed
    path.header.stamp = rospy.Time.now()

    for i in range(mpc_prediction.shape[1]):
        pose = PoseStamped()
        pose.header = path.header
        pose.header.stamp =  rospy.Time.now() + rospy.Duration(0.1*i)  # Assuming 0.1s between predictions

        pose.pose.position.x = mpc_prediction[0, i]
        pose.pose.position.y = mpc_prediction[1, i]
        pose.pose.position.z = 1.7  # Assuming 2D motion

        # Assuming the third row is yaw, if available
        if mpc_prediction.shape[0] > 2:
            yaw = mpc_prediction[2, i]
            quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
            pose.pose.orientation.x = quaternion[0]
            pose.pose.orientation.y = quaternion[1]
            pose.pose.orientation.z = quaternion[2]
            pose.pose.orientation.w = quaternion[3]
        else:
            pose.pose.orientation.w = 1.0  # Default orientation if yaw is not provided

        path.poses.append(pose)

    return path

def create_tree_markers(trees_pos, scores):
    markers = MarkerArray()
    scores = scores.flatten()
    for i, (tree_pos, score) in enumerate(zip(trees_pos, scores)):
        if score == 0:
            continue
        
        # Sphere marker for the tree
        sphere_marker = Marker()
        sphere_marker.header.frame_id = 'map'
        sphere_marker.header.stamp = rospy.Time.now()
        sphere_marker.ns = "tree_markers"
        sphere_marker.id = i * 2
        sphere_marker.type = Marker.CYLINDER
        sphere_marker.action = Marker.ADD
        sphere_marker.pose.position = Point(x=tree_pos[0], y=tree_pos[1], z=0.0)
        sphere_marker.pose.orientation.w = 1.0
        sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = 0.5
        sphere_marker.color.a = 1.0
        sphere_marker.color.r = score
        sphere_marker.color.g = 1.0 - score
        sphere_marker.color.b = 0.0
        markers.markers.append(sphere_marker)
        
        # Text marker for the score
        text_marker = Marker()
        text_marker.header.frame_id = 'map'
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "tree_scores"
        text_marker.id = i * 2 + 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position = Point(x=tree_pos[0], y=tree_pos[1], z=4)
        text_marker.text = f"Score: {score:.2f}"
        text_marker.scale.z = 0.2
        text_marker.color.a = 1.0
        text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0
        markers.markers.append(text_marker)
    
    return markers


# Update the SENSORS dictionary with the new entry for the Detection2D message
SENSORS = [
  { 
    "name":  "trees_poses",
    "type":  GetTreesPoses,
    "topic": "/obj_pose_srv",
    "mode":  "srv",
    "serializer": lambda pose_array_msg: np.array([[pose.position.x, pose.position.y] for pose in pose_array_msg.trees_poses.poses]),
  },
  {
    "name":  "gps",
    "type":  Pose,
    "topic": "agent_0/pose",
    "mode":  "sub",
    "serializer": lambda pose_msg: np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z,
                                          np.arctan2(2.0 * (pose_msg.orientation.w * pose_msg.orientation.z + pose_msg.orientation.x * pose_msg.orientation.y),
                                          1.0 - 2.0 * (pose_msg.orientation.y * pose_msg.orientation.y + pose_msg.orientation.z * pose_msg.orientation.z))]),
  },
  { 
    "name":  "tree_scores",
    "type":  Float32MultiArray,
    "topic": "tree_scores",
    "mode":  "sub",
    "serializer": lambda array_msg: np.array(array_msg.data).reshape(-1,1),
  },
  { 
    "name":  "cmd_pose",
    "type":  Pose,
    "topic": "agent_0/cmd/pose",
    "mode":  "pub",
    "serializer":  lambda arr: Pose(position=Point(x=arr[0,0], y=arr[1,0], z=0.0), orientation=Quaternion(*tf.transformations.quaternion_from_euler(0, 0, arr[2,0]))),
  },
  { 
    "name":  "viz_pred_pose",
    "type":  PoseArray,
    "topic": "viz/pred/traj",
    "mode":  "pub",
    "serializer":  create_pose_array,
  },
  {
      "name": "predicted_path",
      "type": Path,
      "topic": "agent_0/predicted_path",
      "mode": "pub",
      "serializer": lambda arr: create_path_from_mpc_prediction(arr),
  },
  {
      "name": "tree_markers",
      "type": MarkerArray,
      "topic": "agent_0/tree_markers",
      "mode": "pub",
      "serializer": lambda self: create_tree_markers(self["trees_pos"], self["lambda"]),
  },
  { 
    "name":  "yolo_detector",
    "type":  Detection2DArray,
    "topic": "/yolov7/detect",
    "mode":  "sub",
    "serializer": lambda msg: [{ 
                          'id': detection.results[0].id,
                          'score': detection.results[0].score,
                          'center_x': detection.bbox.center.x if detection.bbox else None,
                          'center_y': detection.bbox.center.y if detection.bbox else None,
                          'size_x': detection.bbox.size_x if detection.bbox else None,
                          'size_y': detection.bbox.size_y if detection.bbox else None
                        } for detection in msg.detections],  # Add the callback function here
  }]