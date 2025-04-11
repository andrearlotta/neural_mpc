import rospy
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, CameraInfo
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from image_geometry import PinholeCameraModel
from scipy.spatial import cKDTree
from nmpc_ros.srv import GetTreesPoses
from std_msgs.msg import Float32MultiArray
import tf
import tf.transformations as tf_trans
import message_filters

def weight_value(n_elements, mean_score, midpoint=5, steepness=10.):
    return np.ceil(100 * (mean_score - 0.5) * (0.5 + 0.5 * np.tanh(steepness * (n_elements - midpoint)))) / 100

class DataAssociationNode:
    def __init__(self):
        rospy.init_node('bounding_box_3d_pose', anonymous=True)

        self.bridge = CvBridge()
        self.camera_info = None
        self.camera_matrix = None
        self.depth_image = None
        self.tree_poses = None
        self.publish_visualization = rospy.get_param('~publish_visualization', True)

        detection_sub = message_filters.Subscriber("/yolov7/detect", Detection2DArray)
        depth_image_sub = message_filters.Subscriber("agent_0/camera/depth/image/compressed", CompressedImage)

        self.ts = message_filters.ApproximateTimeSynchronizer([detection_sub, depth_image_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.synchronized_callback)

        self.camera_info_sub = rospy.Subscriber("agent_0/camera/depth/camera_info", CameraInfo, self.camera_info_callback)
        self.scores_pub = rospy.Publisher("agent_0/tree_scores", Float32MultiArray, queue_size=1)
        
        if self.publish_visualization:
            self.marker_scores_pub = rospy.Publisher("agent_0/scores_markers", MarkerArray, queue_size=1)
            self.marker_fruits_pub = rospy.Publisher("agent_0/fruits_markers", MarkerArray, queue_size=1)
        
        self.cam_model = PinholeCameraModel()
        self.tf_listener = tf.TransformListener()
        
        rospy.wait_for_service('/obj_pose_srv')
        self.get_trees_poses = rospy.ServiceProxy('/obj_pose_srv', GetTreesPoses)
        self.update_tree_poses()
        
        rospy.spin()

    def update_tree_poses(self):
        try:
            response = self.get_trees_poses()
            self.tree_poses = np.array([[pose.position.x, pose.position.y] for pose in response.trees_poses.poses])
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.camera_matrix = np.array(self.camera_info.K).reshape(3, 3)
        self.cam_model.fromCameraInfo(msg)
    

    def uint8_to_distance(self, value, min_dist, max_dist):
        value = max(0, min(value, 255))
        fraction = value / 255.0
        distance = max_dist - fraction * (max_dist - min_dist)
        return distance
    
    def associate_fruits_to_trees(self, fruit_positions, fruit_classes, fruit_scores):
        if self.tree_poses is None or len(fruit_positions) == 0:
            return {}
        
        tree_kdtree = cKDTree(self.tree_poses)
        distances, tree_indices = tree_kdtree.query(fruit_positions[:, :2])

        tree_fruit_dict = {i: {"ripe": [], "raw": []} for i in range(len(self.tree_poses))}
        
        for fruit_index, (tree_index, distance) in enumerate(zip(tree_indices, distances)):
            if distance <= 2.5:
                fruit_class = "ripe" if fruit_classes[fruit_index] == "ripe" else "raw"
                tree_fruit_dict[tree_index][fruit_class].append(fruit_scores[fruit_index])
        
        return tree_fruit_dict

    def transform_fruit_positions(self, fruit_positions, header):
        map_fruits_positions = []
        for fruit_pos in fruit_positions:
            point_camera = PointStamped()
            point_camera.point = Point(*fruit_pos)
            point_camera.header.frame_id = 'depth_camera_frame'
            try:
                point_map = self.tf_listener.transformPoint('map', point_camera)
                map_fruits_positions.append([point_map.point.x, point_map.point.y, point_map.point.z])
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr(e)
                continue
        return np.array(map_fruits_positions)
    def synchronized_callback(self, detection_msg, depth_image_msg):
        if self.camera_matrix is None or self.tree_poses is None:
            return
        
        try:
            self.depth_image = self.bridge.compressed_imgmsg_to_cv2(depth_image_msg, desired_encoding="passthrough")[:, :, 0]
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        fruit_positions, fruit_scores, fruit_classes = [], [], []

        for detection in detection_msg.detections:
            bbox = detection.bbox
            xmin = int(bbox.center.x - bbox.size_x / 2)
            xmax = int(bbox.center.x + bbox.size_x / 2)
            ymin = int(bbox.center.y - bbox.size_y / 2)
            ymax = int(bbox.center.y + bbox.size_y / 2)

            xmin, xmax = max(0, xmin), min(self.depth_image.shape[1], xmax)
            ymin, ymax = max(0, ymin), min(self.depth_image.shape[0], ymax)

            depth_roi = self.depth_image[int(bbox.center.y), int(bbox.center.x)]
            non_zero_depths = depth_roi[depth_roi > 11]
            if len(non_zero_depths) == 0:
                continue

            median_depth = np.median(non_zero_depths)
            center_x, center_y = int(bbox.center.x), int(bbox.center.y)

            XYZ = np.array(self.cam_model.projectPixelTo3dRay((center_x, center_y)))
            XYZ *= self.uint8_to_distance(median_depth, 0.05, 20)

            fruit_positions.append(XYZ)
            fruit_scores.append(detection.results[0].score)
            fruit_classes.append("ripe" if detection.results[0].id == 2 else "raw")  # Assuming ID 1 is ripe, others are raw
        
        fruit_positions = np.array(fruit_positions)

        # Transform fruit positions from camera frame to map frame
        map_fruits_positions = self.transform_fruit_positions(fruit_positions, detection_msg.header)

        associated_fruits = self.associate_fruits_to_trees(map_fruits_positions, fruit_classes, fruit_scores)
        try:
            (drone_trans, drone_rot) = self.tf_listener.lookupTransform('map', 'drone_base_link', rospy.Time())
            drone_x, drone_y = drone_trans[0], drone_trans[1]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("Could not get drone position, skipping distance check.")
            drone_x, drone_y = None, None
        tree_scores = np.ones(len(self.tree_poses)) * 0.5
        for i, fruits in associated_fruits.items():
            ripe_scores = fruits["ripe"]
            raw_scores = fruits["raw"]
            tree_x, tree_y = self.tree_poses[i]
            distance = np.sqrt((drone_x - tree_x)**2 + (drone_y - tree_y)**2)
            if distance < 8:
                ripe_value = weight_value(len(ripe_scores), np.mean(ripe_scores) if ripe_scores else 0)
                raw_value = weight_value(len(raw_scores), np.mean(raw_scores) if raw_scores else 0)
                tree_scores[i] = ripe_value - raw_value + 0.5
            
        scores_msg = Float32MultiArray()
        scores_msg.data = tree_scores.tolist()
        self.scores_pub.publish(scores_msg)
        
        if self.publish_visualization:
            markers = MarkerArray()
            for i, (fruit_pos, score) in enumerate(zip(map_fruits_positions, fruit_scores)):
                fruit_marker = Marker()
                fruit_marker.header = detection_msg.header
                fruit_marker.header.frame_id = 'map'
                fruit_marker.ns = "fruit_markers"
                fruit_marker.id =  len(self.tree_poses) * 2 + i * 2
                fruit_marker.type = Marker.SPHERE
                fruit_marker.action = Marker.MODIFY
                fruit_marker.pose.position.x = fruit_pos[0]
                fruit_marker.pose.position.y = fruit_pos[1]
                fruit_marker.pose.position.z = fruit_pos[2]
                fruit_marker.pose.orientation.w = 1.0
                fruit_marker.lifetime = rospy.Duration(0.2)
                fruit_marker.scale.x = fruit_marker.scale.y = fruit_marker.scale.z = 0.1
                fruit_marker.color.a = 1.0
                fruit_marker.color.r = 1.0 if fruit_classes[i] == "ripe" else 0.0
                fruit_marker.color.g = 1.0 if fruit_classes[i] == "raw" else 0.0
                fruit_marker.color.b = 0.0
                markers.markers.append(fruit_marker)
            self.marker_fruits_pub.publish(markers)

if __name__ == '__main__':
    try:
        DataAssociationNode()
    except rospy.ROSInterruptException:
        pass
