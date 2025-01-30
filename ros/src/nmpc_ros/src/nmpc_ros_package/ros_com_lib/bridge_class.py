from nmpc_ros_package.ros_com_lib.publisher_node import PublisherNode
from nmpc_ros_package.ros_com_lib.subscriber_node import SubscriberNode
from nmpc_ros_package.ros_com_lib.client_node import ClientNode
import rospy
import tf2_ros
import numpy as np
import tf

class BridgeClass:
    def __init__(self, components_list = []):
        self.__publishers_dict   = {}
        self.__subscribers_dict = {}
        self.__clients_dict = {}
        self.setup_ros_com(components_list)
        rospy.init_node("sim_bridge", anonymous=True, log_level=rospy.DEBUG)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def add_component(self, sensor):
        if      sensor.get("mode") == "pub": self.__publishers_dict[sensor.get("name")]   =  PublisherNode(sensor)
        elif    sensor.get("mode") == "sub": self.__subscribers_dict[sensor.get("name")]  =  SubscriberNode(sensor)
        elif    sensor.get("mode") == "srv": self.__clients_dict[sensor.get("name")]      =  ClientNode(sensor)

    def setup_ros_com(self, components_list):
        for component_info in components_list:
            self.add_component(component_info)
    
    def get_data(self):
        ret = {}
        for sensor, sub in self.__subscribers_dict.items():
            ret[sensor]= sub.get_data()
        return ret

    def pub_data(self, data_dict):
        for sensor, data in data_dict.items():
            if data is not None: self._pubData(sensor, data)

    def call_server(self, req_dict):
        return {server: self._call_server(server,req) for server, req in req_dict.items()}
    
    def _get_data(self, sensor):
        return self.__subscribers_dict[sensor].get_data()

    def _pubData(self, pub, data):
        self.__publishers_dict[pub].pub_data(data)

    def _call_server(self, server, req):
        return self.__clients_dict[server].call_server(req)
    
    def update_robot_state(self):
        robot_pose = []
        while not len(robot_pose):
            try:
                # Get the transform from 'map' to 'drone_base_link'
                trans = self.tf_buffer.lookup_transform('map', 'drone_base_link', rospy.Time())

                # Extract rotation
                (_, _, yaw) = tf.transformations.euler_from_quaternion([ trans.transform.rotation.x,  trans.transform.rotation.y,  trans.transform.rotation.z,  trans.transform.rotation.w])
                
                # Update x_robot with the transform (x, y, yaw)
                robot_pose = [[trans.transform.translation.x], [trans.transform.translation.y], [yaw]]
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"Failed to get transform: {e}")
            
        return robot_pose