from nmpc_ros.srv import GetTreesPoses
import rospy

class ClientNode:
    def __init__(self, dict) -> None:
        self.setSrvName(dict.get("topic"))
        self.call = dict.get("serializer")
        self.type = dict.get("type")
        
    def getSrvName(self):
        return self.srv_name
    
    def setSrvName(self, name):
        self.srv_name = name
        
    def call_server(self, cmd):
        rospy.wait_for_service(self.getSrvName())
        try:
            srv_call = rospy.ServiceProxy(self.getSrvName(), self.type)
            return self.call(srv_call()) if  cmd is None else self.call(srv_call(*cmd)) 
            # Otherwise, pass the input data to the service call        
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")