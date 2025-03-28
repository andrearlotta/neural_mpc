#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from time import sleep


def create_pose(x, y, z, orientation_w=1.0):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    return pose

def main():
    rospy.init_node('control', anonymous=True)

    x = 0
    y = (-15.0, -10.0, -6.0)
    z = -0.6

    pub_drone_0 = rospy.Publisher('drone_0/cmd/pose', Pose, queue_size=10)
    pub_drone_1 = rospy.Publisher('drone_1/cmd/pose', Pose, queue_size=10)
    pub_drone_2 = rospy.Publisher('drone_2/cmd/pose', Pose, queue_size=10)

    initial_pose_0 = create_pose(0.0, y[0], z)   
    initial_pose_1 = create_pose(0.0, y[1], z)   
    initial_pose_2 = create_pose(0.0, y[2], z)   

    pub_drone_0.publish(initial_pose_0)
    pub_drone_1.publish(initial_pose_1)
    pub_drone_2.publish(initial_pose_2)
    rospy.sleep(3)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        new_pose_0 = create_pose(x, y[0], z)
        new_pose_1 = create_pose(x, y[1], z)
        new_pose_2 = create_pose(x, y[2], z)

        pub_drone_0.publish(new_pose_0)
        pub_drone_1.publish(new_pose_1)
        pub_drone_2.publish(new_pose_2)

        x += 0.1

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
