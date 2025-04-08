#!/usr/bin/env python

import rospy
import numpy as np
from sklearn.cluster import KMeans
from std_msgs.msg import Int32MultiArray
from collections import defaultdict
import threading
import matplotlib.pyplot as plt
import tf
import tf2_ros
from nmpc_ros.srv import GetTreesPoses


class KMeansClusterNode:
    def __init__(self):

        # Inizializza ROS e il rate di esecuzione
        rospy.init_node('kmeans_cluster_node', anonymous=True)
        self.rate = rospy.Rate(1)
        self.n_agents = rospy.get_param('~n_agents', 1)

        self.robot_positions = np.zeros((self.n_agents+1, 2)) # elemento 0 vuoto


        # Posizioni degli alberi (x, y) -> m alberi
        self.tree_positions = self.get_trees_poses()

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # Start the robot state update thread at 30 Hz.
        self.robots_state_thread = threading.Thread(target=self.robots_state_update_thread)
        self.robots_state_thread.daemon = True
        self.robots_state_thread.start()
        self.data_received = False

    def robots_state_update_thread(self):
        """Continuously update the robot's state using TF at 30 Hz."""
        rate = rospy.Rate(30)  # 30 Hz update rate
        while not rospy.is_shutdown():
            for n in range(1, self.n_agents+1):
                try:
                    # Look up the transform from 'map' to 'base_link_n'
                    link_str = 'base_link_' + str(n)
                    trans = self.tf_buffer.lookup_transform('map', link_str, rospy.Time(0))
                    self.robot_positions[n, 0] = trans.transform.translation.x
                    self.robot_positions[n, 1] = trans.transform.translation.y
                    self.data_received = True
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    rospy.logwarn("Failed to get transform: %s", e)
            rate.sleep()
    
    def get_trees_poses(self):
        """
        Calls the GetTreesPoses service and returns tree positions as an (N,2) numpy array.
        The serializer logic is taken from sensors.py.
        """
        rospy.wait_for_service("/obj_pose_srv")
        try:
            trees_srv = rospy.ServiceProxy("/obj_pose_srv", GetTreesPoses)
            response = trees_srv()  # Adjust parameters if needed
            # Using the serializer from sensors.py:
            trees_pos = np.array([[pose.position.x, pose.position.y] for pose in response.trees_poses.poses])
            return trees_pos
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
            return np.array([])

    def run_kmeans(self):
        # Eseguiamo il K-Means per assegnare gli alberi ai robot
        # Escludiamo la prima posizione dei robot (posizione 0) e inizializziamo KMeans
        kmeans = KMeans(n_clusters=self.n_agents, init=self.robot_positions[1:, :], n_init=1)
        kmeans.fit(self.tree_positions)

        # Otteniamo l'indice dell'albero assegnato per ogni posizione
        assigned_clusters = kmeans.labels_

        # Associa ogni robot agli indici degli alberi
        robot_to_trees = defaultdict(list)

        # Per ogni albero, associa l'indice al robot, tenendo conto che i robot partono da indice 1
        for tree_idx, robot_idx in enumerate(assigned_clusters):
            robot_to_trees[robot_idx + 1].append(tree_idx)  # Aggiungiamo 1 per saltare l'indice 0

        ##################   Plotting
        # colors = plt.cm.viridis(np.linspace(0, 1, len(self.robot_positions) - 1))  # Colori da mappa 'viridis', ridotto di 1 per escludere la posizione 0
        # plt.figure(figsize=(8, 6))
        # # Plot dei robot (colorati in base al cluster)
        # for robot_idx, pos in enumerate(self.robot_positions[1:], start=1):  # Iniziamo da 1 per escludere la posizione 0
        #     plt.scatter(pos[0], pos[1], s=100, color=colors[robot_idx - 1], label=f"Robot {robot_idx}", edgecolors='black', marker='X', zorder=5)
        # # Plot degli alberi, colorati in base all'assegnazione del robot (cluster)
        # for tree_idx, pos in enumerate(self.tree_positions):
        #     robot_idx = assigned_clusters[tree_idx]
        #     plt.scatter(pos[0], pos[1], c=[colors[robot_idx]], s=100, edgecolors='black', zorder=2)
        # # Aggiungi legende, titoli e griglia
        # plt.title("Assegnamento Alberi ai Robot tramite K-Means", fontsize=14)
        # plt.xlabel("Coordinata X")
        # plt.ylabel("Coordinata Y")
        # plt.legend()
        # plt.grid(True)
        # # Mostra il grafico
        # plt.show()
        ##################

        # Pubblica i risultati
        for robot_idx, trees in robot_to_trees.items():
            self.publish_cluster(robot_idx, trees)


    def publish_cluster(self, robot_idx, trees):
        # Crea il messaggio da pubblicare
        msg = Int32MultiArray()
        msg.data = trees  # Gli indici degli alberi assegnati

        # Pubblica su /agent_i/cluster (dove i è l'indice del robot)
        topic_name = f"/agent_{robot_idx}/cluster"
        pub = rospy.Publisher(topic_name, Int32MultiArray, queue_size=1)
        pub.publish(msg)

    def spin(self):
        while not rospy.is_shutdown():
            # print(self.robot_positions)
            if self.data_received:
                self.run_kmeans()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = KMeansClusterNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
