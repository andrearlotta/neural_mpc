#!/usr/bin/env python

import rospy
import numpy as np
from sklearn.cluster import KMeans
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from collections import defaultdict
import threading
import matplotlib.pyplot as plt
import tf
import tf2_ros
from nmpc_ros.srv import GetTreesPoses
import os


class KMeansClusterNode:
    def __init__(self):

        # Inizializza ROS e il rate di esecuzione
        rospy.init_node('kmeans_cluster_node', anonymous=True)
        self.rate = rospy.Rate(1)
        self.n_agents = rospy.get_param('~n_agents', 1)

        # Positions
        self.robot_positions = np.zeros((self.n_agents+1, 2)) # elemento 0 vuoto
        self.tree_positions = self.get_trees_poses()

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # Start the robot state update thread at 30 Hz.
        self.robots_state_thread = threading.Thread(target=self.robots_state_update_thread)
        self.robots_state_thread.daemon = True
        self.robots_state_thread.start()
        self.data_received = False # first TF obtained
        
        # subscibers
        self.lambda_value = None
        subscribers_net = []
        for n in range(1, self.n_agents+1):
            topic = f"/agent_{n}/lambda"
            sub = rospy.Subscriber(topic, Float32MultiArray, self.lambda_callback)
            subscribers_net.append(sub)

        self.first_assignment = False   # used to stop first assignment
        self.dict_assignment = None # store agent-list of assigned trees

    def lambda_callback(self, msg):
        """
        Callback for lambda.
        """
        if self.lambda_value is None:
            self.lambda_value = msg.data
        received = msg.data
        lambda_curr = self.lambda_value
        result_max = np.maximum(received, lambda_curr)
        self.lambda_value = result_max

    def robots_state_update_thread(self):
        """Continuously update the robots' state using TF at 30 Hz."""
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

    def run_first_kmeans(self):
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

        self.dict_assignment = robot_to_trees
        
        ##################   Plotting
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.robot_positions) - 1))  # Colori da mappa 'viridis', ridotto di 1 per escludere la posizione 0
        plt.figure(figsize=(8, 6))
        # Plot dei robot (colorati in base al cluster)
        for robot_idx, pos in enumerate(self.robot_positions[1:], start=1):  # Iniziamo da 1 per escludere la posizione 0
            plt.scatter(pos[0], pos[1], s=100, color=colors[robot_idx - 1], label=f"Robot {robot_idx}", edgecolors='black', marker='X', zorder=5)
        # Plot degli alberi, colorati in base all'assegnazione del robot (cluster)
        for tree_idx, pos in enumerate(self.tree_positions):
            robot_idx = assigned_clusters[tree_idx]
            plt.scatter(pos[0], pos[1], c=[colors[robot_idx]], s=100, edgecolors='black', zorder=2)
        # Aggiungi legende, titoli e griglia
        plt.title("K-Means", fontsize=14)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.tight_layout()
        # Calcola i limiti con un margine
        plt.legend(loc='upper right',
                fancybox=True, shadow=True, ncol=5)
        x_min, x_max = self.tree_positions[:, 0].min(), self.tree_positions[:, 0].max()
        y_min, y_max = self.tree_positions[:, 1].min(), self.tree_positions[:, 1].max()
        x_margin = 3
        y_margin = 3
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + 2*y_margin)
        # Mostra il grafico
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(current_dir, "start.svg")
        plt.savefig(filename, format='svg')
        plt.close()
        # plt.show()
        ##################
    
    def rerun_kmeans(self, idx):
        # Trova gli indici originali degli alberi non ancora visitati (lambda < 0.95)
        unvisited_idxs = [i for i, val in enumerate(self.lambda_value) if val < 0.95]

        # Se gli alberi non visitati sono meno dei robot, il clustering non ha senso
        if len(unvisited_idxs) < self.n_agents:
            return

        # Estrai le posizioni solo degli alberi non ancora visitati
        unvisited_positions = self.tree_positions[unvisited_idxs]

        # Inizializza K-Means con i robot (escludendo la posizione 0)
        kmeans = KMeans(
            n_clusters=self.n_agents,
            init=self.robot_positions[1:, :],  # le posizioni iniziali dei robot (escluso indice 0)
            n_init=1
        )
        kmeans.fit(unvisited_positions)

        # Ottieni a quale cluster (robot) è stato assegnato ciascun albero non visitato
        assigned_clusters = kmeans.labels_

        # Crea un dizionario che assegna ad ogni robot una lista di indici originali degli alberi
        robot_to_trees = defaultdict(list)
        for idx_in_subset, robot_idx in enumerate(assigned_clusters):
            original_tree_idx = unvisited_idxs[idx_in_subset]
            robot_to_trees[robot_idx + 1].append(original_tree_idx)  # +1 per saltare l'indice 0 del robot

        # Salva l'assegnamento nel dizionario interno della classe
        self.dict_assignment = robot_to_trees

        ##################   Plotting
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_agents))
        plt.figure(figsize=(8, 6))
        # Plot dei robot (marker 'X'), esclusa la posizione 0
        for robot_idx, pos in enumerate(self.robot_positions[1:], start=1):
            plt.scatter(
                pos[0], pos[1],
                s=100,
                color=colors[robot_idx - 1],
                label=f"Robot {robot_idx}",
                edgecolors='black',
                marker='X',
                zorder=5
            )
        # Plot degli alberi non visitati, colorati in base al robot assegnato
        for i, tree_idx in enumerate(unvisited_idxs):
            pos = self.tree_positions[tree_idx]
            robot_idx = assigned_clusters[i]  # Cluster assegnato
            plt.scatter(
                pos[0], pos[1],
                c=[colors[robot_idx]],
                s=100,
                edgecolors='black',
                zorder=2
            )
        # Aggiungi legende, titoli e griglia
        plt.title("K-Means", fontsize=14)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.tight_layout()
        # Calcola i limiti con un margine
        plt.legend(loc='upper right',
                fancybox=True, shadow=True, ncol=5)
        x_min, x_max = self.tree_positions[:, 0].min(), self.tree_positions[:, 0].max()
        y_min, y_max = self.tree_positions[:, 1].min(), self.tree_positions[:, 1].max()
        x_margin = 3
        y_margin = 3
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + 2*y_margin)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(current_dir, "reassignment_"+str(idx)+".svg")
        plt.savefig(filename, format='svg')
        plt.close()
        # plt.show()
        ##################


    def publish_cluster(self, robot_idx, trees):
        # Crea il messaggio da pubblicare
        msg = Int32MultiArray()
        msg.data = trees  # Gli indici degli alberi assegnati

        # Pubblica su /agent_i/cluster (dove i è l'indice del robot)
        topic_name = f"/agent_{robot_idx}/cluster"
        pub = rospy.Publisher(topic_name, Int32MultiArray, queue_size=1)
        pub.publish(msg)

    def spin(self):
        t_res = 0
        while not rospy.is_shutdown():
            if self.data_received and self.first_assignment == False:
                self.run_first_kmeans()
                self.first_assignment = True

            # Compute reassignment condition
            reassignment_needed = False
            if self.dict_assignment is not None and self.lambda_value is not None:
                for robot_idx, trees in self.dict_assignment.items():
                    if np.all(self.lambda_value[trees] >= 0.95): # reassign if all labdas are sufficiently high
                        reassignment_needed = True

            # Reassignment (if needed)
            if reassignment_needed == True:
                self.rerun_kmeans(t_res)
                t_res+=1

            # Pubblica i risultati
            if self.dict_assignment is not None:
                for robot_idx, trees in self.dict_assignment.items():
                    self.publish_cluster(robot_idx, trees)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = KMeansClusterNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
