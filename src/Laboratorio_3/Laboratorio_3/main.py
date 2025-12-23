#!/usr/bin/env python3
"""
Nodo ROS 2 per l'Evitamento Semplice di Ostacoli utilizzando un sensore LaserScan (LiDAR).

Questo nodo si iscrive al topic /scan per ricevere i dati LiDAR,
monitora la distanza minima in un'area frontale ristretta e,
se viene rilevato un ostacolo vicino, determina la direzione più libera
(sinistra o destra) in base alla scansione completa e ruota per evitarlo.
Altrimenti, il robot avanza.
"""
import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import time
import cv2                                          # type: ignore
import numpy as np                                  # type: ignore
import rclpy                                        # type: ignore
from rclpy.node import Node                         # type: ignore

from sensor_msgs.msg import LaserScan               # type: ignore
from cv_bridge import CvBridge                      # type: ignore
from geometry_msgs.msg import Twist                 # type: ignore

from vision_msgs.msg import (                       # type: ignore
    Detection2D,
    ObjectHypothesisWithPose,
    BoundingBox2D
)


class ObstacleAvoidanceNode(Node):
    """
    Nodo ROS 2 che implementa l'algoritmo di evitamento ostacoli
    "Bug-like" basato sulla lettura del LaserScan (LiDAR).
    """
    def __init__(self):
        super().__init__("obstacle_avoidance")


        # Internal state
        self.current_distance = np.inf   # Distanza minima rilevata nel campo visivo frontale ristretto
        self.obstacle_detected = False
        self.ranges = []                 # Array completo delle distanze dalla scansione LiDAR
 
        self.sub_scan = self.create_subscription(
            LaserScan, "/scan", self.on_scan, 10
        )

        self.cmd_pub = self.create_publisher(
            Twist, "/cmd_vel", 10
        )

        # Control loop (20 Hz)
        self.control_timer = self.create_timer(
            0.05, self.control_loop
        )

        self.get_logger().info(
            f"Obstacle avoidance node ready..."
        )


    # LIDAR callback
    def on_scan(self, msg: LaserScan):
        """
        Callback per la ricezione del messaggio LaserScan.

        Aggiorna l'array completo delle distanze (`self.ranges`) e calcola
        la distanza minima nel settore frontale di $\pm 30$ gradi.

        Args:
            msg (LaserScan): Messaggio ROS 2 contenente le letture del LiDAR.
        """

        # Ottiene le distanze
        ranges = np.array(msg.ranges)

        # Aggiorna campo interno
        self.ranges = ranges


        # TODO: considerando un arco frontale di -30/+30 indici, estrarre la distanza minima presente in tale intervallo
        min_distance = None # TODO


        self.current_distance = min_distance


    def detect_obstacle(self, threshold_dist=0.4):
        """
        Verifica se la distanza minima frontale (`self.current_distance`)
        è inferiore alla soglia specificata, aggiornando `self.obstacle_detected`.

        Args:
            threshold_dist (float): Soglia di distanza (in metri) per considerare
                                    un ostacolo come rilevato.
        """
        
        # Controlla se il robot deve evitare un ostacolo

        # TODO
        
        self.obstacle_detected = False


    # Control loop
    def control_loop(self):
        """
        Loop di controllo periodico che implementa la logica di navigazione.

        1. Verifica la presenza di ostacoli frontali.
        2. Se un ostacolo è rilevato:
           a. Analizza la scansione completa per trovare la direzione più libera
              (a sinistra o a destra) misurando la distanza minima su ciascun lato.
           b. Ruota sul posto nella direzione più libera.
        3. Se nessun ostacolo è rilevato, avanza.
        """

        self.obstacle_detected = False

        # Controlla se vi sono ostacoli vicini
        # TODO

        # Se un ostacolo è stato rilevato: gira
        if self.obstacle_detected:

            # TODO: identifica la metà più libera del campo visivo del robot

            left_min_distance = None  # TODO: distanza min nella metà sinistra
            right_min_distance = None # TODO: distanza min nella metà destra

            # Stabilire il senso di rotazione (orario o antiorario)
            direction = None    # TODO

            # Ruota
            self.publish_twist(0.0, 0.5 * direction)

            return
        
        # Altrimenti, prosegue dritto
        self.publish_twist(0.4, 0)

            

    # Metodi di supporto al movimento
    def publish_twist(self, linear=0.0, angular=0.0):
        """
        Pubblica un comando di velocità lineare e angolare sul topic /cmd_vel.

        Args:
            linear (float): Velocità lineare in X (m/s).
            angular (float): Velocità angolare in Z (rad/s).
        """
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def publish_stop(self):
        """Pubblica un comando di velocità nullo (arresto del robot)."""
        self.publish_twist(0.0, 0.0)


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():
    """
    Funzione principale per l'esecuzione del nodo.
    Inizializza ROS 2, crea il nodo e avvia lo spin.
    Garantisce l'arresto del robot al termine.
    """
    rclpy.init()
    node = ObstacleAvoidanceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()   # assicura l'arresto dei motori
        rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()