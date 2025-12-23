#!/usr/bin/env python3
"""
Nodo ROS 2 di Navigazione Ibrida e Tracciamento (Bug-like)

Questo nodo combina tre compiti:
1. Navigazione di base verso un punto obiettivo (goal_x, goal_y).
2. Evitamento di Ostacoli (LiDAR): Se viene rilevato un ostacolo, il robot entra in modalità
   di evitamento (ruota, poi avanza lateralmente, poi si riallinea).
3. Tracciamento/Interazione con Oggetto (YOLO + Depth): Rileva un oggetto target
   e si ferma ad una distanza di sicurezza da esso.

Il nodo utilizza un loop di controllo per gestire la sequenza di movimento
(normale, evitando ostacoli, o riallineamento).
"""
import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import time
import math
import cv2                                                                       # type: ignore
import numpy as np                                                               # type: ignore
import rclpy                                                                     # type: ignore
from rclpy.node import Node                                                      # type: ignore
from sensor_msgs.msg import Image, LaserScan                                     # type: ignore
from nav_msgs.msg import Odometry                                                # type: ignore
from geometry_msgs.msg import Twist                                              # type: ignore
from cv_bridge import CvBridge                                                   # type: ignore
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy         # type: ignore

try:
    from ultralytics import YOLO                                                 # type: ignore
except ImportError:
    YOLO = None


class YoloObstacleAvoidanceNode(Node):
    """
    Implementa la logica di tracciamento e navigazione con evitamento ostacoli.
    """
    def __init__(self):
        super().__init__("limo_yolo")

        # -----------------------------
        # Dichiarazione parametri
        # -----------------------------
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("model", "yolov8n.pt")
        self.declare_parameter("score_thresh", 0.5)
        self.declare_parameter("filter_class", "elephant")

        self.declare_parameter("goal_x", -0.200)
        self.declare_parameter("goal_y", 2.325)
        
        self.declare_parameter("rate_hz", 20)              
        self.declare_parameter("obstacle_threshold", 0.4) # Distanza LiDAR sotto cui scatta l'evitamento
        self.declare_parameter("rejoin_distance", 0.8)    # Distanza dall'inizio dell'evitamento per passare al riallineamento
        self.declare_parameter("align_tol", 0.05)         # Tolleranza angolare per il riallineamento
        self.declare_parameter("forward_speed", 0.4)
        self.declare_parameter("turn_speed_max", 1.5)
        self.declare_parameter("safe_distance", 0.4)      # Distanza Depth sotto cui il target è "raggiunto"

        # -----------------------------
        # Caricamento parametri
        # -----------------------------
        self.img_topic = self.get_parameter("image_topic").value
        self.model_path = self.get_parameter("model").value
        self.score_thresh = self.get_parameter("score_thresh").value
        self.filter_class = self.get_parameter("filter_class").value.lower()

        # Posizione target (per il riallineamento)
        self.goal_x = self.get_parameter("goal_x").value
        self.goal_y = self.get_parameter("goal_y").value

        # Parametri di controllo
        self.rate_hz = self.get_parameter("rate_hz").value
        self.obstacle_threshold = self.get_parameter("obstacle_threshold").value
        self.rejoin_distance = self.get_parameter("rejoin_distance").value
        self.align_tol = self.get_parameter("align_tol").value
        self.forward_speed = self.get_parameter("forward_speed").value
        self.turn_speed_max = self.get_parameter("turn_speed_max").value
        self.safe_distance = self.get_parameter("safe_distance").value

        # -----------------------------
        # VARIABILI DI STATO
        # -----------------------------
        # Odometria
        self.x = self.y = self.yaw = None
        self.starting_x = self.starting_y = self.starting_yaw = None    # Odometria di partenza

        # Percezione
        self.bridge = CvBridge()
        self.model = None
        self.model_names = {}
        self.target_detected = False                # Flag: Target YOLO rilevato
        self.target_confidence = 0.0                # Confidenza del target YOLO
        self.target_last_seen_time = time.time()    # Ultimo istante in cui è stato visto il target
        self.target_cx = 0                          # Centro X del box target (per profondità)
        self.target_cy = 0                          # Centro Y del box target (per profondità)

        self.current_distance = np.inf      # Distanza minima LiDAR frontale
        self.ranges = []                    # Array completo delle distanze LiDAR

        # Evitamento ostacoli
        self.obstacle_detected = False      # Flag: Ostacolo rilevato da LiDAR
        self.avoiding = False               # Evitamento attivo (movimento laterale)
        self.aligning = False               # Riallineamento verso il goal
        self.start_avoiding_x = None        # Posizione X all'inizio dell'evitamento
        self.start_avoiding_y = None        # Posizione Y all'inizio dell'evitamento

        # Logica di stop
        self.stop = False                   # Flag: Interrompe l'intera navigazione

        # -----------------------------
        # ROS I/O
        # -----------------------------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Topic
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos)            # Riceve odometria
        self.sub_image = self.create_subscription(Image, self.img_topic, self.on_image, 10)             # Riceve immagini dalla camera
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.on_scan, 10)                  # Riceve il LIDAR
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)                                     # Invia velocità
        self.pub_image = self.create_publisher(Image, "/yolo/annotated_image", 10)                      # Pubblica immagini annotate


        self.control_timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)

        self.load_model()
        self.get_logger().info(f"Node loaded. Tracking: {self.filter_class}")

    # -----------------------------
    # CARICAMENTO MODELLO YOLO
    # -----------------------------
    def load_model(self):
        """
        Carica il modello YOLOv8 specificato dal parametro 'model'.
        """
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO not installed")

        t0 = time.time()
        self.model = YOLO(self.model_path)
        self.model.fuse()
        self.model_names = self.model.names
        self.get_logger().info(f"Loaded YOLO model '{self.model_path}' in {time.time() - t0:.2f}s")

    # -----------------------------
    # CALLBACK PER ODOMETRIA
    # -----------------------------
    def odom_callback(self, msg: Odometry):
        """
        Callback per la ricezione dell'odometria. Aggiorna posizione (x, y) e
        orientamento (yaw) del robot. Inizializza la posizione di partenza.

        Args:
            msg (Odometry): Messaggio di odometria ROS 2.
        """
        # Salva posizione e orientamento correnti
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        # Salva posizione e orientamento di partenza
        if self.starting_x is None:
            self.starting_x = self.x
            self.starting_y = self.y
            self.starting_yaw = self.yaw
            self.get_logger().info("Odom initialized")

    # -----------------------------
    # CALLBACK PER CAMERA
    # -----------------------------
    def on_image(self, msg: Image):
        """
        Callback per la ricezione delle immagini a colori.

        Esegue l'inferenza YOLO, chiama `detect_target` per processare i risultati
        e sovrappone i bounding box rilevati sull'immagine prima di ripubblicarla.

        Args:
            msg (Image): Messaggio immagine ROS 2.
        """

        # Converte da ROS a OpenCV
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failed: {e}")
            return

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Lancia il modello YOLO
        results = self.model.predict(
            img_rgb,
            imgsz=320,
            conf=self.score_thresh,
            verbose=False
        )

        # Logica di estrazione e disegno dei box
        if not results:
            return

        r = results[0]
        if r.boxes is None:
            return

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()

        
        self.detect_target(
            xyxy, conf, cls_ids,
            h, w,
            target_class=self.filter_class,
            threshold_score=self.score_thresh
        )

        # Sovrappone all'immagine la bounding box se il bersaglio è stato individuato
        if self.target_detected:

            x = int(self.target_cx)
            y = int(self.target_cy)

            
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                cls_name = str(self.model_names.get(int(cls_ids[i]), "unknown")).lower()

                if cls_name == self.filter_class:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # Traccia il contorno della box
                    cv2.rectangle(
                        img_bgr,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Traccia il centro della box
                    cv2.circle(
                        img_bgr,
                        (self.target_cx, self.target_cy),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    # Aggiunge l'etichetta della classe e la soglia di confidenza
                    label = f"{self.filter_class} {self.target_confidence:.2f}"
                    cv2.putText(
                        img_bgr,
                        label,
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    break

            # Pubblica l'immagine annotata sul topic d'uscita
            out_msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
            self.pub_image.publish(out_msg)

    # -----------------------------
    # CALLBACK PER LIDAR
    # -----------------------------
    def on_scan(self, msg: LaserScan):
        """
        Callback per la ricezione del messaggio LaserScan.

        Aggiorna l'array completo delle distanze (`self.ranges`) e calcola
        la distanza minima nel settore frontale ristretto (qui definito
        come i 60 indici centrali dell'array).

        Args:
            msg (LaserScan): Messaggio ROS 2 contenente le letture del LiDAR.
        """
        self.ranges = np.array(msg.ranges)
        # Calcola la distanza minima nel settore frontale (+/- 30 indici)
        self.current_distance = np.nanmin(self.ranges[len(self.ranges)//2 - 30 : len(self.ranges)//2 + 30])

    # -----------------------------
    # CONTROL HELPERS
    # -----------------------------
    def distance_from_point(self, x0: float, y0: float) -> float:
        """
        Calcola la distanza euclidea dalla posizione corrente (self.x, self.y)
        a un punto dato (x0, y0).
        """
        return math.hypot(self.x - x0, self.y - y0)

    def quaternion_to_yaw(self, qx: float, qy: float, qz: float, qw: float) -> float:
        """
        Calcola l'imbardata (yaw) da un quaternione.
        """
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy*qy + qz*qz)
        return math.atan2(siny, cosy)

    def normalize_angle(self, angle: float) -> float:
        """
        Normalizza un angolo all'intervallo [-pi, pi).
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def angle_error(self, target_angle: float) -> float:
        """
        Calcola l'errore angolare normalizzato tra l'angolo target e lo yaw corrente.
        """
        return self.normalize_angle(target_angle - self.yaw)

    def calculate_target_yaw(self, goal_x: float, goal_y: float) -> float:
        """
        Calcola l'angolo di imbardata (yaw) necessario per puntare verso il goal.
        """
        return math.atan2(goal_y - self.y, goal_x - self.x)

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

    # -----------------------------
    # RILEVAMENTO OSTACOLI
    # -----------------------------
    def detect_obstacle(self):
        """
        Verifica la presenza di ostacoli.

        Prioritizza l'evitamento: un ostacolo è rilevato se la distanza LiDAR
        frontale è inferiore alla soglia, A MENO CHE l'oggetto rilevato da YOLO
        sia proprio in quel punto.
        """

        # Se il target è stato identificato, non è segnalato come ostacolo (self.obstacle_detected = False).
        # TODO
 
        # Rileva un ostacolo generico se la distanza LiDAR frontale è troppo piccola.
        self.obstacle_detected = None   # TODO

        # Se è appena scattato l'evitamento, registra il punto di partenza (usando la posizione corrente).
        if self.obstacle_detected and self.start_avoiding_x is None:
            self.start_avoiding_x = None    # TODO  
            self.start_avoiding_y = None    # TODO


    # -----------------------------
    # TARGET DETECTION
    # -----------------------------
    def detect_target(self, xyxy: np.ndarray, conf: np.ndarray, cls_ids: np.ndarray, image_h: int, image_w: int,
                    target_class: str, threshold_score: float):
        """
        Analizza i risultati YOLO, identifica l'oggetto target specificato
        e aggiorna le variabili di stato (`target_detected`, `target_cx`, ecc.).

        Prioritizza il target e rilassa la soglia di confidenza se è già tracciato.

        Args:
            xyxy (np.ndarray): Coordinate dei bounding box (x1, y1, x2, y2).
            conf (np.ndarray): Punteggi di confidenza.
            cls_ids (np.ndarray): ID delle classi.
            image_h (int): Altezza dell'immagine.
            image_w (int): Larghezza dell'immagine.
            target_class (str): Nome della classe target da tracciare.
            threshold_score (float): Soglia di confidenza minima.
        """

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]

            # Calcola la confidenza della predizione corrente
            score = None    # TODO

            # Calcola la classe corrente
            cls_idx = None  # TODO
            cls_name = str(self.model_names.get(int(cls_idx), "unknown")).lower()

            # Debug: stampa la classe identificata
            # self.get_logger().info(f"Detected: {cls_name} (score={score:.2f})")

            # Prosegue solo se la classe corrente è la classe target e la confidenza è sufficientemente alta
            # TODO

            # IL target è stato localizzato
            self.target_confidence = None   # TODO
            self.target_last_seen_time = time.time()

            # Aggiorna il centro del target
            self.target_cx = None   # TODO
            self.target_cy = None   # TODO

            self.target_detected = None    # TODO

            return  

    # -----------------------------
    # CONTROL LOOP 
    # -----------------------------
    def control_loop(self):
        """
        Loop di controllo principale che esegue le seguenti funzionalità:
        1. Evitamento: se un ostacolo frontale viene rilevato, il robot ruota fino a che la traiettoria non è libera.
        2. Movimento laterale: quando la traiettoria risulta libera, il robot avanza per allontanarsi dall'ostacolo.
        3. Riallineamento: una volta allontanatosi dall'ostacolo, il robot si riallinea al punto target.
        4. Avanzamento: il robot avanza verso il punto target.
        """

        # Attende odometria
        if self.x is None or self.yaw is None:
            self.publish_stop()
            return
        
        if self.stop:
            self.publish_stop()
            return
            
        # Controllo di arrivo 
        if self.target_detected:

            # TODO: se la distanza corrente è minore della distanza di sicurezza:
            # a. pubblicare il comando di stop per il robot
            # b. impostare correttamente la flag self.stop
            # c. ritornare

            pass    # TODO               
            
        # Rileva ostacoli
        # TODO

        # EVITAMENTO (Ruota per liberare il fronte)
        if self.obstacle_detected:

            # TODO: scegli la direzione di rotazione come la direzione più libera (destra o sinistra)
            direction = None    # TODO
            
            # Ruota sul posto.
            self.publish_twist(0.0, 0.5 * direction)
            self.avoiding = True
            self.aligning = False 

            return

        # MOVIMENTO LATERALE (Avanza per allontanarsi dall'ostacolo)
        if self.avoiding:

            # Calcola la distanza dal punto di inizio evitamento
            dist = None     # TODO

            if None:    # TODO: controlla se la distanza è maggiore della distanza-soglia prefissata
                # Distanza di sfollamento laterale raggiunta -> Passa al riallineamento
                self.avoiding = False
                self.start_avoiding_x = self.start_avoiding_y = None
                self.aligning = True
            else:
                # Continua ad avanzare 
                pass    # TODO
            return

        # RIALLINEAMENTO (Orientamento verso la direzione del goal)
        if self.aligning:
            # Calcola l'orientamento target (l'orientamento della posizione di goal)
            target_yaw = None   # TODO

            # Calcola l'errore angolare rispetto a target_yaw
            angle_err = None    # TODO

            # Se l'errore è tollerabile, smette di ruotare e avanza. Altrimenti continua a ruotare
            if None:    # TODO
                # Ruota con controllo proporzionale
                angular_vel = max(min(0.8 * angle_err, self.turn_speed_max), -self.turn_speed_max)

                # TODO: ruota
            else:
                # Allineato -> Passa alla navigazione normale

                # TODO: avanza
                self.aligning = False
            return
        
        # NAVIGAZIONE NORMALE/AVANZAMENTO
        # TODO: avanza normalmente

        return


# -----------------------------
# MAIN
# -----------------------------
def main():
    """
    Funzione principale per l'esecuzione del nodo.
    Inizializza ROS 2, crea il nodo e avvia lo spin.
    Garantisce l'arresto del robot al termine.
    """
    rclpy.init()
    node = YoloObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()