#!/usr/bin/env python3
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


class FiniteStateMachine(Node):
    def __init__(self):
        super().__init__("limo_yolo")

        # PARAMETRI ROS
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("model", "yolov8n.pt")
        self.declare_parameter("score_thresh", 0.7)             # Soglia di confidenza minima da soddisfare
        self.declare_parameter("filter_class", "sports ball")   # Classe target da individuare
        self.declare_parameter("turn_speed_max", 1.5)           # Velocità angolare massima
        self.declare_parameter("bb_ratio_threshold", 0.03)      # Rapporto tra bounding box e immagine da usare come proxy di distanza
        self.declare_parameter("search_straight_distance", 0.8) # Spazio da percorrere in rettilineo in fase FORWARD

        self.img_topic = self.get_parameter("image_topic").value
        self.model_path = self.get_parameter("model").value
        self.score_thresh = self.get_parameter("score_thresh").value
        self.filter_class = self.get_parameter("filter_class").value.lower()
        self.turn_speed_max = self.get_parameter("turn_speed_max").value
        self.bb_ratio_threshold = self.get_parameter("bb_ratio_threshold").value
        self.search_straight_distance = self.get_parameter("search_straight_distance").value

        # CAMERA E IMMAGINE
        self.image_w = None         # Ampiezza immagine (width)

        self.bridge = CvBridge()

        # TARGET
        self.model = None
        self.model_names = {}

        self.target_detected = False    # Indica se il target è individuato
        self.target_centered = False    # Indica se il target è stato centrato
        self.target_box_ratio = None    # Rapporto corrente tra area della bounding box e dell'immagine
        self.target_last_seen_time = time.time() # Timestamp per gestire il timeout (target perso)

        # CONTROLLO MOVIMENTO
        self.rate_hz = 20
        self.forward_speed = 0.3        # Velocità lineare
        self.center_deadband_ratio = 0.1      # ±10% of image width
        self.center_turn_speed = 0.4
        self.align_tol = 0.05           # Errore tollerato in fase di riallineamento

        # VARIABILE DI STATO
        self.state = "FORWARD"  # Indica lo stato interno della FSM

        # ODOMETRIA
        # Odometria corrente
        self.x = None
        self.y = None
        self.yaw = None

        # Odometria di partenza
        self.starting_x = None
        self.starting_y = None
        self.starting_yaw = None

        # Odometria di partenza per la fase di ricerca
        self.search_start_x = None
        self.search_start_y = None
        self.search_start_yaw = None

        # Variabili di stato per la fase di "SCAN"
        self.turning_right = True
        self.turning_left = False
        self.realigning = False

        self.right_target_yaw = None
        self.left_target_yaw = None
        self.realigning_target_yaw = None

        # Odometria di partenza in fase di evitamento ostacolo
        self.start_avoiding_x = None
        self.start_avoiding_y = None
        self.start_avoiding_yaw = None

        # Logica di stop
        self.stopped = False

        # -----------------------------
        # ROS I/O
        # -----------------------------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos)    # Riceve odometria
        self.sub_image = self.create_subscription(Image, self.img_topic, self.on_image, 10)     # Riceve le immagini della camera
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)                             # Invia le velocità
        self.pub_image = self.create_publisher(Image, "/yolo/annotated_image", 10)              # Pubblica le immagini annotate con bounding box

        self.control_timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)           # Ciclo di controllo principale

        self.load_model()
        self.get_logger().info(f"Node loaded. Tracking: {self.filter_class}")

    # -----------------------------
    # CARICAMENTO MODELLO YOLO
    # -----------------------------
    def load_model(self):
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
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

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

        if self.image_w is None:
            self.image_w = w

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Lancia il modello YOLO
        results = self.model.predict(
            img_rgb,
            imgsz=320,
            conf=self.score_thresh,
            verbose=False
        )

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

                    # Disegna il contorno della bounding box
                    cv2.rectangle(
                        img_bgr,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Disegna il punto centrale
                    cv2.circle(
                        img_bgr,
                        (self.target_cx, self.target_cy),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    # Aggiunge l'etichetta della classe
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

            # Pubblica l'immagine annotata
            out_msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
            self.pub_image.publish(out_msg)

    # -----------------------------
    # METODI DI CONTROLLO
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
    
    def compute_center_error(self):
        """
        Calcola l'errore orizzontale normalizzato rispetto al centro dell'immagine:
        -1.0 = il target si trova all'estrema sinistra
         0.0 = il target è perfettamente al centro
        +1.0 = il target si trova all'estrema destra
        
        Ritorna:
            float: Errore normalizzato nell'intervallo [-1.0, 1.0].
        """
        image_center_x = self.image_w / 2.0
        return (self.target_cx - image_center_x) / image_center_x

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
    # RILEVAMENTO TARGET
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

            score = None    # TODO
            cls_idx = None  # TODO
            cls_name = str(self.model_names.get(int(cls_idx), "unknown")).lower()

            # Debug
            # self.get_logger().info(f"Detected: {cls_name} (score={score:.2f})")

            # Prosegue solo se classe e confidenza sono corrette
            # TODO

            # Il target è stato individuato
            self.target_confidence = None   # TODO
            self.target_last_seen_time = time.time()

            # Aggiorna il centro del target
            self.target_cx = None   # TODO
            self.target_cy = None   # TODO

            # Calcola il rapporto di area 
            self.target_box_ratio = None    # TODO

            # Debug log 
            # self.get_logger().info(f"Target ratio: {self.target_box_ratio:.4f}")

            self.target_detected = None     # TODO

            return


    # -----------------------------
    # METODI DI STATO
    # -----------------------------
    def move_forward(self):

        """
        Gestisce la fase di avanzamento rettilineo durante la modalità di ricerca.
        
        Il metodo esegue le seguenti operazioni:
        1. Verifica che lo stato corrente sia effettivamente "FORWARD".
        2. Memorizza la posizione iniziale (odometria) al primo avvio della manovra.
        3. Calcola la distanza percorsa rispetto al punto di partenza.
        4. Se la distanza percorsa è uguale o superiore a 'search_straight_distance':
           - Arresta il robot.
           - Resetta tutti i parametri di navigazione e orientamento.
           - Passa allo stato "SCAN" per iniziare la rotazione di ricerca.
        5. Se la distanza non è stata ancora raggiunta, continua a pubblicare
           un comando di velocità lineare costante.
        """
        
        # Controllo di sicurezza: se lo stato interno non è FORWARD -> ritorna subito
        # TODO

        # Salva odometria di partenza  
        if self.search_start_x is None:
            self.search_start_x = None      # TODO
            self.search_start_y = None      # TODO
            self.search_start_yaw = None    # TODO

        # Avanza per la distanza indicata, poi passa allo stato "SCAN"
        d = None    # TODO: calcola la distanza percorsa da (self.search_start_x, search_start_y)

        if d >= self.search_straight_distance:
            
            # TODO: invia comando per fermare il robot

            # Resetta i parametri di scansione 
            self.turning_right = True
            self.turning_left = False
            self.realigning = False

            self.right_target_yaw = None
            self.left_target_yaw = None
            self.realigning_target_yaw = None

            # Resetta i parametri di stato
            self.search_start_x = None
            self.search_start_y = None
            self.search_start_yaw = None

            self.state = None   # TODO

            return

        # Altrimenti, prosegue dritto
        # TODO

        return

    def scan(self):
        """
        Esegue una manovra di scansione angolare sul posto per cercare il target.
        
        La logica segue una sequenza di tre fasi:
        1. ROTAZIONE A DESTRA: Ruota il robot di 45 gradi verso destra rispetto 
           all'orientamento iniziale.
        2. ROTAZIONE A SINISTRA: Una volta completata la destra, ruota fino a 
           45 gradi a sinistra rispetto all'orientamento di partenza (arco totale di 90°).
        3. RIALLINEAMENTO: Torna all'orientamento originale memorizzato all'inizio 
           della scansione.
        
        Al termine del riallineamento, lo stato viene impostato su "FORWARD" per 
           proseguire l'esplorazione in linea retta.
        """
        # Controllo di sicurezza: se lo stato interno non è SCAN -> ritorna immediatamente
        # TODO

        # Gira a destra
        if self.turning_right:

            if self.right_target_yaw is None:
                self.start_spinning_yaw = None  # TODO: salva l'orientamento corrente per il successivo riallineamento
                self.right_target_yaw = self.yaw - math.pi / 4

            # Se ha raggiunto l'ampiezza desiderata, smette di ruotare
            if None:   # TODO: l'errore angolare deve essere inferiore a self.align_tol
                
                # TODO: ferma il robot

                # Aggiorna le variabili interne
                self.right_target_yaw = None
                self.turning_left = True
                self.turning_right = False

                return
            
            # Altrimenti, ruota verso DESTRA
            self.publish_twist(0.0, -0.2)

            return
        
        # Gira a sinistra
        if self.turning_left:

            if self.left_target_yaw is None:
                self.left_target_yaw = self.start_spinning_yaw + math.pi / 4

            # Se ha raggiunto l'ampiezza desiderata, smette di ruotare
            if None:    # TODO
                
                # TODO: ferma il robot

                self.left_target_yaw = None
                self.realigning = True
                self.turning_left = False

                return
            
            # Altrimenti, ruota verso SINISTRA
            self.publish_twist(0.0, 0.2)
            return
        
        # Riallineamento a orientamento di partenza
        if self.realigning:
            
            # Salva orientamento di partenza per la fase di riallineamento
            if self.realigning_target_yaw is None:
                self.realigning_target_yaw = self.start_spinning_yaw

            # Se ha completato il riallineamento: smette di ruotare e torna allo stato FORWARD
            if None:    # TODO
                self.publish_stop()
                self.realigning_target_yaw = None
                self.realigning = False

                # Resetta le variabili relative allo stato FORWARD
                self.search_start_x = None
                self.search_start_y = None
                self.search_start_yaw = None

                self.state = None   # TODO

                return
            
            # Altrimenti, gira a destra
            self.publish_twist(0.0, -0.2)
            return

        return


    def approach(self):

        """
        Gestisce l'avvicinamento finale al target rilevato e centrato.
        
        Il metodo monitora costantemente il rapporto tra l'area della bounding box 
        e l'area totale dell'immagine (target_box_ratio). 
        - Se il rapporto supera la soglia 'bb_ratio_threshold', il robot considera 
          l'obiettivo raggiunto e passa allo stato "INTERACT".
        - Altrimenti, continua ad avanzare verso il target a velocità costante.
        """
        
        if self.target_box_ratio is not None:

            # TODO: se il robot è abbastanza vicino al target:
            # a. ferma il robot
            # b. cambia stato interno
            # c. ritorna

            pass    # TODO

        # Altrimenti prosegue dritto 
        self.publish_twist(self.forward_speed, 0.0)

        return

    def interact(self):
        """
        Conclude la missione una volta raggiunto l'obiettivo.
        
        Esegue le seguenti azioni finali:
        1. Arresta completamente ogni movimento del robot.
        2. Invia un log informativo indicando il completamento del task.
        3. Imposta il flag 'self.stopped' su True per interrompere il loop di controllo.
        """

        # TODO

        return

    def align_to_target(self):
        """
        Esegue l'allineamento del robot verso l'obiettivo rilevato (Visual Servoing).
        
        Il metodo utilizza l'errore di centraggio calcolato sui pixel dell'immagine:
        1. Calcola l'errore normalizzato tra il centro della bounding box e il 
           centro del frame della camera.
        2. Verifica se l'errore rientra nella 'zona morta' (center_deadband_ratio):
           - Se centrato: arresta la rotazione, imposta il flag 'target_centered' 
             e passa allo stato "APPROACH".
        3. Se non centrato: applica una velocità angolare proporzionale all'errore, 
           limitata dal valore massimo 'turn_speed_max', per orientare il robot 
           verso il target.
        """

        if self.image_w is None or not self.target_detected:
            self.publish_stop()
            return

        error = None    # TODO: calcola l'errore tra centro della bounding box e centro della camera

        # Deadband: il target è centrato
        if None:    # TODO
            self.publish_stop()

            # Debug
            # self.get_logger().info(f"TARGET CENTERED: switching to APPROACH")

            self.target_centered = None     # TODO

            self.state = None   # TODO

            return

        # Rotazione proporzionale
        angular_z = self.center_turn_speed * error

        # Calcola la velocità angolare
        angular_z = max(min(angular_z, self.turn_speed_max),
                        -self.turn_speed_max)

        # TODO: ruota il robot


    # -----------------------------
    # CONTROL LOOP
    # -----------------------------
    def control_loop(self):
        """
        Ciclo di controllo principale eseguito a 20 Hz. 
        Gestisce la logica decisionale del robot e le transizioni tra gli stati.

        Il metodo opera secondo la seguente gerarchia di priorità:
        1. VALIDAZIONE DATI: Verifica la disponibilità di odometria e dati LiDAR.
        2. SICUREZZA: Se il robot è in stato di arresto ('stopped'), pubblica velocità nulla.
        3. GESTIONE TARGET PERSO: Se il target non viene rilevato per più di 1 secondo, 
           resetta i parametri di inseguimento e forza il ritorno allo stato "SCAN".
        4. AGGANCIO TARGET: Se il target è visibile ma non centrato, imposta lo 
           stato "CENTER_TARGET" (a meno che non si sia già in fase di approccio finale).
        5. ESECUZIONE FSM: Smista l'esecuzione ai metodi specifici ("AVOID", "APPROACH", 
           "FORWARD", "SCAN", "CENTER_TARGET", "INTERACT") in base allo stato attivo.
        """

        now = time.time()

        # Attende odometria
        if self.x is None or self.yaw is None or self.ranges is None:
            self.publish_stop()
            return
        
        if self.stopped:
            self.publish_stop()
            return
        
        # Se il bersaglio non è individuato da almeno un secondo, è considerato perso
        if now - self.target_last_seen_time > 1 and self.target_detected:
            self.target_detected = False
            self.target_box_ratio = None
            self.target_centered = False

            # Resetta le variabili di stato SCAN
            self.turning_right = True
            self.turning_left = False
            self.realigning = False

            self.right_target_yaw = None
            self.left_target_yaw = None
            self.realigning_target_yaw = None

            self.state = "SCAN"

            return
        
        # Se il target viene rilevato, occorre centrarlo e avanzare
        if not self.target_centered:
            if self.target_detected and self.state not in ["APPROACH", "INTERACT"]:
                self.state = "CENTER_TARGET"


        # Logica di macchina a stati finiti (FSM)
        # TODO: chiama il metodo appropriato in base allo stato corrente.
        # Stati possibili:
        # a. FORWARD: muove il robot in rettilineo
        # b. SCAN: esegue scansione
        # c. CENTER_TARGET: centra il target nell'immagine della camera
        # d. APPROACH: avanza in rettilineo verso il target
        # e. INTERACT: esegue un'azione una volta raggiunto il target (es. stampa un messaggio)
        
        return

# -----------------------------
# MAIN
# -----------------------------
def main():
    rclpy.init()
    node = FiniteStateMachine()
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
