#!/usr/bin/env python3
"""
Nodo ROS 2 per la Rilevazione e il Tracciamento di Oggetti
utilizzando YOLOv8 (Ultralytics) su un robot mobile (come LIMO).

Questo nodo si iscrive ai topic della telecamera (immagine a colori e profondità),
esegue l'inferenza YOLO per identificare un oggetto target specificato e,
in base alla sua posizione e distanza, invia comandi di velocità per
tracciare l'oggetto o fermarsi se è troppo vicino.
"""
import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import time
import cv2                                 # type: ignore
import numpy as np                         # type: ignore
import rclpy                               # type: ignore
from rclpy.node import Node                # type: ignore
            
from sensor_msgs.msg import Image          # type: ignore
from cv_bridge import CvBridge             # type: ignore
from geometry_msgs.msg import Twist        # type: ignore

from vision_msgs.msg import (              # type: ignore
    Detection2D,
    ObjectHypothesisWithPose,
    BoundingBox2D
)

try:
    from ultralytics import YOLO           # type: ignore
except ImportError:
    YOLO = None


class LimoYoloNode(Node):
    """
    Nodo principale ROS 2 per l'elaborazione delle immagini e il controllo
    del movimento basato sulle rilevazioni YOLO.

    Gestisce l'inferenza YOLO e il loop di controllo 
    per il tracciamento e l'arresto di sicurezza.
    """
    def __init__(self):
        super().__init__("limo_yolo")

        # ----------------------------------------------------
        # Parametri
        # ----------------------------------------------------
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("model", "yolov8n.pt")
        self.declare_parameter("score_thresh", 0.5)               # Soglia di confidenza minima per non scartare una predizione
        self.declare_parameter("filter_class", "sports ball")     # Classe target da individuare: può essere cambiata a "elephant" (# TODO)
        self.declare_parameter("bb_ratio_threshold", 0.03)        # Rapporto tra area della bounding box e area dell'immagine necessario al robot per fermarsi
        self.declare_parameter("turn_speed_max", 1.5)             # Velocità di rotazione massima

        self.img_topic = self.get_parameter("image_topic").value
        self.model_path = self.get_parameter("model").value
        self.score_thresh = self.get_parameter("score_thresh").value
        self.filter_class = self.get_parameter("filter_class").value.lower()
        self.bb_ratio_threshold = self.get_parameter("bb_ratio_threshold").value
        self.turn_speed_max = self.get_parameter("turn_speed_max").value

        # ----------------------------------------------------
        # Variabili di stato
        # ----------------------------------------------------

        # PERCEZIONE E AI (YOLO & CV)
        self.bridge = CvBridge()
        self.model = None               # Modello YOLOv8 caricato in load_model()
        self.model_names = {}           # Dizionario ID -> Nome Classe (es. 0: 'person')
        
        # GEOMETRIA IMMAGINE (Dimensioni sensore)
        self.image_w = None             # Larghezza immagine (px) 
        self.image_h = None             # Altezza immagine (px) 

        # STATO DEL TARGET (Tracking)
        self.target_detected = False    # Flag principale: il target è attualmente visibile?
        self.target_confidence = 0.0    # Confidenza (0.0 - 1.0) dell'ultima rilevazione
        self.target_last_seen_time = time.time() # Timestamp per gestire il timeout (target perso)
        
        # Variabili di posizione (calcolate in detect_target)
        self.target_cx = 0              # Coordinata X del centro del box (pixel)
        self.target_cy = 0              # Coordinata Y del centro del box (pixel)
        self.target_box_ratio = 0.0     # Area occupata dal box rispetto all'immagine (0.0 - 1.0)

        # LOGICA DI CONTROLLO E SICUREZZA
        self.stopped = False            # Stato di blocco (dopo aver raggiunto l'obiettivo)
        
        # Parametri per il centraggio 
        # La "deadband" evita che il robot oscilli continuamente a destra e sinistra
        self.center_deadband_ratio = 0.1 # Tolleranza centrale (10% della larghezza immagine)
        self.target_centered = False    # True se il target è dentro la zona centrale
        self.center_turn_speed = 0.4    # Velocità angolare di rotazione (rad/s)

        # ----------------------------------------------------
        # ROS I/O
        # ----------------------------------------------------
        self.sub_image = self.create_subscription(      
            Image, self.img_topic, self.on_image, 10            # Riceve le immagini dalla camera del robot
        )

        self.cmd_pub = self.create_publisher(
            Twist, "/cmd_vel", 10                               # Invia comandi di velocità al robot
        )

        self.pub_image = self.create_publisher(
            Image,
            "/yolo/annotated_image",                            # Pubblica le immagini con bounding box sovrapposta
            10
        )

        # Control loop (20 Hz)
        self.control_timer = self.create_timer(
            0.05, self.control_loop
        )

        # Carica YOLO
        self.load_model()
        self.get_logger().info(
            f"Limo YOLO node ready. Tracking: {self.filter_class}"
        )

    # ----------------------------------------------------
    # CARICAMENTO YOLO
    # ----------------------------------------------------
    def load_model(self):
        """
        Carica il modello YOLOv8 specificato dal parametro 'model'.
        Esegue la fusione dei modelli per un'inferenza più veloce.

        Raises:
            RuntimeError: Se il pacchetto 'ultralytics' non è installato.
        """
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO not installed")

        t0 = time.time()
        self.model = YOLO(self.model_path)
        self.model.fuse()
        self.model_names = self.model.names

        dt = time.time() - t0
        self.get_logger().info(
            f"Loaded YOLO model '{self.model_path}' in {dt:.2f}s"
        )

    # ----------------------------------------------------
    # CALLBACK DELLA CAMERA
    # ----------------------------------------------------
    def on_image(self, msg: Image):
        """
        Callback per la ricezione delle immagini a colori.

        Esegue i seguenti passaggi:
        1. Conversione dell'immagine ROS in formato OpenCV (BGR).
        2. Esecuzione dell'inferenza YOLO.
        3. Aggiornamento dello stato di rilevamento del target tramite `detect_target`.
        4. Disegno del bounding box e del centro sull'immagine se il target è rilevato.
        5. Pubblicazione dell'immagine annotata.

        Args:
            msg (Image): Messaggio immagine ROS 2.
        """
        
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failed: {e}")
            return

        h, w = img_bgr.shape[:2]

        if self.image_w is None:
            self.image_w = w

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Run YOLO inference
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

        # Sovrappone bounding box sull'immagine se l'oggetto target è stato individuato
        if self.target_detected:

            x = int(self.target_cx)
            y = int(self.target_cy)

            
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                cls_name = str(self.model_names.get(int(cls_ids[i]), "unknown")).lower()

                if cls_name == self.filter_class:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # Contorno della bounding box
                    cv2.rectangle(
                        img_bgr,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Centro della bounding box
                    cv2.circle(
                        img_bgr,
                        (self.target_cx, self.target_cy),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    # Aggiunge la label (etichetta) relativa alla classe
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

            # Pubblica l'immagine con annotazione (bounding box)
            out_msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
            self.pub_image.publish(out_msg)


    def detect_target(self, xyxy: np.ndarray, conf: np.ndarray, cls_ids: np.ndarray, image_h: int, image_w: int,
                    target_class: str, threshold_score: float):
        """
        Analizza i risultati YOLO, identifica il target specificato e aggiorna
        le variabili di stato del nodo (`target_detected`, `target_cx`, ecc.).

        Applica una logica di rilassamento della soglia (metà `threshold_score`)
        se l'oggetto è già stato precedentemente rilevato per migliorarne la stabilità.

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

            # Estrae la confidenza per l'elemento corrente
            score = None    # TODO
            
            # Considera l'elemento corrente solo se la soglia di confidenza è sufficientemente elevata, 
            # altrimenti scarta l'elemento e passa al prossimo
            # TODO
            
            # Rilassa la soglia di confidenza se il bersaglio è stato già individuato (per stabilità)
            if self.target_detected and score < (threshold_score / 2):
                continue

            # L'obiettivo è stato individuato correttamente
            self.target_confidence = None   # TODO
            self.target_last_seen_time = time.time()

            # Calcola il centro della bounding box
            # TODO: salvare le coordinate del centro in (self.target_cx, self.target_cy)


            # Calcola larghezza e altezza del box in pixel

            self.target_box_ratio = None    # TODO

            # Debug log per vedere quanto spazio occupa
            self.get_logger().info(f"Target ratio: {self.target_box_ratio:.4f}")

            self.target_detected = None     # TODO

            return

    # Control loop
    def control_loop(self):
        """
        Loop di controllo periodico (20 Hz) che determina i comandi di movimento.

        Logica:
        1. Se il target è perso per più di 1 secondo, entra in modalità ricerca.
        2. Se il target non è rilevato, il robot ruota sul posto (cerca).
        3. Se il target è rilevato:
            a. Allinea il robot con il target, in modo che il centro della bounding box corrisponda al 
               centro dell'immagine della camera.
            b. Controlla il rapporto tra area della bounding box e area dell'immagine.
            c. Se troppo vicino, pubblica l'arresto e imposta `self.stopped = True`.
        4. Altrimenti, il robot avanza.
        """
        now = time.time()

        if self.stopped:
            return

        # Se il bersaglio non è individuato da almeno un secondo, è considerato perso
        if True:    # TODO
            self.target_detected = False
            self.target_box_ratio = None
            self.target_centered = False

        # Se il bersaglio non è stato individuato, ruota sul posto
        if None:    # TODO
            self.stopped = False

            # TODO: ruotare

            return
        
        # Altrimmenti, allinea il robot al centro della bounding box
        if not self.target_centered:
            error = None    # TODO

            # Deadband: il target è stato centrato
            if None:    # TODO: controlla se il target è stato centrato
                
                self.publish_stop()

                self.get_logger().info(f"TARGET CENTERED")

                self.target_centered = True

                return

            # Rotazione proporzionale
            angular_z = self.center_turn_speed * error

            # Calcola la velocità angolare
            angular_z = max(min(angular_z, self.turn_speed_max),
                            -self.turn_speed_max)

            # Ruota
            # TODO: usare angular_z

            return

        # Controlla se la bounding box è abbastanza grande -> il robot è abbastanza vicino
        if self.target_box_ratio is not None:

            # TODO: se il rapporto tra bounding box e area dell'immagine è abbastanza grande: 
            # a. Pubblicare messaggio di stop per il robot
            # b. Aggiornare la flag self.stopped
            # c. Ritornare

            pass    # TODO


        # Altrimenti avanza
        self.stopped = False
        
        # TODO: avanza

    # ----------------------------------------------------
    # METODI DI SUPPORTO
    # ----------------------------------------------------
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
    node = LimoYoloNode()

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