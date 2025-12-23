#!/usr/bin/env python3
"""
Nodo ROS 2 per pilotare un robot mobile (ad esempio un TurtleBot3)
per seguire percorsi predefiniti come un quadrato, un poligono o un cerchio,
utilizzando il feedback di odometria.
"""
import rclpy                                                                # type: ignore
from rclpy.node import Node                                                 # type: ignore
import threading
import math
import time

from geometry_msgs.msg import Twist                                         # type: ignore
from nav_msgs.msg import Odometry                                           # type: ignore
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy    # type: ignore


# -------------------------------------------------------------------
# Funzioni di supporto
# -------------------------------------------------------------------
def quaternion_to_yaw(qx, qy, qz, qw):
    """
    Calcola l'imbardata (rotazione attorno a Z) da un quaternione.

    Args:
        qx (float): Componente X del quaternione.
        qy (float (float): Componente Y del quaternione.
        qz (float): Componente Z del quaternione.
        qw (float): Componente W del quaternione.

    Returns:
        float: Angolo di imbardata in radianti.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def normalize_angle(angle):
    """
    Normalizza un angolo all'intervallo [-pi, pi).

    Args:
        angle (float): L'angolo in radianti.

    Returns:
        float: L'angolo normalizzato in radianti.
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


# -------------------------------------------------------------------
# Nodo ROS principale
# -------------------------------------------------------------------
class Turtlebot3SquarePath(Node):
    """
    Un nodo ROS 2 che controlla un TurtleBot3 per eseguire percorsi geometrici di base 
    utilizzando l'odometria.

    Gestisce la ricezione dell'odometria e la pubblicazione dei comandi di velocità.
    """
    def __init__(self):
        super().__init__('turtlebot3_square_path_node')

        # QoS per odometria e velocità
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers/Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos)

        # Parametri
        self.rate_hz = 20.0

        # Variabili di stato per odometria
        self.x = None
        self.y = None
        self.yaw = None

        # Flag di controllo
        self.is_running = True

        self.get_logger().info('Node initialized.')

    # -------------------------------------------------------------------
    # Callbacks e metodi di supporto
    # -------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        """
        Estrazione della posa di odometria (x, y, yaw) dal messaggio Odometry.

        Args:
            msg (Odometry): Messaggio di odometria ROS 2.
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

    def publish_twist(self, linear=0.0, angular=0.0):
        """
        Pubblica un comando di velocità lineare e angolare sul topic /cmd_vel.

        Args:
            linear (float): Velocità lineare in X (m/s).
            angular (float): Velocità angolare in Z (rad/s).
        """
        t = Twist()
        t.linear.x = float(linear)
        t.angular.z = float(angular)
        self.cmd_pub.publish(t)

    def publish_stop(self):
        """Pubblica un comando di velocità nullo (arresto del robot)."""
        self.publish_twist(0.0, 0.0)

    def wait_for_odom(self, timeout=5.0):
        """
        Attende fino a quando non vengono ricevute le prime coordinate di odometria (x, y, yaw).

        Args:
            timeout (float): Tempo massimo di attesa in secondi.

        Returns:
            bool: True se l'odometria è stata ricevuta, False altrimenti.
        """
        start = time.time()
        while rclpy.ok() and (self.x is None or self.yaw is None):
            if time.time() - start > timeout:
                return False
            time.sleep(0.05)
        return True

    # -------------------------------------------------------------------
    # Primitive di movimento
    # -------------------------------------------------------------------
    def move_distance(self, distance, linear_vel):
        """
        Avanza o retrocede per una data distanza utilizzando l'odometria per il feedback.

        Il movimento si interrompe quando la distanza target è stata coperta.

        Args:
            distance (float): Distanza da percorrere (m). Positivo per avanti, negativo per indietro.
            linear_vel (float): Velocità lineare massima (m/s).
        """
        if not self.wait_for_odom():
            self.get_logger().warn('No odom — aborting move_distance().')
            return

        start_x, start_y = self.x, self.y
        target = abs(distance)
        direction = 1.0 if distance >= 0 else -1.0
        rate_dt = 1.0 / self.rate_hz

        self.get_logger().info(f'Moving {("forward" if direction > 0 else "backward")} {target:.2f} m')

        while rclpy.ok() and self.is_running:
            dx = self.x - start_x
            dy = self.y - start_y
            traveled = math.hypot(dx, dy)

            if traveled >= target:
                break

            remaining = max(target - traveled, 0.0)
            # Controllo proporzionale per decelerare alla fine
            speed = min(linear_vel, remaining * 1.0 + 0.05)

            self.publish_twist(linear=direction * speed)
            time.sleep(rate_dt)

        self.publish_stop()
        time.sleep(0.05)

    def rotate_angle(self, angle_rad, angular_vel):
        """
        Ruota di un angolo specificato (rad) utilizzando il feedback di imbardata (yaw)
        dell'odometria.

        La rotazione si interrompe quando l'angolo target è raggiunto.

        Args:
            angle_rad (float): Angolo di rotazione in radianti. Positivo per CCW.
            angular_vel (float): Velocità angolare massima (rad/s).
        """
        if not self.wait_for_odom():
            self.get_logger().warn('No odom — aborting rotate_angle().')
            return

        start_yaw = self.yaw
        target_yaw = normalize_angle(start_yaw + angle_rad)
        direction = 1.0 if angle_rad >= 0 else -1.0
        rate_dt = 1.0 / self.rate_hz

        self.get_logger().info(f'Rotating {math.degrees(angle_rad):.1f} degrees')

        while rclpy.ok() and self.is_running:
            if self.yaw is None:
                time.sleep(rate_dt)
                continue

            error = normalize_angle(target_yaw - self.yaw)
            if abs(error) < math.radians(1.5): # Tolleranza di 1.5 gradi
                break

            k = 1.2 # Guadagno Proporzionale (P)
            angular_speed = min(angular_vel, max(0.05, abs(k * error)))

            self.publish_twist(angular=direction * angular_speed)
            time.sleep(rate_dt)

        self.publish_stop()
        time.sleep(0.05)

    # -------------------------------------------------------------------
    # Funzioni di alto livello (TODO)
    # -------------------------------------------------------------------
    def run_square_path(self, side_length, duration, per_rot_duration=0.3):
        """
        Esegue un percorso quadrato con 4 lati.

        La velocità lineare e angolare sono calcolate in base alla durata totale.

        Args:
            side_length (float): Lunghezza di un lato del quadrato (m).
            duration (float): Durata totale desiderata del percorso (s).
            per_rot_duration (float): Frazione della durata totale dedicata alla rotazione
                                      (e.g., 0.3 significa 30% del tempo totale).
        """

        # Attende odometria
        if not self.wait_for_odom(timeout=10.0):
            self.get_logger().error('No odom received — cannot start square.')
            return

        self.get_logger().info('Starting square path.')

        # Inizializza numero di lati e ampiezza degli angoli del quadrato
        n_sides = 4
        angle = math.pi / 2 # 90 gradi in radianti

        # Allocazione del tempo
        # TODO: calcolare il tempo da dedicare a ogni rettilineo e ad ogni rotazione

        # Calcolo delle velocità
        # TODO: calcolare velocità lineare e angolare in base al tempo, alla lunghezza dei lati e all'ampiezza degli angoli


        # Alterna rettilinei e rotazioni per seguire un percorso quadrato
        for i in range(n_sides):
            if not rclpy.ok() or not self.is_running:
                break

            self.get_logger().info(f'Side {i+1}/4: move {side_length:.2f} m')
            
            # TODO: muovere il robot in rettilineo per la lunghezza del lato

            self.get_logger().info(f'Side {i+1}/4: rotate 90°')

            # TODO: far ruotare il robot per l'ampiezza dell'angolo (in radianti)

        self.publish_stop()
        self.get_logger().info('Completed square path.')


    def follow_polygon(self, n_edges, side_length, duration, per_rot_duration=0.3):
        """
        Esegue un percorso poligonale di n lati.

        La velocità lineare e angolare sono calcolate in base alla durata totale.

        Args:
            n_edges (int): Numero di lati del poligono.
            side_length (float): Lunghezza di un lato del poligono (m).
            duration (float): Durata totale desiderata del percorso (s).
            per_rot_duration (float): Frazione della durata totale dedicata alla rotazione.
        """

        # Attende odometria
        if not self.wait_for_odom(timeout=10.0):
            self.get_logger().error('No odom — cannot start polygon.')
            return

        self.get_logger().info(f'Starting polygon with {n_edges} edges.')

        # Allocazione del tempo
        # TODO: calcolare il tempo da dedicare a ogni rettilineo e ad ogni rotazione

        # Calcolo dell'angolo del poligono
        external_angle = None # TODO

        # Calcolo delle velocità
        # TODO: calcolare velocità lineare e angolare

        for i in range(n_edges):
            if not rclpy.ok() or not self.is_running:
                break

            # TODO: muovere il robot in rettilineo (lato)
            # TODO: far ruotare il robot (angolo)

        self.publish_stop()
        self.get_logger().info('Completed polygon path.')


    def follow_circle(self, radius, duration):
        """
        Esegue un percorso circolare a velocità costante (velocità lineare/angolare)
        per una data durata.

        Args:
            radius (float): Raggio del cerchio (m).
            duration (float): Durata totale del percorso (s).
        """
        if not self.wait_for_odom(timeout=10.0):
            self.get_logger().error('No odom — cannot start circle.')
            return

        self.get_logger().info(f'Starting circular path: radius={radius} m, duration={duration}s')

        # Calcola le velocità per completare un giro in 'duration' secondi.
        linear_vel = None  # TODO
        angular_vel = None # TODO

        rate_dt = 1.0 / self.rate_hz
        steps = int(duration * self.rate_hz)

        # Segue il percorso
        for _ in range(steps):
            if not rclpy.ok() or not self.is_running:
                break

            # TODO

            time.sleep(rate_dt)

        self.publish_stop()
        self.get_logger().info('Completed circular path.')


# -------------------------------------------------------------------
# Main entry
# -------------------------------------------------------------------
def main(args=None):
    """
    Funzione principale per l'esecuzione del nodo.
    Inizializza ROS 2, crea il nodo, esegue lo spin in un thread
    separato ed esegue la traiettoria desiderata.
    """
    rclpy.init(args=args)
    node = Turtlebot3SquarePath()

    # Thread in background per ricevere odometria
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        # node.run_square_path(side_length=0.5, duration=12.0)
        node.follow_polygon(n_edges=6, side_length=0.4, duration=10.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.is_running = False
        node.publish_stop()

        node.get_logger().info('Shutting down.')

        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()