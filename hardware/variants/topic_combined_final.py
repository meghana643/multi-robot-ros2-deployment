#!/usr/bin/env python3
# Combined controller:
# 1. Rotation with closest XY point lock
# 2. Lateral movement with closest X lock
# 3. Forward/reverse movement with closest Y lock

import RPi.GPIO as GPIO
import time
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from std_msgs.msg import String

from rclpy.qos import QoSProfile
from rclpy.qos import HistoryPolicy
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import DurabilityPolicy


GPIO.setmode(GPIO.BCM)

FL_IN1, FL_IN2 = 17, 27
FR_IN1, FR_IN2 = 22, 23
RL_IN1, RL_IN2 = 5, 6
RR_IN1, RR_IN2 = 13, 26

ENA_FRONT = 18
ENB_FRONT = 19
ENA_REAR = 12
ENB_REAR = 16

pins = [
    FL_IN1, FL_IN2, FR_IN1, FR_IN2,
    RL_IN1, RL_IN2, RR_IN1, RR_IN2,
    ENA_FRONT, ENB_FRONT, ENA_REAR, ENB_REAR
]

for p in pins:
    GPIO.setup(p, GPIO.OUT)


PWM_FREQ = 100

pwm_fl = GPIO.PWM(ENA_FRONT, PWM_FREQ)
pwm_fr = GPIO.PWM(ENB_FRONT, PWM_FREQ)
pwm_rl = GPIO.PWM(ENA_REAR, PWM_FREQ)
pwm_rr = GPIO.PWM(ENB_REAR, PWM_FREQ)

pwm_fl.start(0)
pwm_fr.start(0)
pwm_rl.start(0)
pwm_rr.start(0)


# ============================================================
# ROTATION CONTROLLER PARAMETERS
# ============================================================

ROT_POSITION_TOL = 0.015

ROT_FORWARD_TOL = 0.010
ROT_LATERAL_TOL = 0.010

ROT_FORWARD_KP = 2.0
ROT_LATERAL_KP = 2.0

ROT_FORWARD_MIN_SPEED = 12
ROT_FORWARD_MAX_SPEED = 15

ROT_LATERAL_MIN_SPEED = 45
ROT_LATERAL_MAX_SPEED = 65

ROT_YAW_SPEED = 25

ROT_FORWARD_SIGN = 1.0
ROT_LATERAL_SIGN = -1.0


# ============================================================
# LATERAL CONTROLLER PARAMETERS
# ============================================================

LAT_STRAFE_SPEED = 100

LAT_TARGET_YAW_DEG = 0.0
LAT_YAW_TOL_DEG = 3

LAT_YAW_KP = 5.0
LAT_YAW_MIN_SPEED = 18
LAT_YAW_MAX_SPEED = 22

LAT_X_TOL = 0.005

LAT_X_KP = 3.0
LAT_X_MIN_SPEED = 16
LAT_X_MAX_SPEED = 22


# ============================================================
# FORWARD / REVERSE CONTROLLER PARAMETERS
# ============================================================

FR_FORWARD_SPEED = 25
FR_REVERSE_SPEED = 25

FR_LATERAL_CORRECTION_SPEED = 55

FR_TARGET_YAW_DEG = 0.0
FR_YAW_TOL_DEG = 5

FR_YAW_KP = 5.0
FR_YAW_MIN_SPEED = 18
FR_YAW_MAX_SPEED = 22

FR_Y_TOL = 0.005


# ============================================================
# ADJACENT TARGET / STOP PARAMETERS
# ============================================================

# Robot is assumed to be at one of the 12 target/intersection points,
# but odom can have a small offset. This radius is used only to identify
# which target point the robot is currently at.
CURRENT_POINT_RADIUS = 0.20

# Points are considered to be on the same row/column if their coordinate
# difference on the fixed axis is within this tolerance. Tune this based
# on your target point spacing/noise.
GRID_LINE_TOL = 0.08

# Final stopping tolerance along the travel axis.
STOP_X_TOL = 0.010
STOP_Y_TOL = 0.010

# Small settling speeds used near the target. These reuse the existing
# motor motion primitives, not a new controller.
FR_STOP_MIN_SPEED = 14
FR_STOP_MAX_SPEED = 20
FR_STOP_KP = 45.0

LAT_STOP_MIN_SPEED = 65
LAT_STOP_MAX_SPEED = 98
LAT_STOP_KP = 110.0

# Finite rotation snap amount. x/c rotate to the next 90-degree heading.
FINITE_ROTATION_DEG = 90.0
ROT_YAW_TOL_DEG = 3.0
ROT_YAW_KP = 3.0
ROT_YAW_MIN_SPEED = 22
ROT_YAW_MAX_SPEED = 28

# If the robot overshoots the exact yaw target due to inertia, stop instead
# of reversing back and forth. This prevents yaw oscillation around 90-degree
# snap targets. Keep this slightly larger than ROT_YAW_TOL_DEG.
ROT_YAW_OVERSHOOT_STOP_DEG = 6.0

# Lower speed close to the target yaw so the robot enters the 3-degree stop
# window instead of jumping over it.
ROT_YAW_SLOW_ZONE_DEG = 12.0
ROT_YAW_FINE_SPEED = 12


# ============================================================
# COMMON PARAMETERS
# ============================================================

LOOP_DT = 0.02

MOTOR_FL_FLIP = False
MOTOR_FR_FLIP = False
MOTOR_RL_FLIP = False
MOTOR_RR_FLIP = False


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def set_speed(speed):
    speed = clamp(speed, 0, 100)

    pwm_fl.ChangeDutyCycle(speed)
    pwm_fr.ChangeDutyCycle(speed)
    pwm_rl.ChangeDutyCycle(speed)
    pwm_rr.ChangeDutyCycle(speed)


def apply(in1, in2, forward, flip):
    if flip:
        forward = not forward

    if forward:
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
    else:
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)


def hold():
    pwm_fl.ChangeDutyCycle(0)
    pwm_fr.ChangeDutyCycle(0)
    pwm_rl.ChangeDutyCycle(0)
    pwm_rr.ChangeDutyCycle(0)

    for pin in [
        FL_IN1, FL_IN2,
        FR_IN1, FR_IN2,
        RL_IN1, RL_IN2,
        RR_IN1, RR_IN2
    ]:
        GPIO.output(pin, GPIO.LOW)


def yaw_ccw(speed):
    set_speed(speed)

    apply(FL_IN1, FL_IN2, True, MOTOR_FL_FLIP)
    apply(FR_IN1, FR_IN2, False, MOTOR_FR_FLIP)
    apply(RL_IN1, RL_IN2, True, MOTOR_RL_FLIP)
    apply(RR_IN1, RR_IN2, False, MOTOR_RR_FLIP)


def yaw_cw(speed):
    set_speed(speed)

    apply(FL_IN1, FL_IN2, False, MOTOR_FL_FLIP)
    apply(FR_IN1, FR_IN2, True, MOTOR_FR_FLIP)
    apply(RL_IN1, RL_IN2, False, MOTOR_RL_FLIP)
    apply(RR_IN1, RR_IN2, True, MOTOR_RR_FLIP)


def move_forward(speed):
    set_speed(speed)

    apply(FL_IN1, FL_IN2, True, MOTOR_FL_FLIP)
    apply(FR_IN1, FR_IN2, True, MOTOR_FR_FLIP)
    apply(RL_IN1, RL_IN2, True, MOTOR_RL_FLIP)
    apply(RR_IN1, RR_IN2, True, MOTOR_RR_FLIP)


def move_backward(speed):
    set_speed(speed)

    apply(FL_IN1, FL_IN2, False, MOTOR_FL_FLIP)
    apply(FR_IN1, FR_IN2, False, MOTOR_FR_FLIP)
    apply(RL_IN1, RL_IN2, False, MOTOR_RL_FLIP)
    apply(RR_IN1, RR_IN2, False, MOTOR_RR_FLIP)


def strafe_left(speed):
    set_speed(speed)

    apply(FL_IN1, FL_IN2, True, MOTOR_FL_FLIP)
    apply(FR_IN1, FR_IN2, False, MOTOR_FR_FLIP)
    apply(RL_IN1, RL_IN2, False, MOTOR_RL_FLIP)
    apply(RR_IN1, RR_IN2, True, MOTOR_RR_FLIP)


def strafe_right(speed):
    set_speed(speed)

    apply(FL_IN1, FL_IN2, False, MOTOR_FL_FLIP)
    apply(FR_IN1, FR_IN2, True, MOTOR_FR_FLIP)
    apply(RL_IN1, RL_IN2, True, MOTOR_RL_FLIP)
    apply(RR_IN1, RR_IN2, False, MOTOR_RR_FLIP)


def quat_multiply(ax, ay, az, aw, bx, by, bz, bw):
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz

    return x, y, z, w


def rotate_world_vector_to_robot(vx, vy, vz, qx, qy, qz, qw):
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)

    if norm == 0.0:
        return 0.0, 0.0, 0.0

    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    iqx = -qx
    iqy = -qy
    iqz = -qz
    iqw = qw

    tx, ty, tz, tw = quat_multiply(
        iqx, iqy, iqz, iqw,
        vx, vy, vz, 0.0
    )

    rx, ry, rz, rw = quat_multiply(
        tx, ty, tz, tw,
        qx, qy, qz, qw
    )

    return rx, ry, rz


def normalize_angle_error(target_deg, current_deg):
    error = target_deg - current_deg

    while error > 180:
        error -= 360

    while error < -180:
        error += 360

    return error


class CombinedRobotController(Node):

    def __init__(self):
        super().__init__("combined_robot_controller")

        self.current_x = None
        self.current_y = None
        self.current_yaw_deg = None

        self.qx = None
        self.qy = None
        self.qz = None
        self.qw = None

        self.target_points = []

        self.active_mode = None

        self.locked_target_x = None
        self.locked_target_y = None
        self.locked_target_id = None
        self.locked_target_yaw_deg = None
        self.prev_yaw_error = None
        self.rotation_direction = None
        self.rotation_direction = None

        self.create_subscription(
            Odometry,
            "/robot0/odom",
            self.odom_callback,
            10
        )

        target_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(
            PoseArray,
            "/target_points",
            self.target_points_callback,
            target_qos
        )

        self.create_subscription(
            String,
            "/robot0/action",
            self.action_callback,
            10
        )

        self.last_action = None

    def action_callback(self, msg):
        raw_action = msg.data.strip()
        action = raw_action.lower().replace("-", "").replace("_", "").replace(" ", "")

        print(f"Received /robot0/action: {raw_action}")

        # Ignore repeated movement commands while that same mode is already running.
        # This prevents a continuously published action from repeatedly re-locking
        # the next target before the robot reaches the current one.
        active_alias = {
            "left": "lateral_left",
            "right": "lateral_right",
            "forward": "forward",
            "backward": "reverse",
            "reverse": "reverse",
            "catchleft": "rotation_left",
            "catchright": "rotation_right",
        }

        requested_mode = active_alias.get(action)

        if requested_mode is not None and self.active_mode == requested_mode:
            print(f"Ignoring repeated action while active: {raw_action}")
            return

        if action == "catchfront":
            # Target yaw is already 0 for forward/lateral controllers.
            # No explicit action is needed here.
            print("Catch-Front received: no motion command issued.")
            return

        if action == "catchleft":
            self.start_rotation_left()

        elif action == "catchright":
            self.start_rotation_right()

        elif action == "left":
            self.start_lateral_left()

        elif action == "right":
            self.start_lateral_right()

        elif action == "forward":
            self.start_forward()

        elif action == "backward" or action == "reverse":
            self.start_reverse()

        elif action == "wait" or action == "stop":
            self.stop_motion()

        else:
            print(f"Unknown action on /robot0/action: {raw_action}")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        self.qx = q.x
        self.qy = q.y
        self.qz = q.z
        self.qw = q.w

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        self.current_yaw_deg = math.degrees(
            math.atan2(siny_cosp, cosy_cosp)
        )

    def target_points_callback(self, msg):
        points = []

        for pose in msg.poses:
            point_id = int(pose.position.z)
            x = pose.position.x
            y = pose.position.y

            points.append((x, y, point_id))

        self.target_points = points

        print(f"Received {len(self.target_points)} target points from /target_points")

    def has_odom(self):
        return (
            self.current_x is not None and
            self.current_y is not None and
            self.current_yaw_deg is not None and
            self.qx is not None and
            self.qy is not None and
            self.qz is not None and
            self.qw is not None
        )

    def clear_locked_target(self):
        self.locked_target_x = None
        self.locked_target_y = None
        self.locked_target_id = None
        self.locked_target_yaw_deg = None
        self.prev_yaw_error = None

    def find_current_grid_point(self):
        if not self.has_odom():
            print("Cannot find current point: waiting for /odom")
            return None

        if len(self.target_points) == 0:
            print("Cannot find current point: waiting for /target_points")
            return None

        closest = min(
            self.target_points,
            key=lambda p: math.sqrt(
                (p[0] - self.current_x) ** 2 +
                (p[1] - self.current_y) ** 2
            )
        )

        dist = math.sqrt(
            (closest[0] - self.current_x) ** 2 +
            (closest[1] - self.current_y) ** 2
        )

        if dist > CURRENT_POINT_RADIUS:
            print(
                f"No current target point inside radius: "
                f"closest_id={closest[2]}, dist={dist:.4f}, "
                f"radius={CURRENT_POINT_RADIUS:.4f}"
            )
            return None

        print(
            f"Current grid point: id={closest[2]}, "
            f"x={closest[0]:.4f}, y={closest[1]:.4f}, "
            f"odom_x={self.current_x:.4f}, odom_y={self.current_y:.4f}, "
            f"dist={dist:.4f}"
        )

        return closest

    def lock_adjacent_world_target(self, direction):
        current = self.find_current_grid_point()

        if current is None:
            return False

        cx, cy, cid = current
        candidates = []

        for p in self.target_points:
            x, y, point_id = p

            if point_id == cid:
                continue

            dx = x - cx
            dy = y - cy

            # w: forward means +x direction. Same y row, nearest larger x.
            if direction == "plus_x":
                if dx > GRID_LINE_TOL and abs(dy) <= GRID_LINE_TOL:
                    candidates.append((dx, p))

            # z: reverse means -x direction. Same y row, nearest smaller x.
            elif direction == "minus_x":
                if dx < -GRID_LINE_TOL and abs(dy) <= GRID_LINE_TOL:
                    candidates.append((abs(dx), p))

            # d: right means +y direction. Same x column, nearest larger y.
            elif direction == "plus_y":
                if dy > GRID_LINE_TOL and abs(dx) <= GRID_LINE_TOL:
                    candidates.append((dy, p))

            # a: left means -y direction. Same x column, nearest smaller y.
            elif direction == "minus_y":
                if dy < -GRID_LINE_TOL and abs(dx) <= GRID_LINE_TOL:
                    candidates.append((abs(dy), p))

        if len(candidates) == 0:
            print(
                f"No adjacent target found for direction={direction} "
                f"from current_id={cid}, x={cx:.4f}, y={cy:.4f}"
            )
            return False

        _, target = min(candidates, key=lambda item: item[0])

        self.locked_target_x = target[0]
        self.locked_target_y = target[1]
        self.locked_target_id = target[2]

        print(
            f"Locked adjacent target: direction={direction}, "
            f"from_id={cid}, to_id={self.locked_target_id}, "
            f"target_x={self.locked_target_x:.4f}, "
            f"target_y={self.locked_target_y:.4f}"
        )

        return True

    def nearest_cardinal_yaw(self):
        if self.current_yaw_deg is None:
            return None

        yaw = self.current_yaw_deg % 360.0
        nearest = round(yaw / 90.0) * 90.0
        nearest = nearest % 360.0

        if nearest > 180.0:
            nearest -= 360.0

        return nearest

    def lock_finite_rotation_target(self, direction):
        if not self.lock_closest_xy_target():
            return False

        base_yaw = self.nearest_cardinal_yaw()

        if base_yaw is None:
            print("Cannot lock yaw target: waiting for /odom")
            return False

        if direction == "ccw":
            target = base_yaw + FINITE_ROTATION_DEG
        else:
            target = base_yaw - FINITE_ROTATION_DEG

        while target > 180.0:
            target -= 360.0

        while target < -180.0:
            target += 360.0

        self.locked_target_yaw_deg = target
        self.prev_yaw_error = None
        self.rotation_direction = direction

        print(
            f"Locked finite rotation: current_yaw={self.current_yaw_deg:.2f}, "
            f"base_yaw={base_yaw:.2f}, target_yaw={target:.2f}"
        )

        return True

    def lock_closest_xy_target(self):
        if not self.has_odom():
            print("Cannot lock XY target: waiting for /odom")
            return False

        if len(self.target_points) == 0:
            print("Cannot lock XY target: waiting for /target_points")
            return False

        closest = min(
            self.target_points,
            key=lambda p: math.sqrt(
                (p[0] - self.current_x) ** 2 +
                (p[1] - self.current_y) ** 2
            )
        )

        self.locked_target_x = closest[0]
        self.locked_target_y = closest[1]
        self.locked_target_id = closest[2]

        print(
            f"Locked closest XY target: "
            f"id={self.locked_target_id}, "
            f"x={self.locked_target_x:.4f}, "
            f"y={self.locked_target_y:.4f}"
        )

        return True

    def lock_closest_x_target(self):
        if self.current_x is None:
            print("Cannot lock X target: waiting for /odom")
            return False

        if len(self.target_points) == 0:
            print("Cannot lock X target: waiting for /target_points")
            return False

        closest = min(
            self.target_points,
            key=lambda p: abs(p[0] - self.current_x)
        )

        self.locked_target_x = closest[0]
        self.locked_target_y = None
        self.locked_target_id = closest[2]

        print(
            f"Locked closest X target: "
            f"id={self.locked_target_id}, "
            f"target_x={self.locked_target_x:.4f}, "
            f"current_x={self.current_x:.4f}"
        )

        return True

    def lock_closest_y_target(self):
        if self.current_y is None:
            print("Cannot lock Y target: waiting for /odom")
            return False

        if len(self.target_points) == 0:
            print("Cannot lock Y target: waiting for /target_points")
            return False

        closest = min(
            self.target_points,
            key=lambda p: abs(p[1] - self.current_y)
        )

        self.locked_target_x = None
        self.locked_target_y = closest[1]
        self.locked_target_id = closest[2]

        print(
            f"Locked closest Y target: "
            f"id={self.locked_target_id}, "
            f"target_y={self.locked_target_y:.4f}, "
            f"current_y={self.current_y:.4f}"
        )

        return True

    # ========================================================
    # MODE START FUNCTIONS
    # ========================================================

    def start_rotation_left(self):
        if self.lock_finite_rotation_target("ccw"):
            self.active_mode = "rotation_left"
            print("FINITE ROTATION LEFT mode active")

    def start_rotation_right(self):
        if self.lock_finite_rotation_target("cw"):
            self.active_mode = "rotation_right"
            print("FINITE ROTATION RIGHT mode active")

    def start_lateral_left(self):
        # a = left = -y direction
        if self.lock_adjacent_world_target("minus_y"):
            self.active_mode = "lateral_left"
            print("LATERAL LEFT mode active toward -y adjacent target")

    def start_lateral_right(self):
        # d = right = +y direction
        if self.lock_adjacent_world_target("plus_y"):
            self.active_mode = "lateral_right"
            print("LATERAL RIGHT mode active toward +y adjacent target")

    def start_forward(self):
        # w = forward = +x direction
        if self.lock_adjacent_world_target("plus_x"):
            self.active_mode = "forward"
            print("FORWARD mode active toward +x adjacent target")

    def start_reverse(self):
        # z = reverse = -x direction
        if self.lock_adjacent_world_target("minus_x"):
            self.active_mode = "reverse"
            print("REVERSE mode active toward -x adjacent target")

    def stop_motion(self):
        self.active_mode = None
        self.clear_locked_target()

        hold()
        print("Stopped and cleared locked target.")

    # ========================================================
    # CONTROL DISPATCH
    # ========================================================

    def control_step(self):
        if self.active_mode is None:
            hold()
            return

        if not self.has_odom():
            print("Waiting for /odom...")
            hold()
            return

        if self.active_mode in ["rotation_left", "rotation_right"]:
            self.rotation_control_step()
            return

        if self.active_mode in ["lateral_left", "lateral_right"]:
            self.lateral_control_step()
            return

        if self.active_mode in ["forward", "reverse"]:
            self.forward_reverse_control_step()
            return

        hold()

    # ========================================================
    # ROTATION CONTROL
    # ========================================================

    def rotation_control_step(self):
        if (
            self.locked_target_x is None or
            self.locked_target_y is None or
            self.locked_target_yaw_deg is None
        ):
            print("No locked XY/yaw target.")
            hold()
            return

        dx_world = self.locked_target_x - self.current_x
        dy_world = self.locked_target_y - self.current_y

        euclidean_error = math.sqrt(dx_world ** 2 + dy_world ** 2)

        # Keep the original rotation-position lock behavior first.
        # If the robot is not exactly at its current intersection, correct XY
        # before rotating.
        if euclidean_error > ROT_POSITION_TOL:
            robot_x, robot_y, _ = rotate_world_vector_to_robot(
                dx_world,
                dy_world,
                0.0,
                self.qx,
                self.qy,
                self.qz,
                self.qw
            )

            robot_forward_error = ROT_FORWARD_SIGN * robot_x
            robot_lateral_error = ROT_LATERAL_SIGN * robot_y

            abs_forward_error = abs(robot_forward_error)
            abs_lateral_error = abs(robot_lateral_error)

            print(
                f"ROT position correction, "
                f"id={self.locked_target_id}, "
                f"error={euclidean_error:.4f}, "
                f"forward_error={robot_forward_error:.4f}, "
                f"lateral_error={robot_lateral_error:.4f}"
            )

            if abs_forward_error >= abs_lateral_error and abs_forward_error > ROT_FORWARD_TOL:
                speed = ROT_FORWARD_KP * abs_forward_error
                speed = clamp(speed, ROT_FORWARD_MIN_SPEED, ROT_FORWARD_MAX_SPEED)

                if robot_forward_error > 0:
                    move_forward(speed)
                else:
                    move_backward(speed)

                return

            if abs_lateral_error > ROT_LATERAL_TOL:
                speed = ROT_LATERAL_KP * abs_lateral_error
                speed = clamp(speed, ROT_LATERAL_MIN_SPEED, ROT_LATERAL_MAX_SPEED)

                if robot_lateral_error > 0:
                    strafe_left(speed)
                else:
                    strafe_right(speed)

                return

        yaw_error = normalize_angle_error(
            self.locked_target_yaw_deg,
            self.current_yaw_deg
        )

        abs_yaw_error = abs(yaw_error)

        print(
            f"ROT finite mode={self.active_mode}, "
            f"direction={self.rotation_direction}, "
            f"current_yaw={self.current_yaw_deg:.2f}, "
            f"target_yaw={self.locked_target_yaw_deg:.2f}, "
            f"yaw_error={yaw_error:.2f}"
        )

        # Stop as soon as yaw is inside the accepted window.
        if abs_yaw_error <= ROT_YAW_TOL_DEG:
            print("Rotation target reached within tolerance. Stopping.")
            self.stop_motion()
            return

        # IMPORTANT:
        # Do not reverse direction after crossing the target.
        # x/c are finite open-loop-direction rotations with feedback stop.
        # If the yaw error sign changes compared with the previous loop,
        # the robot has passed the target, so stop instead of commanding
        # the opposite yaw direction and oscillating.
        if self.prev_yaw_error is not None:
            crossed_target = (
                (self.prev_yaw_error > 0.0 and yaw_error < 0.0) or
                (self.prev_yaw_error < 0.0 and yaw_error > 0.0)
            )

            if crossed_target:
                print("Rotation crossed target. Stopping to avoid oscillation.")
                self.stop_motion()
                return

        speed = ROT_YAW_KP * abs_yaw_error

        if abs_yaw_error <= ROT_YAW_SLOW_ZONE_DEG:
            speed = ROT_YAW_FINE_SPEED
        else:
            speed = clamp(speed, ROT_YAW_MIN_SPEED, ROT_YAW_MAX_SPEED)

        self.prev_yaw_error = yaw_error

        # Command only the originally requested direction.
        # This prevents CW/CCW switching around the target yaw.
        if self.rotation_direction == "ccw":
            yaw_ccw(speed)
        elif self.rotation_direction == "cw":
            yaw_cw(speed)
        else:
            hold()

    # ========================================================
    # LATERAL CONTROL
    # ========================================================

    def lateral_control_step(self):
        if self.locked_target_x is None or self.locked_target_y is None:
            print("No locked XY target for lateral movement.")
            hold()
            return

        yaw_error = normalize_angle_error(
            LAT_TARGET_YAW_DEG,
            self.current_yaw_deg
        )

        abs_yaw_error = abs(yaw_error)

        # Keep original lateral controller structure:
        # 1. correct yaw
        # 2. correct x using forward/reverse
        # 3. strafe left/right
        # New only: use target_y to decide when to stop, and switch strafe
        # direction if it overshoots.
        x_error = self.locked_target_x - self.current_x
        y_error = self.locked_target_y - self.current_y
        abs_x_error = abs(x_error)
        abs_y_error = abs(y_error)

        print(
            f"LAT mode={self.active_mode}, "
            f"id={self.locked_target_id}, "
            f"yaw={self.current_yaw_deg:.2f}, "
            f"yaw_error={yaw_error:.2f}, "
            f"x={self.current_x:.4f}, "
            f"target_x={self.locked_target_x:.4f}, "
            f"x_error={x_error:.4f}, "
            f"y={self.current_y:.4f}, "
            f"target_y={self.locked_target_y:.4f}, "
            f"y_error={y_error:.4f}"
        )

        if abs_yaw_error > LAT_YAW_TOL_DEG:
            speed = LAT_YAW_KP * abs_yaw_error
            speed = clamp(speed, LAT_YAW_MIN_SPEED, LAT_YAW_MAX_SPEED)

            if yaw_error > 0:
                yaw_cw(speed)
            else:
                yaw_ccw(speed)

            return

        if abs_x_error > LAT_X_TOL:
            speed = LAT_X_KP * abs_x_error
            speed = clamp(speed, LAT_X_MIN_SPEED, LAT_X_MAX_SPEED)

            if x_error > 0:
                move_forward(speed)
            else:
                move_backward(speed)

            return

        if abs_y_error <= STOP_Y_TOL:
            print("Lateral target reached. Stopping.")
            self.stop_motion()
            return

        speed = LAT_STOP_KP * abs_y_error
        speed = clamp(speed, LAT_STOP_MIN_SPEED, LAT_STOP_MAX_SPEED)

        # d/right is +y, a/left is -y. If overshoot happens, this switches
        # direction automatically and settles on the target.
        if y_error > 0:
            strafe_right(speed)
        else:
            strafe_left(speed)

    # ========================================================
    # FORWARD / REVERSE CONTROL
    # ========================================================

    def forward_reverse_control_step(self):
        if self.locked_target_x is None or self.locked_target_y is None:
            print("No locked XY target for forward/reverse movement.")
            hold()
            return

        yaw_error = normalize_angle_error(
            FR_TARGET_YAW_DEG,
            self.current_yaw_deg
        )

        abs_yaw_error = abs(yaw_error)

        # Keep original forward/reverse controller structure:
        # 1. correct yaw
        # 2. correct y using lateral strafe
        # 3. move forward/reverse
        # New only: use target_x to decide when to stop, and switch
        # forward/reverse if it overshoots.
        x_error = self.locked_target_x - self.current_x
        y_error = self.locked_target_y - self.current_y
        abs_x_error = abs(x_error)
        abs_y_error = abs(y_error)

        print(
            f"FR mode={self.active_mode}, "
            f"id={self.locked_target_id}, "
            f"yaw={self.current_yaw_deg:.2f}, "
            f"yaw_error={yaw_error:.2f}, "
            f"x={self.current_x:.4f}, "
            f"target_x={self.locked_target_x:.4f}, "
            f"x_error={x_error:.4f}, "
            f"y={self.current_y:.4f}, "
            f"target_y={self.locked_target_y:.4f}, "
            f"y_error={y_error:.4f}"
        )

        if abs_yaw_error > FR_YAW_TOL_DEG:
            speed = FR_YAW_KP * abs_yaw_error
            speed = clamp(speed, FR_YAW_MIN_SPEED, FR_YAW_MAX_SPEED)

            if yaw_error > 0:
                yaw_cw(speed)
            else:
                yaw_ccw(speed)

            return

        if abs_y_error > FR_Y_TOL:
            if y_error > 0:
                strafe_right(FR_LATERAL_CORRECTION_SPEED)
            else:
                strafe_left(FR_LATERAL_CORRECTION_SPEED)

            return

        if abs_x_error <= STOP_X_TOL:
            print("Forward/reverse target reached. Stopping.")
            self.stop_motion()
            return

        speed = FR_STOP_KP * abs_x_error
        speed = clamp(speed, FR_STOP_MIN_SPEED, FR_STOP_MAX_SPEED)

        # w/forward is +x, z/reverse is -x. If overshoot happens, this
        # switches direction automatically and settles on the target.
        if x_error > 0:
            move_forward(speed)
        else:
            move_backward(speed)


def main():
    rclpy.init()
    node = CombinedRobotController()

    try:
        print("\nCombined Robot Controller")
        print("Only one mode runs at a time.")
        print("Target topic : /target_points")
        print("Action topic : /robot0/action")
        print("Action type  : std_msgs/msg/String")
        print("")
        print("Catch-Left  = finite rotation left  (+90 deg snap)")
        print("Catch-Right = finite rotation right (-90 deg snap)")
        print("Left        = lateral left   to adjacent -y target")
        print("Right       = lateral right  to adjacent +y target")
        print("Forward     = forward        to adjacent +x target")
        print("Backward    = reverse        to adjacent -x target")
        print("Wait/Stop   = stop and clear target")
        print("Catch-Front = no command; yaw target remains 0 in controllers\n")

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)
            node.control_step()
            time.sleep(LOOP_DT)

    except KeyboardInterrupt:
        print("Keyboard interrupt received.")

    finally:
        hold()

        for pwm in [pwm_fl, pwm_fr, pwm_rl, pwm_rr]:
            pwm.stop()

        GPIO.cleanup()

        node.destroy_node()
        rclpy.shutdown()

        print("Clean exit")


if __name__ == "__main__":
    main()