# Robust multi-robot ArUco odometry node
# Publishes odometry for robot0, robot1, and robot2 from one RealSense camera.
#
# Main change from the previous version:
# - 4 markers: best measurement, also learns/refreshes each robot marker geometry.
# - 3 markers: estimates the missing marker and learns/refreshes geometry.
# - 2 markers: estimates robot pose from the visible marker pair + learned geometry.
# - 1 marker : estimates robot center from that marker + previous yaw + learned geometry.
# - 0 markers: holds the last pose for a short time, then marks stale.
#
# IMPORTANT:
# The 1-marker and 2-marker modes require the tracker to learn marker geometry first.
# Let each robot show 3 or 4 markers for a few frames at startup if possible.

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import pyrealsense2 as rs
import numpy as np
import cv2
import math
import time
import json
import os


ROBOTS = {
    "robot0": {
        "topic": "/robot0/odom",
        "front_ids": [1, 0],
        "rear_ids": [2, 3],
    },
    "robot1": {
        "topic": "/robot1/odom",
        "front_ids": [5, 4],
        "rear_ids": [6, 7],
    },
    "robot2": {
        "topic": "/robot2/odom",
        "front_ids": [9, 8],
        "rear_ids": [10, 11],
    },
}

YAW_OFFSET = 0.0

IMAGE_W = 1280.0
IMAGE_H = 720.0
IMAGE_CX = IMAGE_W / 2.0
IMAGE_CY = IMAGE_H / 2.0

# EMA smoothing. Lower alpha = smoother but more lag.
ALPHA_POS_4 = 0.50
ALPHA_POS_3 = 0.42
ALPHA_POS_2 = 0.32
ALPHA_POS_1 = 0.18
ALPHA_YAW_4 = 0.55
ALPHA_YAW_3 = 0.45
ALPHA_YAW_2 = 0.30
ALPHA_YAW_1 = 0.12

# Slowly update learned marker geometry when 3/4 marker pose is available.
MODEL_LEARN_ALPHA = 0.08

# If no marker is seen for this long, keep publishing last pose but mark stale.
STALE_TIMEOUT_SEC = 0.75

# Reject impossible jumps in image-normalized odometry per frame.
# Increase if the camera/robot moves very fast.
MAX_NORM_JUMP = 0.35

# Reacquire after several mutually consistent measurements were rejected as a jump.
# This prevents the tracker from remaining locked forever to an old position.
REACQUIRE_CONFIRM_FRAMES = 3
REACQUIRE_MAX_NORM_SPREAD = 0.08

# Detection/bootstrapping controls.
DETECT_UPSCALE = 1.5
BOOTSTRAP_SINGLE_MARKER = False  # Avoid random robot poses from a single false ArUco detection
MODEL_FILE = os.path.expanduser("~/.ros/aruco_robot_marker_models.json")

# False-detection filtering.
VALID_MARKER_IDS = {
    marker_id
    for cfg in ROBOTS.values()
    for marker_id in (cfg["front_ids"] + cfg["rear_ids"])
}
MIN_MARKER_AREA_PX = 220.0
MAX_SIDE_RATIO = 2.2
MIN_CONFIRM_FRAMES = 2
MAX_CONFIRM_CENTER_JUMP_PX = 45.0


def yaw_to_quaternion(yaw):
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def angle_wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def smooth_yaw_ema(old_yaw, new_yaw, alpha):
    err = angle_wrap(new_yaw - old_yaw)
    return angle_wrap(old_yaw + alpha * err)


def rotate_point(p, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def mean_points(points):
    n = float(len(points))
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


class RobotTracker:
    def __init__(self, name, front_ids, rear_ids):
        self.name = name
        self.front_ids = list(front_ids)
        self.rear_ids = list(rear_ids)

        # Local IDs preserve your original geometry:
        # local 0,1 = front markers, local 2,3 = rear markers.
        self.local_to_global = {
            0: self.front_ids[1],
            1: self.front_ids[0],
            2: self.rear_ids[0],
            3: self.rear_ids[1],
        }
        self.global_to_local = {v: k for k, v in self.local_to_global.items()}
        self.required_global_ids = set(self.global_to_local.keys())

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.center_px = None
        self.yaw_initialized = False

        self.quality = "not_initialized"
        self.used_global_centers = {}
        self.marker_global_centers = {}
        self.front_center = None
        self.rear_center = None
        self.robot_center = None
        self.missing_global_id = None

        # Learned geometry: local marker offset from robot center in robot frame.
        # Example: marker_model_local[0] = (x_px_from_center, y_px_from_center)
        self.marker_model_local = {}
        self.model_initialized = False
        self.last_update_time = None

        # Candidate used to recover after a real large displacement or a long occlusion.
        self.pending_reacquire_center = None
        self.pending_reacquire_count = 0

        # True only when a pose was accepted from detections in the current frame.
        self.pose_updated_this_frame = False

        # Load previously calibrated marker-to-center offsets, if available.
        # This lets correct 2-marker odometry work immediately after restarting.
        self.load_marker_model()


    def load_marker_model(self):
        """Load this robot's calibrated marker geometry from disk."""
        try:
            if not os.path.exists(MODEL_FILE):
                return

            with open(MODEL_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            robot_data = data.get(self.name)
            if not robot_data:
                return

            loaded = {}
            for lid_text, offset in robot_data.items():
                lid = int(lid_text)
                if lid in (0, 1, 2, 3) and len(offset) == 2:
                    loaded[lid] = (float(offset[0]), float(offset[1]))

            if all(lid in loaded for lid in (0, 1, 2, 3)):
                self.marker_model_local = loaded
                self.model_initialized = True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Continue without a model; it will be learned again from 3/4 markers.
            self.marker_model_local = {}
            self.model_initialized = False

    def save_marker_model(self):
        """Persist this robot's calibrated geometry for later runs."""
        if not self.model_initialized:
            return

        try:
            directory = os.path.dirname(MODEL_FILE)
            if directory:
                os.makedirs(directory, exist_ok=True)

            data = {}
            if os.path.exists(MODEL_FILE):
                try:
                    with open(MODEL_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    data = {}

            data[self.name] = {
                str(lid): [float(offset[0]), float(offset[1])]
                for lid, offset in self.marker_model_local.items()
            }

            tmp_file = MODEL_FILE + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_file, MODEL_FILE)
        except OSError:
            pass

    def pose_from_uncalibrated_diagonal_pair(self, local_centers):
        """
        Estimate center from an uncalibrated diagonal marker pair.

        For the parallelogram layout used by this node, local pairs (0, 2)
        and (1, 3) are opposite corners. Their midpoint is the robot center.
        Yaw cannot be recovered from two markers without calibrated geometry,
        so the previous yaw is retained.
        """
        if len(local_centers) != 2 or not self.yaw_initialized:
            return None

        pair = frozenset(local_centers.keys())
        if pair not in (frozenset((0, 2)), frozenset((1, 3))):
            return None

        pts = list(local_centers.values())
        center = (
            0.5 * (pts[0][0] + pts[1][0]),
            0.5 * (pts[0][1] + pts[1][1]),
        )

        # Front/rear positions are not observable without a calibrated model.
        return center, self.odom_yaw, center, center

    def estimate_missing_marker_local(self, local_centers):
        ids_present = set(local_centers.keys())
        required_ids = {0, 1, 2, 3}

        if len(ids_present) != 3:
            return local_centers, None

        missing = list(required_ids - ids_present)
        if len(missing) != 1:
            return local_centers, None

        missing_id = missing[0]
        mc = dict(local_centers)

        # Parallelogram relations for rectangular four-marker layout.
        if missing_id == 0 and all(i in mc for i in [1, 2, 3]):
            p1, p2, p3 = mc[1], mc[2], mc[3]
            mc[0] = (p1[0] + p3[0] - p2[0], p1[1] + p3[1] - p2[1])
        elif missing_id == 1 and all(i in mc for i in [0, 2, 3]):
            p0, p2, p3 = mc[0], mc[2], mc[3]
            mc[1] = (p0[0] + p2[0] - p3[0], p0[1] + p2[1] - p3[1])
        elif missing_id == 2 and all(i in mc for i in [0, 1, 3]):
            p0, p1, p3 = mc[0], mc[1], mc[3]
            mc[2] = (p1[0] + p3[0] - p0[0], p1[1] + p3[1] - p0[1])
        elif missing_id == 3 and all(i in mc for i in [0, 1, 2]):
            p0, p1, p2 = mc[0], mc[1], mc[2]
            mc[3] = (p0[0] + p2[0] - p1[0], p0[1] + p2[1] - p1[1])

        return mc, missing_id

    def compute_center_and_yaw_from_full_local(self, local_centers):
        p0 = local_centers[0]
        p1 = local_centers[1]
        p2 = local_centers[2]
        p3 = local_centers[3]

        center = (
            (p0[0] + p1[0] + p2[0] + p3[0]) / 4.0,
            (p0[1] + p1[1] + p2[1] + p3[1]) / 4.0,
        )
        front = ((p1[0] + p0[0]) / 2.0, (p1[1] + p0[1]) / 2.0)
        rear = ((p2[0] + p3[0]) / 2.0, (p2[1] + p3[1]) / 2.0)

        yaw = math.atan2(front[1] - rear[1], front[0] - rear[0])
        yaw = angle_wrap(yaw + YAW_OFFSET)
        return center, yaw, front, rear

    def update_marker_model(self, full_local_centers, center, yaw):
        # Convert observed marker offsets into robot frame by rotating by -yaw.
        new_model = {}
        for lid, pt in full_local_centers.items():
            dx = pt[0] - center[0]
            dy = pt[1] - center[1]
            new_model[lid] = rotate_point((dx, dy), -yaw)

        if not self.model_initialized:
            self.marker_model_local = new_model
            self.model_initialized = True
            self.save_marker_model()
            return

        # EMA update avoids model jitter while adapting to perspective/camera changes.
        for lid, new_off in new_model.items():
            old_off = self.marker_model_local.get(lid, new_off)
            self.marker_model_local[lid] = (
                old_off[0] + MODEL_LEARN_ALPHA * (new_off[0] - old_off[0]),
                old_off[1] + MODEL_LEARN_ALPHA * (new_off[1] - old_off[1]),
            )

        self.save_marker_model()

    def pose_from_model_and_visible_markers(self, local_centers):
        if not self.model_initialized:
            return None

        present = [lid for lid in local_centers.keys() if lid in self.marker_model_local]
        if len(present) == 0:
            return None

        # Estimate yaw from all visible marker pairs if at least 2 markers are visible.
        yaw_candidates = []
        if len(present) >= 2:
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a = present[i]
                    b = present[j]
                    obs_vec = sub(local_centers[b], local_centers[a])
                    model_vec = sub(self.marker_model_local[b], self.marker_model_local[a])

                    obs_len = math.hypot(obs_vec[0], obs_vec[1])
                    model_len = math.hypot(model_vec[0], model_vec[1])
                    if obs_len < 5.0 or model_len < 5.0:
                        continue

                    obs_ang = math.atan2(obs_vec[1], obs_vec[0])
                    model_ang = math.atan2(model_vec[1], model_vec[0])
                    yaw_candidates.append(angle_wrap(obs_ang - model_ang + YAW_OFFSET))

        if yaw_candidates:
            # Circular average around the previous yaw when available.
            if self.yaw_initialized:
                # Choose pair-yaw values close to previous yaw to avoid 180-degree jumps.
                unwrapped = [self.odom_yaw + angle_wrap(y - self.odom_yaw) for y in yaw_candidates]
                yaw = angle_wrap(sum(unwrapped) / len(unwrapped))
            else:
                yaw = angle_wrap(sum(yaw_candidates) / len(yaw_candidates))
        elif self.yaw_initialized:
            # One marker: yaw is not observable, so hold previous yaw.
            yaw = self.odom_yaw
        else:
            return None

        # Estimate robot center from each visible marker:
        # observed_marker = center + R(yaw) * model_offset
        center_candidates = []
        for lid in present:
            rotated_offset = rotate_point(self.marker_model_local[lid], yaw - YAW_OFFSET)
            center_candidates.append(sub(local_centers[lid], rotated_offset))

        center = mean_points(center_candidates)

        # Reconstruct all four marker centers for overlay and front/rear calculation.
        reconstructed = {}
        for lid, off in self.marker_model_local.items():
            reconstructed[lid] = add(center, rotate_point(off, yaw - YAW_OFFSET))

        front = ((reconstructed[1][0] + reconstructed[0][0]) / 2.0,
                 (reconstructed[1][1] + reconstructed[0][1]) / 2.0)
        rear = ((reconstructed[2][0] + reconstructed[3][0]) / 2.0,
                (reconstructed[2][1] + reconstructed[3][1]) / 2.0)

        return center, yaw, front, rear, reconstructed

    def update_ema_position(self, raw_center, alpha, now):
        if raw_center is None or not np.all(np.isfinite(raw_center)):
            self.quality += " invalid_center"
            return False

        force_reacquire = (
            self.last_update_time is not None
            and (now - self.last_update_time) > STALE_TIMEOUT_SEC
        )

        if self.center_px is not None and not force_reacquire:
            old_norm_x = (self.center_px[0] - IMAGE_CX) / IMAGE_CX
            old_norm_y = (self.center_px[1] - IMAGE_CY) / IMAGE_CY
            new_norm_x = (raw_center[0] - IMAGE_CX) / IMAGE_CX
            new_norm_y = (raw_center[1] - IMAGE_CY) / IMAGE_CY
            jump = math.hypot(new_norm_x - old_norm_x, new_norm_y - old_norm_y)

            if jump > MAX_NORM_JUMP:
                # Do not compare forever only against the old accepted pose. Confirm that
                # several rejected measurements agree with one another, then reacquire.
                if self.pending_reacquire_center is None:
                    self.pending_reacquire_center = raw_center
                    self.pending_reacquire_count = 1
                else:
                    px, py = self.pending_reacquire_center
                    pending_dx = (raw_center[0] - px) / IMAGE_CX
                    pending_dy = (raw_center[1] - py) / IMAGE_CY
                    pending_spread = math.hypot(pending_dx, pending_dy)

                    if pending_spread <= REACQUIRE_MAX_NORM_SPREAD:
                        n = self.pending_reacquire_count
                        self.pending_reacquire_center = (
                            (px * n + raw_center[0]) / (n + 1),
                            (py * n + raw_center[1]) / (n + 1),
                        )
                        self.pending_reacquire_count += 1
                    else:
                        self.pending_reacquire_center = raw_center
                        self.pending_reacquire_count = 1

                if self.pending_reacquire_count < REACQUIRE_CONFIRM_FRAMES:
                    self.quality += f" rejected_jump({self.pending_reacquire_count})"
                    return False

                raw_center = self.pending_reacquire_center
                force_reacquire = True

        if self.center_px is None or force_reacquire:
            self.center_px = raw_center
            if force_reacquire:
                self.quality += " reacquired"
        else:
            old_x, old_y = self.center_px
            new_x, new_y = raw_center
            self.center_px = (
                old_x + alpha * (new_x - old_x),
                old_y + alpha * (new_y - old_y),
            )

        self.pending_reacquire_center = None
        self.pending_reacquire_count = 0

        cx, cy = self.center_px
        self.odom_x = (cx - IMAGE_CX) / IMAGE_CX
        self.odom_y = (cy - IMAGE_CY) / IMAGE_CY
        return True

    def accept_pose(self, center, yaw, front, rear, used_local_centers, quality, alpha_pos, alpha_yaw, now):
        self.quality = quality
        ok = self.update_ema_position(center, alpha_pos, now)
        if not ok:
            return False

        if not self.yaw_initialized:
            self.odom_yaw = yaw
            self.yaw_initialized = True
        else:
            self.odom_yaw = smooth_yaw_ema(self.odom_yaw, yaw, alpha_yaw)

        self.robot_center = center
        self.front_center = front
        self.rear_center = rear
        self.used_global_centers = {
            self.local_to_global[lid]: pt for lid, pt in used_local_centers.items()
        }
        self.last_update_time = now
        self.pose_updated_this_frame = True
        return True

    def bootstrap_pose_from_visible_markers(self, local_centers):
        """Return a provisional pose before the four-marker geometry is learned."""
        if not local_centers:
            return None

        present = list(local_centers.keys())
        center = mean_points([local_centers[lid] for lid in present])

        front_pts = [local_centers[lid] for lid in present if lid in (0, 1)]
        rear_pts = [local_centers[lid] for lid in present if lid in (2, 3)]

        if front_pts and rear_pts:
            front = mean_points(front_pts)
            rear = mean_points(rear_pts)
            yaw = angle_wrap(math.atan2(front[1] - rear[1], front[0] - rear[0]) + YAW_OFFSET)
            center = ((front[0] + rear[0]) / 2.0, (front[1] + rear[1]) / 2.0)
        elif self.yaw_initialized:
            yaw = self.odom_yaw
            front = center
            rear = center
        else:
            # Orientation cannot be observed from only front markers, only rear markers,
            # or one marker. Still publish/show the measured location immediately.
            yaw = 0.0
            front = center
            rear = center

        return center, yaw, front, rear

    def compute_pose_from_global_markers(self, all_marker_centers, now):
        # Reset every frame. This prevents stale front/rear labels from being drawn
        # when the robot was not reliably localized in the current image.
        self.pose_updated_this_frame = False

        self.marker_global_centers = {
            gid: all_marker_centers[gid]
            for gid in self.required_global_ids
            if gid in all_marker_centers
        }

        local_centers = {
            self.global_to_local[gid]: center
            for gid, center in self.marker_global_centers.items()
        }

        num_seen = len(local_centers)
        self.missing_global_id = None

        if num_seen == 4:
            center, yaw, front, rear = self.compute_center_and_yaw_from_full_local(local_centers)
            accepted = self.accept_pose(
                center, yaw, front, rear, local_centers,
                "4-marker", ALPHA_POS_4, ALPHA_YAW_4, now
            )
            if accepted:
                self.update_marker_model(local_centers, center, yaw)
            return

        if num_seen == 3:
            full_local, missing_lid = self.estimate_missing_marker_local(local_centers)
            if all(i in full_local for i in [0, 1, 2, 3]):
                center, yaw, front, rear = self.compute_center_and_yaw_from_full_local(full_local)
                self.missing_global_id = self.local_to_global[missing_lid]
                quality = f"3-marker + est id {self.missing_global_id}"
                accepted = self.accept_pose(
                    center, yaw, front, rear, full_local,
                    quality, ALPHA_POS_3, ALPHA_YAW_3, now
                )
                if accepted:
                    self.update_marker_model(full_local, center, yaw)
                return

        if num_seen in [1, 2]:
            pose = self.pose_from_model_and_visible_markers(local_centers)
            if pose is not None:
                center, yaw, front, rear, reconstructed = pose
                alpha_pos = ALPHA_POS_2 if num_seen == 2 else ALPHA_POS_1
                alpha_yaw = ALPHA_YAW_2 if num_seen == 2 else ALPHA_YAW_1
                quality = f"{num_seen}-marker model-est"
                self.accept_pose(center, yaw, front, rear, reconstructed, quality, alpha_pos, alpha_yaw, now)
                return

            # Before calibration, only a diagonal marker pair has an exact center:
            # local pairs (0,2) and (1,3). Adjacent pairs do NOT have the robot center
            # at their midpoint, so they are rejected until geometry is learned.
            if num_seen == 2:
                diagonal_pose = self.pose_from_uncalibrated_diagonal_pair(local_centers)
                if diagonal_pose is not None:
                    center, yaw, front, rear = diagonal_pose
                    self.accept_pose(
                        center, yaw, front, rear, local_centers,
                        "2-marker diagonal-center (yaw held)",
                        ALPHA_POS_2, ALPHA_YAW_2, now
                    )
                    return

            self.quality = f"{num_seen}-marker waiting_for_geometry"
            self.used_global_centers = dict(self.marker_global_centers)
            return

        # 0 markers: hold pose briefly.
        if self.last_update_time is not None and (now - self.last_update_time) <= STALE_TIMEOUT_SEC:
            self.quality = "0-marker hold"
        else:
            self.quality = "0-marker stale"
        self.used_global_centers = dict(self.marker_global_centers)


def draw_robot_overlay(img, tracker):
    h, w = img.shape[:2]

    def valid_point(pt, margin=20):
        if pt is None or not np.all(np.isfinite(pt)):
            return False
        return -margin <= pt[0] < w + margin and -margin <= pt[1] < h + margin

    # Draw marker labels only when their coordinates are reasonable.
    for marker_id, pt in tracker.used_global_centers.items():
        if not valid_point(pt):
            continue
        x, y = int(pt[0]), int(pt[1])
        if marker_id in tracker.marker_global_centers:
            color = (0, 255, 255)
            label = f"{tracker.name} id={marker_id}"
        else:
            color = (0, 0, 255)
            label = f"{tracker.name} est id={marker_id}"
        cv2.circle(img, (x, y), 6, color, -1)
        cv2.putText(img, label, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    # Keep the filtered robot center visible during the short hold period, but hide it
    # after the track becomes stale.
    if tracker.center_px is not None and "stale" not in tracker.quality and valid_point(tracker.center_px):
        mx, my = map(int, tracker.center_px)
        cv2.circle(img, (mx, my), 9, (255, 255, 255), -1)
        heading_len = 90
        hx = int(mx + heading_len * math.cos(tracker.odom_yaw))
        hy = int(my + heading_len * math.sin(tracker.odom_yaw))
        cv2.arrowedLine(img, (mx, my), (hx, hy), (255, 0, 255), 3)
        cv2.putText(img, tracker.name, (mx + 8, my), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)

    return img


class MultiArucoOdom(Node):
    def __init__(self):
        super().__init__("multi_aruco_odom_node")

        self.trackers = {
            name: RobotTracker(name, cfg["front_ids"], cfg["rear_ids"])
            for name, cfg in ROBOTS.items()
        }
        self.odom_pubs = {
            name: self.create_publisher(Odometry, cfg["topic"], 10)
            for name, cfg in ROBOTS.items()
        }

        self.bridge = CvBridge()
        self.raw_image_pub = self.create_publisher(Image, "/topview/raw", 10)
        self.labeled_image_pub = self.create_publisher(Image, "/topview/labeled", 10)

        self.timer = self.create_timer(0.03, self.update)

        self.frame_counter = 0
        self.last_fps_print = time.perf_counter()
        self.sampling_rate = 0.0
        self.last_status_print = 0.0

        # Marker detections must remain consistent for several frames before use.
        self.marker_confirm_state = {}

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, int(IMAGE_W), int(IMAGE_H), rs.format.bgr8, 15)
        self.pipeline.start(config)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()

        # Slightly more permissive detection for partially visible/smaller markers.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 35
        params.adaptiveThreshWinSizeStep = 4
        params.minMarkerPerimeterRate = 0.025
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate = 0.03
        params.minDistanceToBorder = 1
        params.minMarkerDistanceRate = 0.01
        params.errorCorrectionRate = 0.5
        params.detectInvertedMarker = False

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _valid_marker_shape(self, pts):
        """Reject tiny, highly stretched, or malformed quadrilaterals."""
        if pts is None or len(pts) != 4:
            return False

        area = abs(cv2.contourArea(pts.astype(np.float32)))
        if area < MIN_MARKER_AREA_PX:
            return False

        side_lengths = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            side_lengths.append(float(np.linalg.norm(p2 - p1)))

        min_side = min(side_lengths)
        max_side = max(side_lengths)

        if min_side < 5.0:
            return False

        if max_side / min_side > MAX_SIDE_RATIO:
            return False

        if not cv2.isContourConvex(pts.astype(np.float32)):
            return False

        return True

    def _temporally_confirm_markers(self, candidates):
        """
        Require a marker to appear consistently before it is accepted.

        Existing confirmed markers remain accepted while their center moves
        reasonably. A sudden new location must be confirmed again.
        """
        confirmed = {}
        visible_ids = set(candidates.keys())

        # Remove state for IDs that disappeared.
        for marker_id in list(self.marker_confirm_state.keys()):
            if marker_id not in visible_ids:
                del self.marker_confirm_state[marker_id]

        for marker_id, pts in candidates.items():
            center = np.mean(pts, axis=0)
            state = self.marker_confirm_state.get(marker_id)

            if state is None:
                self.marker_confirm_state[marker_id] = {
                    "center": center,
                    "count": 1,
                }
                continue

            distance = float(np.linalg.norm(center - state["center"]))

            if distance <= MAX_CONFIRM_CENTER_JUMP_PX:
                state["count"] += 1
                state["center"] = center
            else:
                # Treat a large instantaneous jump as a new candidate.
                state["count"] = 1
                state["center"] = center

            if state["count"] >= MIN_CONFIRM_FRAMES:
                confirmed[marker_id] = pts

        return confirmed

    def detect_markers_robust(self, bgr_img):
        """Detect and reject unknown, tiny, distorted, and one-frame markers."""
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        variants = [(gray, 1.0), (self.clahe.apply(gray), 1.0)]

        if DETECT_UPSCALE > 1.0:
            enlarged = cv2.resize(
                variants[-1][0],
                None,
                fx=DETECT_UPSCALE,
                fy=DETECT_UPSCALE,
                interpolation=cv2.INTER_CUBIC,
            )
            variants.append((enlarged, DETECT_UPSCALE))

        best_by_id = {}

        for image_variant, scale in variants:
            corners, ids, _ = self.detector.detectMarkers(image_variant)

            if ids is None:
                continue

            for corner, marker_id in zip(corners, ids.flatten()):
                marker_id = int(marker_id)

                # Never accept an ID that is not assigned to a robot.
                if marker_id not in VALID_MARKER_IDS:
                    continue

                pts = corner[0].astype(np.float32) / scale

                if not self._valid_marker_shape(pts):
                    continue

                area = abs(cv2.contourArea(pts))
                previous = best_by_id.get(marker_id)

                if previous is None or area > previous[0]:
                    best_by_id[marker_id] = (area, pts)

        candidates = {
            marker_id: value[1]
            for marker_id, value in best_by_id.items()
        }

        confirmed = self._temporally_confirm_markers(candidates)

        if not confirmed:
            return [], None

        ordered_ids = sorted(confirmed.keys())
        out_corners = [
            confirmed[marker_id].reshape(1, 4, 2)
            for marker_id in ordered_ids
        ]
        out_ids = np.array(
            ordered_ids,
            dtype=np.int32,
        ).reshape(-1, 1)

        return out_corners, out_ids

    def update_sampling_rate(self):
        self.frame_counter += 1
        now = time.perf_counter()
        elapsed = now - self.last_fps_print
        if elapsed >= 1.0:
            self.sampling_rate = self.frame_counter / elapsed
            self.frame_counter = 0
            self.last_fps_print = now

    def publish_odom(self, name, tracker):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_odom"
        msg.child_frame_id = f"{name}/base_link"
        msg.pose.pose.position.x = tracker.odom_x
        msg.pose.pose.position.y = tracker.odom_y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = yaw_to_quaternion(tracker.odom_yaw)

        # Rough covariance by quality. Larger = less trusted.
        if "4-marker" in tracker.quality:
            pos_var, yaw_var = 0.0025, 0.01
        elif "3-marker" in tracker.quality:
            pos_var, yaw_var = 0.006, 0.03
        elif "2-marker" in tracker.quality:
            pos_var, yaw_var = 0.015, 0.08
        elif "1-marker" in tracker.quality:
            pos_var, yaw_var = 0.04, 0.25
        else:
            pos_var, yaw_var = 0.2, 1.0

        msg.pose.covariance[0] = pos_var
        msg.pose.covariance[7] = pos_var
        msg.pose.covariance[35] = yaw_var

        self.odom_pubs[name].publish(msg)

    def update(self):
        self.update_sampling_rate()
        now = time.perf_counter()

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError as exc:
            self.get_logger().warning(f"RealSense frame timeout: {exc}")
            return

        frame = frames.get_color_frame()
        if not frame:
            return

        img = np.asanyarray(frame.get_data())
        raw_img = img.copy()
        corners, ids = self.detect_markers_robust(img)

        all_marker_centers = {}

        if ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                marker_id = int(marker_id)
                pts = corners[i][0]
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                all_marker_centers[marker_id] = (cx, cy)

        detected_id_text = "detected IDs: " + (", ".join(map(str, sorted(all_marker_centers))) if all_marker_centers else "none")
        cv2.putText(img, detected_id_text, (20, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        h, w = img.shape[:2]
        cx0, cy0 = w // 2, h // 2
        cv2.circle(img, (cx0, cy0), 5, (255, 255, 255), -1)
        cv2.arrowedLine(img, (cx0, cy0), (cx0 + 120, cy0), (0, 0, 255), 3)
        cv2.putText(img, "+X image/right", (cx0 + 125, cy0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.arrowedLine(img, (cx0, cy0), (cx0, cy0 + 100), (0, 255, 0), 3)
        cv2.putText(img, "+Y image/down", (cx0 + 10, cy0 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        y_text = 30
        for name, tracker in self.trackers.items():
            tracker.compute_pose_from_global_markers(all_marker_centers, now)
            self.publish_odom(name, tracker)
            img = draw_robot_overlay(img, tracker)

            cv2.putText(
                img,
                f"{name}: x={tracker.odom_x:.3f} y={tracker.odom_y:.3f} "
                f"yaw={math.degrees(tracker.odom_yaw):.1f} {tracker.quality}",
                (20, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )
            y_text += 30

        cv2.putText(img, f"rate: {self.sampling_rate:.2f} Hz",
                    (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

        raw_msg = self.bridge.cv2_to_imgmsg(raw_img, encoding="bgr8")
        raw_msg.header.stamp = self.get_clock().now().to_msg()
        raw_msg.header.frame_id = "camera_color_frame"
        self.raw_image_pub.publish(raw_msg)

        labeled_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        labeled_msg.header.stamp = raw_msg.header.stamp
        labeled_msg.header.frame_id = "camera_color_frame"
        self.labeled_image_pub.publish(labeled_msg)

        cv2.imshow("Robust Multi Aruco Odom View", img)
        cv2.waitKey(1)

        # Printing every frame can slow detection and make marker loss more likely.
        if now - self.last_status_print >= 1.0:
            self.last_status_print = now
            print(f"rate={self.sampling_rate:.2f} Hz")
            for name, tracker in self.trackers.items():
                print(
                    f"{name}: x={tracker.odom_x:.3f}, y={tracker.odom_y:.3f}, "
                    f"yaw={math.degrees(tracker.odom_yaw):.2f} deg, "
                    f"markers={len(tracker.marker_global_centers)}, "
                    f"model={tracker.model_initialized}, quality={tracker.quality}"
                )


def main():
    rclpy.init()
    node = MultiArucoOdom()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()