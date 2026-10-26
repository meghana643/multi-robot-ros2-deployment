#!/usr/bin/env python3
"""
lidar_fov2_os_viz.py

Visualize each robot's lbforaging-style local observation window (sight=2 => 5x5)
using the SAME grid/world mapping as board_server.py.

Subscribes:
  /board/food_grid   (Int32MultiArray length 100, row-major)
TF:
  world -> robotX_base_link

Publishes:
  /robotX/os_markers   (MarkerArray)  # shows:
      - FOV window outline
      - obstacle cubes inside FOV
      - food spheres + "L#" text inside FOV
"""

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.duration import Duration

from std_msgs.msg import Int32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf2_ros import Buffer, TransformListener


# =============================
# MUST MATCH board_server.py
# =============================
N = 10
CELL = 1.0
CENTERED = True

# Obstacles in (row, col) — MUST match board_server.py
OBSTACLES_RC = [
    (1, 7), (3, 8), (6, 7), (8, 6),
    (2, 4), (1, 3), (3, 2), (5, 1),
    (7, 2), (9, 4),
]

SIGHT = 2  # FOV radius in cells => 5x5

# ✅ FOV outline color per robot
ROBOT_FOV_COLOR = {
    "robot1": (1.0, 0.0, 0.0),  # red
    "robot2": (0.0, 1.0, 0.0),  # green
    "robot3": (1.0, 0.0, 0.0),  # red: level 1
}


def board_origin() -> Tuple[float, float]:
    if CENTERED:
        half = 0.5 * N * CELL
        return (-half, -half)  # bottom-left corner
    return (0.0, 0.0)


def cell_center(row: int, col: int) -> Tuple[float, float]:
    ox, oy = board_origin()
    x = ox + (col + 0.5) * CELL
    y = oy + (row + 0.5) * CELL
    return (x, y)


def world_to_rc(x: float, y: float) -> Tuple[int, int]:
    ox, oy = board_origin()
    col = int(math.floor((x - ox) / CELL))
    row = int(math.floor((y - oy) / CELL))
    row = max(0, min(N - 1, row))
    col = max(0, min(N - 1, col))
    return (row, col)


class FOVOSViz(Node):
    def __init__(self):
        super().__init__("lidar_fov2_os_viz")

        # TF
        self.tfbuf = Buffer()
        self.tfl = TransformListener(self.tfbuf, self)

        # Latest food grid (row-major, length N*N)
        self.food_grid: List[int] = [0] * (N * N)

        # Subscribe to the *real* food state (updates when collected)
        self.create_subscription(Int32MultiArray, "/board/food_grid", self.on_food_grid, 10)

        # Per-robot marker publishers (latched)
        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self.robots = [
            {"name": "robot1", "base": "robot1_base_link"},
            {"name": "robot2", "base": "robot2_base_link"},
            {"name": "robot3", "base": "robot3_base_link"},
        ]

        self.pub = {
            r["name"]: self.create_publisher(MarkerArray, f'/{r["name"]}/os_markers', qos)
            for r in self.robots
        }

        self.create_timer(0.1, self.tick)  # 10 Hz

        self.get_logger().info("✅ lidar_fov2_os_viz running (publishing /robotX/os_markers)")

    def on_food_grid(self, msg: Int32MultiArray):
        data = list(msg.data)
        if len(data) != N * N:
            self.get_logger().warn(f"/board/food_grid len={len(data)} != {N*N}")
            return
        self.food_grid = data

    def _lookup_robot_xy(self, base_frame: str):
        t = self.tfbuf.lookup_transform("world", base_frame, rclpy.time.Time())
        x = t.transform.translation.x
        y = t.transform.translation.y
        return x, y

    def _color_for_level(self, lvl: int):
        # L1 green, L2 yellow, L3+ red
        if lvl == 1:
            return (0.20, 0.85, 0.25)
        if lvl == 2:
            return (0.95, 0.75, 0.25)
        return (0.92, 0.25, 0.25)

    def _clear_all(self, robot_name: str):
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = f"{robot_name}_os"
        m.id = 0
        m.action = Marker.DELETEALL
        ma.markers.append(m)
        self.pub[robot_name].publish(ma)

    def tick(self):
        now = self.get_clock().now().to_msg()

        for r in self.robots:
            name = r["name"]
            base = r["base"]

            try:
                rx, ry = self._lookup_robot_xy(base)
            except Exception:
                self._clear_all(name)
                continue

            rr, rc = world_to_rc(rx, ry)

            # FOV bounds in grid
            r0 = max(0, rr - SIGHT)
            r1 = min(N - 1, rr + SIGHT)
            c0 = max(0, rc - SIGHT)
            c1 = min(N - 1, rc + SIGHT)

            ma = MarkerArray()

            # ---- Clear old markers ----
            clear = Marker()
            clear.header.frame_id = "world"
            clear.header.stamp = now
            clear.ns = f"{name}_os"
            clear.id = 0
            clear.action = Marker.DELETEALL
            ma.markers.append(clear)

            mid = 1

            # ---- 1) Draw FOV window outline in WORLD ----
            ox, oy = board_origin()
            left = ox + c0 * CELL
            right = ox + (c1 + 1) * CELL
            bottom = oy + r0 * CELL
            top = oy + (r1 + 1) * CELL

            outline = Marker()
            outline.header.frame_id = "world"
            outline.header.stamp = now
            outline.ns = f"{name}_fov"
            outline.id = mid; mid += 1
            outline.type = Marker.LINE_STRIP
            outline.action = Marker.ADD
            outline.scale.x = 0.06
            outline.pose.orientation.w = 1.0

            # ✅ per-robot outline color
            cr, cg, cb = ROBOT_FOV_COLOR.get(name, (0.2, 0.8, 1.0))
            outline.color.r = cr
            outline.color.g = cg
            outline.color.b = cb
            outline.color.a = 0.95

            outline.lifetime = Duration(seconds=0).to_msg()

            p1 = Point(x=left,  y=bottom, z=0.02)
            p2 = Point(x=right, y=bottom, z=0.02)
            p3 = Point(x=right, y=top,    z=0.02)
            p4 = Point(x=left,  y=top,    z=0.02)
            outline.points = [p1, p2, p3, p4, p1]
            ma.markers.append(outline)

            # ---- 2) Obstacles inside FOV ----
            for (orow, ocol) in OBSTACLES_RC:
                if not (r0 <= orow <= r1 and c0 <= ocol <= c1):
                    continue
                x, y = cell_center(orow, ocol)

                m = Marker()
                m.header.frame_id = "world"
                m.header.stamp = now
                m.ns = f"{name}_obs"
                m.id = mid; mid += 1
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = float(x)
                m.pose.position.y = float(y)
                m.pose.position.z = 0.05
                m.pose.orientation.w = 1.0
                m.scale.x = CELL * 0.7
                m.scale.y = CELL * 0.7
                m.scale.z = 0.1
                m.color.r = 0.65
                m.color.g = 0.65
                m.color.b = 0.65
                m.color.a = 0.55
                m.lifetime = Duration(seconds=0).to_msg()
                ma.markers.append(m)

            # ---- 3) Foods inside FOV (from /board/food_grid) ----
            for row in range(r0, r1 + 1):
                for col in range(c0, c1 + 1):
                    lvl = int(self.food_grid[row * N + col])
                    if lvl <= 0:
                        continue

                    x, y = cell_center(row, col)
                    cr, cg, cb = self._color_for_level(lvl)

                    # sphere
                    s = Marker()
                    s.header.frame_id = "world"
                    s.header.stamp = now
                    s.ns = f"{name}_food"
                    s.id = mid; mid += 1
                    s.type = Marker.SPHERE
                    s.action = Marker.ADD
                    s.pose.position.x = float(x)
                    s.pose.position.y = float(y)
                    s.pose.position.z = 0.20
                    s.pose.orientation.w = 1.0
                    s.scale.x = s.scale.y = s.scale.z = CELL * 0.28
                    s.color.r = cr
                    s.color.g = cg
                    s.color.b = cb
                    s.color.a = 1.0
                    s.lifetime = Duration(seconds=0).to_msg()
                    ma.markers.append(s)

                    # label
                    t = Marker()
                    t.header.frame_id = "world"
                    t.header.stamp = now
                    t.ns = f"{name}_food_text"
                    t.id = mid; mid += 1
                    t.type = Marker.TEXT_VIEW_FACING
                    t.action = Marker.ADD
                    t.pose.position.x = float(x)
                    t.pose.position.y = float(y)
                    t.pose.position.z = 0.50
                    t.pose.orientation.w = 1.0
                    t.scale.z = 0.28
                    t.color.r = t.color.g = t.color.b = 1.0
                    t.color.a = 1.0
                    t.text = f"L{lvl}"
                    t.lifetime = Duration(seconds=0).to_msg()
                    ma.markers.append(t)

            self.pub[name].publish(ma)


def main():
    rclpy.init()
    node = FOVOSViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
