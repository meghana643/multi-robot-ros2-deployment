#!/usr/bin/env python3
"""
fov2_local_debug_print.py

Print, per robot, what is inside its local FOV window (sight=2) using CLIPPED local coords.

- Uses SAME mapping as your board_server.py:
    N=10, CELL=1.0, CENTERED=True
    world_to_rc uses bottom-left as (row=0, col=0)

Subscribes:
  /board/food_grid (Int32MultiArray, len 100, row-major)

Reads TF:
  world -> robotX_base_link

Prints:
  - robot global cell (rr,rc)
  - clipped window bounds and size HxW (can be <5x5 at edges)
  - obstacles in local coords (y,x)
  - foods in local coords (y,x,lvl)
  - other robots in local coords (y,x)

This is ONLY for debug printing (does not publish markers).
"""

import math
from typing import List, Tuple, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from tf2_ros import Buffer, TransformListener


# ===== MUST MATCH board_server.py =====
N = 10
CELL = 1.0
CENTERED = True

SIGHT = 2  # fov radius => (2*SIGHT+1) = 5 nominal, clipped near edges

OBSTACLES_RC = [
    (1, 7), (3, 8), (6, 7), (8, 6),
    (2, 4), (1, 3), (3, 2), (5, 1),
    (7, 2), (9, 4),
]


def board_origin() -> Tuple[float, float]:
    if CENTERED:
        half = 0.5 * N * CELL
        return (-half, -half)  # bottom-left corner in world
    return (0.0, 0.0)


def world_to_rc(x: float, y: float) -> Tuple[int, int]:
    """Same as board_server.py: row increases with world +y, col increases with world +x."""
    ox, oy = board_origin()
    col = int(math.floor((x - ox) / CELL))
    row = int(math.floor((y - oy) / CELL))
    row = max(0, min(N - 1, row))
    col = max(0, min(N - 1, col))
    return row, col


def clipped_bounds(rr: int, rc: int, sight: int) -> Tuple[int, int, int, int]:
    r0 = max(0, rr - sight)
    r1 = min(N - 1, rr + sight)
    c0 = max(0, rc - sight)
    c1 = min(N - 1, rc + sight)
    return r0, r1, c0, c1


def to_local(row: int, col: int, r0: int, c0: int) -> Tuple[int, int]:
    """Local coords inside clipped window: (y,x) with origin at (r0,c0)."""
    return row - r0, col - c0


class FOV2LocalDebugPrint(Node):
    def __init__(self):
        super().__init__("fov2_local_debug_print")

        self.food_grid: List[int] = [0] * (N * N)
        self.have_food = False

        self.create_subscription(Int32MultiArray, "/board/food_grid", self.on_food_grid, 10)

        self.tfbuf = Buffer()
        self.tfl = TransformListener(self.tfbuf, self)

        self.robots = [
            {"name": "robot1", "base": "robot1_base_link"},
            {"name": "robot2", "base": "robot2_base_link"},
            {"name": "robot3", "base": "robot3_base_link"},
        ]

        # print at ~2 Hz (change if you want)
        self.create_timer(0.5, self.tick)

        self.get_logger().info("✅ fov2_local_debug_print running (prints local clipped coords)")

    def on_food_grid(self, msg: Int32MultiArray):
        data = list(msg.data)
        if len(data) != N * N:
            self.get_logger().warn(f"/board/food_grid len={len(data)} != {N*N}")
            return
        self.food_grid = data
        self.have_food = True

    def lookup_xy(self, frame: str) -> Optional[Tuple[float, float]]:
        try:
            t = self.tfbuf.lookup_transform("world", frame, rclpy.time.Time())
            return (t.transform.translation.x, t.transform.translation.y)
        except Exception:
            return None

    def tick(self):
        if not self.have_food:
            self.get_logger().info("waiting for /board/food_grid ...")
            return

        # precompute all robot global cells first (so we can report other robots per robot)
        robot_cells: Dict[str, Tuple[int, int]] = {}
        for r in self.robots:
            xy = self.lookup_xy(r["base"])
            if xy is None:
                continue
            rr, rc = world_to_rc(xy[0], xy[1])
            robot_cells[r["name"]] = (rr, rc)

        if not robot_cells:
            self.get_logger().info("waiting for TF world -> robotX_base_link ...")
            return

        lines = []
        lines.append("========== FOV2 LOCAL DEBUG ==========")

        for r in self.robots:
            name = r["name"]
            if name not in robot_cells:
                lines.append(f"{name}: (no TF)")
                continue

            rr, rc = robot_cells[name]
            r0, r1, c0, c1 = clipped_bounds(rr, rc, SIGHT)
            H = (r1 - r0 + 1)
            W = (c1 - c0 + 1)

            # obstacles inside window -> local (y,x)
            obs_local = []
            for (orow, ocol) in OBSTACLES_RC:
                if r0 <= orow <= r1 and c0 <= ocol <= c1:
                    ly, lx = to_local(orow, ocol, r0, c0)
                    obs_local.append((ly, lx))
            obs_local.sort()

            # foods inside window -> local (y,x,lvl)
            food_local = []
            for row in range(r0, r1 + 1):
                for col in range(c0, c1 + 1):
                    lvl = int(self.food_grid[row * N + col])
                    if lvl > 0:
                        ly, lx = to_local(row, col, r0, c0)
                        food_local.append((ly, lx, lvl))
            food_local.sort()

            # other robots inside window -> local (y,x)
            others_local = []
            for other_name, (orow, ocol) in robot_cells.items():
                if other_name == name:
                    continue
                if r0 <= orow <= r1 and c0 <= ocol <= c1:
                    ly, lx = to_local(orow, ocol, r0, c0)
                    others_local.append((other_name, ly, lx))
            others_local.sort(key=lambda x: x[0])

            # print block
            r_local_y = rr - r0
            r_local_x = rc - c0

            lines.append(
                f"{name}: local_robot(y,x)=({r_local_y},{r_local_x}) "
                f"| window r[{r0}..{r1}] c[{c0}..{c1}] size={H}x{W}"
            )

            lines.append(f"  obstacles_local(y,x): {obs_local if obs_local else '[]'}")
            lines.append(f"  foods_local(y,x,lvl): {food_local if food_local else '[]'}")
            lines.append(f"  other_robots_local:  {others_local if others_local else '[]'}")

        self.get_logger().info("\n" + "\n".join(lines))


def main():
    rclpy.init()
    node = FOV2LocalDebugPrint()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

