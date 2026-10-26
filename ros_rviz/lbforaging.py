#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32MultiArray

from tf2_ros import Buffer, TransformListener
import numpy as np
import math


class LBForagingObsDebug(Node):
    """
    Builds the EXACT 21-dim obs per agent (LBF style).
    ✅ Prints it AND publishes it to:
        /robot1/rl_obs
        /robot2/rl_obs
        /robot3/rl_obs
    """

    def __init__(self):
        super().__init__("lbforaging_obs_debug")

        # ----- board config (MUST match board_server) -----
        self.N = 10
        self.CELL = 1.0
        self.SIGHT = 2
        self.ORIGIN_X = -5.0
        self.ORIGIN_Y = -5.0

        self.player_levels = {
            "robot1": 1,
            "robot2": 2,
            "robot3": 3,
        }

        self.robots = ["robot1", "robot2", "robot3"]

        # food grid (row-major)
        self.food_grid = np.zeros((self.N, self.N), dtype=np.int32)
        self.have_food = False

        self.create_subscription(Int32MultiArray, "/board/food_grid", self.on_food_grid, 10)

        # ✅ publishers for obs (so rl_loop can read them)
        self.obs_pub = {
            "robot1": self.create_publisher(Float32MultiArray, "/robot1/rl_obs", 10),
            "robot2": self.create_publisher(Float32MultiArray, "/robot2/rl_obs", 10),
            "robot3": self.create_publisher(Float32MultiArray, "/robot3/rl_obs", 10),
        }

        self.tfbuf = Buffer()
        self.tfl = TransformListener(self.tfbuf, self)

        self.create_timer(0.5, self.tick)
        self.get_logger().info("✅ lbforaging_obs_debug running (PRINT + PUBLISH /robotX/rl_obs)")

    # ---------------------------------------------------
    # Food grid
    # ---------------------------------------------------
    def on_food_grid(self, msg: Int32MultiArray):
        arr = np.array(msg.data, dtype=np.int32)
        if arr.size != self.N * self.N:
            return
        self.food_grid = arr.reshape((self.N, self.N))
        self.have_food = True

    # ---------------------------------------------------
    # Coordinate transforms
    # ---------------------------------------------------
    def world_to_grid(self, x, y):
        col = int(math.floor((x - self.ORIGIN_X) / self.CELL))
        row = int(math.floor((y - self.ORIGIN_Y) / self.CELL))
        row = int(np.clip(row, 0, self.N - 1))
        col = int(np.clip(col, 0, self.N - 1))
        return row, col

    def clipped_bounds(self, r, c):
        r0 = max(0, r - self.SIGHT)
        r1 = min(self.N - 1, r + self.SIGHT)
        c0 = max(0, c - self.SIGHT)
        c1 = min(self.N - 1, c + self.SIGHT)
        return r0, r1, c0, c1

    def to_local(self, r0, c0, r, c):
        return (r - r0, c - c0)

    # ---------------------------------------------------
    # Robot state
    # ---------------------------------------------------
    def get_robot_state(self, name):
        try:
            tf = self.tfbuf.lookup_transform("world", f"{name}_base_link", rclpy.time.Time())
        except Exception:
            return None

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        row, col = self.world_to_grid(x, y)
        lvl = int(self.player_levels[name])
        return row, col, lvl

    # ---------------------------------------------------
    # Build + print + publish obs
    # ---------------------------------------------------
    def tick(self):
        if not self.have_food:
            return

        states = {}
        for r in self.robots:
            s = self.get_robot_state(r)
            if s is None:
                return  # wait until all TF available
            states[r] = s

        for agent in self.robots:
            obs = self.build_obs(agent, states)

            # print
            self.get_logger().info(f"{agent} obs (len=21): {obs}")

            # publish
            msg = Float32MultiArray()
            msg.data = [float(x) for x in obs]
            self.obs_pub[agent].publish(msg)

    def build_obs(self, agent, states):
        center_r, center_c, _ = states[agent]
        r0, r1, c0, c1 = self.clipped_bounds(center_r, center_c)

        local_food = self.food_grid[r0:r1 + 1, c0:c1 + 1]

        obs = []

        # -------- foods (max 4) --------
        ys, xs = np.nonzero(local_food)
        for i in range(4):
            if i < len(ys):
                y, x = int(ys[i]), int(xs[i])
                lvl = int(local_food[y, x])
                obs.extend([float(y), float(x), float(lvl)])
            else:
                obs.extend([-1.0, -1.0, 0.0])

        # -------- players (self first) --------
        players = []
        for name, (r, c, lvl) in states.items():
            if not (r0 <= r <= r1 and c0 <= c <= c1):
                continue
            ly, lx = self.to_local(r0, c0, r, c)
            players.append((name == agent, int(ly), int(lx), int(lvl)))

        players.sort(key=lambda x: 0 if x[0] else 1)

        for i in range(3):
            if i < len(players):
                _, y, x, lvl = players[i]
                obs.extend([float(y), float(x), float(lvl)])
            else:
                obs.extend([-1.0, -1.0, 0.0])

        return obs


def main():
    rclpy.init()
    node = LBForagingObsDebug()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

