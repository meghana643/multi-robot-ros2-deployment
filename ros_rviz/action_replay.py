#!/usr/bin/env python3
"""
action_replay.py
Replay a fixed action sequence into RViz2 simulation and print "obs" each step.

Publishes:
  /rl_actions : Int32MultiArray  (len=3)  [robot1, robot2, robot3]

Subscribes:
  /board/debug_robot_cells : String json {"robot1":[r,c],...}  (from board_server)
  /board/food_grid         : Int32MultiArray row-major len=N*N (from board_server)
  /robotX/level            : Int32 (optional; if not present defaults to 1)
  /task_done               : String (move completion)
  /board/food_events       : String (food_collected events)

Behavior:
  - waits for MOVE actions via /task_done
  - waits for LOAD action via /board/food_events containing that robot
  - prints per-step "observation": robot cell, level, adjacent food, load feasibility
  - per-robot timeout so it won't get stuck
"""

import time
import json
from typing import Dict, Tuple, Optional, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, String, Int32

# ---------- GRID CONFIG (must match board_server) ----------
N = 10

# actions: 0=NONE,1=NORTH,2=SOUTH,3=WEST,4=EAST,5=LOAD
NONE, NORTH, SOUTH, WEST, EAST, LOAD = 0, 1, 2, 3, 4, 5
MOVE_ACTIONS = {NORTH, SOUTH, WEST, EAST}

ROBOTS = ["robot1", "robot2", "robot3"]

ACTION_PLAN = [
    # Robot1 moves beside the L1 food at (2,2).
    # Robot2 and Robot3 begin approaching the L2 food at (7,5).
    [EAST,  WEST,  NORTH],

    # Robot1 independently collects the L1 food.
    [LOAD,  WEST,  EAST],

    # Direct synchronized approach. Robot2 targets (7,6), and Robot3
    # targets (7,4), placing them on opposite sides of the L2 food.
    [WEST,  SOUTH, EAST],
    [SOUTH, SOUTH, EAST],
    [SOUTH, SOUTH, SOUTH],

    # Both level-1 robots LOAD simultaneously; 1 + 1 satisfies level 2.
    [SOUTH, LOAD,  LOAD],

    # Continue directly toward the remaining tasks. Robot2 and Robot3
    # approach opposite sides of the L2 food at (5,5).
    [EAST,  NORTH, NORTH],
    [EAST,  WEST,  NORTH],

    # Robot2 and Robot3 cooperatively collect the second L2 food.
    [EAST,  LOAD,  LOAD],

    # Robot1 approaches the remaining L1 food at (2,5). Robots 2 and 3
    # continue along valid, non-oscillating paths while Robot1 finishes.
    [EAST,  EAST,  WEST],
    [NORTH, NORTH, NORTH],
    [LOAD,  NORTH, EAST],
]


def action_name(a: int) -> str:
    return {0: "NONE", 1: "NORTH", 2: "SOUTH", 3: "WEST", 4: "EAST", 5: "LOAD"}.get(a, str(a))


def in_bounds(rc: Tuple[int, int]) -> bool:
    r, c = rc
    return 0 <= r < N and 0 <= c < N


def nbrs_n_s_w_e(rc: Tuple[int, int]) -> List[Tuple[int, int]]:
    r, c = rc
    # LBF priority: N, S, W, E
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


class ActionReplay(Node):
    def __init__(self):
        super().__init__("action_replay")

        self.pub = self.create_publisher(Int32MultiArray, "/rl_actions", 10)

        # state from board_server
        self.robot_cells: Dict[str, Tuple[int, int]] = {r: (0, 0) for r in ROBOTS}
        self.food_grid: Optional[List[int]] = None
        self.robot_levels: Dict[str, int] = {r: 1 for r in ROBOTS}

        # waiting flags
        self.wait_move = {r: False for r in ROBOTS}
        self.wait_load = {r: False for r in ROBOTS}

        # timeouts
        self.deadline_move = {r: 0.0 for r in ROBOTS}
        self.deadline_load = {r: 0.0 for r in ROBOTS}
        # Movement advances only after /task_done confirms physical arrival.
        # A short timeout would overwrite an unfinished waypoint.
        self.move_timeout_s = None
        self.load_timeout_s = 1.2

        self.step_idx = 0

        # Subscribers
        self.create_subscription(String, "/board/debug_robot_cells", self.on_cells, 10)
        self.create_subscription(Int32MultiArray, "/board/food_grid", self.on_food_grid, 10)

        self.create_subscription(Int32, "/robot1/level", lambda m: self._on_level("robot1", m), 10)
        self.create_subscription(Int32, "/robot2/level", lambda m: self._on_level("robot2", m), 10)
        self.create_subscription(Int32, "/robot3/level", lambda m: self._on_level("robot3", m), 10)

        self.create_subscription(String, "/task_done", self.on_task_done, 10)
        self.create_subscription(String, "/board/food_events", self.on_food_event, 10)

        self.timer = self.create_timer(0.05, self.tick)

        self.get_logger().info(f"✅ action_replay ready. Steps={len(ACTION_PLAN)}")

    # ---------------- subscriptions ----------------
    def _on_level(self, rn: str, msg: Int32):
        self.robot_levels[rn] = int(msg.data)

    def on_cells(self, msg: String):
        try:
            data = json.loads(msg.data or "{}")
            for r in ROBOTS:
                if r in data and isinstance(data[r], (list, tuple)) and len(data[r]) == 2:
                    self.robot_cells[r] = (int(data[r][0]), int(data[r][1]))
        except Exception:
            pass

    def on_food_grid(self, msg: Int32MultiArray):
        self.food_grid = list(msg.data)

    def on_task_done(self, msg: String):
        s = msg.data or ""
        for r in ROBOTS:
            if r in s:
                self.wait_move[r] = False
                self.deadline_move[r] = 0.0
                self.get_logger().info(f"  ✅ {r} arrived ({s})")
                return

    def on_food_event(self, msg: String):
        raw = msg.data or ""
        try:
            data = json.loads(raw)
        except Exception:
            self.get_logger().info(f"🍏 food_event (non-json): {raw}")
            return

        if data.get("event") != "food_collected":
            return

        by = data.get("by", [])
        if not isinstance(by, list):
            return

        for r in ROBOTS:
            if r in by and self.wait_load[r]:
                self.wait_load[r] = False
                self.deadline_load[r] = 0.0
                self.get_logger().info(f"🍏 {r} load success -> food_collected")

    # ---------------- helpers ----------------
    def _grid_lvl(self, rc: Tuple[int, int]) -> int:
        if self.food_grid is None:
            return 0
        r, c = rc
        if not in_bounds(rc):
            return 0
        idx = r * N + c
        if idx < 0 or idx >= len(self.food_grid):
            return 0
        return int(self.food_grid[idx])

    def _adjacent_food_info(self, rn: str) -> Dict:
        """Return which adjacent food exists and which one LBF would target (N,S,W,E)."""
        rcell = self.robot_cells[rn]
        adj = []
        target = None
        for nb in nbrs_n_s_w_e(rcell):
            lvl = self._grid_lvl(nb)
            if lvl > 0:
                adj.append({"cell": list(nb), "lvl": lvl})
                if target is None:
                    target = {"cell": list(nb), "lvl": lvl}
        return {"robot_cell": list(rcell), "adj_foods": adj, "target": target}

    def _print_step_obs(self, acts: List[int]):
        """Print a compact per-step observation + whether LOAD should succeed."""
        # If we don’t have food grid yet, we can’t print properly
        if self.food_grid is None:
            self.get_logger().warn("No /board/food_grid received yet (waiting...)")
            return

        # Per robot info
        lines = []
        for i, r in enumerate(ROBOTS):
            info = self._adjacent_food_info(r)
            lvl = self.robot_levels.get(r, 1)
            a = acts[i]
            tgt = info["target"]
            sees = "YES" if tgt is not None else "NO"
            tgt_str = f"target={tgt['cell']} L{tgt['lvl']}" if tgt else "target=None"
            lines.append(
                f"  {r}: cell={info['robot_cell']} level={lvl} act={action_name(a)}({a}) adj_food? {sees} {tgt_str}"
            )

        # Predict LOAD outcome per targeted food (using current state + actions)
        # group loaders by the food they would target
        target_groups: Dict[Tuple[int, int], List[str]] = {}
        for i, r in enumerate(ROBOTS):
            if acts[i] != LOAD:
                continue
            tgt = self._adjacent_food_info(r)["target"]
            if tgt is None:
                continue
            cell = tuple(tgt["cell"])
            target_groups.setdefault(cell, []).append(r)

        pred_lines = []
        for food_cell, loaders in target_groups.items():
            food_level = self._grid_lvl(food_cell)
            level_sum = sum(int(self.robot_levels.get(r, 1)) for r in loaders)
            ok = (level_sum >= food_level) and (food_level > 0)
            pred_lines.append(
                f"  LOAD-group food={list(food_cell)} foodL={food_level} loaders={loaders} sumLevels={level_sum} -> {'COLLECT' if ok else 'NO'}"
            )

        self.get_logger().info("🧠 OBS (from /board/debug_robot_cells + /board/food_grid):\n" + "\n".join(lines))
        if pred_lines:
            self.get_logger().info("🔎 LOAD prediction:\n" + "\n".join(pred_lines))

    def _skip_timeouts(self, now):
        for r in ROBOTS:
            if self.wait_move[r] and self.deadline_move[r] > 0.0 and now > self.deadline_move[r]:
                self.wait_move[r] = False
                self.deadline_move[r] = 0.0
                self.get_logger().warn(f"⏱️ {r} MOVE timed out. Skipping wait.")

        for r in ROBOTS:
            if self.wait_load[r] and self.deadline_load[r] > 0.0 and now > self.deadline_load[r]:
                self.wait_load[r] = False
                self.deadline_load[r] = 0.0
                self.get_logger().warn(f"⏱️ {r} LOAD timed out (no collection). Continuing.")

    # ---------------- main loop ----------------
    def tick(self):
        now = time.time()

        if self.step_idx >= len(ACTION_PLAN):
            self.get_logger().info("🏁 Finished replay. (Node stays alive; not shutting down)")
            self.timer.cancel()
            return

        self._skip_timeouts(now)

        if any(self.wait_move.values()) or any(self.wait_load.values()):
            return

        acts = ACTION_PLAN[self.step_idx]
        if len(acts) != 3:
            self.get_logger().error(f"Step {self.step_idx}: invalid action length {len(acts)}")
            self.timer.cancel()
            return

        # ✅ print "obs" BEFORE publishing this step
        self.get_logger().info(f"\n================= STEP {self.step_idx:03d} =================")
        self._print_step_obs([int(acts[0]), int(acts[1]), int(acts[2])])

        out = Int32MultiArray()
        out.data = [int(acts[0]), int(acts[1]), int(acts[2])]
        self.pub.publish(out)
        self.get_logger().info(f"🎬 Published /rl_actions = {out.data}")

        # set waiting flags for this step
        for i, r in enumerate(ROBOTS):
            a = out.data[i]

            if a in MOVE_ACTIONS:
                self.wait_move[r] = True
                self.deadline_move[r] = 0.0
            else:
                self.wait_move[r] = False
                self.deadline_move[r] = 0.0

            if a == LOAD:
                self.wait_load[r] = True
                self.deadline_load[r] = now + self.load_timeout_s
            else:
                self.wait_load[r] = False
                self.deadline_load[r] = 0.0

        self.step_idx += 1


def main():
    rclpy.init()
    node = ActionReplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
