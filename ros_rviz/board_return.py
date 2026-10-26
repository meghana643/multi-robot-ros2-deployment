#!/usr/bin/env python3
# return.py — board_server (RViz markers + RL actions + LBF semantics)
#
# ✅ This version fixes BOTH issues you observed:
# 1) While one robot is returning, OTHER robots STILL take RL actions (no global freeze).
# 2) Returning redundancy reduced:
#    - Return uses BFS shortest path (removes useless loops)
#    - Path compression removes immediate back/forth
#    - Replan ONLY if next step is blocked OR robot stalls too long
#    - Reserved-next-cell prevents other robots from stepping into returner's next cell
#
# Topics used:
#   /rl_actions (Int32MultiArray)           <- from rl_loop
#   /board/init_cells (String, transient)  <- from dynamic_multi_sim (home/start cells)
#   /{robot}/waypoint (PoseStamped)        <- to dynamic_multi_sim
#   /board/action_feedback (String json)   <- to rl_loop (hidden reset)
#   /board/food_events (String json)       <- debug / print

import math
import json
import random
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
)
from rclpy.duration import Duration

from std_msgs.msg import Int32, Int32MultiArray, String
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener

# ============================================================
# Grid settings (MATCH EVERYWHERE)
# ============================================================
N = 10
CELL = 1.0
CENTERED = True

# actions: 0=NONE,1=NORTH,2=SOUTH,3=WEST,4=EAST,5=LOAD
NONE, NORTH, SOUTH, WEST, EAST, LOAD = 0, 1, 2, 3, 4, 5

ACTION_NAMES = {0: "NONE", 1: "NORTH", 2: "SOUTH", 3: "WEST", 4: "EAST", 5: "LOAD"}


def board_origin():
    if CENTERED:
        half = 0.5 * N * CELL
        return -half, -half
    return 0.0, 0.0


def cell_center(row: int, col: int):
    ox, oy = board_origin()
    x = ox + (col + 0.5) * CELL
    y = oy + (row + 0.5) * CELL
    return x, y


def world_to_rc(x: float, y: float):
    ox, oy = board_origin()
    col = int(math.floor((x - ox) / CELL))
    row = int(math.floor((y - oy) / CELL))
    row = max(0, min(N - 1, row))
    col = max(0, min(N - 1, col))
    return row, col


def cell_id(row: int, col: int) -> int:
    return row * N + col


class BoardServer(Node):
    def __init__(self):
        super().__init__("board_server")

        self.robots = ["robot1", "robot2", "robot3"]

        # Robot levels
        # All robots are level 1; level-2 food therefore requires cooperation.
        self.robot_levels = {"robot1": 1, "robot2": 1, "robot3": 1}

        # Obstacles
        self.obstacles_rc = [
            (1, 7), (3, 8), (6, 7), (8, 6),
            (2, 4), (1, 3), (3, 2), (5, 1),
            (7, 2), (9, 4),
        ]

        # Foods
        self.foods = [
            {"rc": (2, 5), "lvl": 1},
            {"rc": (5, 5), "lvl": 2},
            {"rc": (7, 5), "lvl": 2},
            {"rc": (2, 2), "lvl": 1},
        ]
        self.collected_set = set()
        self.collected_visuals = []
        self.collected_visual_seconds = 2.0

        # TF
        self.tfbuf = Buffer()
        self.tfl = TransformListener(self.tfbuf, self)

        # Discrete state (grid truth)
        self.disc_cells = {rn: (0, 0) for rn in self.robots}
        self._disc_inited = False

        # init/home cells from /board/init_cells
        self.start_cells = None
        self.have_start_cells = False
        q_init = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(String, "/board/init_cells", self.on_init_cells, q_init)

        # stuck fix
        self.stuck = {rn: 0 for rn in self.robots}
        self.stuck_threshold = 3
        # Keep the hard-coded RViz demonstration deterministic. Otherwise a
        # robot intentionally receiving NONE for three steps is moved randomly.
        self.enable_stuck_rescue = False

        # -------------------------------------------------------
        # RETURN MODE (serial per-returner, but others keep acting)
        # -------------------------------------------------------
        self.returning_robot = None
        self.ret_queue = []          # FIFO
        self.ret_path = []           # list of actions for active return
        self.ret_target = None       # target cell we are moving to (wait using TF)
        self.ret_arrival_tol = 0.15
        self._ret_last_cmd_time = 0.0
        self._ret_reissue_sec = 1.5

        # ✅ reserved cell (next step) so other robots don't step into it
        self.ret_reserved_cell = None

        # ✅ stall detection (avoid replan spam / redundancy)
        self.ret_stall_timeout = 4.0
        self._ret_last_progress_time = time.time()
        self._ret_last_cell = None

        # Publishers
        qos_markers = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.marker_pub = self.create_publisher(MarkerArray, "/board/markers", qos_markers)
        self.food_grid_pub = self.create_publisher(Int32MultiArray, "/board/food_grid", 10)
        self.food_map_pub = self.create_publisher(String, "/board/food_map_json", 10)
        self.food_evt_pub = self.create_publisher(String, "/board/food_events", 10)
        self.action_fb_pub = self.create_publisher(String, "/board/action_feedback", 10)
        self.debug_cells_pub = self.create_publisher(String, "/board/debug_robot_cells", 10)

        self.wp_pub = {rn: self.create_publisher(PoseStamped, f"/{rn}/waypoint", 10) for rn in self.robots}
        self.level_pub = {rn: self.create_publisher(Int32, f"/{rn}/level", 1) for rn in self.robots}

        self.create_subscription(Int32MultiArray, "/rl_actions", self.on_actions, 10)

        self.create_timer(0.25, self.publish_everything)

        self.get_logger().info("✅ board_server up (return=BFS + reserved cell + less redundancy)")

    # -------------------------
    # init cells callback
    # -------------------------
    def on_init_cells(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.start_cells = {k: (int(v[0]), int(v[1])) for k, v in data.items()}
            self.have_start_cells = True
            self.get_logger().info(f"✅ Received start_cells from /board/init_cells: {self.start_cells}")
        except Exception as e:
            self.get_logger().error(f"❌ Failed parsing /board/init_cells: {e}")

    # -------------------------
    # helpers
    # -------------------------
    def in_bounds(self, rc):
        r, c = rc
        return 0 <= r < N and 0 <= c < N

    def move_target(self, rc, a):
        r, c = rc
        if a == NORTH: return (r - 1, c)
        if a == SOUTH: return (r + 1, c)
        if a == WEST:  return (r, c - 1)
        if a == EAST:  return (r, c + 1)
        return (r, c)

    def robot_world_xy(self, rn):
        try:
            t = self.tfbuf.lookup_transform("world", f"{rn}_base_link", rclpy.time.Time())
            return float(t.transform.translation.x), float(t.transform.translation.y)
        except Exception:
            return None

    def publish_waypoint_to_cell(self, rn, rc):
        x, y = cell_center(rc[0], rc[1])
        ps = PoseStamped()
        ps.header.frame_id = "world"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        self.wp_pub[rn].publish(ps)

    # -------------------------
    # BFS path planner (grid)
    # -------------------------
    def _neighbors(self, rc):
        r, c = rc
        return [
            ((r - 1, c), NORTH),
            ((r + 1, c), SOUTH),
            ((r, c - 1), WEST),
            ((r, c + 1), EAST),
        ]

    def _compress_actions(self, acts):
        # remove immediate back-and-forth: e.g., NORTH then SOUTH, WEST then EAST
        inv = {NORTH: SOUTH, SOUTH: NORTH, WEST: EAST, EAST: WEST}
        out = []
        for a in acts:
            if out and inv.get(a, None) == out[-1]:
                out.pop()
            else:
                out.append(a)
        return out

    def plan_bfs(self, start_rc, goal_rc, blocked_cells):
        if start_rc == goal_rc:
            return []

        q = deque([start_rc])
        prev = {start_rc: None}  # cell -> (prev_cell, action_used)

        while q:
            cur = q.popleft()
            for nb, act in self._neighbors(cur):
                if not self.in_bounds(nb):
                    continue
                if nb in blocked_cells:
                    continue
                if nb in prev:
                    continue
                prev[nb] = (cur, act)
                if nb == goal_rc:
                    q.clear()
                    break
                q.append(nb)

        if goal_rc not in prev:
            return None

        # reconstruct
        acts = []
        cur = goal_rc
        while cur != start_rc:
            p, a = prev[cur]
            acts.append(a)
            cur = p
        acts.reverse()
        return acts

    # -------------------------
    # Unblocker: move a blocking robot aside 1 step
    # -------------------------
    def move_robot_aside_one_step(self, blocker_rn, avoid_cells):
        cur = self.disc_cells[blocker_rn]
        candidates = []
        for nb, act in self._neighbors(cur):
            if not self.in_bounds(nb):
                continue
            if nb in avoid_cells:
                continue
            candidates.append((nb, act))
        if not candidates:
            return False
        nb, act = random.choice(candidates)
        self.disc_cells[blocker_rn] = nb
        self.publish_waypoint_to_cell(blocker_rn, nb)
        self.get_logger().warn(f"🧹 moved blocker {blocker_rn} aside -> {nb} via {ACTION_NAMES[act]}({act})")
        return True

    # =======================================================
    # RETURN QUEUE (serial for who returns, others keep acting)
    # =======================================================
    def enqueue_return(self, group):
        for rn in group:
            if rn not in self.robots:
                continue
            if rn == self.returning_robot:
                continue
            if rn in self.ret_queue:
                continue
            self.ret_queue.append(rn)
        self._maybe_start_return()

    def _maybe_start_return(self):
        if self.returning_robot is not None:
            return
        if not self.ret_queue:
            return
        if not self.have_start_cells or not self._disc_inited:
            return

        rn = self.ret_queue.pop(0)
        self.returning_robot = rn
        self.ret_path = []
        self.ret_target = None
        self.ret_reserved_cell = None
        self._ret_last_cmd_time = 0.0

        self._ret_last_cell = self.disc_cells[rn]
        self._ret_last_progress_time = time.time()

        self.get_logger().info(f"🏠 Starting RETURN for {rn}: {self.disc_cells[rn]} -> {self.start_cells[rn]}")
        self._replan_return()

    def _replan_return(self):
        rn = self.returning_robot
        if rn is None:
            return False

        cur = self.disc_cells[rn]
        home = self.start_cells[rn]

        obstacle_cells = set(self.obstacles_rc)
        food_cells = {tuple(f["rc"]) for f in self.foods if tuple(f["rc"]) not in self.collected_set}
        other_robots = set(self.disc_cells.values()) - {cur}

        blocked = obstacle_cells | food_cells | other_robots

        plan = self.plan_bfs(cur, home, blocked)
        if plan is not None:
            plan = self._compress_actions(plan)
            self.ret_path = plan
            self.ret_reserved_cell = None
            self.get_logger().info(f"🏠 {rn} RETURN plan length: {len(plan)}")
            return True

        # If no path, try to move a nearby blocking robot aside (basic unblock)
        for nb, _act in self._neighbors(cur):
            if nb in other_robots:
                blocker = None
                for name, cell in self.disc_cells.items():
                    if name != rn and cell == nb:
                        blocker = name
                        break
                if blocker:
                    avoid = obstacle_cells | food_cells | set(self.disc_cells.values())
                    moved = self.move_robot_aside_one_step(blocker, avoid_cells=avoid)
                    if moved:
                        # replan after moving blocker
                        other_robots2 = set(self.disc_cells.values()) - {self.disc_cells[rn]}
                        blocked2 = obstacle_cells | food_cells | other_robots2
                        plan2 = self.plan_bfs(self.disc_cells[rn], home, blocked2)
                        if plan2 is not None:
                            plan2 = self._compress_actions(plan2)
                            self.ret_path = plan2
                            self.ret_reserved_cell = None
                            self.get_logger().info(f"🏠 {rn} RETURN replan after unblock length: {len(plan2)}")
                            return True

        self.get_logger().warn(f"⚠️ {rn} RETURN: no path found. Will retry soon.")
        self.ret_path = []
        self.ret_reserved_cell = None
        return False

    def _finish_return(self):
        rn = self.returning_robot
        self.get_logger().info(f"✅ {rn} finished RETURN (at home)")
        self.returning_robot = None
        self.ret_path = []
        self.ret_target = None
        self.ret_reserved_cell = None
        self._ret_last_cmd_time = 0.0
        self._maybe_start_return()

    def _step_return_once(self):
        rn = self.returning_robot
        if rn is None:
            return

        # 1) wait for reaching current TF target
        if self.ret_target is not None:
            xy = self.robot_world_xy(rn)
            if xy is None:
                if time.time() - self._ret_last_cmd_time > self._ret_reissue_sec:
                    self.publish_waypoint_to_cell(rn, self.ret_target)
                    self._ret_last_cmd_time = time.time()
                return

            x, y = xy
            cx, cy = cell_center(self.ret_target[0], self.ret_target[1])
            if math.hypot(x - cx, y - cy) <= self.ret_arrival_tol:
                self.ret_target = None
                # progress marker
                self._ret_last_progress_time = time.time()
                self._ret_last_cell = self.disc_cells[rn]
            else:
                if time.time() - self._ret_last_cmd_time > self._ret_reissue_sec:
                    self.publish_waypoint_to_cell(rn, self.ret_target)
                    self._ret_last_cmd_time = time.time()
                return

        # 2) done?
        if self.disc_cells[rn] == self.start_cells[rn]:
            self._finish_return()
            return

        # 3) stall detection (if discrete cell isn't changing for too long -> replan)
        now_t = time.time()
        if self._ret_last_cell is None:
            self._ret_last_cell = self.disc_cells[rn]
            self._ret_last_progress_time = now_t

        if self.disc_cells[rn] != self._ret_last_cell:
            self._ret_last_cell = self.disc_cells[rn]
            self._ret_last_progress_time = now_t
        elif (now_t - self._ret_last_progress_time) > self.ret_stall_timeout:
            self.get_logger().warn(f"⚠️ {rn} RETURN stalled -> replanning")
            self.ret_path = []
            self.ret_target = None
            self.ret_reserved_cell = None
            self._replan_return()
            self._ret_last_progress_time = now_t
            return

        # 4) ensure we have a plan
        if not self.ret_path:
            ok = self._replan_return()
            if not ok:
                return

        # 5) compute reserved next cell (so others won't step into it)
        a0 = self.ret_path[0]
        cur = self.disc_cells[rn]
        nxt = self.move_target(cur, a0)
        self.ret_reserved_cell = nxt

        # 6) if next is blocked now, replan
        obstacle_cells = set(self.obstacles_rc)
        food_cells = {tuple(f["rc"]) for f in self.foods if tuple(f["rc"]) not in self.collected_set}
        occupied = set(self.disc_cells.values()) - {cur}

        if (not self.in_bounds(nxt)) or (nxt in obstacle_cells) or (nxt in food_cells) or (nxt in occupied):
            self.get_logger().warn(f"⚠️ {rn} RETURN next step blocked -> replanning (cur={cur}, nxt={nxt})")
            self.ret_path = []
            self.ret_reserved_cell = None
            self._replan_return()
            return

        # 7) execute one step
        a = self.ret_path.pop(0)
        self.disc_cells[rn] = nxt
        self.ret_target = nxt
        self.publish_waypoint_to_cell(rn, nxt)
        self._ret_last_cmd_time = time.time()

    # =======================================================
    # Publish: markers + food_grid + levels
    # =======================================================
    def publish_everything(self):
        now = self.get_clock().now().to_msg()
        now_s = float(self.get_clock().now().nanoseconds) * 1e-9

        # init disc_cells from TF once TF is available
        if not self._disc_inited:
            ok = True
            new_cells = {}
            for rn in self.robots:
                cell = self.robot_cell_tf(rn)
                if cell == (0, 0):
                    ok = False
                    break
                new_cells[rn] = cell
            if ok:
                self.disc_cells.update(new_cells)
                self._disc_inited = True
                self.get_logger().info(f"✅ disc_cells initialized from TF: {self.disc_cells}")
            else:
                return

        # progress return
        if self.returning_robot is not None:
            self._step_return_once()
        else:
            self._maybe_start_return()

        # prune visuals
        self.collected_visuals = [c for c in self.collected_visuals if c["t_end"] > now_s]

        ma = MarkerArray()

        # Remove markers from the previous redraw. Without DELETEALL, an
        # infinite-lifetime food marker remains visible after collection.
        clear = Marker()
        clear.header.frame_id = "world"
        clear.header.stamp = now
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        mid = 1

        # obstacles
        for (row, col) in self.obstacles_rc:
            x, y = cell_center(row, col)
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = now
            m.ns = "obstacle"
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
            m.color.r = m.color.g = m.color.b = 0.6
            m.color.a = 1.0
            m.lifetime = Duration(seconds=0).to_msg()
            ma.markers.append(m)

        # foods + labels
        for food in self.foods:
            (row, col), lvl = food["rc"], int(food["lvl"])
            if (row, col) in self.collected_set:
                continue
            x, y = cell_center(row, col)

            s = Marker()
            s.header.frame_id = "world"
            s.header.stamp = now
            s.ns = "food"
            s.id = mid; mid += 1
            s.type = Marker.SPHERE
            s.action = Marker.ADD
            s.pose.position.x = float(x)
            s.pose.position.y = float(y)
            s.pose.position.z = 0.12
            s.pose.orientation.w = 1.0
            s.scale.x = s.scale.y = s.scale.z = CELL * 0.6

            if lvl == 1:
                s.color.r, s.color.g, s.color.b = 0.2, 0.85, 0.25
            elif lvl == 2:
                s.color.r, s.color.g, s.color.b = 0.95, 0.75, 0.25
            else:
                s.color.r, s.color.g, s.color.b = 0.92, 0.25, 0.25
            s.color.a = 1.0

            s.lifetime = Duration(seconds=0).to_msg()
            ma.markers.append(s)

            t = Marker()
            t.header.frame_id = "world"
            t.header.stamp = now
            t.ns = "food_text"
            t.id = mid; mid += 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(x)
            t.pose.position.y = float(y)
            t.pose.position.z = 0.60
            t.pose.orientation.w = 1.0
            t.scale.z = 0.35
            t.color.r = t.color.g = t.color.b = 1.0
            t.color.a = 1.0
            t.text = f"L{lvl}"
            t.lifetime = Duration(seconds=0).to_msg()
            ma.markers.append(t)

        # collection highlight
        for c in self.collected_visuals:
            (row, col) = c["rc"]
            lvl = int(c["lvl"])
            x, y = cell_center(row, col)
            cid = cell_id(row, col)

            txt = Marker()
            txt.header.frame_id = "world"
            txt.header.stamp = now
            txt.ns = "food_collected_text"
            txt.id = 9000 + cid
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(x)
            txt.pose.position.y = float(y)
            txt.pose.position.z = 1.0
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.8
            txt.color.r = 0.1
            txt.color.g = 1.0
            txt.color.b = 0.1
            txt.color.a = 1.0
            txt.text = f"✅ L{lvl}"
            txt.lifetime = Duration(seconds=0).to_msg()
            ma.markers.append(txt)

            ring = Marker()
            ring.header.frame_id = "world"
            ring.header.stamp = now
            ring.ns = "food_collected_ring"
            ring.id = 9100 + cid
            ring.type = Marker.CYLINDER
            ring.action = Marker.ADD
            ring.pose.position.x = float(x)
            ring.pose.position.y = float(y)
            ring.pose.position.z = 0.02
            ring.pose.orientation.w = 1.0
            ring.scale.x = CELL * 0.95
            ring.scale.y = CELL * 0.95
            ring.scale.z = 0.03
            ring.color.r = 0.1
            ring.color.g = 1.0
            ring.color.b = 0.1
            ring.color.a = 0.6
            ring.lifetime = Duration(seconds=0).to_msg()
            ma.markers.append(ring)

        self.marker_pub.publish(ma)

        # food grid
        grid = [0] * (N * N)
        for food in self.foods:
            (row, col), lvl = food["rc"], int(food["lvl"])
            if (row, col) in self.collected_set:
                continue
            grid[row * N + col] = lvl
        gmsg = Int32MultiArray()
        gmsg.data = grid
        self.food_grid_pub.publish(gmsg)

        # food map json
        triples = []
        for food in self.foods:
            (row, col), lvl = food["rc"], int(food["lvl"])
            if (row, col) in self.collected_set:
                continue
            x, y = cell_center(row, col)
            triples.append([float(x), float(y), int(lvl)])
        self.food_map_pub.publish(String(data=json.dumps(triples)))

        # robot levels
        for rn, pub in self.level_pub.items():
            imsg = Int32()
            imsg.data = int(self.robot_levels[rn])
            pub.publish(imsg)

        # debug cells
        self.debug_cells_pub.publish(String(data=json.dumps(self.disc_cells)))

    # =======================================================
    # Actions from RL
    # =======================================================
    def on_actions(self, msg: Int32MultiArray):
        if not self._disc_inited or not self.have_start_cells:
            return

        acts_in = list(msg.data)
        while len(acts_in) < 3:
            acts_in.append(0)
        acts_in = acts_in[:3]

        # If a robot is returning, override ONLY THAT robot's action (others still act)
        forced_by_return = {rn: False for rn in self.robots}
        if self.returning_robot is not None:
            r_idx = self.robots.index(self.returning_robot)
            acts_in[r_idx] = NONE
            forced_by_return[self.returning_robot] = True  # rl_loop should reset hidden for it

        cur = dict(self.disc_cells)

        obstacle_cells = set(self.obstacles_rc)
        food_pos_to_idx = {tuple(f["rc"]): i for i, f in enumerate(self.foods)}
        food_cells = {tuple(f["rc"]) for f in self.foods if tuple(f["rc"]) not in self.collected_set}

        def adjacent_food_exists(rc):
            rr, cc = rc
            cand = [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]
            return any(nb in food_cells for nb in cand)

        def valid_moves(rc):
            moves = []
            for a in (NORTH, SOUTH, WEST, EAST):
                tgt = self.move_target(rc, a)
                if self.in_bounds(tgt) and (tgt not in obstacle_cells) and (tgt not in food_cells):
                    # also block stepping into return reserved cell
                    if self.ret_reserved_cell is not None and tgt == self.ret_reserved_cell:
                        continue
                    moves.append(a)
            if adjacent_food_exists(rc):
                moves.append(LOAD)
            moves.append(NONE)
            return moves

        # sanitize
        sanitized = []
        for rn, a in zip(self.robots, acts_in):
            rcell = cur[rn]

            if a == NONE:
                sanitized.append(NONE)
                continue

            if a == LOAD:
                sanitized.append(LOAD if adjacent_food_exists(rcell) else NONE)
                continue

            if a in (NORTH, SOUTH, WEST, EAST):
                tgt = self.move_target(rcell, a)

                # prevent stepping into returner's reserved next cell
                if self.ret_reserved_cell is not None and rn != self.returning_robot and tgt == self.ret_reserved_cell:
                    sanitized.append(NONE)
                    continue

                if (not self.in_bounds(tgt)) or (tgt in obstacle_cells) or (tgt in food_cells):
                    sanitized.append(NONE)
                else:
                    sanitized.append(a)
                continue

            sanitized.append(NONE)

        acts = sanitized

        # stuck + force valid move
        forced_flags = {rn: False for rn in self.robots}
        for i, rn in enumerate(self.robots):
            # returning robot is controlled elsewhere; don't run stuck logic on it
            if rn == self.returning_robot:
                self.stuck[rn] = 0
                continue

            self.stuck[rn] = (self.stuck[rn] + 1) if (acts[i] == NONE) else 0
            if self.enable_stuck_rescue and self.stuck[rn] >= self.stuck_threshold:
                cand = valid_moves(cur[rn])
                acts[i] = random.choice(cand) if cand else NONE
                forced_flags[rn] = True
                self.stuck[rn] = 0
                self.get_logger().info(f"🛟 {rn} stuck -> forcing {ACTION_NAMES.get(acts[i], acts[i])}({acts[i]})")

        # feedback to rl_loop
        for rn, raw_a, san_a in zip(self.robots, acts_in, acts):
            forced = bool(forced_flags[rn] or forced_by_return[rn])
            valid = (raw_a == san_a) and (not forced)
            fb = {"robot": rn, "raw": int(raw_a), "sanitized": int(san_a), "valid": bool(valid), "forced": bool(forced)}
            self.action_fb_pub.publish(String(data=json.dumps(fb)))

        # collision resolution
        collisions = {}
        for rn, a in zip(self.robots, acts):
            tgt = self.move_target(cur[rn], a) if a in (NORTH, SOUTH, WEST, EAST) else cur[rn]
            collisions.setdefault(tgt, []).append(rn)

        final = dict(cur)
        for cell, rns in collisions.items():
            if len(rns) == 1:
                final[rns[0]] = cell
            else:
                for rn in rns:
                    final[rn] = cur[rn]

        # commit discrete move
        self.disc_cells.update(final)

        # publish waypoints for executed moves
        for rn, a in zip(self.robots, acts):
            if a not in (NORTH, SOUTH, WEST, EAST):
                continue
            if final[rn] == cur[rn]:
                continue
            self.publish_waypoint_to_cell(rn, final[rn])

        # LOAD resolution
        cells_after_move = final
        loading_robots = {rn for rn, a in zip(self.robots, acts) if a == LOAD}

        def adjacent_food_location_like_lbf(r_cell):
            rr, cc = r_cell
            cand = [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]
            for nb in cand:
                if nb in food_cells:
                    return nb
            return None

        target_groups = {}
        for rn in loading_robots:
            food_cell = adjacent_food_location_like_lbf(cells_after_move[rn])
            if food_cell is None:
                continue
            target_groups.setdefault(food_cell, []).append(rn)

        newly_collected = []
        for food_cell, group in target_groups.items():
            fidx = food_pos_to_idx.get(food_cell, None)
            if fidx is None:
                continue
            food_level = int(self.foods[fidx]["lvl"])
            level_sum = sum(int(self.robot_levels[rn]) for rn in group)
            if level_sum >= food_level:
                newly_collected.append((food_cell, food_level, group, level_sum))

        if newly_collected:
            now_s = float(self.get_clock().now().nanoseconds) * 1e-9
            for (food_cell, food_level, group, level_sum) in newly_collected:
                if food_cell in self.collected_set:
                    continue
                self.collected_set.add(food_cell)

                self.collected_visuals.append({
                    "rc": (food_cell[0], food_cell[1]),
                    "lvl": int(food_level),
                    "t_end": now_s + self.collected_visual_seconds,
                })

                evt = String()
                evt.data = json.dumps({
                    "event": "food_collected",
                    "cell": [int(food_cell[0]), int(food_cell[1])],
                    "level": int(food_level),
                    "by": list(group),
                    "level_sum": int(level_sum),
                    "needed": int(food_level),
                })
                self.food_evt_pub.publish(evt)

                # Return-to-origin is disabled for the current RViz replay.
                # Robots remain beside the collected task and continue from
                # their current cells when the next actions are published.

            self.publish_everything()

    # =======================================================
    # TF helper
    # =======================================================
    def robot_cell_tf(self, rn: str):
        try:
            t = self.tfbuf.lookup_transform("world", f"{rn}_base_link", rclpy.time.Time())
            return world_to_rc(t.transform.translation.x, t.transform.translation.y)
        except Exception:
            return (0, 0)


def main():
    rclpy.init()
    node = BoardServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
