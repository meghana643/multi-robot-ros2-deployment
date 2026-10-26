#!/usr/bin/env python3
import json
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray, String, Header

import torch
import torch.nn as nn


class ActorGRU(nn.Module):
    def __init__(self, obs_dim: int, rnn_input_size: int, hidden_size: int, action_dim: int):
        super().__init__()
        self.first_layer = nn.Linear(obs_dim, rnn_input_size)
        self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.final_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, obs_1d: torch.Tensor, h: torch.Tensor):
        x = torch.relu(self.first_layer(obs_1d))
        x = x.unsqueeze(1)
        y, new_h = self.rnn(x, h)
        y = y.squeeze(1)
        logits = self.final_layer(y)
        return logits, new_h


class MultiAgentActor(nn.Module):
    def __init__(self, n_agents: int, obs_dim: int, rnn_input_size: int, hidden_size: int, action_dim: int):
        super().__init__()
        self.independent = nn.ModuleList([
            ActorGRU(obs_dim, rnn_input_size, hidden_size, action_dim)
            for _ in range(n_agents)
        ])


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
        for k in ["state_dict", "model", "model_state_dict", "policy", "actor"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    raise RuntimeError("Could not find a usable state_dict in the checkpoint.")


def _infer_actor_shapes(sd: dict):
    w0 = sd["actor.independent.0.first_layer.weight"]
    rnn_input_size, obs_dim = int(w0.shape[0]), int(w0.shape[1])
    wih = sd["actor.independent.0.rnn.weight_ih_l0"]
    hidden_size = int(wih.shape[0] // 3)
    wf = sd["actor.independent.0.final_layer.weight"]
    action_dim = int(wf.shape[0])
    return obs_dim, rnn_input_size, hidden_size, action_dim


ACTION_NAMES = {0: "NONE", 1: "NORTH", 2: "SOUTH", 3: "WEST", 4: "EAST", 5: "LOAD"}


class RLLoop(Node):
    def __init__(self):
        super().__init__("rl_loop")

        self.robots = ["robot1", "robot2", "robot3"]
        self.n_agents = 3
        self.device = torch.device("cpu")

        self.checkpoint_path = "/home/smartubuntu3/MARL_MAPPO_meghana/marlbase/outputs/lbforaging:Foraging-10x10-3p-4f-v3/mappo/c8a74dc2/checkpoints/model_s9869880.pt"

        self.tick_period_s = 0.2
        self.ack_collect_window_s = 0.35
        self.max_wait_s = 8.0

        raw = torch.load(self.checkpoint_path, map_location=self.device)
        sd = _extract_state_dict(raw)
        obs_dim, rnn_in, hidden, action_dim = _infer_actor_shapes(sd)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden

        self.get_logger().info(
            f"✅ Loaded checkpoint. Inferred dims: obs_dim={obs_dim}, rnn_in={rnn_in}, hidden={hidden}, action_dim={action_dim}"
        )

        self.actor = MultiAgentActor(self.n_agents, obs_dim, rnn_in, hidden, action_dim).to(self.device)

        actor_sd = {}
        for k, v in sd.items():
            if k.startswith("actor."):
                actor_sd[k[len("actor."):]] = v
        self.actor.load_state_dict(actor_sd, strict=True)
        self.actor.eval()

        self.current_obs = {r: None for r in self.robots}
        self.h = {r: torch.zeros((1, 1, self.hidden_size), dtype=torch.float32, device=self.device) for r in self.robots}

        # stop condition
        self.total_food = None
        self.collected_count = 0
        self.done = False

        # step gating
        self.step_idx = 0
        self.waiting = False
        self.wait_for = set()
        self.finished = set()
        self.wait_started_t = 0.0

        self.collecting_acks = False
        self.ack_collect_until_t = 0.0
        self.acked_this_step = set()

        # subs: obs
        for r in self.robots:
            self.create_subscription(Float32MultiArray, f"/{r}/rl_obs", self._mk_obs_cb(r), 10)

        # subs: food + events
        self.create_subscription(Int32MultiArray, "/board/food_grid", self.on_food_grid, 10)
        self.create_subscription(String, "/board/food_events", self.on_food_event, 10)

        # subs: movement sync
        self.create_subscription(String, "/task_done", self.on_task_done, 10)
        for r in self.robots:
            self.create_subscription(Header, f"/{r}/waypoint_ack", self._mk_ack_cb(r), 10)

        # ✅ Option B: listen to action feedback and reset hidden on invalid/forced
        self.create_subscription(String, "/board/action_feedback", self.on_action_feedback, 10)

        # pub
        self.action_pub = self.create_publisher(Int32MultiArray, "/rl_actions", 10)

        self.timer = self.create_timer(self.tick_period_s, self.tick)

        self.get_logger().info("✅ rl_loop running (smooth-sync + hidden reset on invalid/forced)")

    def _mk_obs_cb(self, robot_name):
        def cb(msg: Float32MultiArray):
            obs = np.asarray(msg.data, dtype=np.float32)
            if obs.shape[0] != self.obs_dim:
                self.get_logger().warn(f"{robot_name}: got obs len {obs.shape[0]} expected {self.obs_dim}")
                return
            self.current_obs[robot_name] = obs
        return cb

    def _mk_ack_cb(self, robot_name):
        def cb(_msg: Header):
            if self.collecting_acks:
                self.acked_this_step.add(robot_name)
        return cb

    def on_action_feedback(self, msg: String):
        # {"robot":"robot1","raw":4,"sanitized":0,"valid":false,"forced":false}
        try:
            fb = json.loads(msg.data)
            r = fb.get("robot", "")
            valid = bool(fb.get("valid", True))
            forced = bool(fb.get("forced", False))
        except Exception:
            return

        if r not in self.h:
            return

        # Reset hidden state if invalid or forced (break the RNN loop)
        if (not valid) or forced:
            self.h[r].zero_()
            self.get_logger().info(f"🧠 reset hidden for {r} (valid={valid}, forced={forced})")

    def on_food_grid(self, msg: Int32MultiArray):
        arr = np.asarray(msg.data, dtype=np.int32)
        if arr.size == 0:
            return
        if self.total_food is None:
            self.total_food = int(np.count_nonzero(arr))
            self.get_logger().info(f"🍴 Total foods detected: {self.total_food}")

    def on_food_event(self, msg: String):
        self.get_logger().info(f"🍉 FOOD EVENT: {msg.data}")
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        if data.get("event") == "food_collected":
            self.collected_count += 1
            if self.total_food is not None and self.collected_count >= self.total_food and not self.done:
                self.done = True
                self.get_logger().info("✅ All food collected. Stopping RL loop.")
                try:
                    self.timer.cancel()
                except Exception:
                    pass
                out = Int32MultiArray()
                out.data = [0, 0, 0]
                self.action_pub.publish(out)

    def on_task_done(self, msg: String):
        try:
            data = json.loads(msg.data)
            r = data.get("robot", "")
        except Exception:
            return

        if not self.waiting:
            return

        if r in self.wait_for:
            self.finished.add(r)

        if self.finished == self.wait_for:
            self.waiting = False
            self.wait_for.clear()
            self.finished.clear()
            self.get_logger().info("✅ All moved robots reached waypoint -> next RL step")

    @torch.no_grad()
    def _compute_actions(self):
        actions = []
        action_strs = []
        for i, r in enumerate(self.robots):
            obs_np = self.current_obs[r]
            obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, new_h = self.actor.independent[i](obs, self.h[r])
            self.h[r] = new_h
            a = int(torch.argmax(logits, dim=-1).item())
            actions.append(a)
            action_strs.append(f"{r}:{ACTION_NAMES.get(a, str(a))}({a})")
        return actions, action_strs

    def tick(self):
        if self.done:
            return

        now = time.time()

        if self.waiting:
            if (now - self.wait_started_t) > self.max_wait_s:
                self.get_logger().warn("⚠️ Wait timeout. Continuing to next RL step.")
                self.waiting = False
                self.wait_for.clear()
                self.finished.clear()
            else:
                return

        if self.collecting_acks:
            if now >= self.ack_collect_until_t:
                self.collecting_acks = False
                if len(self.acked_this_step) > 0:
                    self.waiting = True
                    self.wait_for = set(self.acked_this_step)
                    self.finished = set()
                    self.wait_started_t = now
                    self.get_logger().info(f"⏳ Waiting for: {sorted(self.wait_for)}")
                self.acked_this_step.clear()
            return

        if any(self.current_obs[r] is None for r in self.robots):
            return

        self.step_idx += 1
        actions, action_strs = self._compute_actions()

        self.get_logger().info(f"STEP {self.step_idx} | " + "  ".join(action_strs))

        out = Int32MultiArray()
        out.data = actions
        self.action_pub.publish(out)

        self.collecting_acks = True
        self.ack_collect_until_t = now + self.ack_collect_window_s
        self.acked_this_step.clear()


def main():
    rclpy.init()
    node = RLLoop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

