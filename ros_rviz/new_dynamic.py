#!/usr/bin/env python3
# dynamic_multi_sim.py — holonomic multi-robot TF + waypoint follower (SMOOTH)
# ✅ Publishes /board/init_cells (latched) with the initial spawn cells

import os, time, math, subprocess, shlex, json

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
)
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from std_msgs.msg import String, Header
from tf2_ros.transform_broadcaster import TransformBroadcaster


class DynamicMultiSim(Node):
    def __init__(self):
        super().__init__('dynamic_multi_sim')

        self.urdf_dir = os.environ.get('URDF_DIR', '/home/smartubuntu3/meghana')

        self.dt = 0.05

        # ✅ slower = smoother visuals
        self.k_lin = 0.6
        self.max_v = 0.25

        self.goal_tol = 0.08
        self.stop_tol = 0.01

        # SAME grid mapping: row->y, col->x, centered 10x10
        self.robots = {
            'robot1': {'base_link': 'robot1_base_link', 'pose': [-3.5, -3.5, 0.0, 0.0], 'vel':[0.0,0.0], 'queue':[], 'going':False,
                       'override': {'vx':0.0,'vy':0.0,'until':0.0}, 'last_log':0.0},
            'robot2': {'base_link': 'robot2_base_link', 'pose': [3.5, -0.5, 0.0, 0.0], 'vel':[0.0,0.0], 'queue':[], 'going':False,
                       'override': {'vx':0.0,'vy':0.0,'until':0.0}, 'last_log':0.0},
            'robot3': {'base_link': 'robot3_base_link', 'pose': [-3.5,  2.5, 0.0, 0.0], 'vel':[0.0,0.0], 'queue':[], 'going':False,
                       'override': {'vx':0.0,'vy':0.0,'until':0.0}, 'last_log':0.0},
        }

        self.tf_broadcaster = TransformBroadcaster(self)

        # publishes {"robot":"robotX"} when robot reaches goal
        self.pub_task_done = self.create_publisher(String, '/task_done', 10)

        # ✅ NEW: init cells publisher (latched)
        q_init = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # latched
        )
        self.init_cells_pub = self.create_publisher(String, "/board/init_cells", q_init)
        self._init_cells_sent = False
        self.create_timer(0.5, self._publish_init_cells_once)

        q_desc = QoSProfile(depth=1)
        q_desc.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.desc_pub = {r: self.create_publisher(String, f'/{r}/robot_description', q_desc)
                         for r in self.robots.keys()}

        self.pub_wp_ack = {}
        for r in self.robots:
            self.pub_wp_ack[r] = self.create_publisher(Header, f'/{r}/waypoint_ack', 10)
            self.create_subscription(Twist, f'/{r}/cmd_vel', self._mk_cmd_cb(r), 10)
            self.create_subscription(PoseStamped, f'/{r}/waypoint', self._mk_wp_cb(r), 10)
            self.create_subscription(PoseStamped, f'/{r}/goal', self._mk_wp_cb(r), 10)

        self.urdf_cache = {}
        self.rsp_procs = []
        self._launch_rsp_for_robots()
        self._publish_descriptions_once()

        self.timer = self.create_timer(self.dt, self.step)
        self.get_logger().info("✅ DynamicMultiSim running (SMOOTH)")

    # ---- MUST match board mapping: centered 10x10, CELL=1.0 ----
    def _world_to_rc(self, x: float, y: float):
        N = 10
        CELL = 1.0
        half = 0.5 * N * CELL
        ox, oy = -half, -half  # (-5,-5)

        col = int(math.floor((x - ox) / CELL))
        row = int(math.floor((y - oy) / CELL))
        row = max(0, min(N - 1, row))
        col = max(0, min(N - 1, col))
        return row, col

    def _publish_init_cells_once(self):
        if self._init_cells_sent:
            return

        init = {}
        for name, data in self.robots.items():
            x, y, z, yaw = data["pose"]
            r, c = self._world_to_rc(float(x), float(y))
            init[name] = [int(r), int(c)]

        self.init_cells_pub.publish(String(data=json.dumps(init)))
        self.get_logger().info(f"📌 Published /board/init_cells: {init}")
        self._init_cells_sent = True

    def _mk_cmd_cb(self, name):
        def cb(msg: Twist):
            now = time.time()
            ov = self.robots[name]['override']
            ov['vx'] = float(max(-self.max_v, min(self.max_v, msg.linear.x)))
            ov['vy'] = float(max(-self.max_v, min(self.max_v, msg.linear.y)))
            ov['until'] = now + 0.25
        return cb

    def _mk_wp_cb(self, name):
        def cb(msg: PoseStamped):
            gx = float(msg.pose.position.x)
            gy = float(msg.pose.position.y)
            self.robots[name]['queue'] = [(gx, gy)]
            self.robots[name]['going'] = True

            hdr = Header()
            hdr.stamp = msg.header.stamp
            hdr.frame_id = 'ack'
            self.pub_wp_ack[name].publish(hdr)

            self.get_logger().info(f"📍 {name}: waypoint -> ({gx:.2f}, {gy:.2f})")
        return cb

    def _load_urdf_text(self, name: str) -> str:
        urdf_path = os.path.join(self.urdf_dir, f'{name}.urdf')
        xacro_path = os.path.join(self.urdf_dir, f'{name}.urdf.xacro')

        if os.path.isfile(urdf_path):
            with open(urdf_path, 'r') as f:
                return f.read()

        if os.path.isfile(xacro_path):
            cmd = f'xacro {shlex.quote(xacro_path)}'
            out = subprocess.check_output(cmd, shell=True, env=os.environ).decode('utf-8')
            return out

        self.get_logger().warn(f"URDF for {name} not found in {self.urdf_dir}.")
        return f"<robot name='{name}_empty'/>"

    def _launch_rsp_for_robots(self):
        for name in self.robots.keys():
            urdf_text = self._load_urdf_text(name)
            self.urdf_cache[name] = urdf_text

            env_ns = {**os.environ, 'ROS_NAMESPACE': name}
            proc = subprocess.Popen(
                [
                    'ros2', 'run', 'robot_state_publisher', 'robot_state_publisher',
                    '--ros-args',
                    '-p', f'robot_description:={urdf_text}',
                    '-p', 'use_sim_time:=true',
                    '-r', '/tf:=/tf',
                    '-r', '/tf_static:=/tf_static'
                ],
                env=env_ns
            )
            self.rsp_procs.append(proc)
            self.get_logger().info(f"📦 robot_state_publisher in namespace '{name}'")

    def _publish_descriptions_once(self):
        for name, urdf in self.urdf_cache.items():
            self.desc_pub[name].publish(String(data=urdf))

        self.create_timer(
            1.0,
            lambda: [self.desc_pub[n].publish(String(data=u)) for n, u in self.urdf_cache.items()]
        )

    def step(self):
        now_msg = self.get_clock().now().to_msg()
        now_s = time.time()

        for name, data in self.robots.items():
            x, y, z, yaw = data['pose']
            vx, vy = data['vel']
            ov = data['override']

            used_override = False
            if ov['until'] > now_s:
                vx, vy, used_override = ov['vx'], ov['vy'], True

            if not used_override and data['going'] and data['queue']:
                gx, gy = data['queue'][0]
                dx, dy = gx - x, gy - y
                dist = math.hypot(dx, dy)

                if dist < self.goal_tol:
                    data['queue'].pop(0)
                    if not data['queue']:
                        data['going'] = False
                        vx, vy = 0.0, 0.0
                        self.pub_task_done.publish(String(data=f'{{"robot":"{name}"}}'))
                else:
                    speed = min(self.max_v, self.k_lin * dist)
                    ux, uy = dx / dist, dy / dist
                    vx, vy = speed * ux, speed * uy

            if not data['going'] and not used_override:
                vx *= 0.8
                vy *= 0.8
                if abs(vx) < self.stop_tol:
                    vx = 0.0
                if abs(vy) < self.stop_tol:
                    vy = 0.0

            x += vx * self.dt
            y += vy * self.dt

            data['pose'] = [x, y, z, yaw]
            data['vel'] = [vx, vy]

            tf = TransformStamped()
            tf.header.stamp = now_msg
            tf.header.frame_id = 'world'
            tf.child_frame_id = data['base_link']
            tf.transform.translation.x = x
            tf.transform.translation.y = y
            tf.transform.translation.z = z
            tf.transform.rotation.z = 0.0
            tf.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(tf)

    def destroy_node(self):
        for p in getattr(self, 'rsp_procs', []):
            try:
                p.terminate()
            except Exception:
                pass
        return super().destroy_node()


def main():
    rclpy.init()
    node = DynamicMultiSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
