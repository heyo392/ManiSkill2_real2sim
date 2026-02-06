from collections import OrderedDict
from typing import List, Optional

import numpy as np
import cv2
import sapien.core as sapien
from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name
from transforms3d.euler import euler2quat
from mani_skill2_real2sim.utils.common import random_choice
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult
from mani_skill2_real2sim.utils.sapien_utils import (
    get_pairwise_contacts,
    compute_total_impulse,
)

from .base_env import CustomOtherObjectsInSceneEnv, CustomSceneEnv
from .open_drawer_in_scene import OpenDrawerInSceneEnv


class PlaceObjectInClosedDrawerInSceneEnv(OpenDrawerInSceneEnv):

    def __init__(
        self,
        force_advance_subtask_time_steps: int = 100,
        target_drawer_number: int = -1,
        **kwargs,
    ):
        self.model_id = None
        self.model_scale = None
        self.model_bbox_size = None
        self.obj = None
        self.obj_init_options = {}

        self.force_advance_subtask_time_steps = force_advance_subtask_time_steps
        self.target_drawer_number = target_drawer_number

        # Make drawers easier to pull by default unless explicitly overridden
        if "cabinet_joint_friction" not in kwargs:
            kwargs["cabinet_joint_friction"] = 0.003

        # Bake local defaults for this scene; still user-overridable via kwargs
        kwargs.setdefault(
            "robot", "google_robot_static_twice_finger_friction"
        )
        kwargs.setdefault("scene_name", "dummy_drawer")
        kwargs.setdefault("station_name", "mk_station_flush")
        kwargs.setdefault(
            "control_mode",
            "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
        )
        kwargs.setdefault("model_ids", "opened_coke_can")

        super().__init__(**kwargs)

    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.contact_offset = (
            0.005
        )  # avoid "false-positive" collisions with other objects
        return scene_config
    
    def _set_model(self, model_id, model_scale):
        """Set the model id and scale. If not provided, choose one randomly from self.model_ids."""
        reconfigure = False

        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
            # Prefer a can by default if randomly choosing; do not override explicit requests
            if model_id == "apple":
                preferred_cans = [
                    "opened_pepsi_can",
                    "opened_coke_can",
                    "opened_sprite_can",
                    "opened_fanta_can",
                    "opened_redbull_can",
                ]
                available_cans = [mid for mid in self.model_ids if mid in preferred_cans]
                if len(available_cans) > 0:
                    model_id = random_choice(available_cans, self._episode_rng)
        if model_id != self.model_id:
            self.model_id = model_id
            reconfigure = True

        if model_scale is None:
            model_scales = self.model_db[self.model_id].get("scales")
            if model_scales is None:
                model_scale = 1.0
            else:
                model_scale = random_choice(model_scales, self._episode_rng)
        if model_scale != self.model_scale:
            self.model_scale = model_scale
            reconfigure = True

        model_info = self.model_db[self.model_id]
        if "bbox" in model_info:
            bbox = model_info["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.model_bbox_size = bbox_size * self.model_scale
        else:
            self.model_bbox_size = None

        return reconfigure

    def _load_model(self):
        model_info = self.model_db[self.model_id]
        # Primitive support via model_db
        if isinstance(model_info, dict) and (model_info.get("primitive") is not None):
            prim = model_info.get("primitive")
            density = model_info.get("density", 1000)
            color = model_info.get("color", [0.6, 0.6, 0.6])
            material = self._scene.create_physical_material(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0,
            )
            builder = self._scene.create_actor_builder()
            if prim == "box":
                half_size = np.asarray(model_info.get("half_size", [0.05, 0.05, 0.05]), dtype=np.float32)
                half_size = half_size * float(self.model_scale)
                builder.add_box_collision(half_size=half_size, material=material, density=density)
                builder.add_box_visual(half_size=half_size, color=color)
            else:
                raise NotImplementedError(f"Unsupported primitive: {prim}")
            self.obj = builder.build()
            self.obj.name = self.model_id
            return

        density = model_info.get("density", 1000)

        self.obj = self._build_actor_helper(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            density=density,
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0,
            ),
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id

    def _load_actors(self):
        super()._load_actors()
        self._load_model()
        self.obj.set_damping(0.1, 0.1)

    def _initialize_actors(self):
        # The object will fall from a certain initial height
        obj_init_xy = self.obj_init_options.get("init_xy", None)
        if obj_init_xy is None:
            obj_init_xy = self._episode_rng.uniform([-0.10, -0.00], [-0.05, 0.1], [2])
        obj_init_z = self.obj_init_options.get("init_z", self.scene_table_height)
        obj_init_z = obj_init_z + 0.5  # let object fall onto the table
        obj_init_rot_quat = self.obj_init_options.get("init_rot_quat", [1, 0, 0, 0])
        p = np.hstack([obj_init_xy, obj_init_z])
        q = obj_init_rot_quat

        # Rotate along z-axis
        if self.obj_init_options.get("init_rand_rot_z", False):
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = qmult(euler2quat(0, 0, ori), q)

        # Rotate along a random axis by a small angle
        if (
            init_rand_axis_rot_range := self.obj_init_options.get(
                "init_rand_axis_rot_range", 0.0
            )
        ) > 0:
            axis = self._episode_rng.uniform(-1, 1, 3)
            axis = axis / max(np.linalg.norm(axis), 1e-6)
            ori = self._episode_rng.uniform(0, init_rand_axis_rot_range)
            q = qmult(q, axangle2quat(axis, ori, True))
        self.obj.set_pose(sapien.Pose(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later in _initialize_agent (in base_env.py)
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # Lock rotation around x and y to let the target object fall onto the table
        self.obj.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)

        # Unlock motion
        self.obj.lock_motion(0, 0, 0, 0, 0, 0)
        # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
        self.obj.set_pose(self.obj.pose)
        self.obj.set_velocity(np.zeros(3))
        self.obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(self.obj.velocity)
        ang_vel = np.linalg.norm(self.obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(1.5)

        # Record the object height after it settles
        self.obj_height_after_settle = self.obj.pose.p[2]

    def _initialize_agent(self):
        super()._initialize_agent()
        # Optional: set initial end-effector (TCP) pose w.r.t robot base using IK
        # Accept either robot_init_options: {ee_init_pos, ee_init_quat} or {ee_init_pose: {p, q}}
        rio = getattr(self, "robot_init_options", {}) or {}
        ee_p = None
        ee_q = None
        if isinstance(rio.get("ee_init_pose"), dict):
            pose_dict = rio["ee_init_pose"]
            ee_p = pose_dict.get("p", None)
            ee_q = pose_dict.get("q", None)
        else:
            ee_p = rio.get("ee_init_pos", rio.get("ee_init_p", None))
            ee_q = rio.get("ee_init_quat", rio.get("ee_init_q", None))

        if ee_p is not None and ee_q is not None:
            ee_p = np.asarray(ee_p, dtype=np.float32)
            ee_q = np.asarray(ee_q, dtype=np.float32)
            target_pose = sapien.Pose(p=ee_p, q=ee_q)

            controller = self.agent.controller.controllers["arm"]
            cur_qpos = self.agent.robot.get_qpos()
            init_arm_qpos = controller.compute_ik(target_pose)
            cur_qpos[controller.joint_indices] = init_arm_qpos
            self.agent.reset(cur_qpos)

    def get_object_world_position(self) -> np.ndarray:
        if self.obj is None:
            raise RuntimeError("Target object has not been created; call reset() first.")
        return np.asarray(self.obj.pose.p, dtype=np.float32)

    def get_object_world_orientation(self) -> np.ndarray:
        """Return the object's world orientation quaternion as float32 array of shape (4,).

        The quaternion ordering matches SAPIEN's Pose.q ordering as returned by the engine.
        """
        if self.obj is None:
            raise RuntimeError("Target object has not been created; call reset() first.")
        return np.asarray(self.obj.pose.q, dtype=np.float32)

    def _add_object_position_to_info(self, info: dict) -> dict:
        p = self.get_object_world_position()
        info["object_position_world"] = [float(p[0]), float(p[1]), float(p[2])]
        return info

    def _add_object_orientation_to_info(self, info: dict) -> dict:
        q = self.get_object_world_orientation()
        info["object_orientation_world"] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        return info

    def get_object_robot_pose(self):
        """Return object's pose in robot base frame as (position, quaternion).

        Position shape (3,), quaternion shape (4,) with SAPIEN's Pose.q ordering.
        """
        if self.obj is None:
            raise RuntimeError("Target object has not been created; call reset() first.")
        base_in_world = self.agent.robot.pose
        obj_in_world = self.obj.pose
        obj_in_base = base_in_world.inv() * obj_in_world
        p = np.asarray(obj_in_base.p, dtype=np.float32)
        q = np.asarray(obj_in_base.q, dtype=np.float32)
        return p, q

    def _add_object_pose_robot_to_info(self, info: dict) -> dict:
        p, q = self.get_object_robot_pose()
        info["object_position_robot"] = [float(p[0]), float(p[1]), float(p[2])]
        info["object_orientation_robot"] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        return info

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        self.set_episode_rng(seed)

        # set objects
        self.obj_init_options = options.get("obj_init_options", {})
        model_scale = options.get("model_scale", None)
        model_id = options.get("model_id", None)
        reconfigure = options.get("reconfigure", False)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure
        options["reconfigure"] = reconfigure

        # Handle drawer_number option (use instance variable as default)
        drawer_number = options.get("drawer_number", self.target_drawer_number)
        self.target_drawer_number = drawer_number

        obs, info = super().reset(seed=self._episode_seed, options=options)
        
        # Override drawer_id AFTER parent reset (parent always randomly selects a drawer)
        if self.target_drawer_number in [0, 1, 2]:
            self.drawer_id = self.drawer_ids[self.target_drawer_number]
            # Update joint_idx to match the selected drawer
            self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")
        
        self.drawer_link: sapien.Link = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        # Prefer the URDF-marked bottom: local z near -0.06; fallback to lowest local z
        def _select_bottom_collision_shape(link: sapien.Link, expected_local_z: float = -0.06, tol: float = 1e-3):
            shapes = link.get_collision_shapes()
            if len(shapes) == 0:
                return None
            # First pass: exact match to expected bottom z from URDF
            for s in shapes:
                local_z = float(s.get_local_pose().p[2])
                if abs(local_z - expected_local_z) <= tol:
                    return s
            # Fallback: choose the shape with minimal local z
            min_z = float("inf")
            chosen = None
            for s in shapes:
                local_z = float(s.get_local_pose().p[2])
                if local_z < min_z:
                    min_z = local_z
                    chosen = s
            return chosen

        self.drawer_collision = _select_bottom_collision_shape(self.drawer_link) if self.drawer_link is not None else None
        
        if self.target_drawer_number in [0, 1, 2]:
            qpos = self.art_obj.get_qpos()
            qpos[self.joint_idx] = 0.18
            self.art_obj.set_qpos(qpos)
            # Regenerate observation after opening drawer so it reflects the open state
            obs = self.get_obs()
        
        info = self._add_object_position_to_info(info)
        info = self._add_object_orientation_to_info(info)
        info = self._add_object_pose_robot_to_info(info)
        return obs, info

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged evaluation configs under visual matching setup
        overlay_ids = ["a0", "b0", "c0"]
        rgb_overlay_paths = [
            str(ASSET_DIR / f"real_inpainting/open_drawer_{i}.png") for i in overlay_ids
        ]
        robot_init_xs = [0.644, 0.652, 0.665]
        robot_init_ys = [-0.179, 0.009, 0.224]
        robot_init_rotzs = [-0.03, 0, 0]
        idx_chosen = self._episode_rng.choice(len(overlay_ids))

        options["robot_init_options"] = {
            "init_xy": [robot_init_xs[idx_chosen], robot_init_ys[idx_chosen]],
            "init_rot_quat": (
                sapien.Pose(q=euler2quat(0, 0, robot_init_rotzs[idx_chosen]))
                * sapien.Pose(q=[0, 0, 0, 1])
            ).q,
        }
        self.rgb_overlay_img = (
            cv2.cvtColor(cv2.imread(rgb_overlay_paths[idx_chosen]), cv2.COLOR_BGR2RGB)
            / 255
        )
        new_urdf_version = self._episode_rng.choice(
            [
                "",
                "recolor_tabletop_visual_matching_1",
                "recolor_tabletop_visual_matching_2",
                "recolor_cabinet_visual_matching_1",
            ]
        )
        if new_urdf_version != self.urdf_version:
            self.urdf_version = new_urdf_version
            self._configure_agent()
            return True
        return False

    def _initialize_episode_stats(self):
        self.cur_subtask_id = 0 # 0: open drawer, 1: place object into drawer
        self.episode_stats = OrderedDict(
            phase=0,
            qpos=0.0,
            is_drawer_open=False,
            has_contact=0,
        )

    def evaluate(self, **kwargs):
        # Drawer
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = qpos
        is_drawer_open = qpos >= 0.15
        self.episode_stats["is_drawer_open"] = self.episode_stats["is_drawer_open"] or is_drawer_open

        # Check whether the object contacts with the drawer
        contact_infos = get_pairwise_contacts(
            self._scene.get_contacts(),
            self.obj,
            self.drawer_link,
            collision_shape1=self.drawer_collision,
        )
        total_impulse = compute_total_impulse(contact_infos)
        has_contact = np.linalg.norm(total_impulse) > 1e-6
        self.episode_stats["has_contact"] += has_contact

        success = (self.cur_subtask_id == 1) and (qpos >= 0.05) and (self.episode_stats["has_contact"] >= 1)

        # Expose current phase for downstream evaluators
        self.episode_stats["phase"] = int(self.cur_subtask_id)

        return dict(success=success, episode_stats=self.episode_stats)

    def advance_to_next_subtask(self):
        self.cur_subtask_id = 1

    def step(self, action):
        result = super().step(action)
        # Append object position into the returned info dict without altering upstream API
        if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[-1], dict):
            result_list = list(result)
            result_list[-1] = self._add_object_position_to_info(result_list[-1])
            result_list[-1] = self._add_object_orientation_to_info(result_list[-1])
            result_list[-1] = self._add_object_pose_robot_to_info(result_list[-1])
            return tuple(result_list)
        return result
    
    def get_language_instruction(self, **kwargs):
        if self.cur_subtask_id == 0:
            return f"open {self.drawer_id} drawer"
        else:
            model_name = self._get_instruction_obj_name(self.model_id)
            return f"place {model_name} into {self.drawer_id} drawer"
        
    def is_final_subtask(self):
        return self.cur_subtask_id == 1


@register_env("PlaceIntoClosedDrawerCustomInScene-v0", max_episode_steps=200)
class PlaceIntoClosedDrawerCustomInSceneEnv(
    PlaceObjectInClosedDrawerInSceneEnv, CustomOtherObjectsInSceneEnv
):
    DEFAULT_MODEL_JSON = "info_pick_custom_baked_tex_v1.json"
    drawer_ids = ["top", "middle", "bottom"]


@register_env("PlaceIntoClosedTopDrawerCustomInScene-v0", max_episode_steps=200)
class PlaceIntoClosedTopDrawerCustomInSceneEnv(PlaceIntoClosedDrawerCustomInSceneEnv):
    drawer_ids = ["top"]


@register_env("PlaceIntoClosedMiddleDrawerCustomInScene-v0", max_episode_steps=200)
class PlaceIntoClosedMiddleDrawerCustomInSceneEnv(
    PlaceIntoClosedDrawerCustomInSceneEnv
):
    drawer_ids = ["middle"]


@register_env("PlaceIntoClosedBottomDrawerCustomInScene-v0", max_episode_steps=200)
class PlaceIntoClosedBottomDrawerCustomInSceneEnv(
    PlaceIntoClosedDrawerCustomInSceneEnv
):
    drawer_ids = ["bottom"]


class PlaceRetrieveFromDrawerInSceneEnv(PlaceObjectInClosedDrawerInSceneEnv):

    def __init__(
        self,
        button_impulse_threshold: float = 1.0,
        top_xy_half_extent: Optional[List[float]] = None,
        top_height_offset: float = 0.15,
        **kwargs,
    ):
        self.cur_subtask_id = 0  # 0: place-in, 1: close, 2: retrieve+place-top
        self._button = None
        self._button_impulse_threshold = button_impulse_threshold
        # Region on cabinet top in cabinet frame (later converted to world)
        if top_xy_half_extent is None:
            top_xy_half_extent = [0.25, 0.20]
        self._top_xy_half_extent = np.array(top_xy_half_extent, dtype=np.float32)
        self._top_height_offset = float(top_height_offset)
        super().__init__(**kwargs)

    def _load_actors(self):
        super()._load_actors()
        # Add a simple kinematic reset button at a fixed world pose near the cabinet
        # Chosen to be reachable but not obstructive
        builder = self._scene.create_actor_builder()
        half_size = np.array([0.02, 0.02, 0.01])
        builder.add_box_visual(half_size=half_size, color=[1.0, 0.2, 0.2])
        builder.add_box_collision(half_size=half_size)
        # self._button = builder.build_kinematic(name="reset_button")
        # Place the button on top of the table plane
        # self._button.set_pose(sapien.Pose([0.35, -0.10, self.scene_table_height + 0.02]))

    def _initialize_episode_stats(self):
        # Track per-component rewards and bookkeeping
        self.cur_subtask_id = 0  # 0: close with object, 1: press button, 2: retrieve+place on top
        # Internal cumulative progress flags (not exposed in episode_stats)
        self._completed_task1 = False
        self._completed_task2 = False
        self._completed_task3 = False
        self.episode_stats = OrderedDict(
            phase=0,
            inside_any_drawer=False,
            closed_drawer_with_object=False,
            button_pressed=False,
            retrieved_and_on_top=False,
            cube_in_drawer=-1,
            drawers_closed=[],
            target_drawer_number=-1,
        )

    def _get_drawer_link_and_collision(self, drawer_id: str):
        link = get_entity_by_name(self.art_obj.get_links(), f"{drawer_id}_drawer")
        if link is None:
            return None, None
        # Choose the interior bottom collision shape by URDF bottom z (hardcoded) with fallback
        def _select_bottom_collision_shape(link: sapien.Link, expected_local_z: float = -0.06, tol: float = 1e-3):
            shapes = link.get_collision_shapes()
            if len(shapes) == 0:
                return None
            for s in shapes:
                local_z = float(s.get_local_pose().p[2])
                if abs(local_z - expected_local_z) <= tol:
                    return s
            min_z = float("inf")
            chosen = None
            for s in shapes:
                local_z = float(s.get_local_pose().p[2])
                if local_z < min_z:
                    min_z = local_z
                    chosen = s
            return chosen
        return link, _select_bottom_collision_shape(link)

    def _is_object_inside_specific_drawer(self, drawer_link: sapien.Link, drawer_collision: Optional[sapien.CollisionShape]):
        """Contact-based drawer membership: object is considered inside if it contacts the drawer bin.

        Uses pairwise contact impulse between the object and the specified drawer link/shape.
        """
        if drawer_link is None or drawer_collision is None:
            return False
        contacts = self._scene.get_contacts()
        contact_infos = get_pairwise_contacts(
            contacts,
            self.obj,
            drawer_link,
            collision_shape1=drawer_collision,
        )
        total_impulse = compute_total_impulse(contact_infos)
        return np.linalg.norm(total_impulse) > 1e-6

    def _is_object_on_cabinet_top(self):
        """Contact-based support check indicating the object is supported from below by cabinet top.

        Implementation: accumulate upward (world-z) impulse on the object from cabinet links excluding drawer links.
        """
        drawer_names = [f"{did}_drawer" for did in getattr(self, "drawer_ids", ["top", "middle", "bottom"])]
        non_drawer_links = [
            link for link in self.art_obj.get_links() if link.get_name() not in drawer_names
        ]
        if len(non_drawer_links) == 0:
            return False

        contacts = self._scene.get_contacts()
        up_impulse_z = 0.0
        for c in contacts:
            if c.actor0 == self.obj and isinstance(c.actor1, sapien.Link) and c.actor1 in non_drawer_links:
                up_impulse_z += float(np.sum([p.impulse[2] for p in c.points]))
            elif c.actor1 == self.obj and isinstance(c.actor0, sapien.Link) and c.actor0 in non_drawer_links:
                up_impulse_z += float(np.sum([-p.impulse[2] for p in c.points]))

        return up_impulse_z > 1e-6

    def _after_simulation_step(self):
        # Button press detection (instantaneous, no teleport)
        if self._button is not None:
            contacts = self._scene.get_contacts()
            # Detect any contact between the button and a robot link that presses from above (top-only).
            pressed = False
            for c in contacts:
                if c.actor0 == self._button and isinstance(c.actor1, sapien.Link) and c.actor1.get_articulation() == self.agent.robot:
                    # If the button receives any downward impulse (negative world-z on the button), count as press
                    if float(np.sum([p.impulse[2] for p in c.points])) < 0.0:
                        pressed = True
                        break
                elif c.actor1 == self._button and isinstance(c.actor0, sapien.Link) and c.actor0.get_articulation() == self.agent.robot:
                    if float(np.sum([-p.impulse[2] for p in c.points])) < 0.0:
                        pressed = True
                        break
            # Reflect current contact state regardless of phase (debug-friendly)
            self.episode_stats["button_pressed"] = bool(pressed)
        return super()._after_simulation_step()

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Hardcode button position based on middle drawer at qpos=0 (closed)
        # We compute this once to anchor it to the closed drawer position
        mid_link = get_entity_by_name(self.art_obj.get_links(), "middle_drawer")
        if self._button is not None and mid_link is not None:
            # Save the current joint state
            current_qpos = self.art_obj.get_qpos().copy()
            # Temporarily set drawer to fully closed position
            closed_qpos = np.zeros_like(current_qpos)
            self.art_obj.set_qpos(closed_qpos)
            # Calculate world position with drawer closed
            offset_local = np.array([0.3, -0.15, 0.0], dtype=np.float32)
            world_p = (mid_link.pose * sapien.Pose(p=offset_local)).p
            button_pos = np.array([world_p[0], world_p[1], self.scene_table_height + 0.02])
            # Restore the original joint state
            self.art_obj.set_qpos(current_qpos)
            # Set button to the computed position (stays fixed even as drawer moves)
            self._button.set_pose(sapien.Pose(button_pos))
        return obs, info

    def advance_to_next_subtask(self):
        self.cur_subtask_id += 1

    def evaluate(self, **kwargs):
        # Evaluate across all drawers or specific drawer based on target_drawer_number
        qpos_all = self.art_obj.get_qpos()
        placed_any = False
        closed_with_object = False

        # Debug info: check all drawers
        cube_in_drawer = -1  # -1 means not in any drawer
        drawers_closed = []  # List of drawer indices that are closed
        
        # Check all drawers for debugging
        all_drawer_ids = getattr(self, "drawer_ids", ["top", "middle", "bottom"])
        for drawer_idx, did in enumerate(all_drawer_ids):
            link, col = self._get_drawer_link_and_collision(did)
            if link is None:
                continue
            joint_name = f"{did}_drawer_joint"
            if joint_name not in self.joint_names:
                continue
            joint_idx = self.joint_names.index(joint_name)
            
            # Check if cube is in this drawer
            inside = self._is_object_inside_specific_drawer(link, col)
            if inside:
                cube_in_drawer = drawer_idx
            
            # Check if this drawer is closed
            if qpos_all[joint_idx] <= 0.01:
                drawers_closed.append(drawer_idx)

        # Determine which drawers to check for success
        if self.target_drawer_number in [0, 1, 2]:
            # Only check the specific drawer
            drawers_to_check = [self.drawer_ids[self.target_drawer_number]]
        else:
            # Check all drawers (-1 case)
            drawers_to_check = all_drawer_ids

        for did in drawers_to_check:
            link, col = self._get_drawer_link_and_collision(did)
            if link is None:
                continue
            joint_name = f"{did}_drawer_joint"
            if joint_name not in self.joint_names:
                continue
            joint_idx = self.joint_names.index(joint_name)
            inside = self._is_object_inside_specific_drawer(link, col)
            if inside:
                placed_any = True
                if qpos_all[joint_idx] <= 0.005:
                    closed_with_object = True

        on_top = self._is_object_on_cabinet_top()
        inside_any = placed_any
        retrieved_and_on_top = (not inside_any) and on_top


        # Phase progression in order: (0) close with object -> (1) press button -> (2) retrieve+place on top
        if self.cur_subtask_id == 0:
            if closed_with_object and inside_any:
                self._completed_task1 = True
                self.cur_subtask_id += 1
        elif self.cur_subtask_id == 1:
            if not self.target_drawer_number in drawers_closed:
                self._completed_task2 = True
                self.cur_subtask_id += 1
        elif self.cur_subtask_id == 2:
            if retrieved_and_on_top:
                self._completed_task3 = True

        # Cache predicates for debugging
        self.episode_stats["inside_any_drawer"] = inside_any
        self.episode_stats["closed_drawer_with_object"] = closed_with_object

        # Phase 2 completion condition
        self.episode_stats["retrieved_and_on_top"] = retrieved_and_on_top

        # Debug info
        self.episode_stats["cube_in_drawer"] = cube_in_drawer
        self.episode_stats["drawers_closed"] = drawers_closed
        self.episode_stats["target_drawer_number"] = self.target_drawer_number

        success = self._completed_task3

        # Expose current phase in stats
        self.episode_stats["phase"] = int(self.cur_subtask_id)
        return dict(success=success, episode_stats=self.episode_stats)

    def compute_dense_reward(self, info, **kwargs):
        # Cumulative reward: sum latched task completions (stable, non-fluctuating)
        r = 0.0
        r += 1.0 if self._completed_task1 else 0.0
        r += 1.0 if self._completed_task2 else 0.0
        r += 1.0 if self._completed_task3 else 0.0
        return r

    def compute_normalized_dense_reward(self, **kwargs):
        denom = 3.0
        return self.compute_dense_reward(**kwargs) / denom

    def get_language_instruction(self, **kwargs):
        if self.cur_subtask_id == 0:
            model_name = self._get_instruction_obj_name(self.model_id)
            return f"place {model_name} in a drawer and close it"
        elif self.cur_subtask_id == 1:
            return "press the button on top"
        else:
            model_name = self._get_instruction_obj_name(self.model_id)
            return f"retrieve {model_name} and place it on top"

    def is_final_subtask(self):
        return self.cur_subtask_id == 2


@register_env("PlaceRetrieveDrawerCustomInScene-v0", max_episode_steps=260)
class PlaceRetrieveDrawerCustomInSceneEnv(PlaceRetrieveFromDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_baked_tex_v1.json"
    drawer_ids = ["top", "middle", "bottom"]
