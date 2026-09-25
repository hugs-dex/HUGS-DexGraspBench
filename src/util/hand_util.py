import os
import pdb
import logging

import trimesh
import numpy as np
import mujoco
import transforms3d.quaternions as tq

from .rot_util import interplote_pose, interplote_qpos
from .viewer_util import create_debug_viewer


class MjHO:
    hand_prefix: str = "child-"
    min_object_mesh_scaled_volume: float = 1e-14
    _object_mesh_scaled_volume_cache = {}

    def __init__(
        self,
        obj_path,
        obj_scale,
        obj_density,
        mj_arena_memory_bytes,
        hand_xml_path,
        hand_mocap,
        exclude_table_contact,
        friction_coef,
        has_floor_z0,
        debug_render=False,
        debug_viewer=False,
        viewer_config=None,
        viewer_session=None,
    ):
        self.hand_mocap = hand_mocap
        self.spec = mujoco.MjSpec()
        self.spec.memory = self._normalize_mj_arena_memory_bytes(mj_arena_memory_bytes)
        self.spec.meshdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.spec.option.timestep = 0.004
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY
        self.b_debug_render = debug_render
        self.b_debug_viewer = debug_viewer
        self.viewer_config = viewer_config
        self.viewer_session = viewer_session
        self.debug_viewer = None
        self.debug_render = None
        self.debug_images = []
        self.skipped_tiny_object_meshes = []
        if self.b_debug_render or self.b_debug_viewer:
            self.spec.add_texture(
                type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                # rgb1=[0.3, 0.5, 0.7],
                # rgb2=[0.3, 0.5, 0.7],
                rgb1=[1.0, 1.0, 1.0],
                rgb2=[1.0, 1.0, 1.0],
                width=512,
                height=512,
            )
            self.spec.worldbody.add_light(
                name="spotlight",
                pos=[0, -1, 2],
                castshadow=False,
            )
            self.spec.worldbody.add_camera(name="closeup", pos=[0.75, 1.0, 1.0], xyaxes=[-1, 0, 0, 0, -1, 1])

        self._add_hand(hand_xml_path, hand_mocap)
        self._add_object(obj_path, obj_scale, obj_density, has_floor_z0)
        self._set_friction(friction_coef)
        self.spec.add_key()
        if exclude_table_contact is not None:
            for body_name in exclude_table_contact:
                self.spec.add_exclude(bodyname1="world", bodyname2=f"{self.hand_prefix}{body_name}")

        # Get ready for simulation
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        self.ext_force_on_obj = None
        self.target_qpos_a = np.zeros((self.model.nu))

    def _normalize_mj_arena_memory_bytes(self, mj_arena_memory_bytes):
        """Normalize the configured MuJoCo arena memory size.

        Args:
            mj_arena_memory_bytes: Arena capacity from config in bytes. `None`
                means MuJoCo should use its default compiled size.

        Returns:
            Positive integer byte count that can be assigned to `MjSpec.memory`.
        """

        if mj_arena_memory_bytes is None:
            return self.spec.memory
        normalized_bytes = int(mj_arena_memory_bytes)
        if normalized_bytes <= 0:
            raise ValueError(
                f"mj_arena_memory_bytes must be a positive integer number of bytes, got {mj_arena_memory_bytes}."
            )
        return normalized_bytes

    def _init_after_first_fk(self):
        # For ctrl
        qpos2ctrl_matrix = np.zeros((self.model.nu, self.model.nv))
        mujoco.mju_sparse2dense(
            qpos2ctrl_matrix,
            self.data.actuator_moment,
            self.data.moment_rownnz,
            self.data.moment_rowadr,
            self.data.moment_colind,
        )
        self._qpos2ctrl_matrix = qpos2ctrl_matrix[..., :-6]

    def _init_viewer_and_render(self):
        try:
            if self.b_debug_viewer:
                if self.viewer_session is None:
                    self.debug_viewer = create_debug_viewer(
                        self.model,
                        self.data,
                        self.viewer_config,
                        frame_sleep_seconds=self.spec.option.timestep,
                    )
                else:
                    self.debug_viewer = self.viewer_session.attach(
                        self.model,
                        self.data,
                        frame_sleep_seconds=self.spec.option.timestep,
                    )

            if self.b_debug_render:
                self.debug_render = mujoco.Renderer(self.model, 480, 640)
                self.debug_options = mujoco.MjvOption()
                mujoco.mjv_defaultOption(self.debug_options)
                self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
                self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
                self.debug_images = []
        except Exception:
            self.close_view_and_render()
            raise

        return

    def wait_for_viewer_client(self):
        """Wait for an interactive viewer client when required by the selected backend."""

        if self.debug_viewer is not None:
            self.debug_viewer.wait_for_client()

    def hold_viewer_on_finish(self):
        """Keep the final simulated frame visible according to viewer configuration."""

        if self.debug_viewer is not None:
            self.debug_viewer.hold_on_finish()

    def reset(self):
        self.ext_force_on_obj = None
        self.target_qpos_a = np.zeros((self.model.nu))
        self.debug_images = []

    def _add_hand(self, xml_path, mocap_base):
        # Read hand xml
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self.spec.meshdir

        for g in child_spec.geoms:
            # This solimp and solref comes from the Shadow Hand xml
            # They can generate larger force with smaller penetration
            # The body will be more "rigid" and less "soft"
            g.solimp[:3] = [0.5, 0.99, 0.0001]
            g.solref[:2] = [0.005, 1]

        attach_frame = self.spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(child_spec.worldbody, self.hand_prefix, "")
        # Add freejoint and mocap of hand root
        if mocap_base:
            child_world.add_freejoint(name="hand_freejoint")
            self.spec.worldbody.add_body(name="mocap_body", mocap=True)
            self.spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name1="mocap_body",
                name2=f"{self.hand_prefix}world",
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                solimp=[0.9, 0.95, 0.001, 0.5, 2],
                data=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            )
        return

    def _get_object_mesh_scaled_volume(self, file_path, obj_scale):
        """Calculate the mesh volume after applying the object scale.

        Args:
            file_path: Path to a convex-piece OBJ mesh file.
            obj_scale: Uniform scale used when adding the object mesh to MuJoCo.

        Returns:
            Tuple `(scaled_volume, scaled_extents)` where `scaled_volume` is the
            absolute mesh volume after scaling and `scaled_extents` is the scaled
            axis-aligned bounding-box size.
        """

        cache_key = (os.path.abspath(file_path), float(obj_scale))
        if cache_key in self._object_mesh_scaled_volume_cache:
            return self._object_mesh_scaled_volume_cache[cache_key]

        mesh = trimesh.load(file_path, force="mesh", process=False)
        scaled_volume = abs(float(mesh.volume)) * (float(obj_scale) ** 3)
        scaled_extents = np.asarray(mesh.extents, dtype=np.float64) * float(obj_scale)
        self._object_mesh_scaled_volume_cache[cache_key] = (scaled_volume, scaled_extents)
        return scaled_volume, scaled_extents

    def _should_skip_object_mesh(self, obj_path, file_path, mesh_name, obj_scale):
        """Decide whether a convex object mesh is too small to add to MuJoCo.

        Args:
            obj_path: Object asset directory used for warning messages.
            file_path: Path to a convex-piece OBJ mesh file.
            mesh_name: MuJoCo mesh name derived from the OBJ filename.
            obj_scale: Uniform scale used when adding the object mesh to MuJoCo.

        Returns:
            Boolean indicating whether this mesh should be skipped.
        """

        scaled_volume, scaled_extents = self._get_object_mesh_scaled_volume(file_path, obj_scale)
        if scaled_volume >= self.min_object_mesh_scaled_volume:
            return False

        skip_record = {
            "obj_path": obj_path,
            "mesh": mesh_name,
            "scaled_volume": scaled_volume,
            "threshold": self.min_object_mesh_scaled_volume,
            "scaled_extents": scaled_extents.tolist(),
        }
        self.skipped_tiny_object_meshes.append(skip_record)
        if self.b_debug_viewer or self.b_debug_render:
            logging.warning(
                "Skip tiny object convex mesh in MuJoCo scene: obj_path=%s, mesh=%s, "
                "scaled_volume=%.6e, threshold=%.6e, scaled_extents=%s",
                obj_path,
                mesh_name,
                scaled_volume,
                self.min_object_mesh_scaled_volume,
                scaled_extents.tolist(),
            )
        return True

    def _add_object(self, obj_path, obj_scale, obj_density, has_floor_z0):
        if has_floor_z0:
            floor_geom = self.spec.worldbody.add_geom(
                name="object_collision_floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=[0, 0, 0],
                size=[0, 0, 1.0],
                rgba=[1.0, 1.0, 1.0, 0.8],  # transparent
            )

        obj_body = self.spec.worldbody.add_body(name="object")
        obj_body.add_freejoint(name="obj_freejoint")
        parts_folder = os.path.join(obj_path, "urdf/meshes")
        for file in os.listdir(parts_folder):
            if not file.endswith(".obj"):
                continue
            file_path = os.path.join(parts_folder, file)
            mesh_name = file.replace(".obj", "")
            mesh_id = mesh_name.replace("convex_piece_", "")

            if self._should_skip_object_mesh(obj_path, file_path, mesh_name, obj_scale):
                continue

            self.spec.add_mesh(
                name=mesh_name,
                file=file_path,
                scale=[obj_scale, obj_scale, obj_scale],
            )
            obj_body.add_geom(
                name=f"object_visual_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=0,
                contype=0,
                conaffinity=0,
            )
            obj_body.add_geom(
                name=f"object_collision_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=obj_density,
                rgba=[0.925, 0.7, 0.42, 1.0],  # yellow
            )

        return

    def _normalize_friction_pair(self, friction_coef):
        """Convert a configured friction value to the MuJoCo two-value friction pair used here.

        Args:
            friction_coef: Sequence-like config value containing sliding and torsional friction.

        Returns:
            Numpy array with two float values suitable for assigning to `geom.friction[:2]`.
        """

        friction_pair = np.asarray(friction_coef, dtype=np.float64).reshape(-1)
        if len(friction_pair) < 2:
            raise ValueError(f"Expected at least two friction coefficients, got {friction_coef}")
        return friction_pair[:2]

    def _set_friction(self, friction_coef):
        """Set MuJoCo contact friction for all simulation geoms.

        Args:
            friction_coef: Two-value friction pair applied to all geoms.

        Returns:
            None.
        """

        friction_pair = self._normalize_friction_pair(friction_coef)
        self.spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.noslip_iterations = 2
        self.spec.option.impratio = 10
        for g in self.spec.geoms:
            g.friction[:2] = friction_pair
            g.condim = 4
        return

    def _qpos2ctrl(self, hand_qpos):
        if self.hand_mocap:
            return self._qpos2ctrl_matrix[:, 6:] @ hand_qpos[7:]
        else:
            return self._qpos2ctrl_matrix @ hand_qpos

    def get_obj_pose(self):
        return self.data.qpos[-7:].copy()

    def get_contact_info(self, hand_qpos, obj_pose, obj_margin=0):
        # Set margin and gap to detect contact
        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = obj_margin

        # Set pose and qpos for hand and object
        self.reset_pose_qpos(hand_qpos, obj_pose)
        # self.udpate_debug_viewer()

        object_id = self.model.nbody - 1
        hand_id = self.model.nbody - 2
        # Body id 0 is MuJoCo's world body. It must never be counted as a hand
        # body, otherwise tabletop floor contacts can be mixed into hand-hand or
        # hand-object metrics for mocap hands.
        world_id = 0

        # Processing all contact information
        ho_contact = []
        hh_contact = []
        for contact in self.data.contact:
            body1_id = self.model.geom(contact.geom1).bodyid
            body2_id = self.model.geom(contact.geom2).bodyid
            body1_name = self.model.body(self.model.geom(contact.geom1).bodyid).name
            body2_name = self.model.body(self.model.geom(contact.geom2).bodyid).name
            # hand and object
            if (body1_id > world_id and body1_id <= hand_id and body2_id == object_id) or (
                body2_id > world_id and body2_id <= hand_id and body1_id == object_id
            ):
                # keep body1=hand and body2=object
                if body2_id == object_id:
                    contact_normal = contact.frame[0:3]
                    hand_body_name = body1_name.removeprefix(self.hand_prefix)
                    obj_body_name = body2_name
                else:
                    contact_normal = -contact.frame[0:3]
                    hand_body_name = body2_name.removeprefix(self.hand_prefix)
                    obj_body_name = body1_name
                ho_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact_normal,
                        "body1_name": hand_body_name,
                        "body2_name": obj_body_name,
                    }
                )
            # hand and hand
            elif body1_id > world_id and body1_id < hand_id and body2_id > world_id and body2_id < hand_id:
                hh_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact.frame[0:3],
                        "body1_name": body1_name,
                        "body2_name": body2_name,
                    }
                )
            # else:
            #     print(body1_name, body2_name, body1_id, body2_id)

        # Set margin and gap back
        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = 0
        return ho_contact, hh_contact

    def _body_pair_is_excluded(self, body1_id, body2_id):
        """Check whether MuJoCo has an explicit contact exclude for two bodies.

        Args:
            body1_id: First MuJoCo body id.
            body2_id: Second MuJoCo body id.

        Returns:
            True if the model contains a `<contact><exclude ...>` entry for the
            unordered body pair; otherwise False.
        """

        low_id, high_id = sorted((int(body1_id), int(body2_id)))
        signature = low_id + (high_id << 16)
        return bool(np.any(self.model.exclude_signature == signature))

    def _body_pair_is_ancestor_related(self, body1_id, body2_id):
        """Check whether two bodies are on the same kinematic ancestor chain.

        Args:
            body1_id: First MuJoCo body id.
            body2_id: Second MuJoCo body id.

        Returns:
            True if either body is an ancestor of the other. These pairs are
            skipped for the self-distance metric because neighboring links often
            meet at joints and do not represent meaningful finger-finger clearance.
        """

        body1_id = int(body1_id)
        body2_id = int(body2_id)
        parent_id = body1_id
        while parent_id > 0:
            parent_id = int(self.model.body_parentid[parent_id])
            if parent_id == body2_id:
                return True

        parent_id = body2_id
        while parent_id > 0:
            parent_id = int(self.model.body_parentid[parent_id])
            if parent_id == body1_id:
                return True
        return False

    def _geom_pair_can_collide(self, geom1_id, geom2_id):
        """Check whether two geoms are valid self-distance candidates.

        Args:
            geom1_id: First MuJoCo geom id.
            geom2_id: Second MuJoCo geom id.

        Returns:
            True when the pair is contact-enabled, not explicitly excluded, and
            not on the same ancestor chain; otherwise False.
        """

        body1_id = int(self.model.geom_bodyid[geom1_id])
        body2_id = int(self.model.geom_bodyid[geom2_id])
        if body1_id == body2_id:
            return False
        geom1_can_hit_geom2 = (self.model.geom_contype[geom1_id] & self.model.geom_conaffinity[geom2_id]) != 0
        geom2_can_hit_geom1 = (self.model.geom_contype[geom2_id] & self.model.geom_conaffinity[geom1_id]) != 0
        if not geom1_can_hit_geom2 and not geom2_can_hit_geom1:
            return False
        if self._body_pair_is_excluded(body1_id, body2_id):
            return False
        if self._body_pair_is_ancestor_related(body1_id, body2_id):
            return False
        return True

    def _get_hand_collision_geom_ids(self, valid_body_names=None):
        """Collect contact-enabled hand geoms used by self-distance evaluation.

        Args:
            valid_body_names: Optional collection of body names without the
                attached-hand prefix. When provided, only geoms whose body names
                are in the collection are included.

        Returns:
            List of MuJoCo geom ids that belong to the hand model and can
            participate in collision checks.
        """

        hand_id = self.model.nbody - 2
        valid_body_set = set(valid_body_names) if valid_body_names is not None else None
        geom_ids = []
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            if body_id <= 0 or body_id > hand_id:
                continue
            body_name = self.model.body(body_id).name.removeprefix(self.hand_prefix)
            if valid_body_set is not None and body_name not in valid_body_set:
                continue
            if self.model.geom_contype[geom_id] == 0 and self.model.geom_conaffinity[geom_id] == 0:
                continue
            geom_ids.append(geom_id)
        return geom_ids

    def get_hand_hand_signed_distance(self, hand_qpos, obj_pose, valid_body_names=None, distmax=1.0):
        """Compute nearest hand-hand geom distance at one hand/object pose.

        Args:
            hand_qpos: Hand qpos in MuJoCo joint order.
            obj_pose: Object pose appended to the MuJoCo qpos.
            valid_body_names: Optional collection of body names without the
                attached-hand prefix. When provided, only these hand bodies are
                considered.
            distmax: Maximum distance searched by `mujoco.mj_geomDistance`.

        Returns:
            Dictionary with `self_signed_dist`, nearest geom/body names, and the
            number of candidate pairs. The sign convention is penetration-positive
            and clearance-negative. This result is always computed from
            `mujoco.mj_geomDistance` and does not depend on MuJoCo contact records.
        """

        self.reset_pose_qpos(hand_qpos, obj_pose)

        geom_ids = self._get_hand_collision_geom_ids(valid_body_names=valid_body_names)
        nearest_distance = None
        nearest_fromto = None
        nearest_pair = None
        candidate_pair_num = 0
        fromto = np.zeros(6, dtype=np.float64)
        for i, geom1_id in enumerate(geom_ids):
            for geom2_id in geom_ids[i + 1 :]:
                if not self._geom_pair_can_collide(geom1_id, geom2_id):
                    continue
                candidate_pair_num += 1
                distance = float(mujoco.mj_geomDistance(self.model, self.data, geom1_id, geom2_id, distmax, fromto))
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_fromto = fromto.copy()
                    nearest_pair = (geom1_id, geom2_id)

        if nearest_distance is None:
            return {
                "self_signed_dist": np.nan,
                "self_signed_dist_source": "geom_distance",
                "self_signed_dist_geom_ids": [-1, -1],
                "self_signed_dist_geom_pair": ["", ""],
                "self_signed_dist_body_pair": ["", ""],
                "self_signed_dist_fromto": np.full(6, np.nan, dtype=np.float64),
                "self_signed_dist_candidate_pair_num": 0,
            }

        geom1_id, geom2_id = nearest_pair
        body1_id = int(self.model.geom_bodyid[geom1_id])
        body2_id = int(self.model.geom_bodyid[geom2_id])
        geom1_name = self.model.geom(geom1_id).name or f"geom_{geom1_id}"
        geom2_name = self.model.geom(geom2_id).name or f"geom_{geom2_id}"
        return {
            "self_signed_dist": -nearest_distance,
            "self_signed_dist_source": "geom_distance",
            "self_signed_dist_geom_ids": [int(geom1_id), int(geom2_id)],
            "self_signed_dist_geom_pair": [
                geom1_name,
                geom2_name,
            ],
            "self_signed_dist_body_pair": [
                self.model.body(body1_id).name.removeprefix(self.hand_prefix),
                self.model.body(body2_id).name.removeprefix(self.hand_prefix),
            ],
            "self_signed_dist_fromto": nearest_fromto,
            "self_signed_dist_candidate_pair_num": candidate_pair_num,
        }

    def get_joint_names(self):
        model = self.model
        """获取所有关节名称"""
        joint_names = []
        for i in range(model.njnt - 1):  # exclude the obj_freejoint
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_name = joint_name.replace(self.hand_prefix, "")
            joint_names.append(joint_name)
        return joint_names

    def set_ext_force_on_obj_single_step(self, ext_force):
        """
        Only valid for the next simulation step.
        """
        self.data.xfrc_applied[-1] = ext_force
        return

    def set_ext_force_on_obj(self, ext_force):
        """
        The force will be constantly kept.
        """
        self.ext_force_on_obj = ext_force

    def reset_pose_qpos(self, hand_qpos, obj_pose, set_ctrl=True):
        # set key frame
        self.model.key_qpos[0] = np.concatenate([hand_qpos, obj_pose], axis=0)
        if set_ctrl:
            self.model.key_ctrl[0] = self._qpos2ctrl(hand_qpos)
        self.model.key_qvel[0] = 0
        self.model.key_act[0] = 0
        if self.hand_mocap:
            self.model.key_mpos[0] = hand_qpos[:3]
            self.model.key_mquat[0] = hand_qpos[3:7]

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        return

    def control_hand_with_interp(self, hand_qpos1, hand_qpos2, step_outer=10, step_inner=10):
        if self.hand_mocap:
            pose_interp = interplote_pose(hand_qpos1[:7], hand_qpos2[:7], step_outer)
        qpos_interp = interplote_qpos(self._qpos2ctrl(hand_qpos1), self._qpos2ctrl(hand_qpos2), step_outer)
        for j in range(step_outer):
            if self.hand_mocap:
                self.data.mocap_pos[0] = pose_interp[j, :3]
                self.data.mocap_quat[0] = pose_interp[j, 3:7]
            self.data.ctrl[:] = qpos_interp[j]
            mujoco.mj_forward(self.model, self.data)
            self.control_hand_step(step_inner)
        return

    def control_hand_step(self, step_inner):
        for _ in range(step_inner):
            if self.ext_force_on_obj is not None:
                self.set_ext_force_on_obj_single_step(ext_force=self.ext_force_on_obj)
            mujoco.mj_step(self.model, self.data)

        if self.debug_render is not None:
            # self.debug_render.update_scene(self.data, "closeup", self.debug_options)
            self.debug_render.update_scene(self.data, camera=self.cam, scene_option=self.debug_options)
            pixels = self.debug_render.render()
            self.debug_images.append(pixels)

        if self.debug_viewer is not None:
            self.debug_viewer.sync()

        return

    def udpate_debug_viewer(self):
        if self.debug_viewer is not None:
            self.debug_viewer.sync()

    def close_view_and_render(self):
        if self.debug_viewer is not None:
            if self.viewer_session is None:
                self.debug_viewer.close()
            self.debug_viewer = None
        if self.debug_render is not None:
            self.debug_render.close()
            self.debug_render = None


class RobotKinematics:
    def __init__(self, xml_path):
        spec = mujoco.MjSpec.from_file(xml_path)
        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)

        self.mesh_geom_info = {}
        for i in range(self.mj_model.ngeom):
            geom = self.mj_model.geom(i)
            mesh_id = geom.dataid
            if mesh_id != -1:
                mjm = self.mj_model.mesh(mesh_id)
                vert = self.mj_model.mesh_vert[mjm.vertadr[0] : mjm.vertadr[0] + mjm.vertnum[0]]
                face = self.mj_model.mesh_face[mjm.faceadr[0] : mjm.faceadr[0] + mjm.facenum[0]]
                body_name = self.mj_model.body(geom.bodyid).name
                mesh_name = mjm.name
                self.mesh_geom_info[f"{body_name}_{mesh_name}"] = {
                    "vert": vert,
                    "face": face,
                    "geom_id": i,
                }

        return

    def forward_kinematics(self, q):
        self.mj_data.qpos = q
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        return

    def get_init_meshes(self):
        init_mesh_lst = []
        mesh_name_lst = []
        for k, v in self.mesh_geom_info.items():
            mesh_name_lst.append(k)
            init_mesh_lst.append(trimesh.Trimesh(vertices=v["vert"], faces=v["face"]))
        return mesh_name_lst, init_mesh_lst

    def get_poses(self, root_pose):
        geom_poses = np.zeros((len(self.mesh_geom_info), 7))
        root_rot = tq.quat2mat(root_pose[3:])
        root_trans = root_pose[:3]
        for i, v in enumerate(self.mesh_geom_info.values()):
            geom_trans = self.mj_data.geom_xpos[v["geom_id"]]
            geom_rot = self.mj_data.geom_xmat[v["geom_id"]].reshape(3, 3)
            geom_poses[i, :3] = root_rot @ geom_trans + root_trans
            geom_poses[i, 3:] = tq.mat2quat(root_rot @ geom_rot)
        return geom_poses

    def get_posed_meshes(self, root_pose):
        root_rot = tq.quat2mat(root_pose[3:])
        root_trans = root_pose[:3]
        full_tm = []
        for k, v in self.mesh_geom_info.items():
            geom_rot = self.mj_data.geom_xmat[v["geom_id"]].reshape(3, 3)
            geom_trans = self.mj_data.geom_xpos[v["geom_id"]]
            posed_vert = (v["vert"] @ geom_rot.T + geom_trans) @ root_rot.T + root_trans
            posed_tm = trimesh.Trimesh(vertices=posed_vert, faces=v["face"])
            full_tm.append(posed_tm)
        full_tm = trimesh.util.concatenate(full_tm)
        return full_tm
