import os
import sys
import types
import unittest
from unittest import mock

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from util import viewer_util


class FakeNativeHandle:
    def __init__(self):
        self.cam = types.SimpleNamespace(
            lookat=np.zeros(3),
            distance=0.0,
            azimuth=0.0,
            elevation=0.0,
        )
        self.running = True
        self.sync_count = 0
        self.close_count = 0

    def is_running(self):
        return self.running

    def sync(self):
        self.sync_count += 1

    def close(self):
        self.close_count += 1
        self.running = False


class FakeViserServer:
    instances = []

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client_snapshots = []
        self.clients = {}
        self.stop_count = 0
        self.scene = FakeSceneApi()
        self.gui = FakeGuiApi()
        self.client_connect_callbacks = []
        self._websock_server = types.SimpleNamespace(_server_thread=FakeServerThread())
        self.__class__.instances.append(self)

    def get_port(self):
        return self.port

    def get_clients(self):
        if self.client_snapshots:
            return self.client_snapshots.pop(0)
        return self.clients

    def stop(self):
        self.stop_count += 1

    def on_client_connect(self, callback):
        self.client_connect_callbacks.append(callback)
        return callback


class FakeSceneApi:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class FakeGuiApi:
    def __init__(self):
        self.reset_count = 0
        self.markdown = []

    def reset(self):
        self.reset_count += 1

    def add_markdown(self, content, **kwargs):
        self.markdown.append((content, kwargs))
        return types.SimpleNamespace(content=content)


class FakeServerThread:
    def __init__(self):
        self.alive = True
        self.join_count = 0

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        self.join_count += 1
        self.alive = False


class FakeMjviserScene:
    instances = []

    def __init__(self, server, model, num_envs):
        self.server = server
        self.model = model
        self.num_envs = num_envs
        self.gui_kwargs = None
        self.updated_data = []
        self.__class__.instances.append(self)

    def create_visualization_gui(self, **kwargs):
        self.gui_kwargs = kwargs

    def update_from_mjdata(self, data):
        self.updated_data.append(data)


class ViewerConfigTest(unittest.TestCase):
    def test_default_backend_is_mjviser(self):
        config = viewer_util.normalize_debug_viewer_config(None)

        self.assertEqual(config.backend, "mjviser")
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 8080)
        self.assertTrue(config.wait_for_client)
        self.assertTrue(config.hold_on_finish)
        self.assertFalse(config.playlist_enabled)
        self.assertEqual(config.playlist_interval_seconds, 1.0)

    def test_invalid_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported debug viewer backend"):
            viewer_util.normalize_debug_viewer_config({"backend": "unknown"})

    def test_invalid_port_and_boolean_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "port"):
            viewer_util.normalize_debug_viewer_config({"port": 0})
        with self.assertRaisesRegex(ValueError, "port"):
            viewer_util.normalize_debug_viewer_config({"port": 70000})
        with self.assertRaisesRegex(ValueError, "wait_for_client"):
            viewer_util.normalize_debug_viewer_config({"wait_for_client": "false"})
        with self.assertRaisesRegex(ValueError, "playlist.interval_seconds"):
            viewer_util.normalize_debug_viewer_config({"playlist": {"interval_seconds": -1}})
        with self.assertRaisesRegex(ValueError, "finite"):
            viewer_util.normalize_debug_viewer_config({"playlist": {"interval_seconds": float("nan")}})
        with self.assertRaisesRegex(ValueError, "supported only"):
            viewer_util.normalize_debug_viewer_config(
                {"backend": "mujoco", "playlist": {"enabled": True}}
            )


class ViewerBackendTest(unittest.TestCase):
    def setUp(self):
        FakeViserServer.instances.clear()
        FakeMjviserScene.instances.clear()

    def test_native_backend_wraps_launch_passive(self):
        handle = FakeNativeHandle()
        native_module = types.SimpleNamespace(launch_passive=mock.Mock(return_value=handle))

        with mock.patch.object(viewer_util.importlib, "import_module", return_value=native_module):
            viewer = viewer_util.create_debug_viewer(
                model="model",
                data="data",
                config={"backend": "mujoco", "hold_on_finish": False},
            )
            viewer.sync()
            viewer.close()
            viewer.close()

        native_module.launch_passive.assert_called_once_with("model", "data")
        np.testing.assert_allclose(handle.cam.lookat, [0.5, 0.0, -0.2])
        self.assertEqual(handle.cam.distance, 1.5)
        self.assertEqual(handle.cam.azimuth, 180)
        self.assertEqual(handle.cam.elevation, -20)
        self.assertEqual(handle.sync_count, 1)
        self.assertEqual(handle.close_count, 1)

    def test_mjviser_backend_updates_existing_simulation_data_and_closes_server(self):
        modules = {
            "viser": types.SimpleNamespace(ViserServer=FakeViserServer),
            "mjviser": types.SimpleNamespace(ViserMujocoScene=FakeMjviserScene),
        }

        with mock.patch.object(viewer_util.importlib, "import_module", side_effect=modules.__getitem__):
            viewer = viewer_util.create_debug_viewer(
                model="model",
                data="data",
                config={
                    "backend": "mjviser",
                    "host": "127.0.0.1",
                    "port": 9000,
                    "wait_for_client": False,
                    "hold_on_finish": False,
                },
            )
            viewer.sync()
            viewer.close()
            viewer.close()

        server = FakeViserServer.instances[0]
        scene = FakeMjviserScene.instances[0]
        self.assertEqual((server.host, server.port), ("127.0.0.1", 9000))
        self.assertEqual(scene.num_envs, 1)
        self.assertEqual(scene.updated_data, ["data", "data"])
        self.assertEqual(scene.gui_kwargs["camera_distance"], 1.5)
        self.assertEqual(server.stop_count, 1)
        self.assertEqual(server._websock_server._server_thread.join_count, 1)

    def test_mjviser_waits_for_connection_and_holds_until_disconnect(self):
        modules = {
            "viser": types.SimpleNamespace(ViserServer=FakeViserServer),
            "mjviser": types.SimpleNamespace(ViserMujocoScene=FakeMjviserScene),
        }

        with mock.patch.object(viewer_util.importlib, "import_module", side_effect=modules.__getitem__):
            viewer = viewer_util.create_debug_viewer(model="model", data="data", config={"backend": "mjviser"})

        viewer.server.client_snapshots = [{}, {1: object()}, {1: object()}, {1: object()}, {}]
        with mock.patch.object(viewer_util.time, "sleep") as sleep:
            viewer.wait_for_client()
            viewer.hold_on_finish()
        self.assertGreaterEqual(sleep.call_count, 2)
        viewer.close()

    def test_missing_mjviser_dependency_has_actionable_error(self):
        with mock.patch.object(viewer_util.importlib, "import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(RuntimeError, "mujoco==3.6.0"):
                viewer_util.create_debug_viewer(model="model", data="data", config={"backend": "mjviser"})

    def test_playlist_session_reuses_server_and_resets_scene_between_models(self):
        modules = {
            "viser": types.SimpleNamespace(ViserServer=FakeViserServer),
            "mjviser": types.SimpleNamespace(ViserMujocoScene=FakeMjviserScene),
        }
        config = {
            "backend": "mjviser",
            "wait_for_client": False,
            "hold_on_finish": False,
            "playlist": {"enabled": True, "interval_seconds": 0.25},
        }

        with mock.patch.object(viewer_util.importlib, "import_module", side_effect=modules.__getitem__):
            session = viewer_util.MjviserDebugViewerSession(config)
            client = types.SimpleNamespace(camera=types.SimpleNamespace(position=None, look_at=None))
            session.server.client_connect_callbacks[0](client)
            session.begin_grasp(0, 2, 3, "/input/first.npy")
            first_viewer = session.attach("model-1", "data-1", frame_sleep_seconds=0)
            with mock.patch.object(viewer_util.time, "sleep") as sleep:
                session.hold_on_finish()
            session.begin_grasp(1, 2, 7, "/input/second.npy")
            second_viewer = session.attach("model-2", "data-2", frame_sleep_seconds=0)
            session.hold_on_finish()
            session.close()

        server = FakeViserServer.instances[0]
        self.assertIs(first_viewer, session)
        self.assertIs(second_viewer, session)
        self.assertEqual(len(FakeViserServer.instances), 1)
        self.assertIsInstance(client.camera.position, np.ndarray)
        self.assertEqual(client.camera.position.shape, (3,))
        self.assertIsInstance(client.camera.look_at, np.ndarray)
        self.assertEqual(client.camera.look_at.shape, (3,))
        self.assertEqual(len(FakeMjviserScene.instances), 2)
        self.assertEqual(server.scene.reset_count, 1)
        self.assertEqual(server.gui.reset_count, 1)
        self.assertEqual(len(server.gui.markdown), 2)
        self.assertIn("Grasp 2/2", server.gui.markdown[-1][0])
        sleep.assert_called_once_with(0.25)
        self.assertEqual(server.stop_count, 1)

    def test_playlist_waits_for_client_only_once(self):
        modules = {
            "viser": types.SimpleNamespace(ViserServer=FakeViserServer),
            "mjviser": types.SimpleNamespace(ViserMujocoScene=FakeMjviserScene),
        }
        config = {"backend": "mjviser", "playlist": {"enabled": True}}

        with mock.patch.object(viewer_util.importlib, "import_module", side_effect=modules.__getitem__):
            session = viewer_util.MjviserDebugViewerSession(config)

        session.server.client_snapshots = [{}, {1: object()}]
        with mock.patch.object(viewer_util.time, "sleep") as sleep:
            session.wait_for_client()
            session.wait_for_client()
        sleep.assert_called_once_with(0.1)
        session.close()

    def test_playlist_clears_partial_scene_when_attach_fails(self):
        scene_factory = mock.Mock(side_effect=RuntimeError("scene failed"))
        modules = {
            "viser": types.SimpleNamespace(ViserServer=FakeViserServer),
            "mjviser": types.SimpleNamespace(ViserMujocoScene=scene_factory),
        }
        config = {"backend": "mjviser", "playlist": {"enabled": True}}

        with mock.patch.object(viewer_util.importlib, "import_module", side_effect=modules.__getitem__):
            session = viewer_util.MjviserDebugViewerSession(config)
            session.begin_grasp(0, 1, 0, "/input/grasp.npy")
            with self.assertRaisesRegex(RuntimeError, "scene failed"):
                session.attach("model", "data", frame_sleep_seconds=0)

        server = FakeViserServer.instances[0]
        self.assertEqual(server.scene.reset_count, 1)
        self.assertEqual(server.gui.reset_count, 1)
        self.assertIsNone(session.scene)
        self.assertIsNone(session.data)
        session.close()


if __name__ == "__main__":
    unittest.main()
