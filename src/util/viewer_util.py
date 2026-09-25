"""Debug viewer backends for MuJoCo evaluation."""

from __future__ import annotations

import importlib
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


SUPPORTED_DEBUG_VIEWER_BACKENDS = ("mjviser", "mujoco")


@dataclass(frozen=True)
class DebugViewerConfig:
    """Normalized configuration shared by debug viewer backends."""

    backend: str = "mjviser"
    host: str = "127.0.0.1"
    port: int = 8080
    wait_for_client: bool = True
    hold_on_finish: bool = True
    playlist_enabled: bool = False
    playlist_interval_seconds: float = 1.0


def _read_config_value(config: Any, name: str, default: Any) -> Any:
    """Read one value from a mapping-like or attribute-like config object."""

    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def normalize_debug_viewer_config(config: Any) -> DebugViewerConfig:
    """Validate and normalize the Hydra debug viewer configuration.

    Args:
        config: Mapping-like or attribute-like object containing viewer fields.

    Returns:
        A normalized immutable viewer configuration.

    Raises:
        ValueError: If the backend, host, port, or boolean fields are invalid.
    """

    if isinstance(config, DebugViewerConfig):
        return config

    backend = str(_read_config_value(config, "backend", "mjviser")).strip().lower()
    if backend not in SUPPORTED_DEBUG_VIEWER_BACKENDS:
        supported = ", ".join(SUPPORTED_DEBUG_VIEWER_BACKENDS)
        raise ValueError(f"Unsupported debug viewer backend {backend!r}. Expected one of: {supported}.")

    host = str(_read_config_value(config, "host", "127.0.0.1")).strip()
    if not host:
        raise ValueError("Debug viewer host must be a non-empty string.")

    raw_port = _read_config_value(config, "port", 8080)
    if isinstance(raw_port, bool):
        raise ValueError(f"Debug viewer port must be an integer in [1, 65535], got {raw_port!r}.")
    try:
        port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Debug viewer port must be an integer in [1, 65535], got {raw_port!r}.") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"Debug viewer port must be in [1, 65535], got {port}.")

    wait_for_client = _read_config_value(config, "wait_for_client", True)
    hold_on_finish = _read_config_value(config, "hold_on_finish", True)
    if not isinstance(wait_for_client, bool):
        raise ValueError(f"viewer.wait_for_client must be boolean, got {wait_for_client!r}.")
    if not isinstance(hold_on_finish, bool):
        raise ValueError(f"viewer.hold_on_finish must be boolean, got {hold_on_finish!r}.")

    playlist_config = _read_config_value(config, "playlist", None)
    playlist_enabled = _read_config_value(playlist_config, "enabled", False)
    if not isinstance(playlist_enabled, bool):
        raise ValueError(f"viewer.playlist.enabled must be boolean, got {playlist_enabled!r}.")

    raw_interval_seconds = _read_config_value(playlist_config, "interval_seconds", 1.0)
    if isinstance(raw_interval_seconds, bool):
        raise ValueError(
            "viewer.playlist.interval_seconds must be a non-negative number, "
            f"got {raw_interval_seconds!r}."
        )
    try:
        playlist_interval_seconds = float(raw_interval_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "viewer.playlist.interval_seconds must be a non-negative number, "
            f"got {raw_interval_seconds!r}."
        ) from exc
    if not math.isfinite(playlist_interval_seconds) or playlist_interval_seconds < 0:
        raise ValueError(
            "viewer.playlist.interval_seconds must be finite and non-negative, "
            f"got {playlist_interval_seconds}."
        )
    if playlist_enabled and backend != "mjviser":
        raise ValueError("viewer.playlist.enabled=true is supported only with viewer.backend=mjviser.")

    return DebugViewerConfig(
        backend=backend,
        host=host,
        port=port,
        wait_for_client=wait_for_client,
        hold_on_finish=hold_on_finish,
        playlist_enabled=playlist_enabled,
        playlist_interval_seconds=playlist_interval_seconds,
    )


def _import_mjviser_modules() -> tuple[Any, Any]:
    """Import the pinned browser-viewer dependencies on demand."""

    try:
        viser = importlib.import_module("viser")
        mjviser = importlib.import_module("mjviser")
    except ImportError as exc:
        raise RuntimeError(
            "The mjviser debug viewer requires the pinned packages "
            "mujoco==3.6.0, mjviser==0.0.14, and viser==1.0.27."
        ) from exc
    return viser, mjviser


def _print_mjviser_server_url(server: Any, config: DebugViewerConfig) -> None:
    """Print the browser URL and loopback SSH forwarding hint."""

    actual_port = server.get_port() if hasattr(server, "get_port") else config.port
    print(f"mjviser debug viewer: http://{config.host}:{actual_port}", flush=True)
    if config.host in {"127.0.0.1", "localhost"}:
        print(
            f"Remote server access: ssh -L {actual_port}:127.0.0.1:{actual_port} <server>",
            flush=True,
        )


def _stop_viser_server(server: Any) -> None:
    """Stop Viser and wait briefly for its fixed-version server thread."""

    server.stop()

    # Viser 1.0.27 waits only 0.1 seconds in stop(). A connected HTTP client
    # can keep the background thread alive briefly, which prevents the next
    # debug run from reusing the configured port.
    websock_server = getattr(server, "_websock_server", None)
    server_thread = getattr(websock_server, "_server_thread", None)
    if server_thread is not None and server_thread.is_alive():
        server_thread.join(timeout=2.0)
    if server_thread is not None and server_thread.is_alive():
        logging.warning("Viser server thread did not stop within 2 seconds; port release may be delayed.")


class MujocoDebugViewer:
    """Adapter around MuJoCo's native passive viewer."""

    def __init__(self, model: Any, data: Any, config: DebugViewerConfig, frame_sleep_seconds: float) -> None:
        viewer_module = importlib.import_module("mujoco.viewer")
        self.config = config
        self.frame_sleep_seconds = max(float(frame_sleep_seconds), 0.0)
        self.handle = viewer_module.launch_passive(model, data)
        self.handle.cam.lookat[:] = [0.5, 0.0, -0.2]
        self.handle.cam.distance = 1.5
        self.handle.cam.azimuth = 180
        self.handle.cam.elevation = -20

    def wait_for_client(self) -> None:
        """Return immediately because the native viewer opens synchronously."""

    def sync(self) -> None:
        """Push the latest MuJoCo state to the native viewer."""

        if self.handle is None or not self.handle.is_running():
            return
        self.handle.sync()
        if self.frame_sleep_seconds > 0:
            time.sleep(self.frame_sleep_seconds)

    def hold_on_finish(self) -> None:
        """Keep the final frame visible until the native window is closed."""

        if not self.config.hold_on_finish or self.handle is None or not self.handle.is_running():
            return
        print("MuJoCo evaluation finished. Close the viewer window or press Ctrl+C to continue.", flush=True)
        try:
            while self.handle.is_running():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Close the MuJoCo debug viewer and continue saving evaluation results.", flush=True)

    def close(self) -> None:
        """Close the native viewer if it is still active."""

        if self.handle is not None:
            self.handle.close()
            self.handle = None


class MjviserDebugViewer:
    """Read-only browser viewer backed by mjviser and Viser."""

    def __init__(self, model: Any, data: Any, config: DebugViewerConfig, frame_sleep_seconds: float) -> None:
        self.config = config
        self.data = data
        self.frame_sleep_seconds = max(float(frame_sleep_seconds), 0.0)
        self.server = None
        self.scene = None
        viser, mjviser = _import_mjviser_modules()

        try:
            self.server = viser.ViserServer(host=config.host, port=config.port)
            self.scene = mjviser.ViserMujocoScene(self.server, model, num_envs=1)
            self.scene.create_visualization_gui(
                camera_distance=1.5,
                camera_azimuth=180.0,
                camera_elevation=-20.0,
            )
            self.scene.update_from_mjdata(data)
        except Exception:
            self.close()
            raise

        _print_mjviser_server_url(self.server, config)

    def _clients(self) -> dict:
        """Return a snapshot of connected browser clients."""

        if self.server is None:
            return {}
        return self.server.get_clients()

    def wait_for_client(self) -> None:
        """Wait until a browser client connects when requested by config."""

        if not self.config.wait_for_client:
            return
        print("Waiting for an mjviser browser client. Press Ctrl+C to abort.", flush=True)
        while not self._clients():
            time.sleep(0.1)
        print("mjviser client connected; starting evaluation.", flush=True)

    def sync(self) -> None:
        """Push the latest MuJoCo state to connected browser clients."""

        if self.scene is None:
            return
        self.scene.update_from_mjdata(self.data)
        if self.frame_sleep_seconds > 0:
            time.sleep(self.frame_sleep_seconds)

    def hold_on_finish(self) -> None:
        """Keep the final frame available until clients disconnect or Ctrl+C is pressed."""

        if not self.config.hold_on_finish or not self._clients():
            return
        print("MuJoCo evaluation finished. Close the browser tab or press Ctrl+C to continue.", flush=True)
        try:
            while self._clients():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stop holding the mjviser final frame and continue saving evaluation results.", flush=True)

    def close(self) -> None:
        """Stop the Viser server and release its listening port."""

        self.scene = None
        if self.server is not None:
            server = self.server
            self.server = None
            _stop_viser_server(server)


class MjviserDebugViewerSession:
    """Task-level mjviser session that replaces scenes without restarting Viser."""

    def __init__(self, config: Any) -> None:
        self.config = normalize_debug_viewer_config(config)
        if self.config.backend != "mjviser" or not self.config.playlist_enabled:
            raise ValueError(
                "MjviserDebugViewerSession requires viewer.backend=mjviser and viewer.playlist.enabled=true."
            )

        viser, self._mjviser = _import_mjviser_modules()
        self.server = None
        self.scene = None
        self.data = None
        self.frame_sleep_seconds = 0.0
        self._wait_completed = False
        self._sequence_position = 0
        self._sequence_total = 0
        self._input_index = 0
        self._input_path = ""

        try:
            self.server = viser.ViserServer(host=self.config.host, port=self.config.port)

            @self.server.on_client_connect
            def _initialize_camera(client: Any) -> None:
                self._set_client_camera(client)

            _print_mjviser_server_url(self.server, self.config)
        except Exception:
            self.close()
            raise

    def begin_grasp(self, sequence_position: int, sequence_total: int, input_index: int, input_path: str) -> None:
        """Record metadata for the next evaluator attached to this session."""

        self._sequence_position = int(sequence_position)
        self._sequence_total = int(sequence_total)
        self._input_index = int(input_index)
        self._input_path = str(input_path)

    def _set_client_camera(self, client: Any) -> None:
        """Initialize one connected browser camera for the current scene."""

        client.camera.position = np.array([1.40953893, 0.0, -0.51303021], dtype=np.float64)
        client.camera.look_at = np.zeros(3, dtype=np.float64)

    def attach(self, model: Any, data: Any, frame_sleep_seconds: float) -> "MjviserDebugViewerSession":
        """Replace the current scene with a new model while keeping the server alive."""

        if self.server is None:
            raise RuntimeError("Cannot attach a MuJoCo scene after the mjviser playlist session is closed.")

        if self.scene is not None:
            self.server.scene.reset()
            self.server.gui.reset()
            self.scene = None

        self.data = data
        self.frame_sleep_seconds = max(float(frame_sleep_seconds), 0.0)
        try:
            self.scene = self._mjviser.ViserMujocoScene(self.server, model, num_envs=1)
            input_name = self._input_path.rsplit("/", 1)[-1]
            self.server.gui.add_markdown(
                "\n".join(
                    [
                        f"### Grasp {self._sequence_position + 1}/{self._sequence_total}",
                        f"Input index: `{self._input_index}`",
                        f"File: `{input_name}`",
                    ]
                ),
                order=-1000.0,
            )
            for client in self.server.get_clients().values():
                self._set_client_camera(client)
            self.scene.update_from_mjdata(data)
        except Exception:
            self.server.scene.reset()
            self.server.gui.reset()
            self.scene = None
            self.data = None
            raise
        print(
            f"mjviser playlist grasp {self._sequence_position + 1}/{self._sequence_total}: "
            f"input index {self._input_index}, {self._input_path}",
            flush=True,
        )
        return self

    def _clients(self) -> dict:
        """Return a snapshot of connected browser clients."""

        if self.server is None:
            return {}
        return self.server.get_clients()

    def wait_for_client(self) -> None:
        """Wait at most once before the playlist begins."""

        if self._wait_completed or not self.config.wait_for_client:
            return
        print("Waiting for an mjviser browser client. Press Ctrl+C to abort.", flush=True)
        while not self._clients():
            time.sleep(0.1)
        self._wait_completed = True
        print("mjviser client connected; starting playlist.", flush=True)

    def sync(self) -> None:
        """Push the current evaluator's MuJoCo data to the active scene."""

        if self.scene is None or self.data is None:
            return
        self.scene.update_from_mjdata(self.data)
        if self.frame_sleep_seconds > 0:
            time.sleep(self.frame_sleep_seconds)

    def hold_on_finish(self) -> None:
        """Pause between grasps and hold only the final playlist frame."""

        is_final_grasp = self._sequence_position + 1 >= self._sequence_total
        if not is_final_grasp:
            if self.config.playlist_interval_seconds > 0:
                print(
                    "Playlist grasp finished; loading the next grasp in "
                    f"{self.config.playlist_interval_seconds:g} seconds.",
                    flush=True,
                )
                time.sleep(self.config.playlist_interval_seconds)
            return

        if not self.config.hold_on_finish or not self._clients():
            return
        print("Mjviser playlist finished. Close the browser tab or press Ctrl+C to continue.", flush=True)
        try:
            while self._clients():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stop holding the mjviser playlist final frame and continue saving results.", flush=True)

    def close(self) -> None:
        """Close the shared Viser server and release the configured port."""

        self.scene = None
        self.data = None
        if self.server is not None:
            server = self.server
            self.server = None
            _stop_viser_server(server)


def create_debug_viewer(model: Any, data: Any, config: Any, frame_sleep_seconds: float = 0.0) -> Any:
    """Create the configured passive debug viewer without taking over simulation."""

    normalized_config = normalize_debug_viewer_config(config)
    if normalized_config.backend == "mujoco":
        return MujocoDebugViewer(model, data, normalized_config, frame_sleep_seconds)
    return MjviserDebugViewer(model, data, normalized_config, frame_sleep_seconds)
