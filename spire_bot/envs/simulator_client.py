"""Small JSON-line client for the STS2 headless simulator."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


class SimulatorClientError(RuntimeError):
    """Raised when the simulator process cannot be used safely."""


def find_dotnet() -> str:
    """Return a usable dotnet executable path."""
    candidates = [
        Path.home() / ".dotnet-arm64" / "dotnet",
        Path.home() / ".dotnet" / "dotnet",
        "dotnet",
    ]
    for candidate in candidates:
        path = str(candidate)
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return path

    resolved = shutil.which("dotnet")
    if resolved:
        return resolved
    return "dotnet"


@dataclass
class SimulatorClient:
    """Owns one headless STS2 simulator subprocess."""

    project_path: Path | None = None
    dotnet_path: str | None = None
    game_dir: str | None = None
    no_build: bool = True

    def __post_init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        self.project_path = self.project_path or (
            repo_root / "external" / "sts2-cli" / "src" / "Sts2Headless" / "Sts2Headless.csproj"
        )
        self.dotnet_path = self.dotnet_path or find_dotnet()
        self.proc: subprocess.Popen[str] | None = None

    def start(self) -> JsonDict:
        """Start the simulator and return its initial ready message."""
        if self.proc is not None and self.proc.poll() is None:
            raise SimulatorClientError("Simulator process is already running.")

        env = os.environ.copy()
        if self.game_dir:
            env["STS2_GAME_DIR"] = self.game_dir

        cmd = [self.dotnet_path, "run"]
        if self.no_build:
            cmd.append("--no-build")
        cmd.extend(["--project", str(self.project_path)])

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        ready = self.read()
        if ready.get("type") != "ready":
            raise SimulatorClientError(f"Expected simulator ready message, got: {ready}")
        return ready

    def read(self) -> JsonDict:
        """Read the next JSON object from simulator stdout."""
        proc = self._require_process()
        assert proc.stdout is not None

        while True:
            line = proc.stdout.readline()
            if not line:
                message = "EOF from simulator process"
                if proc.poll() is not None and proc.stderr is not None:
                    stderr = proc.stderr.read().strip()
                    if stderr:
                        message += f": {stderr[-2000:]}"
                raise SimulatorClientError(message)

            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)

    def send(self, command: JsonDict) -> JsonDict:
        """Send one command dict and return one response dict."""
        if self.proc is None or self.proc.poll() is not None:
            self.start()

        proc = self._require_process()
        assert proc.stdin is not None

        proc.stdin.write(json.dumps(command) + "\n")
        proc.stdin.flush()
        return self.read()

    def start_run(
        self,
        character: str = "Silent",
        seed: str = "silent-poc",
        ascension: int = 0,
        lang: str = "en",
    ) -> JsonDict:
        return self.send(
            {
                "cmd": "start_run",
                "character": character,
                "seed": seed,
                "ascension": ascension,
                "lang": lang,
            }
        )

    def act(self, action: str, **args: Any) -> JsonDict:
        command: JsonDict = {"cmd": "action", "action": action}
        if args:
            command["args"] = args
        return self.send(command)

    def enter_room(self, room_type: str, **kwargs: Any) -> JsonDict:
        return self.send({"cmd": "enter_room", "type": room_type, **kwargs})

    def get_map(self) -> JsonDict:
        return self.send({"cmd": "get_map"})

    def set_player(self, **kwargs: Any) -> JsonDict:
        return self.send({"cmd": "set_player", **kwargs})

    def set_draw_order(self, cards: list[str]) -> JsonDict:
        return self.send({"cmd": "set_draw_order", "cards": cards})

    def close(self) -> None:
        """Ask the simulator to quit, then clean up the process."""
        if self.proc is None:
            return

        proc = self.proc
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass

        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            self.proc = None

    def _require_process(self) -> subprocess.Popen[str]:
        if self.proc is None:
            raise SimulatorClientError("Simulator process has not been started.")
        return self.proc

    def __enter__(self) -> "SimulatorClient":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
