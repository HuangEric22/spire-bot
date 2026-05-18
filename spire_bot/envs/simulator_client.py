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


def default_game_dir_candidates() -> list[str]:
    """Return common STS2 install paths for Windows, WSL, and macOS."""
    return [
        r"C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2",
        r"C:\Program Files\Steam\steamapps\common\Slay the Spire 2",
        "/mnt/c/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2",
        "/mnt/c/Program Files/Steam/steamapps/common/Slay the Spire 2",
        "~/Library/Application Support/Steam/steamapps/common/Slay the Spire 2/"
        "SlayTheSpire2.app/Contents/Resources/data_sts2_macos_arm64",
    ]


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
    game_dir_candidates: list[str] | None = None
    no_build: bool = True
    cwd: Path | None = None

    def __post_init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        self.project_path = self.project_path or (
            repo_root / "external" / "sts2-cli" / "src" / "Sts2Headless" / "Sts2Headless.csproj"
        )
        self.dotnet_path = self.dotnet_path or find_dotnet()
        self.cwd = self.cwd or repo_root
        self.proc: subprocess.Popen[str] | None = None
        self._last_start_cmd: list[str] = []
        self._last_game_dir: str | None = None

    def start(self) -> JsonDict:
        """Start the simulator and return its initial ready message."""
        if self.proc is not None and self.proc.poll() is None:
            raise SimulatorClientError("Simulator process is already running.")

        env = os.environ.copy()
        game_dir = self._resolve_game_dir()
        self._last_game_dir = game_dir
        if game_dir:
            env["STS2_GAME_DIR"] = game_dir

        cmd = [self.dotnet_path, "run"]
        if self.no_build:
            cmd.append("--no-build")
        cmd.extend(["--project", str(self.project_path)])
        self._last_start_cmd = cmd

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
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

        skipped_lines: list[str] = []
        while True:
            line = proc.stdout.readline()
            if not line:
                message = "EOF from simulator process"
                if self._last_start_cmd:
                    message += f"; cmd: {self._last_start_cmd}"
                if self.cwd:
                    message += f"; cwd: {self.cwd}"
                if self._last_game_dir:
                    message += f"; STS2_GAME_DIR: {self._last_game_dir}"
                if skipped_lines:
                    message += f"; last stdout: {' | '.join(skipped_lines[-5:])}"
                try:
                    if proc.poll() is None:
                        proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                if proc.stderr is not None:
                    try:
                        stderr = proc.stderr.read().strip()
                    except Exception:
                        stderr = ""
                    if stderr:
                        message += f"; stderr: {stderr[-4000:]}"
                raise SimulatorClientError(message)

            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
            if line:
                skipped_lines.append(line[:400])

    def send(self, command: JsonDict) -> JsonDict:
        """Send one command dict and return one response dict."""
        if self.proc is None or self.proc.poll() is not None:
            self.start()

        proc = self._require_process()
        assert proc.stdin is not None

        line = json.dumps(command)
        proc.stdin.write(line + "\n")
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

    def _resolve_game_dir(self) -> str | None:
        candidates = []
        if self.game_dir:
            candidates.append(self.game_dir)
        candidates.extend(self.game_dir_candidates or default_game_dir_candidates())

        for candidate in candidates:
            expanded = os.path.expandvars(os.path.expanduser(candidate))
            if Path(expanded).exists():
                return expanded

        return self.game_dir

    def __enter__(self) -> "SimulatorClient":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
