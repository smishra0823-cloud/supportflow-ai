from __future__ import annotations

import argparse
import atexit
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.25)
        return connection.connect_ex((host, port)) == 0


def wait_until_ready(url: str, process: subprocess.Popen[bytes], timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"A demo service stopped during startup with exit code {process.returncode}.")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the SupportFlow app and API together.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--app-port", type=int, default=8503)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def stop_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> None:
    args = parse_args()
    if not (ROOT / "artifacts" / "model_bundle" / "metadata.json").exists():
        raise SystemExit("The trained model is missing. Run `python -m scripts.train_and_evaluate` first.")

    app_url = f"http://{args.host}:{args.app_port}"
    api_url = f"http://{args.host}:{args.api_port}"
    processes: list[subprocess.Popen[bytes]] = []
    atexit.register(stop_processes, processes)

    if port_is_open(args.host, args.api_port):
        print(f"API is already running: {api_url}/docs", flush=True)
    else:
        api_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api:app",
                "--host",
                args.host,
                "--port",
                str(args.api_port),
            ],
            cwd=ROOT,
        )
        processes.append(api_process)
        wait_until_ready(f"{api_url}/health", api_process)
        print(f"API ready: {api_url}/docs", flush=True)

    if port_is_open(args.host, args.app_port):
        print(f"App is already running: {app_url}", flush=True)
    else:
        app_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.address",
                args.host,
                "--server.port",
                str(args.app_port),
                "--server.headless",
                "true",
            ],
            cwd=ROOT,
        )
        processes.append(app_process)
        wait_until_ready(f"{app_url}/_stcore/health", app_process)
        print(f"App ready: {app_url}", flush=True)

    if not args.no_browser:
        webbrowser.open(app_url)
        webbrowser.open(f"{api_url}/docs")

    if not processes:
        print("Both services were already running; no new processes were started.", flush=True)
        return

    print("Press Ctrl+C to stop the services started by this command.", flush=True)
    try:
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    main()
