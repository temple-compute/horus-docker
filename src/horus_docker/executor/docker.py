#
# horus_docker
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Docker executor implementation for Horus.
"""

import asyncio
from typing import TYPE_CHECKING, ClassVar

import docker
from docker.errors import DockerException
from horus_builtin.runtime.command import CommandRuntime
from horus_runtime.core.executor.base import BaseExecutor, RuntimeFilterType
from horus_runtime.core.task.exceptions import TaskExecutionError
from horus_runtime.logging import horus_logger
from pydantic import Field

from horus_docker.i18n import tr as _

if TYPE_CHECKING:
    from horus_runtime.core.task.base import BaseTask


class DockerExecutor(BaseExecutor):
    """
    Runs the task's command inside a Docker container.

    The command produced by the :class:`CommandRuntime` is executed through
    ``/bin/sh -c`` inside a fresh container created from :attr:`image`, so the
    same shell semantics as the local shell executor apply (pipes, globbing,
    ``&&``, ...).
    """

    kind: str = "docker_executor"
    kind_name: ClassVar[str] = "Docker Executor"
    kind_description: ClassVar[str] = _(
        "Executes a command inside a Docker container."
    )

    runtimes: ClassVar[RuntimeFilterType] = (CommandRuntime,)

    image: str
    """
    The Docker image to use for executing tasks (e.g. ``python:3.13-slim``).
    """

    env: dict[str, str] = Field(default_factory=dict)
    """
    Environment variables to set inside the container, as ``NAME -> value``.
    """

    volumes: dict[str, str] = Field(default_factory=dict)
    """
    Bind mounts as ``host_path -> container_path`` (mounted read-write),
    mirroring the ``-v host:container`` flag of the Docker CLI.
    """

    ports: dict[str, str] = Field(default_factory=dict)
    """
    Port mappings as ``host_port -> container_port``, mirroring the
    ``-p host:container`` flag of the Docker CLI. TCP is assumed.
    """

    working_dir: str | None = None
    """
    Working directory to run the command from inside the container.
    """

    network: str | None = None
    """
    Name of the Docker network to attach the container to.
    """

    entrypoint: str | list[str] | None = None
    """
    Override for the image's ``ENTRYPOINT``.
    """

    user: str | None = None
    """
    User (``name``/``uid``, optionally ``uid:gid``) to run the command as.
    """

    auto_remove: bool = True
    """
    Whether to remove the container once execution finishes.
    """

    def _docker_volumes(self) -> dict[str, dict[str, str]]:
        """
        Convert :attr:`volumes` into the structure expected by docker-py.
        """
        return {
            host: {"bind": container, "mode": "rw"}
            for host, container in self.volumes.items()
        }

    def _docker_ports(self) -> dict[str, int]:
        """
        Convert :attr:`ports` into the ``container_port -> host_port`` mapping
        expected by docker-py.
        """
        return {
            f"{container}/tcp": int(host)
            for host, container in self.ports.items()
        }

    def _run_container(self, command: list[str]) -> tuple[int, str]:
        """
        Synchronously run the container to completion and return its exit code
        and combined logs. This is blocking and is meant to be dispatched to a
        worker thread via :func:`asyncio.to_thread`.
        """
        client = docker.from_env()
        try:
            container = client.containers.run(
                self.image,
                command=command,
                detach=True,
                environment=self.env or None,
                volumes=self._docker_volumes() or None,
                ports=self._docker_ports() or None,
                working_dir=self.working_dir,
                network=self.network,
                entrypoint=self.entrypoint,
                user=self.user,
            )
            try:
                result = container.wait()
                logs = container.logs().decode(errors="replace")
            finally:
                if self.auto_remove:
                    container.remove(force=True)
            return int(result.get("StatusCode", 1)), logs
        finally:
            client.close()

    async def _execute(self, task: "BaseTask") -> None:
        """
        Run the task's command inside a Docker container.
        """
        assert isinstance(task.runtime, CommandRuntime)
        prepared_command = await task.runtime.setup_runtime(task)
        command = ["/bin/sh", "-c", prepared_command]

        horus_logger.log.debug(
            _(
                "Running task %(task_id)s in Docker image %(image)s: "
                "%(command)s"
            )
            % {
                "task_id": task.id,
                "image": self.image,
                "command": prepared_command,
            }
        )

        try:
            status_code, logs = await asyncio.to_thread(
                self._run_container, command
            )

            # Debug the output logs
            horus_logger.log.debug(
                _("Logs from task %(task_id)s Docker container:\n%(logs)s")
                % {"task_id": task.id, "logs": logs.strip()}
            )

        except DockerException as exc:
            horus_logger.log.error(
                _("Docker execution failed for task %(task_id)s: %(error)s")
                % {"task_id": task.id, "error": str(exc)}
            )
            raise TaskExecutionError(
                _("Docker execution failed: %(error)s") % {"error": str(exc)}
            ) from exc

        if status_code != 0:
            horus_logger.log.error(
                _(
                    "Container for task %(task_id)s exited with code "
                    "%(code)s. Logs: %(logs)s"
                )
                % {
                    "task_id": task.id,
                    "code": status_code,
                    "logs": logs.strip(),
                }
            )
            raise TaskExecutionError(
                _("Container exited with code %(code)s")
                % {"code": status_code}
            )
