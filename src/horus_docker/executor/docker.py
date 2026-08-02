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
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from horus_builtin.runtime.command import CommandRuntime
from horus_builtin.runtime.substitution import substitute
from horus_runtime.core.executor.base import BaseExecutor, RuntimeFilterType
from horus_runtime.core.resources import ResourceScope
from horus_runtime.core.task.exceptions import TaskExecutionError
from horus_runtime.logging import horus_logger
from horus_runtime.settings import runtime_settings
from pydantic import Field, PrivateAttr

from horus_docker.executor.resources import ContainerScope
from horus_docker.i18n import tr as _

if TYPE_CHECKING:
    from horus_runtime.core.target.base import BaseTarget
    from horus_runtime.core.target.channel import ChannelProcess
    from horus_runtime.core.task.base import BaseTask

CIDFILE_NAME = ".horus_container_id"
"""
Name of the file, under the task's working directory, that ``docker run``
writes the started container's id to.
"""


class DockerExecutor(BaseExecutor):
    """
    Runs the task's command inside a Docker container.
    """

    kind: str = "docker"
    kind_name: ClassVar[str] = "Docker Executor"
    kind_description: ClassVar[str] = _(
        "Executes a command inside a Docker container on the task target."
    )

    runtimes: ClassVar[RuntimeFilterType] = (CommandRuntime,)

    image: str
    """
    Docker image tag (e.g. ``python:3.13-slim``).  When :attr:`dockerfile` is
    set this becomes the tag assigned to the locally-built image.
    """

    env: dict[str, str] = Field(default_factory=dict)
    """
    Environment variables to set inside the container, as ``NAME -> value``.
    Keys and values support ``$id`` / ``${id}`` / ``${id.attr}`` /
    ``${task.attr}`` artifact substitution (see :attr:`volumes`).
    """

    volumes: dict[str, str] = Field(default_factory=dict)
    """
    Bind mounts as ``host_path -> container_path`` (read-write).

    Both sides support ``$id`` / ``${id}`` / ``${id.attr}`` / ``${task.attr}``
    placeholders, resolved the same way a ``command`` string is (see
    :mod:`horus_builtin.runtime.substitution`) — ``$id`` renders to the
    artifact's absolute path on the task's target. This lets an explicit
    mount reference a task's own input/output paths, e.g.::

        volumes:
          "${protein}": /data/protein.pdb

    mounts the ``protein`` artifact's host path at a fixed, host-independent
    path inside the container, so the command doesn't need to know the host
    layout. Unknown placeholders are left as-is. Explicit entries here take
    precedence over the auto-mounted artifact parent dirs (see
    :meth:`_docker_run_cmd`).
    """

    ports: dict[str, str] = Field(default_factory=dict)
    """
    Port mappings as ``host_port -> container_port``.
    """

    working_dir: str | None = None
    """
    Working directory inside the container (``-w`` flag). Supports artifact
    substitution, see :attr:`volumes`.
    """

    network: str | None = None
    """
    Docker network to attach the container to.
    """

    entrypoint: str | None = None
    """
    Override for the image's ``ENTRYPOINT`` (``--entrypoint`` flag). Supports
    artifact substitution, see :attr:`volumes`.
    """

    user: str | None = None
    """
    User (``name``, ``uid``, or ``uid:gid``) to run the command as.
    """

    gpus: str | None = None
    """
    Value passed to ``docker run --gpus`` (e.g. ``all``, ``device=0``).
    Requires the NVIDIA Container Toolkit on the target.
    """

    auto_remove: bool = True
    """
    Add ``--rm`` so the container is removed when it exits.
    """

    dockerfile: str | None = None
    """
    Inline Dockerfile content.  Supports ``$id`` / ``${task.attr}``
    substitution for artifacts and task attributes.
    """

    build_context: str | None = None
    """
    Build context path on the target for ``docker build``.  Defaults to
    ``task.working_dir`` when :attr:`dockerfile` is set.
    """

    _container_name: str | None = PrivateAttr(default=None)
    """
    Predictable container name assigned at execute-time so
    :meth:`cancel_execution` can stop it by name.
    """

    _target: "BaseTarget | None" = PrivateAttr(default=None)
    """
    Target the container was started on, so :meth:`cancel_execution` stops it
    where it actually runs instead of on the orchestrator's host.
    """

    @staticmethod
    def _cidfile_path(task: "BaseTask") -> str:
        """Path on the target where ``docker run`` writes the container id."""
        return (Path(task.working_dir) / CIDFILE_NAME).as_posix()

    @staticmethod
    def _sub(value: str, task: "BaseTask | None") -> str:
        """Render ``$``/``${}`` artifact placeholders in *value*."""
        return substitute(value, task) if task is not None else value

    @classmethod
    def _sub_dict(
        cls, mapping: dict[str, str], task: "BaseTask | None"
    ) -> dict[str, str]:
        """Apply :meth:`_sub` to every key and value of *mapping*."""
        return {
            cls._sub(k, task): cls._sub(v, task) for k, v in mapping.items()
        }

    def _volumes_and_env(
        self, task: "BaseTask | None"
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Return the bind mounts and container environment for *task*."""
        # ponytail: auto-mount artifact parent dirs; explicit volumes win
        auto_mounts: dict[str, str] = {}
        env = self._sub_dict(self.env, task)
        if task is not None:
            for artifact in (*task.inputs, *task.outputs):
                host_dir = str(artifact.path.parent)
                auto_mounts[host_dir] = host_dir
            # The container is a different filesystem: the side-artifacts
            # directory the runtime collects from must be reachable at the
            # same path inside it, or scripts write into a throwaway layer.
            side_dir = task.side_artifacts_dir
            auto_mounts[side_dir] = side_dir
            env.setdefault(runtime_settings.SIDE_ARTIFACTS_DIR_ENV, side_dir)
        explicit_volumes = self._sub_dict(self.volumes, task)
        return {**auto_mounts, **explicit_volumes}, env

    def _docker_run_cmd(
        self, prepared_command: str, task: "BaseTask | None" = None
    ) -> str:
        """Return the full ``docker run`` CLI command string."""
        merged_volumes, env = self._volumes_and_env(task)
        working_dir = (
            self._sub(self.working_dir, task) if self.working_dir else None
        )
        entrypoint = (
            self._sub(self.entrypoint, task)
            if self.entrypoint is not None
            else None
        )
        user = self._sub(self.user, task) if self.user else None

        parts = ["docker", "run"]
        if self.auto_remove:
            parts.append("--rm")
        if self._container_name is not None:
            parts += ["--name", self._container_name]
        if task is not None:
            # --cidfile is the CLI's own contract for "tell me which container
            # you started"; an observer needs the id because the workload runs
            # under the daemon, not under this client process.
            parts += ["--cidfile", shlex.quote(self._cidfile_path(task))]
        if self.gpus:
            parts += ["--gpus", shlex.quote(self.gpus)]
        for k, v in env.items():
            parts += ["-e", shlex.quote(f"{k}={v}")]
        for host, container in merged_volumes.items():
            parts += ["-v", shlex.quote(f"{host}:{container}")]
        for host_p, cont_p in self.ports.items():
            parts += ["-p", shlex.quote(f"{host_p}:{cont_p}")]
        if working_dir:
            parts += ["-w", shlex.quote(working_dir)]
        if self.network:
            parts += ["--network", shlex.quote(self.network)]
        if entrypoint is not None:
            parts += ["--entrypoint", shlex.quote(entrypoint)]
        if user:
            parts += ["-u", shlex.quote(user)]
        parts += [
            shlex.quote(self.image),
            "/bin/sh",
            "-c",
            shlex.quote(prepared_command),
        ]
        return " ".join(parts)

    async def _build_image(self, task: "BaseTask") -> None:
        """
        Render :attr:`dockerfile`, upload it to the target, and build the
        image tagged as :attr:`image`.
        """
        rendered = substitute(self.dockerfile or "", task)
        build_dir = f"{task.working_dir}/.horus_docker"
        dockerfile_path = f"{build_dir}/Dockerfile"

        await task.target.mkdir(build_dir)
        await task.target.put_file(rendered.encode(), dockerfile_path)

        context = self.build_context or task.working_dir
        build_cmd = (
            f"docker build"
            f" -t {shlex.quote(self.image)}"
            f" -f {shlex.quote(dockerfile_path)}"
            f" {shlex.quote(str(context))}"
        )
        proc = await task.target.run_command(build_cmd)
        stdout, stderr = await proc.communicate()
        out = stdout.decode(errors="replace").strip() if stdout else ""
        err = stderr.decode(errors="replace").strip() if stderr else ""
        if out:
            horus_logger.log.debug(out)
        if err:
            horus_logger.log.debug(err)
        if proc.returncode != 0:
            raise TaskExecutionError(
                _("docker build failed with exit code %(code)s")
                % {"code": proc.returncode}
            )

    async def _execute(self, task: "BaseTask") -> None:
        """
        Build the image (if :attr:`dockerfile` is set), run the task's command
        inside a container on the target, then clean up the image.
        """
        if not isinstance(task.runtime, CommandRuntime):
            raise TaskExecutionError(
                _("DockerExecutor only supports CommandRuntime runtimes.")
            )
        prepared_command = await task.runtime.setup_runtime(task)

        if self.dockerfile:
            await self._build_image(task)

        self._container_name = f"horus-{task.id}"
        self._target = task.target
        # docker refuses to start when the cidfile already exists, so a stale
        # one from a previous run of this task would wedge it forever.
        await task.target.remove(self._cidfile_path(task))
        run_cmd = self._docker_run_cmd(prepared_command, task)
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

        proc = await task.target.run_command(
            run_cmd,
            cwd=task.working_dir,
            env={
                runtime_settings.SIDE_ARTIFACTS_DIR_ENV: (
                    task.side_artifacts_dir
                )
            },
        )

        try:
            stdout, stderr = await proc.communicate()
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise

        out = stdout.decode(errors="replace").strip() if stdout else ""
        err = stderr.decode(errors="replace").strip() if stderr else ""
        if out:
            horus_logger.log.info(out)
        if err:
            horus_logger.log.warning(err)

        try:
            if proc.returncode != 0:
                horus_logger.log.error(
                    _(
                        "Container for task %(task_id)s exited with code "
                        "%(code)s. Output: %(out)s"
                    )
                    % {
                        "task_id": task.id,
                        "code": proc.returncode,
                        "out": (out or err).strip(),
                    }
                )

                raise TaskExecutionError(
                    _("Container exited with code %(code)s")
                    % {"code": proc.returncode}
                )
        finally:
            self._container_name = None
            self._target = None
            if self.dockerfile:
                try:
                    rmi = await task.target.run_command(
                        f"docker rmi -f {shlex.quote(self.image)}"
                    )
                    await rmi.wait()
                except Exception:
                    pass

    async def resource_scope(
        self, task: "BaseTask", process: "ChannelProcess | None" = None
    ) -> ResourceScope:
        """
        Report the container, not the spawned process tree.

        ``docker run`` is a thin client: the workload is reparented under the
        daemon's supervisor in another process group, so the spawned tree
        holds nothing worth measuring.  *process* is ignored for exactly that
        reason.
        """
        del process
        return ContainerScope(
            container_id=await self._read_container_id(task),
            cidfile=self._cidfile_path(task),
        )

    async def _read_container_id(self, task: "BaseTask") -> str | None:
        """
        Read the started container's id from the cidfile, or ``None``.

        Best effort: the container may not have started yet, and a
        measurement concern must never fail a task.
        """
        try:
            raw = await task.target.get_file(self._cidfile_path(task))
        except Exception:
            return None
        return raw.decode(errors="replace").strip() or None

    async def cancel_execution(self) -> None:
        """Stop the running container so it does not become orphaned.

        Called by ``BaseTarget.cancel()`` before ``CancelledError`` is
        injected.  If no container is currently running (e.g. the task
        finished before the cancel arrived) this is a safe no-op.
        """
        if self._container_name is None or self._target is None:
            return
        name = self._container_name
        target = self._target
        self._container_name = None  # clear before stop — idempotent
        # Goes through the target channel: the container runs on the target,
        # so a local `docker stop` would stop nothing (or the wrong thing).
        try:
            proc = await target.run_command(
                f"docker stop {shlex.quote(name)}", detach=False
            )
            await proc.wait()
        except Exception as exc:
            horus_logger.log.warning(
                _("Failed to stop container %(name)s: %(err)s")
                % {"name": name, "err": exc}
            )
