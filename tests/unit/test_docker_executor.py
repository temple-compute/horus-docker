#
# horus_docker
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""Unit tests for DockerExecutor."""

import shlex
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from horus_builtin.artifact.file import FileArtifact
from horus_builtin.runtime.command import CommandRuntime
from horus_builtin.task.horus_task import HorusTask
from horus_runtime.context import HorusContext
from horus_runtime.core.executor.base import BaseExecutor
from horus_runtime.core.task.exceptions import TaskExecutionError

from horus_docker.executor.docker import DockerExecutor

_IMAGE = "python:3.13-slim"
_NONZERO_CODE = 2


def _make_mock_proc(
    returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""
) -> AsyncMock:
    """Return an AsyncMock ChannelProcess."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.wait = AsyncMock(return_value=returncode)
    return proc


def _make_mock_target(proc: AsyncMock | None = None) -> MagicMock:
    """Return a mock target whose run_command returns *proc*."""
    target = MagicMock()
    target.run_command = AsyncMock(return_value=proc or _make_mock_proc())
    target.put_file = AsyncMock()
    target.mkdir = AsyncMock()
    return target


@pytest.mark.unit
class TestDockerExecutorRegistration:
    """Verify DockerExecutor registers correctly in the AutoRegistry."""

    def test_registered_under_kind(self) -> None:
        """DockerExecutor must appear in BaseExecutor.registry."""
        assert "docker" in BaseExecutor.registry
        assert BaseExecutor.registry["docker"] is DockerExecutor

    def test_kind_field(self) -> None:
        """The kind field must equal 'docker'."""
        assert DockerExecutor(image=_IMAGE).kind == "docker"

    def test_runtimes_filter(self) -> None:
        """CommandRuntime must be listed as a supported runtime."""
        assert CommandRuntime in DockerExecutor.runtimes


@pytest.mark.unit
class TestDockerExecutorDefaults:
    """Verify default field values for DockerExecutor."""

    def test_required_image_field(self) -> None:
        """The image field must be stored as provided."""
        assert DockerExecutor(image=_IMAGE).image == _IMAGE

    def test_optional_dicts_default_to_empty(self) -> None:
        """env, volumes, and ports must default to empty dicts."""
        executor = DockerExecutor(image=_IMAGE)
        assert executor.env == {}
        assert executor.volumes == {}
        assert executor.ports == {}

    def test_optional_fields_default_to_none(self) -> None:
        """
        working_dir, network, entrypoint, user, dockerfile must default to
        None.
        """
        executor = DockerExecutor(image=_IMAGE)
        assert executor.working_dir is None
        assert executor.network is None
        assert executor.entrypoint is None
        assert executor.user is None
        assert executor.dockerfile is None
        assert executor.build_context is None

    def test_auto_remove_defaults_to_true(self) -> None:
        """auto_remove must default to True."""
        assert DockerExecutor(image=_IMAGE).auto_remove is True


@pytest.mark.unit
class TestDockerRunCmd:
    """Verify _docker_run_cmd() builds correct CLI strings."""

    def test_minimal_command(self) -> None:
        """
        Minimal executor produces 'docker run --rm <image> /bin/sh -c <cmd>'.
        """
        cmd = DockerExecutor(image=_IMAGE)._docker_run_cmd("echo hello")
        parts = shlex.split(cmd)
        assert parts[:3] == ["docker", "run", "--rm"]
        assert shlex.quote(_IMAGE) in cmd
        assert parts[-3:] == ["/bin/sh", "-c", "echo hello"]

    def test_auto_remove_false_omits_rm(self) -> None:
        """auto_remove=False must not include --rm."""
        cmd = DockerExecutor(image=_IMAGE, auto_remove=False)._docker_run_cmd(
            "true"
        )
        assert "--rm" not in cmd

    def test_env_vars_included(self) -> None:
        """Each env var must appear as a -e flag."""
        cmd = DockerExecutor(image=_IMAGE, env={"FOO": "bar"})._docker_run_cmd(
            "true"
        )
        assert "-e" in cmd
        assert "FOO=bar" in cmd

    def test_volume_mount_included(self) -> None:
        """Volumes must appear as -v host:container flags."""
        cmd = DockerExecutor(
            image=_IMAGE, volumes={"/data": "/mnt/data"}
        )._docker_run_cmd("true")
        assert "-v" in cmd
        assert "/data:/mnt/data" in cmd

    def test_port_mapping_included(self) -> None:
        """Ports must appear as -p host:container flags."""
        cmd = DockerExecutor(
            image=_IMAGE, ports={"8080": "80"}
        )._docker_run_cmd("true")
        assert "-p" in cmd
        assert "8080:80" in cmd

    def test_working_dir_included(self) -> None:
        """-w flag must appear when working_dir is set."""
        cmd = DockerExecutor(image=_IMAGE, working_dir="/app")._docker_run_cmd(
            "true"
        )
        assert "-w" in cmd
        assert "/app" in cmd

    def test_network_included(self) -> None:
        """--network flag must appear when network is set."""
        cmd = DockerExecutor(image=_IMAGE, network="host")._docker_run_cmd(
            "true"
        )
        assert "--network" in cmd
        assert "host" in cmd

    def test_user_included(self) -> None:
        """-u flag must appear when user is set."""
        cmd = DockerExecutor(image=_IMAGE, user="1000:1000")._docker_run_cmd(
            "true"
        )
        assert "-u" in cmd
        assert "1000:1000" in cmd

    def test_entrypoint_string_included(self) -> None:
        """--entrypoint must appear when entrypoint is a string."""
        cmd = DockerExecutor(
            image=_IMAGE, entrypoint="/custom"
        )._docker_run_cmd("true")
        assert "--entrypoint" in cmd
        assert "/custom" in cmd

    def test_special_chars_in_command_quoted(self) -> None:
        """Shell metacharacters in the command must be quoted."""
        cmd = DockerExecutor(image=_IMAGE)._docker_run_cmd("echo $HOME && ls")
        # The full command string must not expose bare $ or && outside quotes
        parts = shlex.split(cmd)
        # Last arg is the command passed to sh -c — must be exact
        assert parts[-1] == "echo $HOME && ls"


@pytest.mark.unit
class TestBuildImage:
    """Verify _build_image() interacts with the target channel correctly."""

    def _make_task(self, dockerfile: str) -> tuple[HorusTask, MagicMock]:
        executor = DockerExecutor(image="myapp:test", dockerfile=dockerfile)
        task = HorusTask(
            id="t1",
            name="build_test",
            executor=executor,
            runtime=CommandRuntime(command="true"),
        )
        mock_target = _make_mock_target()
        return task, mock_target

    @pytest.mark.asyncio
    async def test_uploads_dockerfile_to_target(
        self, horus_context: HorusContext
    ) -> None:
        """put_file must be called with the rendered Dockerfile bytes."""
        del horus_context
        task, mock_target = self._make_task("FROM scratch\nRUN true")
        with patch.object(task, "target", mock_target):
            await DockerExecutor(
                image="myapp:test", dockerfile="FROM scratch\nRUN true"
            )._build_image(task)
        mock_target.put_file.assert_called_once()
        content_arg = mock_target.put_file.call_args[0][0]
        assert b"FROM scratch" in content_arg

    @pytest.mark.asyncio
    async def test_creates_build_directory(
        self, horus_context: HorusContext
    ) -> None:
        """Mkdir must be called before put_file."""
        del horus_context
        task, mock_target = self._make_task("FROM scratch")
        with patch.object(task, "target", mock_target):
            await DockerExecutor(
                image="myapp:test", dockerfile="FROM scratch"
            )._build_image(task)
        mock_target.mkdir.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_on_nonzero_build_exit(
        self, horus_context: HorusContext
    ) -> None:
        """A non-zero docker build exit code must raise TaskExecutionError."""
        del horus_context
        task, mock_target = self._make_task("FROM scratch")
        mock_target.run_command = AsyncMock(
            return_value=_make_mock_proc(returncode=1, stderr=b"build failed")
        )
        with patch.object(task, "target", mock_target):
            with pytest.raises(
                TaskExecutionError, match="docker build failed"
            ):
                await DockerExecutor(
                    image="myapp:test", dockerfile="FROM scratch"
                )._build_image(task)


@pytest.mark.unit
class TestDockerExecutorExecute:
    """Verify DockerExecutor._execute() end-to-end behaviour."""

    def _make_task(
        self, executor: DockerExecutor, command: str = "echo hello"
    ) -> HorusTask:
        return HorusTask(
            id="test-task",
            name="test_task",
            executor=executor,
            runtime=CommandRuntime(command=command),
        )

    @pytest.mark.asyncio
    async def test_execute_success(self, horus_context: HorusContext) -> None:
        """A zero exit code must complete without raising."""
        del horus_context
        executor = DockerExecutor(image=_IMAGE)
        task = self._make_task(executor)
        mock_target = _make_mock_target(_make_mock_proc(stdout=b"hello\n"))
        with patch.object(task, "target", mock_target):
            await executor._execute(task)

    @pytest.mark.asyncio
    async def test_execute_nonzero_exit_raises(
        self, horus_context: HorusContext
    ) -> None:
        """A non-zero container exit code must raise TaskExecutionError."""
        del horus_context
        executor = DockerExecutor(image=_IMAGE)
        task = self._make_task(executor, command="exit 1")
        mock_target = _make_mock_target(
            _make_mock_proc(returncode=1, stderr=b"something went wrong")
        )
        with patch.object(task, "target", mock_target):
            with pytest.raises(
                TaskExecutionError, match="Container exited with code 1"
            ):
                await executor._execute(task)

    @pytest.mark.asyncio
    async def test_execute_calls_run_command_on_target(
        self, horus_context: HorusContext
    ) -> None:
        """_execute must delegate to task.target.run_command."""
        del horus_context
        executor = DockerExecutor(image=_IMAGE)
        task = self._make_task(executor, command="echo hello")
        mock_target = _make_mock_target()
        with patch.object(task, "target", mock_target):
            await executor._execute(task)
        mock_target.run_command.assert_called_once()
        cmd_arg = mock_target.run_command.call_args[0][0]
        assert "docker run" in cmd_arg
        assert _IMAGE in cmd_arg

    @pytest.mark.asyncio
    async def test_execute_with_dockerfile_builds_then_runs(
        self, horus_context: HorusContext
    ) -> None:
        """With dockerfile set, build then run commands must both be issued."""
        del horus_context
        executor = DockerExecutor(
            image="myapp:test", dockerfile="FROM scratch"
        )
        task = self._make_task(executor)
        mock_target = _make_mock_target()
        with patch.object(task, "target", mock_target):
            await executor._execute(task)
        # build + run + rmi = 3 run_command calls
        assert mock_target.run_command.call_count == 3
        calls = [c[0][0] for c in mock_target.run_command.call_args_list]
        assert any("docker build" in c for c in calls)
        assert any("docker run" in c for c in calls)
        assert any("docker rmi" in c for c in calls)

    @pytest.mark.asyncio
    async def test_execute_without_dockerfile_no_build(
        self, horus_context: HorusContext
    ) -> None:
        """Without dockerfile, only one run_command call must be made."""
        del horus_context
        executor = DockerExecutor(image=_IMAGE)
        task = self._make_task(executor)
        mock_target = _make_mock_target()
        with patch.object(task, "target", mock_target):
            await executor._execute(task)
        assert mock_target.run_command.call_count == 1

    @pytest.mark.asyncio
    async def test_auto_mounts_artifact_parent_dirs(
        self, horus_context: HorusContext
    ) -> None:
        """
        Artifact parent dirs must be bind-mounted at the same container path.
        """
        del horus_context
        executor = DockerExecutor(image=_IMAGE)
        task = HorusTask(
            id="test-task",
            name="test_task",
            executor=executor,
            runtime=CommandRuntime(command="true"),
            inputs=[
                FileArtifact(id="inp", path=Path("/data/results/input.pdb"))
            ],
            outputs=[
                FileArtifact(id="out", path=Path("/data/results/output.pdb"))
            ],
        )
        mock_target = _make_mock_target()
        with patch.object(task, "target", mock_target):
            await executor._execute(task)
        cmd = mock_target.run_command.call_args[0][0]
        assert "-v" in cmd
        assert "/data/results:/data/results" in cmd

    @pytest.mark.asyncio
    async def test_explicit_volumes_override_auto_mounts(
        self, horus_context: HorusContext
    ) -> None:
        """
        An explicit volume must override the auto-mount for the same host path.
        """
        del horus_context
        executor = DockerExecutor(
            image=_IMAGE, volumes={"/data/results": "/mnt/custom"}
        )
        task = HorusTask(
            id="test-task",
            name="test_task",
            executor=executor,
            runtime=CommandRuntime(command="true"),
            inputs=[
                FileArtifact(id="inp", path=Path("/data/results/input.pdb"))
            ],
            outputs=[],
        )
        mock_target = _make_mock_target()
        with patch.object(task, "target", mock_target):
            await executor._execute(task)
        cmd = mock_target.run_command.call_args[0][0]
        assert "/data/results:/mnt/custom" in cmd
        assert "/data/results:/data/results" not in cmd

    @pytest.mark.asyncio
    async def test_shared_parent_dir_mounted_once(
        self, horus_context: HorusContext
    ) -> None:
        """
        Two artifacts sharing a parent dir must produce exactly one -v flag.
        """
        del horus_context
        executor = DockerExecutor(image=_IMAGE)
        task = HorusTask(
            id="test-task",
            name="test_task",
            executor=executor,
            runtime=CommandRuntime(command="true"),
            inputs=[FileArtifact(id="a", path=Path("/shared/a.pdb"))],
            outputs=[FileArtifact(id="b", path=Path("/shared/b.pdb"))],
        )
        mock_target = _make_mock_target()
        with patch.object(task, "target", mock_target):
            await executor._execute(task)
        cmd = mock_target.run_command.call_args[0][0]
        assert cmd.count("/shared:/shared") == 1

    @pytest.mark.asyncio
    async def test_execute_rmi_failure_does_not_raise(
        self, horus_context: HorusContext
    ) -> None:
        """A docker rmi failure after a successful run must not raise."""
        del horus_context
        executor = DockerExecutor(
            image="myapp:test", dockerfile="FROM scratch"
        )
        task = self._make_task(executor)
        call_count = 0

        async def _run_command(cmd: str, **_: object) -> AsyncMock:
            nonlocal call_count
            call_count += 1
            if "rmi" in cmd:
                raise OSError("rmi failed")
            return _make_mock_proc()

        mock_target = MagicMock()
        mock_target.run_command = _run_command
        mock_target.put_file = AsyncMock()
        mock_target.mkdir = AsyncMock()
        with patch.object(task, "target", mock_target):
            await executor._execute(task)  # must not raise
