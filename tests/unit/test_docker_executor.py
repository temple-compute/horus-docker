#
# horus_docker
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""Unit tests for DockerExecutor."""

from unittest.mock import MagicMock, patch

import pytest
from docker.errors import DockerException
from horus_builtin.runtime.command import CommandRuntime
from horus_builtin.task.horus_task import HorusTask
from horus_runtime.context import HorusContext
from horus_runtime.core.executor.base import BaseExecutor
from horus_runtime.core.task.exceptions import TaskExecutionError

from horus_docker.executor.docker import DockerExecutor

_IMAGE = "python:3.13-slim"
_NONZERO_CODE = 2


@pytest.mark.unit
class TestDockerExecutorRegistration:
    """Verify DockerExecutor registers correctly in the AutoRegistry."""

    def test_registered_under_kind(self) -> None:
        """DockerExecutor must appear in BaseExecutor.registry."""
        assert "docker_executor" in BaseExecutor.registry
        assert BaseExecutor.registry["docker_executor"] is DockerExecutor

    def test_kind_field(self) -> None:
        """The kind field must equal 'docker_executor'."""
        assert DockerExecutor(image=_IMAGE).kind == "docker_executor"

    def test_runtimes_filter(self) -> None:
        """CommandRuntime must be listed as a supported runtime."""
        assert CommandRuntime in DockerExecutor.runtimes


@pytest.mark.unit
class TestDockerExecutorDefaults:
    """Verify default field values for DockerExecutor."""

    def test_required_image_field(self) -> None:
        """The image field must be stored as provided."""
        assert DockerExecutor(image=_IMAGE).image == _IMAGE

    def test_optional_fields_default_to_empty(self) -> None:
        """env, volumes, and ports must default to empty dicts."""
        executor = DockerExecutor(image=_IMAGE)
        assert executor.env == {}
        assert executor.volumes == {}
        assert executor.ports == {}

    def test_optional_fields_default_to_none(self) -> None:
        """working_dir, network, entrypoint, user must default to None."""
        executor = DockerExecutor(image=_IMAGE)
        assert executor.working_dir is None
        assert executor.network is None
        assert executor.entrypoint is None
        assert executor.user is None

    def test_auto_remove_defaults_to_true(self) -> None:
        """auto_remove must default to True."""
        assert DockerExecutor(image=_IMAGE).auto_remove is True


@pytest.mark.unit
class TestDockerVolumeMapping:
    """Verify _docker_volumes() conversion to docker-py format."""

    def test_empty_volumes(self) -> None:
        """No volumes should produce an empty dict."""
        assert DockerExecutor(image=_IMAGE)._docker_volumes() == {}

    def test_single_volume(self) -> None:
        """A single host->container mapping must use bind/mode keys."""
        executor = DockerExecutor(
            image=_IMAGE,
            volumes={"/host/data": "/container/data"},
        )
        assert executor._docker_volumes() == {
            "/host/data": {"bind": "/container/data", "mode": "rw"}
        }

    def test_multiple_volumes(self) -> None:
        """Multiple volumes must all be converted correctly."""
        executor = DockerExecutor(
            image=_IMAGE, volumes={"/a": "/x", "/b": "/y"}
        )
        assert executor._docker_volumes() == {
            "/a": {"bind": "/x", "mode": "rw"},
            "/b": {"bind": "/y", "mode": "rw"},
        }


@pytest.mark.unit
class TestDockerPortMapping:
    """Verify _docker_ports() conversion to docker-py format."""

    def test_empty_ports(self) -> None:
        """No ports should produce an empty dict."""
        assert DockerExecutor(image=_IMAGE)._docker_ports() == {}

    def test_single_port(self) -> None:
        """host_port->container_port must become container/tcp->int(host)."""
        executor = DockerExecutor(image=_IMAGE, ports={"8080": "80"})
        assert executor._docker_ports() == {"80/tcp": 8080}

    def test_multiple_ports(self) -> None:
        """Multiple port mappings must all be converted correctly."""
        executor = DockerExecutor(
            image=_IMAGE, ports={"8080": "80", "5432": "5432"}
        )
        assert executor._docker_ports() == {
            "80/tcp": 8080,
            "5432/tcp": 5432,
        }

    def test_host_port_is_int(self) -> None:
        """The host port value must be an integer, not a string."""
        executor = DockerExecutor(image=_IMAGE, ports={"9000": "9000"})
        host_port = next(iter(executor._docker_ports().values()))
        assert isinstance(host_port, int)


@pytest.mark.unit
class TestRunContainer:
    """Verify _run_container() behaviour with a mocked Docker client."""

    def _make_mock_client(
        self, status_code: int = 0, logs: bytes = b""
    ) -> tuple[MagicMock, MagicMock]:
        """Return a (client_mock, container_mock) pair."""
        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": status_code}
        mock_container.logs.return_value = logs
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        return mock_client, mock_container

    def test_returns_exit_code_and_decoded_logs(self) -> None:
        """Exit code and decoded log bytes must be returned as a tuple."""
        executor = DockerExecutor(image=_IMAGE)
        mock_client, _ = self._make_mock_client(status_code=0, logs=b"hello\n")
        with patch(
            "horus_docker.executor.docker.docker.from_env",
            return_value=mock_client,
        ):
            code, logs = executor._run_container(
                ["/bin/sh", "-c", "echo hello"]
            )
        assert code == 0
        assert logs == "hello\n"

    def test_returns_nonzero_exit_code(self) -> None:
        """A non-zero StatusCode must be forwarded as the returned code."""
        executor = DockerExecutor(image=_IMAGE)
        mock_client, _ = self._make_mock_client(
            status_code=_NONZERO_CODE, logs=b"error"
        )
        with patch(
            "horus_docker.executor.docker.docker.from_env",
            return_value=mock_client,
        ):
            code, _ = executor._run_container(["/bin/sh", "-c", "exit 2"])
        assert code == _NONZERO_CODE

    def test_missing_status_code_defaults_to_1(self) -> None:
        """An empty result dict must default to exit code 1."""
        executor = DockerExecutor(image=_IMAGE)
        mock_container = MagicMock()
        mock_container.wait.return_value = {}
        mock_container.logs.return_value = b""
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        with patch(
            "horus_docker.executor.docker.docker.from_env",
            return_value=mock_client,
        ):
            code, _ = executor._run_container(["/bin/sh", "-c", "true"])
        assert code == 1

    def test_removes_container_when_auto_remove_true(self) -> None:
        """container.remove(force=True) called when auto_remove=True."""
        executor = DockerExecutor(image=_IMAGE, auto_remove=True)
        mock_client, mock_container = self._make_mock_client()
        with patch(
            "horus_docker.executor.docker.docker.from_env",
            return_value=mock_client,
        ):
            executor._run_container(["/bin/sh", "-c", "true"])
        mock_container.remove.assert_called_once_with(force=True)

    def test_skips_remove_when_auto_remove_false(self) -> None:
        """container.remove must NOT be called when auto_remove=False."""
        executor = DockerExecutor(image=_IMAGE, auto_remove=False)
        mock_client, mock_container = self._make_mock_client()
        with patch(
            "horus_docker.executor.docker.docker.from_env",
            return_value=mock_client,
        ):
            executor._run_container(["/bin/sh", "-c", "true"])
        mock_container.remove.assert_not_called()

    def test_closes_client_on_success(self) -> None:
        """client.close() must be called even after a successful run."""
        executor = DockerExecutor(image=_IMAGE)
        mock_client, _ = self._make_mock_client()
        with patch(
            "horus_docker.executor.docker.docker.from_env",
            return_value=mock_client,
        ):
            executor._run_container(["/bin/sh", "-c", "true"])
        mock_client.close.assert_called_once()

    def test_closes_client_on_docker_exception(self) -> None:
        """client.close() must be called even on DockerException."""
        executor = DockerExecutor(image=_IMAGE)
        mock_client = MagicMock()
        mock_client.containers.run.side_effect = DockerException(
            "daemon not running"
        )
        with patch(
            "horus_docker.executor.docker.docker.from_env",
            return_value=mock_client,
        ):
            with pytest.raises(DockerException):
                executor._run_container(["/bin/sh", "-c", "true"])
        mock_client.close.assert_called_once()


@pytest.mark.unit
class TestDockerExecutorExecute:
    """Verify DockerExecutor._execute() end-to-end behaviour."""

    def _make_task(
        self, executor: DockerExecutor, command: str = "echo hello"
    ) -> HorusTask:
        """Return a minimal HorusTask wired to the given executor."""
        return HorusTask(
            id="test-task",
            name="test_task",
            executor=executor,
            runtime=CommandRuntime(command=command),
        )

    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        horus_context: HorusContext,
    ) -> None:
        """A zero exit code must complete without raising."""
        del horus_context  # Unused in this test, but required by fixture
        executor = DockerExecutor(image=_IMAGE)
        task = self._make_task(executor)
        with patch.object(
            executor, "_run_container", return_value=(0, "hello\n")
        ):
            await executor._execute(task)

    @pytest.mark.asyncio
    async def test_execute_nonzero_exit_raises(
        self,
        horus_context: HorusContext,
    ) -> None:
        """A non-zero exit code must raise TaskExecutionError."""
        del horus_context  # Unused in this test, but required by fixture
        executor = DockerExecutor(image=_IMAGE)
        task = self._make_task(executor, command="exit 1")
        with patch.object(
            executor,
            "_run_container",
            return_value=(1, "something went wrong"),
        ):
            with pytest.raises(
                TaskExecutionError, match="Container exited with code 1"
            ):
                await executor._execute(task)

    @pytest.mark.asyncio
    async def test_execute_docker_exception_raises(
        self,
        horus_context: HorusContext,
    ) -> None:
        """DockerException in _run_container must raise TaskExecutionError."""
        del horus_context  # Unused in this test, but required by fixture
        executor = DockerExecutor(image=_IMAGE)
        task = self._make_task(executor)
        with patch.object(
            executor,
            "_run_container",
            side_effect=DockerException("connection refused"),
        ):
            with pytest.raises(
                TaskExecutionError, match="Docker execution failed"
            ):
                await executor._execute(task)

    @pytest.mark.asyncio
    async def test_execute_passes_sh_wrapped_command(
        self,
        horus_context: HorusContext,
    ) -> None:
        """The command must be wrapped with /bin/sh -c before dispatch."""
        del horus_context  # Unused in this test, but required by fixture
        executor = DockerExecutor(image=_IMAGE)
        task = self._make_task(executor, command="echo hello")
        with patch.object(
            executor, "_run_container", return_value=(0, "")
        ) as mock_run:
            await executor._execute(task)
        mock_run.assert_called_once_with(["/bin/sh", "-c", "echo hello"])
