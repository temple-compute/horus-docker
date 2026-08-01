#
# horus_docker
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""Unit tests for DockerExecutor.cancel_execution()."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from horus_docker.executor.docker import DockerExecutor

_IMAGE = "python:3.13-slim"


def _armed_executor() -> tuple[DockerExecutor, MagicMock, AsyncMock]:
    """Return an executor mid-run, its mock target, and the stop process."""
    executor = DockerExecutor(image=_IMAGE)
    executor._container_name = "horus-test-task"
    proc = AsyncMock()
    target = MagicMock()
    target.run_command = AsyncMock(return_value=proc)
    executor._target = target
    return executor, target, proc


@pytest.mark.unit
class TestCancelExecution:
    """Verify DockerExecutor.cancel_execution() stops the container."""

    async def test_cancel_execution_stops_via_target_channel(self) -> None:
        """The stop must be issued on the target, not on the orchestrator."""
        executor, target, proc = _armed_executor()

        await executor.cancel_execution()

        target.run_command.assert_awaited_once_with(
            "docker stop horus-test-task", detach=False
        )
        proc.wait.assert_awaited_once()

    async def test_cancel_execution_noop_when_no_container(self) -> None:
        """cancel_execution must be a no-op when no container is running."""
        executor = DockerExecutor(image=_IMAGE)
        target = MagicMock()
        target.run_command = AsyncMock()
        executor._target = target

        await executor.cancel_execution()

        target.run_command.assert_not_called()

    async def test_cancel_execution_noop_without_target(self) -> None:
        """No target means nothing was started here; must not blow up."""
        executor = DockerExecutor(image=_IMAGE)
        executor._container_name = "horus-test-task"

        await executor.cancel_execution()  # must not raise

    async def test_cancel_execution_clears_container_name(self) -> None:
        """cancel_execution must clear _container_name before stopping."""
        executor, _target, _proc = _armed_executor()

        await executor.cancel_execution()

        assert executor._container_name is None

    async def test_cancel_execution_idempotent(self) -> None:
        """Second cancel_execution call must be a no-op (name cleared)."""
        executor, target, _proc = _armed_executor()

        await executor.cancel_execution()
        await executor.cancel_execution()

        assert target.run_command.await_count == 1

    async def test_cancel_execution_swallows_channel_errors(self) -> None:
        """A failed stop must not mask the cancellation being delivered."""
        executor, target, _proc = _armed_executor()
        target.run_command = AsyncMock(side_effect=OSError("channel down"))

        await executor.cancel_execution()  # must not raise
