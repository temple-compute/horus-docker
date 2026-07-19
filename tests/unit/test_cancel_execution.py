#
# horus_docker
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""Unit tests for DockerExecutor.cancel_execution()."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from horus_docker.executor.docker import DockerExecutor

_IMAGE = "python:3.13-slim"


@pytest.mark.unit
class TestCancelExecution:
    """Verify DockerExecutor.cancel_execution() stops the container."""

    async def test_cancel_execution_calls_docker_stop(self) -> None:
        """cancel_execution must call 'docker stop <name>'."""
        executor = DockerExecutor(image=_IMAGE)
        executor._container_name = "horus-test-task"

        mock_proc = AsyncMock()
        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ) as mock_exec:
            await executor.cancel_execution()
            mock_exec.assert_called_once_with(
                "docker",
                "stop",
                "horus-test-task",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            mock_proc.wait.assert_called_once()

    async def test_cancel_execution_noop_when_no_container(self) -> None:
        """cancel_execution must be a no-op when no container is running."""
        executor = DockerExecutor(image=_IMAGE)
        # _container_name is None by default

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            await executor.cancel_execution()
            mock_exec.assert_not_called()

    async def test_cancel_execution_clears_container_name(self) -> None:
        """cancel_execution must clear _container_name before stopping."""
        executor = DockerExecutor(image=_IMAGE)
        executor._container_name = "horus-test-task"

        mock_proc = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await executor.cancel_execution()

        assert executor._container_name is None

    async def test_cancel_execution_idempotent(self) -> None:
        """Second cancel_execution call must be a no-op (name cleared)."""
        executor = DockerExecutor(image=_IMAGE)
        executor._container_name = "horus-test-task"

        mock_proc = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await executor.cancel_execution()
            # second call — _container_name is already None
            await executor.cancel_execution()

        assert mock_proc.wait.call_count == 1  # only called once
