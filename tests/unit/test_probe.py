#
# horus_docker
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Unit tests for the container resource probe.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from horus_resource_monitor.wrapper import ATTR_EXACT

from horus_docker.executor.resources import ContainerScope
from horus_docker.probe import ContainerProbe, parse_stats


def _returns(value: bytes) -> Any:
    """An async stub returning *value*."""

    async def stub(*_args: object, **_kwargs: object) -> bytes:
        return value

    return stub


def _raises() -> Any:
    """An async stub that always fails."""

    async def stub(*_args: object, **_kwargs: object) -> bytes:
        raise FileNotFoundError("nope")

    return stub


def _fake_task(payload: bytes) -> Any:
    """A stand-in task whose target answers every read with *payload*."""

    class _Proc:
        async def communicate(self) -> tuple[bytes, bytes]:
            return payload, b""

    async def run_command(cmd: str, **kwargs: object) -> _Proc:
        del cmd, kwargs
        return _Proc()

    return SimpleNamespace(
        id="fake",
        name="fake",
        side_artifacts_dir="/tmp/side",
        target=SimpleNamespace(
            get_file=_returns(payload) if payload else _raises(),
            run_command=run_command,
        ),
    )


@pytest.mark.unit
class TestContainerProbe:
    """Container stats are read from the runtime, or not at all."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("12.5MiB / 7.7GiB 3.14%", (12.5 * 1024, 3.14)),
            ("1GiB / 2GiB 0.00%", (1024 * 1024, 0.0)),
            ("", None),
            ("garbage", None),
            ("12.5PiB / 1GiB 1%", None),
            ("MiB / 7.7GiB 3.14%", None),
        ],
    )
    def test_parse_stats(
        self, text: str, expected: tuple[float, float] | None
    ) -> None:
        """Docker's human-readable output is parsed, or refused cleanly."""
        assert parse_stats(text) == expected

    @pytest.mark.asyncio
    async def test_samples_the_container(self) -> None:
        """With an id, each sample is one query to the container runtime."""
        probe = ContainerProbe()
        await probe.start(
            _fake_task(b""), ContainerScope(container_id="deadbeef")
        )
        probe._task = _fake_task(b"64MiB / 2GiB 50.00%")

        await probe.sample_once()
        await probe.stop()

        rows = await probe.drain()
        assert rows and rows[0]["rss_kb"] == 64 * 1024
        assert rows[0]["attribution"] == ATTR_EXACT

    @pytest.mark.asyncio
    async def test_cidfile_is_read_from_the_target(self) -> None:
        """
        The id is not known when the probe starts, only once the runtime has
        written the cidfile.
        """
        probe = ContainerProbe()
        await probe.start(
            _fake_task(b"cafe1234"), ContainerScope(cidfile="/tmp/cid")
        )
        assert await probe._resolve_id() == "cafe1234"

    @pytest.mark.asyncio
    async def test_no_id_produces_nothing(self) -> None:
        """
        An unidentifiable container degrades to no samples, never to an
        exception that would take the task down with it.
        """
        probe = ContainerProbe()
        await probe.start(_fake_task(b""), ContainerScope())
        await probe.sample_once()
        await probe.stop()
        assert await probe.drain() == []
