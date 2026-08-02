#
# horus_docker
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Probe for work that runs inside a Docker container.

The container CLI is a thin client: the workload is reparented under the
container daemon's supervisor, in another process group, so walking the
spawned tree measures the client and nothing else. The daemon is the only one
who knows, so we ask it — through ``task.target.run_command``, which means the
same code measures a container on a remote host as on this one.
"""

import re
import time
from typing import TYPE_CHECKING, Any, ClassVar

from horus_resource_monitor.probe.base import SLOW_INTERVAL, PollingProbe
from horus_runtime.core.resources import ResourceScope
from horus_runtime.core.target.base import BaseTarget
from horus_runtime.logging import horus_logger
from pydantic import PrivateAttr

from horus_docker.executor.resources import ContainerScope
from horus_docker.i18n import tr as _

if TYPE_CHECKING:
    from horus_runtime.core.task.base import BaseTask

#: Prints one line of "<mem used> / <mem limit> <cpu>%".
STATS_COMMAND = (
    "docker stats --no-stream --format '{{.MemUsage}} {{.CPUPerc}}'"
)

_KB = 1024.0
_PCT = 100.0

# Both the binary units docker prints and the decimal ones it prints for some
# drivers, all reduced to kilobytes.
_UNITS = {
    "b": 1 / _KB,
    "kib": 1.0,
    "kb": 1000 / _KB,
    "mib": _KB,
    "mb": 1000 * 1000 / _KB,
    "gib": _KB * _KB,
    "gb": 1000 * 1000 * 1000 / _KB,
    "tib": _KB * _KB * _KB,
}

_MEM = re.compile(r"^([0-9.]+)([A-Za-z]+)$")

# "<used> <cpu>%" is the shortest usable form of the stats line.
_MIN_TOKENS = 2


def parse_stats(text: str) -> tuple[float, float] | None:
    """
    Parse one ``docker stats`` line into ``(rss_kb, cpu_pct)``.

    Returns ``None`` for anything unrecognisable, which includes the empty
    output docker gives for a container that has not started yet.
    """
    tokens = text.strip().splitlines()[0].split() if text.strip() else []
    if len(tokens) < _MIN_TOKENS:
        return None
    mem = _MEM.match(tokens[0])
    if mem is None or mem.group(2).lower() not in _UNITS:
        return None
    try:
        rss_kb = float(mem.group(1)) * _UNITS[mem.group(2).lower()]
        cpu_pct = float(tokens[-1].rstrip("%"))
    except ValueError:
        return None
    return rss_kb, cpu_pct


class ContainerProbe(PollingProbe):
    """
    Polls the container runtime for the task's container.
    """

    kind: str = "container"
    priority: ClassVar[int] = 100

    _container_id: str | None = PrivateAttr(default=None)
    _cidfile: str | None = PrivateAttr(default=None)
    _cpu_s: float = PrivateAttr(default=0.0)
    _last_poll: float = PrivateAttr(default=0.0)

    @classmethod
    def supports(cls, scope: ResourceScope, target: BaseTarget) -> bool:
        """Containers, local or remote — the daemon answers either way."""
        del target
        return isinstance(scope, ContainerScope)

    def interval(self, elapsed: float) -> float:
        """
        A fixed second, not the adaptive cadence.

        Each sample is a process spawn and a round-trip to the daemon; running
        that ten times a second would cost more than the task it watches.
        """
        del elapsed
        return SLOW_INTERVAL

    async def start(self, task: "BaseTask", scope: ResourceScope) -> None:
        """Note where the container id is, or will be."""
        if isinstance(scope, ContainerScope):
            self._container_id = scope.container_id
            self._cidfile = scope.cidfile
        await super().start(task, scope)

    async def _resolve_id(self) -> str | None:
        """
        The container id, read from the cidfile the first time it exists.

        The file appears only once the container has started, so a miss is
        normal early on and is retried on the next sample.
        """
        if self._container_id or self._cidfile is None or self._task is None:
            return self._container_id
        try:
            raw = await self._task.target.get_file(self._cidfile)
        except Exception:
            return None
        self._container_id = raw.decode("utf-8", "replace").strip() or None
        return self._container_id

    async def _sample(self) -> dict[str, Any] | None:
        """Ask the daemon for the container's current usage."""
        container = await self._resolve_id()
        if container is None or self._task is None:
            return None

        proc = await self._task.target.run_command(
            f"{STATS_COMMAND} {container}"
        )
        stdout, _stderr = await proc.communicate()
        stats = parse_stats(stdout.decode("utf-8", "replace"))
        if stats is None:
            return None

        rss_kb, cpu_pct = stats
        # docker reports a percentage, not the counter the rest of the plugin
        # speaks, so integrate it back into seconds over the polling gap.
        now = time.time()
        if self._last_poll:
            self._cpu_s += cpu_pct / _PCT * (now - self._last_poll)
        self._last_poll = now
        return {
            "t": now,
            "rss_kb": rss_kb,
            "cpu_s": self._cpu_s,
            "cpu_pct": cpu_pct,
        }

    async def stop(self) -> None:
        """Stop polling, saying so when we never found the container."""
        if self._container_id is None:
            horus_logger.log.debug(
                _("No container id for task %(task)s; nothing was measured")
                % {"task": self._task.id if self._task else "?"}
            )
        await super().stop()
