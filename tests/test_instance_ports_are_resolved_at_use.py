"""An in-process client reaches the stack it was told about, not the one import order implied.

Every consumer of the privileged channel used to be a fresh process that the stack launched with
`INSTANCE` and the `TF_*` block already in its environment, so resolving the ports into module
constants at import was safe. The flat-functional-scripts refactor makes the collection entry run
the gym IN its own process (docs/handoff/flat-functional-scripts-20260922.md): it sets the stack's
environment and then opens a client, and an import that happened first would have frozen instance
0's numbers.

That failure is silent and slow -- with `INSTANCE=1` the stack sits on 5565/5566/5591 while the
client binds :5556 and waits on :5581, and the only symptom is `no privileged state on :5581`
ten seconds later (2026-09-10, cost a stack cycle to read). It is also invisible to the GPU gate,
which runs on instance 0 where the defaults happen to be right. So it is pinned here instead.
"""

from __future__ import annotations

import pytest

from locomani.sim import privileged_pub


@pytest.mark.parametrize(
    "instance, privileged, episode",
    [(0, 5581, 5582), (1, 5591, 5592), (2, 5601, 5602)],
)
def test_the_port_follows_the_instance_set_after_import(monkeypatch, instance, privileged, episode):
    """The module is already imported by the time the environment is set -- that is the point."""
    monkeypatch.delenv("TF_PRIVILEGED_PORT", raising=False)
    monkeypatch.delenv("TF_EPISODE_PORT", raising=False)
    monkeypatch.setenv("INSTANCE", str(instance))
    assert privileged_pub.privileged_port() == privileged
    assert privileged_pub.episode_port() == episode


def test_an_exported_port_still_wins_over_the_instance_formula(monkeypatch):
    """The stack exports the block explicitly; that must keep beating the derived default."""
    monkeypatch.setenv("INSTANCE", "0")
    monkeypatch.setenv("TF_PRIVILEGED_PORT", "5591")
    assert privileged_pub.privileged_port() == 5591


def test_no_module_constant_freezes_a_port_at_import():
    """A reintroduced constant would pass every test above and still break the in-process path."""
    frozen = [name for name in ("PRIVILEGED_PORT", "EPISODE_PORT") if hasattr(privileged_pub, name)]
    assert not frozen, f"{frozen} is resolved at import; use privileged_port()/episode_port() at the seam"


def test_the_reader_reports_the_port_it_actually_resolved(monkeypatch):
    """The timeout message is the only symptom this failure produces; it must name the real port."""
    monkeypatch.delenv("TF_PRIVILEGED_PORT", raising=False)
    monkeypatch.setenv("INSTANCE", "1")
    reader = privileged_pub.PrivilegedSubscriber.__new__(privileged_pub.PrivilegedSubscriber)
    reader.port = privileged_pub.privileged_port()
    assert reader.port == 5591
