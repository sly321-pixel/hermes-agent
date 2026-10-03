"""Mem0 provider shutdown must never close the backend underneath in-flight work (#90728, #107517).

``sync_turn`` runs ``backend.add(..., infer=True)`` on a background thread. OSS extraction with a
local LLM takes far longer than the old 5s ``shutdown()`` join (field data: p50 17s, max 140s), so
every one-shot cron session closed the backend mid-extraction: mem0 2.0.10 ``Memory.close()`` sets
``self.db = None`` and the still-running add() then failed its vector inserts ("Cannot send a
request, as the client has been closed") and ``save_messages`` ("'NoneType' object has no attribute
'save_messages'"), dropping the turn's extracted memories.

The contract locked here: shutdown stays bounded (cron teardown is itself capped at 10s), stops
accepting new work, and hands the close to whichever in-flight operation finishes last.
"""

import json
import logging
import threading

from plugins.memory.mem0 import Mem0MemoryProvider


class ClosableBackend:
    """Backend that, like mem0's ``Memory``, breaks if used after ``close()``."""

    def __init__(self, *, block_add=False, block_search=False):
        self.closed = False
        self.close_calls = 0
        self.events = []
        self.add_entered, self.add_release = threading.Event(), threading.Event()
        self.search_entered, self.search_release = threading.Event(), threading.Event()
        self._block_add, self._block_search = block_add, block_search

    def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
        self.add_entered.set()
        if self._block_add:
            assert self.add_release.wait(30), "test never released add()"
        if self.closed:  # what mem0 does at Phase 8 once close() nulled self.db
            raise AttributeError("'NoneType' object has no attribute 'save_messages'")
        self.events.append("add_done")
        return {"results": []}

    def search(self, query, *, filters, top_k=10, rerank=False):
        self.search_entered.set()
        if self._block_search:
            assert self.search_release.wait(30), "test never released search()"
        if self.closed:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        self.events.append("search_done")
        return [{"id": "m1", "memory": "likes tea"}]

    def close(self):
        self.close_calls += 1
        self.closed = True
        self.events.append("close")


def _provider(backend, monkeypatch=None, wait=0.05):
    provider = Mem0MemoryProvider()
    provider.initialize("test-session")
    provider._user_id, provider._agent_id = "u123", "hermes"
    provider._backend = backend
    provider._shutdown_wait = wait  # keep the bounded join short; the contract does not depend on it
    return provider


def _join(thread):
    if thread is not None:
        thread.join(timeout=30)
        assert not thread.is_alive()


class TestShutdownWithInflightSync:

    def test_backend_closes_only_after_the_inflight_extraction_finishes(self, caplog):
        backend = ClosableBackend(block_add=True)
        provider = _provider(backend)
        provider.sync_turn("remember: code ABC-1234", "OK", session_id="s1")
        assert backend.add_entered.wait(30)

        with caplog.at_level(logging.WARNING, logger="plugins.memory.mem0"):
            provider.shutdown()
            # Bounded: shutdown returned while extraction is still parked, and did not close under it.
            assert provider._sync_thread.is_alive()
            assert not backend.closed

            backend.add_release.set()
            _join(provider._sync_thread)

        assert backend.events == ["add_done", "close"]
        assert backend.close_calls == 1
        assert "Mem0 sync failed" not in caplog.text
        assert provider._consecutive_failures == 0
        assert provider._backend is None

    def test_no_new_work_is_accepted_once_shutdown_started(self):
        # A prefetch keeps the backend open past shutdown(); nothing new may start in that window.
        backend = ClosableBackend(block_search=True)
        provider = _provider(backend)
        provider.on_turn_start(1, "what do I drink?")
        assert backend.search_entered.wait(30)
        provider.shutdown()
        assert not backend.closed

        provider.sync_turn("late", "turn", session_id="s1")
        assert provider._sync_thread is None  # no extraction spawned against a closing backend
        assert not backend.add_entered.is_set()
        assert provider.prefetch("another query") == ""

        backend.search_release.set()
        _join(provider._prefetch_thread)
        assert backend.events == ["search_done", "close"]

    def test_late_sync_during_inflight_sync_returns_without_waiting(self, monkeypatch):
        backend = ClosableBackend(block_add=True)
        provider = _provider(backend)
        provider.sync_turn("first", "turn", session_id="s1")
        assert backend.add_entered.wait(30)
        first = provider._sync_thread
        provider.shutdown()

        joined = []
        monkeypatch.setattr(first, "join", lambda timeout=None: joined.append(timeout))
        provider.sync_turn("late", "turn", session_id="s1")
        assert joined == []  # refused up front, not after the 5s wait on the previous turn
        assert provider._sync_thread is first

        monkeypatch.undo()
        backend.add_release.set()
        _join(first)
        assert backend.events == ["add_done", "close"]

    def test_interpreter_exit_hook_does_not_close_under_inflight_sync(self, caplog):
        backend = ClosableBackend(block_add=True)
        provider = _provider(backend)
        provider.sync_turn("remember", "OK", session_id="s1")
        assert backend.add_entered.wait(30)

        with caplog.at_level(logging.WARNING, logger="plugins.memory.mem0"):
            provider._shutdown_backend()  # the atexit hook
        assert not backend.closed
        assert "still running" in caplog.text  # exit drops it: say so instead of failing later

        backend.add_release.set()
        _join(provider._sync_thread)
        assert backend.events == ["add_done", "close"]


class TestShutdownIdle:

    def test_idle_provider_closes_immediately(self):
        backend = ClosableBackend()
        provider = _provider(backend)
        provider.sync_turn("hi", "there", session_id="s1")
        _join(provider._sync_thread)
        provider.shutdown()
        assert backend.events == ["add_done", "close"]
        assert provider._backend is None

    def test_shutdown_is_idempotent(self):
        backend = ClosableBackend()
        provider = _provider(backend)
        provider.shutdown()
        provider.shutdown()
        provider._shutdown_backend()
        assert backend.close_calls == 1

    def test_reinitialize_after_shutdown_accepts_work_again(self):
        old = ClosableBackend()
        provider = _provider(old)
        provider.shutdown()
        provider.initialize("next-session")
        new = ClosableBackend()
        provider._backend = new
        provider.sync_turn("after", "reinit", session_id="s2")
        _join(provider._sync_thread)
        assert new.events == ["add_done"]
        assert old.close_calls == 1


class TestShutdownWithInflightToolCall:

    def test_tool_call_in_flight_defers_close(self):
        backend = ClosableBackend(block_search=True)
        provider = _provider(backend)
        out = {}
        caller = threading.Thread(
            target=lambda: out.setdefault("r", provider.handle_tool_call("mem0_search", {"query": "tea"})))
        caller.start()
        assert backend.search_entered.wait(30)

        provider.shutdown()
        assert not backend.closed

        backend.search_release.set()
        _join(caller)
        assert json.loads(out["r"])["count"] == 1
        assert backend.events == ["search_done", "close"]

    def test_tool_call_after_shutdown_reports_unavailable(self):
        provider = _provider(ClosableBackend())
        provider.shutdown()
        assert "error" in json.loads(provider.handle_tool_call("mem0_search", {"query": "tea"}))


class TestShutdownWaitConfig:

    def test_default_wait_is_unchanged(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = Mem0MemoryProvider()
        provider.initialize("s")
        assert provider._shutdown_wait == 5.0

    def test_shutdown_wait_secs_from_mem0_json(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "mem0.json").write_text('{"shutdown_wait_secs": 90}')
        provider = Mem0MemoryProvider()
        provider.initialize("s")
        assert provider._shutdown_wait == 90.0

    def test_invalid_shutdown_wait_falls_back_to_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "mem0.json").write_text('{"shutdown_wait_secs": "soon"}')
        provider = Mem0MemoryProvider()
        provider.initialize("s")
        assert provider._shutdown_wait == 5.0
