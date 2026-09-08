"""
End-to-end browser test of the React UI (backend + frontend, real Playwright
Chromium). Assumes both servers are already running:

    python run_api_ui.py          # backend on 127.0.0.1:8010
    cd frontend && npm run dev    # frontend on localhost:5173

Set XRAG_E2E_SKIP=1 to skip (e.g. CI without the servers up).
"""

import json
import os
import unittest
import urllib.request

from playwright.sync_api import sync_playwright

FRONTEND_URL = "http://localhost:5173"
BACKEND_URL = "http://127.0.0.1:8010"
SKIP = os.environ.get("XRAG_E2E_SKIP") == "1"


def _warm_up_backend() -> None:
    """Force lazy model loading (embedding + reranker) before timing anything.

    Without this, whichever test happens to run first pays the one-time cold-
    start cost inside its own timeout budget instead of the suite's.
    """
    req = urllib.request.Request(
        f"{BACKEND_URL}/ui/chat",
        data=json.dumps({
            "question": "warmup", "session_id": "warmup", "arm": "C_hybrid_rerank",
            "corpus": "statutes", "chat_history": [], "memory_enabled": False,
            "deep_analysis": False,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    # The remote LLM endpoint has been observed taking >150s on a single
    # call; urllib's timeout applies per socket read (i.e. per gap between
    # SSE events), not to the whole request, but generous margin still helps
    # since this call also pays one-time cold model loading on top.
    with urllib.request.urlopen(req, timeout=300) as resp:
        resp.read()


@unittest.skipIf(SKIP, "XRAG_E2E_SKIP=1")
class TestUIEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _warm_up_backend()
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.page = self.browser.new_page()

    def tearDown(self):
        self.page.close()

    def test_chat_streams_an_answer_with_evidence(self):
        page = self.page
        page.goto(FRONTEND_URL)
        page.get_by_placeholder("Ask a question…").fill("What is the punishment for murder?")
        page.get_by_role("button", name="Send").click()

        # Evidence panel populates on the `chunks` SSE event, before the
        # first token -- this is the moment the plan calls out as the demo.
        page.wait_for_selector(".evidence-chunk", timeout=60_000)
        self.assertGreater(page.locator(".evidence-chunk").count(), 0)

        # Wait for the streamed answer to actually contain text (not just the
        # "…" placeholder) and for the Send button to re-enable (turn done).
        page.wait_for_function(
            "document.querySelectorAll('.msg-assistant')[document.querySelectorAll('.msg-assistant').length-1].textContent.length > 5",
            timeout=180_000,
        )
        page.wait_for_selector(".chat-input button:not([disabled])", timeout=180_000)

    def test_ircot_arm_reachable_end_to_end(self):
        page = self.page
        page.goto(FRONTEND_URL)
        page.locator(".chat-controls select").first.select_option("D_ircot")
        page.get_by_placeholder("Ask a question…").fill("What did Kesavananda Bharati decide?")
        page.get_by_role("button", name="Send").click()

        # IRCoT does up to 4 sequential retrieve-reason hops, each with its own
        # remote LLM call; the remote endpoint has been observed to take up to
        # ~160s on a single call, so this generously bounds the worst case
        # rather than the typical case.
        page.wait_for_selector(".chat-input button:not([disabled])", timeout=420_000)
        # ircot metadata is echoed back on the `strategy` event and rendered
        # in the evidence panel's strategy-meta block.
        meta_text = page.locator(".strategy-meta").inner_text()
        self.assertIn("ircot", meta_text.lower())

    def test_graph_tab_renders_both_scopes(self):
        page = self.page
        page.goto(FRONTEND_URL)
        page.get_by_role("button", name="Graph").click()

        page.wait_for_function("window.__graphNodeCount > 0", timeout=15_000)

        page.locator(".graph-toolbar select").select_option("memory")
        page.wait_for_function("window.__graphNodeCount > 0", timeout=15_000)

    def test_graph_tab_responsive_during_chat_turn(self):
        """A chat turn in flight must not block other routes/tabs.

        Matches the concurrency requirement noted for the SSE backend design:
        the chat route is a sync `def` running in Starlette's threadpool, so
        it must not freeze the event loop for the graph fetch.
        """
        chat_page = self.browser.new_page()
        chat_page.goto(FRONTEND_URL)
        chat_page.get_by_placeholder("Ask a question…").fill("Define culpable homicide.")
        chat_page.get_by_role("button", name="Send").click()

        graph_page = self.browser.new_page()
        graph_page.goto(FRONTEND_URL)
        graph_page.get_by_role("button", name="Graph").click()
        graph_page.wait_for_function("window.__graphNodeCount > 0", timeout=10_000)

        chat_page.close()
        graph_page.close()


if __name__ == "__main__":
    unittest.main()
