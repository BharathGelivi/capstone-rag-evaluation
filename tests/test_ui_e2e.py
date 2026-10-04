"""Browser regressions with a fully mocked backend: no models, data writes or LLM calls.
Start the frontend on port 5173 and set XRAG_E2E=1 to opt in.
"""
import json
import os
import unittest
from playwright.sync_api import sync_playwright

CONFIG = {"arms": ["C_hybrid_rerank", "D_ircot", "F_graphrag"],
          "corpora": ["statutes", "judgments", "both"], "device": "CPU",
          "arms_by_corpus": {"statutes": ["C_hybrid_rerank", "D_ircot"],
                             "judgments": ["C_hybrid_rerank", "D_ircot", "F_graphrag"],
                             "both": ["C_hybrid_rerank"]}}
SESSIONS = [{"session_id": "session_a", "title": "First session"},
            {"session_id": "session_b", "title": "Second session"}]

FETCH_SCRIPT = r"""
window.__chatRequests = [];
window.__chatMode = "done";
const originalFetch = window.fetch.bind(window);
window.fetch = async (url, init) => {
  if (!String(url).endsWith("/ui/chat")) return originalFetch(url, init);
  window.__chatRequests.push(JSON.parse(init.body));
  const mode = window.__chatMode;
  const encoder = new TextEncoder();
  const stream = new ReadableStream({start(controller) {
    const emit = event => controller.enqueue(encoder.encode("data: " + JSON.stringify(event) + "\n\n"));
    emit({event:"chunks", chunks:[{chunk_id:"c1", source_file:"fixture.pdf", page_number:1,
          similarity_score:0.8, text:"Fixture evidence"}], retrieval_time:0.01});
    emit({event:"token", text:"Fixture answer"});
    if (mode === "hold") {
      init.signal?.addEventListener("abort", () => controller.error(new DOMException("Aborted", "AbortError")));
      return;
    }
    if (mode !== "incomplete") emit({event:"done", trace_id:"fixture", total_time:0.1});
    controller.close();
  }});
  return new Response(stream, {headers:{"Content-Type":"text/event-stream"}});
};
"""

@unittest.skipUnless(os.environ.get("XRAG_E2E") == "1", "Set XRAG_E2E=1 with frontend running")
class TestUIEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.page = self.browser.new_page()
        self.page.add_init_script(FETCH_SCRIPT)
        self.page.route("http://127.0.0.1:8010/**", self.fixture)
        self.addCleanup(self.page.close)

    def fixture(self, route):
        path = route.request.url.split("8010")[-1].split("?")[0]
        if path == "/ui/config": body = CONFIG
        elif path == "/ui/sessions" and route.request.method == "POST": body = {"session_id":"session_new", "title":"New Session"}
        elif path == "/ui/sessions": body = SESSIONS
        elif path.endswith("/messages"): body = []
        elif path == "/ui/memory": body = {"memories":[], "searched":False}
        elif path == "/ui/graph": body = {"nodes":[{"id":"a", "type":"document", "label":"A", "degree":1},
                                                       {"id":"b", "type":"chunk", "label":"B", "degree":1}],
                                         "edges":[{"source":"a", "target":"b", "kind":"contains"}], "truncated":False}
        elif path.startswith("/ui/trace"):
            route.fulfill(status=404, json={"detail":"No trace"}); return
        else: body = {}
        route.fulfill(json=body)

    def open_chat(self):
        self.page.goto("http://127.0.0.1:5173")
        self.page.get_by_role("button", name="Chat", exact=True).click()
        self.page.wait_for_selector(".input-textarea")
        self.page.wait_for_function("document.querySelector('.topbar-select')?.options.length > 0")

    def send(self, question="Fixture question"):
        self.page.get_by_role("textbox", name="Question", exact=True).fill(question)
        self.page.get_by_role("button", name="Send message", exact=True).click()

    def test_answer_and_evidence_render(self):
        self.open_chat(); self.send()
        self.page.wait_for_function("document.querySelector('.msg-row.assistant')?.textContent.includes('Fixture answer')")
        self.page.get_by_role("button", name="Details").click()
        self.assertIn("Fixture evidence", self.page.locator(".chunk-card").inner_text())
        self.assertEqual(self.page.evaluate("window.__chatRequests.length"), 1)

    def test_session_switch_cancels_previous_turn(self):
        self.open_chat(); self.page.evaluate("window.__chatMode='hold'"); self.send()
        self.page.wait_for_function("window.__chatRequests.length===1")
        self.page.get_by_role("button", name="Second session").click()
        self.page.wait_for_function("document.querySelectorAll('.msg-row').length===0")
        self.assertFalse(self.page.get_by_role("textbox", name="Question", exact=True).is_disabled())

    def test_double_submission_is_guarded(self):
        self.open_chat(); self.page.evaluate("window.__chatMode='hold'")
        self.page.get_by_role("textbox", name="Question", exact=True).fill("One question")
        self.page.locator(".send-btn").evaluate("button => { button.click(); button.click(); }")
        self.page.wait_for_function("window.__chatRequests.length===1")
        self.assertEqual(self.page.evaluate("window.__chatRequests.length"), 1)

    def test_incomplete_stream_is_retryable_and_history_is_clean(self):
        self.open_chat(); self.page.evaluate("window.__chatMode='incomplete'"); self.send()
        self.page.wait_for_selector(".retry-btn")
        self.page.evaluate("window.__chatMode='done'")
        self.page.get_by_role("button", name="Retry").click()
        self.page.wait_for_function("window.__chatRequests.length===2")
        self.assertEqual(self.page.evaluate("window.__chatRequests[1].chat_history"), [])

    def test_graph_is_reachable_during_another_page_turn(self):
        self.open_chat(); self.page.evaluate("window.__chatMode='hold'"); self.send()
        self.page.get_by_role("button", name="Knowledge Graph", exact=True).click()
        self.page.wait_for_function("window.__graphNodeCount===2")

    def test_configuration_reports_device_and_compatible_arms(self):
        self.open_chat()
        self.assertIn("CPU", self.page.locator(".sidebar-footer").inner_text())
        self.assertNotIn("F_graphrag", self.page.locator(".topbar-select").first.inner_text())
        self.page.locator(".topbar-select").nth(1).select_option("judgments")
        self.assertIn("F_graphrag", self.page.locator(".topbar-select").first.inner_text())

    def test_memory_http_error_is_visible(self):
        self.page.unroute("http://127.0.0.1:8010/**")
        def routes(route):
            if "/ui/memory" in route.request.url: route.fulfill(status=500, json={})
            else: self.fixture(route)
        self.page.route("http://127.0.0.1:8010/**", routes)
        self.page.goto("http://127.0.0.1:5173")
        self.page.get_by_role("button", name="Memory", exact=True).click()
        self.page.wait_for_selector('[role="alert"]')
        self.assertIn("Unable to load memories", self.page.get_by_role("alert").inner_text())
