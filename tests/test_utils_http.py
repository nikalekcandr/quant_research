"""Http: a 429 pauses every thread (not just the unlucky one), widens the gap, and later relaxes."""
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.utils import Http

STATE = {"mode": "429_once", "hits": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        STATE["hits"] += 1
        if self.path.startswith("/missing"):
            self.send_response(404)
            self.end_headers()
            return
        if STATE["mode"] == "429_once" and STATE["hits"] == 1:
            self.send_response(429)
            self.send_header("Retry-After", "1")
            self.end_headers()
            return
        if STATE["mode"] == "429_always":
            self.send_response(429)
            self.end_headers()
            return
        body = json.dumps({"ok": True, "hits": STATE["hits"]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class HttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}/x"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        STATE.update(mode="429_once", hits=0)

    def test_429_is_retried_after_the_cooldown_and_widens_the_gap(self):
        http = Http(min_interval=0.05, retries=3)
        t0 = time.time()
        out = http.get(self.url)
        self.assertTrue(out["ok"])
        self.assertGreaterEqual(time.time() - t0, 1.0)      # it honoured Retry-After: 1
        self.assertEqual(http.stats["429"], 1)
        self.assertGreater(http.min_interval, http.base_interval)

    def test_repeated_429_gives_up_with_a_clear_message(self):
        STATE["mode"] = "429_always"
        http = Http(min_interval=0.01, retries=2)
        with self.assertRaises(RuntimeError) as ctx:
            http.get(self.url)
        self.assertIn("429", str(ctx.exception))
        self.assertEqual(http.stats["429"], 2)

    def test_gap_relaxes_after_a_clean_streak(self):
        http = Http(min_interval=0.2, retries=2, recover_after=1)
        http._penalise(None)
        widened = http.min_interval
        self.assertGreater(widened, 0.2)
        STATE["mode"] = "ok"
        http.get(self.url)
        self.assertLess(http.min_interval, widened)
        self.assertGreaterEqual(http.min_interval, 0.2)     # never faster than asked

    def test_404_is_none_not_an_error(self):
        http = Http(min_interval=0.0, retries=1)
        self.assertIsNone(http.get(self.url.replace("/x", "/missing")))
        self.assertEqual(http.stats["429"], 0)


if __name__ == "__main__":
    unittest.main()
