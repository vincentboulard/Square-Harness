"""The configured model endpoint cannot redirect local manuscript traffic."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mathagent.agent import Ollama, AgentError


class ModelNetworkBoundaryTests(unittest.TestCase):
    def test_model_redirect_is_not_followed(self):
        paths = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                paths.append(self.path)
                if self.path == '/api/tags':
                    self.send_response(302)
                    self.send_header('Location', f'http://127.0.0.1:{self.server.server_port}/unapproved')
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps({'models': []}).encode())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with self.assertRaises(AgentError):
                Ollama(f'http://127.0.0.1:{server.server_port}').models()
            self.assertEqual(paths, ['/api/tags'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
