"""Offline HTTP contract and Agent integration tests for web search."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

from corecoder.agent import Agent
from corecoder.llm import LLMResponse, ToolCall
from corecoder.permissions import Permission
from corecoder.tools import ALL_TOOLS
from corecoder.tools import searchTool as search_module
from corecoder.tools.searchTool import WebSearchTool


class WebSearchTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"TAVILY_API_KEY": "", "SERPAPI_API_KEY": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.dotenv = patch.object(search_module, "_load_dotenv")
        self.dotenv.start()
        self.addCleanup(self.dotenv.stop)
        self.requests = []
        self.responses = []
        self.opener = patch.object(search_module, "build_opener")
        opener = self.opener.start()
        self.addCleanup(self.opener.stop)
        opener.return_value.open.side_effect = self.respond

    def respond(self, request, timeout):
        self.assertEqual(timeout, 20)
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, bytes):
            return io.BytesIO(response)
        return io.BytesIO(json.dumps(response).encode())

    def test_tavily_request_and_result(self):
        self.responses = [{"results": [
            {"title": "Python", "url": "https://docs.python.org/3/", "content": "Documentation"},
            {"title": "Duplicate", "url": "https://docs.python.org/3/"},
            {"title": "Invalid", "url": "javascript:alert(1)"},
        ]}]
        result = WebSearchTool(tavily_key="test-key").execute("  Python 文档  ")
        request = self.requests[0]
        self.assertEqual(request.full_url, "https://api.tavily.com/search")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(request.data)["query"], "Python 文档")
        self.assertEqual(result.count("https://docs.python.org/3/"), 1)
        self.assertIn("Documentation", result)
        self.assertNotIn("javascript", result)

    def test_serpapi_request_and_limits(self):
        self.responses = [{"organic_results": [
            {"title": "A", "link": "https://example.com/a", "snippet": "x" * 2000},
            {"title": "B", "link": "https://example.com/b"},
        ]}]
        result = WebSearchTool(serpapi_key="test-secret").execute("Python 中文 & API", 1, "serpapi")
        params = parse_qs(urlsplit(self.requests[0].full_url).query)
        self.assertEqual(params["engine"], ["google"])
        self.assertEqual(params["q"], ["Python 中文 & API"])
        self.assertEqual(params["api_key"], ["test-secret"])
        self.assertIn("x" * 1000, result)
        self.assertNotIn("x" * 1001, result)
        self.assertNotIn("https://example.com/b", result)
        self.assertNotIn("test-secret", result)

    def test_missing_configuration_is_silent(self):
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            tool = WebSearchTool()
            result = tool.execute("Python")
        self.assertEqual(out.getvalue(), "")
        self.assertIn("TAVILY_API_KEY", result)
        self.assertEqual(self.requests, [])

    def test_lazy_environment_and_explicit_precedence(self):
        tool = WebSearchTool(tavily_key="explicit")
        os.environ["TAVILY_API_KEY"] = "later-env-key"
        os.environ["SERPAPI_API_KEY"] = "serp-env"
        self.assertEqual(tool._setup_backends(), {"tavily": "explicit", "serpapi": "serp-env"})

    def test_real_dotenv_after_construction(self):
        from corecoder.config import _load_dotenv

        tool = WebSearchTool()
        # Keep all temporary files on the repository drive.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent) as directory:
            Path(directory, ".env").write_text("TAVILY_API_KEY=dotenv-test\n", encoding="utf-8")
            os.environ.pop("TAVILY_API_KEY", None)
            previous = os.getcwd()
            try:
                os.chdir(directory)
                with patch.object(search_module, "_load_dotenv", _load_dotenv):
                    self.assertEqual(tool._setup_backends(), {"tavily": "dotenv-test"})
            finally:
                os.chdir(previous)

    def test_hybrid_falls_back_on_http_error(self):
        self.responses = [
            HTTPError("https://example.com/?api_key=secret", 429, "secret", {}, None),
            {"organic_results": [{"title": "OK", "link": "https://example.com/"}]},
        ]
        result = WebSearchTool(tavily_key="a", serpapi_key="b").execute("Python")
        self.assertIn("HTTP 429", result)
        self.assertIn("(serpapi)", result)
        self.assertNotIn("secret", result)

    def test_hybrid_falls_back_on_empty_results(self):
        self.responses = [
            {"results": []},
            {"organic_results": [{"link": "https://example.com/"}]},
        ]
        result = WebSearchTool(tavily_key="a", serpapi_key="b").execute("Python")
        self.assertIn("tavily: no results", result)
        self.assertIn("(serpapi)", result)

    def test_hybrid_stops_after_success(self):
        self.responses = [{"results": [{"url": "https://example.com/"}]}]
        WebSearchTool(tavily_key="a", serpapi_key="b").execute("Python")
        self.assertEqual(len(self.requests), 1)

    def test_explicit_provider_never_falls_back(self):
        result = WebSearchTool(serpapi_key="b").execute("Python", backend="tavily")
        self.assertIn("TAVILY_API_KEY", result)
        self.assertEqual(self.requests, [])

    def test_validation_before_network(self):
        for kwargs in ({"query": " "}, {"query": "x" * 401}, {"query": 5},
                       {"query": "x", "max_results": True}, {"query": "x", "max_results": 11},
                       {"query": "x", "backend": "unknown"}):
            with self.subTest(kwargs=kwargs):
                self.assertTrue(WebSearchTool().execute(**kwargs).startswith("Error:"))
        self.assertEqual(self.requests, [])

    def test_errors_are_text_and_redacted(self):
        for response in (TimeoutError("secret"), URLError("secret"), b"not json",
                         {"error": "secret"}, [], {"results": None}, b"x" * (2 * 1024 * 1024 + 1)):
            with self.subTest(response_type=type(response).__name__):
                self.responses = [response]
                result = WebSearchTool(tavily_key="key").execute("Python", backend="tavily")
                self.assertTrue(result.startswith("Error:"))
                self.assertNotIn("secret", result)

    def test_registry_and_agent_round_trip_in_plan_mode(self):
        self.assertEqual(sum(tool.name == "web_search" for tool in ALL_TOOLS), 1)
        tool = WebSearchTool(tavily_key="key")
        self.assertIsNone(Permission().check("web_search", {"query": "Python"}))
        self.responses = [{"results": [{"url": "https://docs.python.org/3/", "content": "Python docs"}]}]

        class ScriptedLLM:
            def chat(self, messages, tools=None, **kwargs):
                if messages[-1]["role"] == "tool":
                    return LLMResponse(content=messages[-1]["content"])
                return LLMResponse(tool_calls=[ToolCall("search-1", "web_search", {"query": "Python"})])

        agent = Agent(llm=ScriptedLLM(), tools=[tool], permission=Permission())
        agent.plan_mode = True
        self.assertIn("https://docs.python.org/3/", agent.chat("Search Python docs"))
        self.assertEqual(agent.messages[-2]["tool_call_id"], "search-1")


if __name__ == "__main__":
    unittest.main()
