"""A separate installed environment can make and record a call without research."""

import json
import subprocess
import sys
import textwrap


def test_client_call_from_an_unrelated_working_directory(tmp_path):
    program = textwrap.dedent("""
        import asyncio
        import importlib.util
        import json
        import sys

        import httpx2
        from thesis_model_clients import TypeSafeHTTPDecisionClient, decision_result_to_dict
        from typesafe_sdk import Noul

        assert importlib.util.find_spec("thesis_research") is None

        async def main():
            requests = []
            def reply(request):
                requests.append(request)
                return httpx2.Response(200, json={
                    "model": "standalone-v1",
                    "usage": {"input_tokens": 9, "output_tokens": 0},
                    "answers": {"q": {"type": "noul", "noul": 0.8}},
                })

            async with TypeSafeHTTPDecisionClient(
                "standalone-v1", "local", base_url="http://localhost:9999",
                _transport=httpx2.MockTransport(reply),
            ) as client:
                state = {"message": "An independent use case"}
                result = await client.evaluate(state, {"q": Noul()})

            assert len(requests) == 1
            assert json.loads(requests[0].content)["state"] == state
            assert "Authorization" not in requests[0].headers
            assert not any(name.startswith("thesis_research") for name in sys.modules)
            print(json.dumps(decision_result_to_dict(
                result, answer_output_mode="native_probabilities",
            )))

        asyncio.run(main())
    """)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    assert record["answers"]["q"]["noul"] == 0.8
    assert record["requested_model"] == "standalone-v1"
    assert record["answer_output_mode"] == "native_probabilities"
    assert record["usage"]["cost_usd"] is None
