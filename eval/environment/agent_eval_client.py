#!/usr/bin/env python3
"""Minimal eval client the agent uses to probe the landscape server.

The agent calls this script with a config string and depth:
    python agent_eval_client.py "config_string" 2

It prints the JSON response from the landscape server. The agent has no
other way to interact with the evaluation environment — no file reads,
no shell commands. This is the ONLY window into the hidden landscape.
"""

import json
import sys
import urllib.request


def evaluate(config: str, depth: int, url: str = "http://127.0.0.1:8080/evaluate") -> dict:
    """POST to the landscape server and return the parsed response."""
    data = json.dumps({"c": config, "depth": depth}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <config_string> <depth>", file=sys.stderr)
        sys.exit(1)
    result = evaluate(sys.argv[1], int(sys.argv[2]))
    print(json.dumps(result))
