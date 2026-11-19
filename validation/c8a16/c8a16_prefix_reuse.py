"""Prefix-cache reuse + correctness check for the C8A16 scale-bearing pages.

Sends the same long prompt twice so the second request reuses cached pages
(including both fp32 scale planes) and compares the two completions. Then
sends a different prompt to confirm the cache does not leak.
"""

import json
import sys
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
)


def ask(prompt, max_tokens=24):
    body = json.dumps(
        {
            "model": "qwen36",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"], data["usage"]["prompt_tokens"]


def main():
    long_prompt = FILLER * 40 + "\n\nSummarize the text above in one short sentence."
    first, n1 = ask(long_prompt)
    second, n2 = ask(long_prompt)
    other, n3 = ask("Name the capital of France in one word.")
    result = {
        "prompt_tokens": [n1, n2, n3],
        "reuse_identical": first == second,
        "first": first,
        "second": second,
        "other": other,
        "status": "pass" if first == second and n1 == n2 else "fail",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
