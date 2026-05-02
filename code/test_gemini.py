"""Debug test - see raw Gemini response for a real ticket."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from agent.llm_client import LLMClient
from agent.prompt_templates import build_system_prompt

client = LLMClient()
system_prompt = build_system_prompt("Claude")

messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": (
            "TICKET ID: TEST_001\n"
            "COMPANY: Claude\n"
            "ISSUE: I lost access to my Claude team workspace after our IT admin removed my seat. "
            "Please restore my access immediately.\n\n"
            "CORPUS EXCERPT [1] (source: claude/account.md, company: claude, product_area: account_access):\n"
            "Workspace access is managed by your workspace administrator. "
            "Contact your IT admin to restore access or request a new seat.\n\n"
            "chunks end here"
        ),
    },
]

print("=== SYSTEM PROMPT (first 300 chars) ===")
print(system_prompt[:300])
print("\n=== CALLING GEMINI ===")
result = client._generate_gemini_new(messages=messages)
print("\n=== RAW RESPONSE ===")
print(result)
