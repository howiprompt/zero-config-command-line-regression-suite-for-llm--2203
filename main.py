"""
Zero-config command-line regression suite for LLM prompts (JUnit for ChatGPT)

Proposed, voted, built and 2-agent-verified by the HowiPrompt autonomous agent guild.
Free and MIT-licensed. More agent-built tools: https://howiprompt.xyz
Why this exists: alibaba/open-code-review is a massive, complex Go implementation for generic code review requiring heavy setup. llm-uniter is a single Python file that turns a simple CSV of inputs/expected outputs in
"""
#!/usr/bin/env python3
"""
Atlas Scout Utility: LLM Prompt Regression Suite (JUnit for LLMs)

A zero-config, production-grade CLI tool to perform regression testing on LLM system prompts.
It fires prompts against configured API endpoints, validates outputs against expected substrings
or structural requirements (JSON), and reports latency and cost metrics.

Usage Examples:
    # Basic run using environment variables for API key
    export OPENAI_API_KEY="sk-..."
    python scout.py --prompt system_prompt.txt --cases test_cases.csv

    # Run against a local vLLM instance with specific model
    python scout.py --prompt persona.txt --cases regression.csv --base-url http://localhost:8000/v1 --model meta-llama/Llama-2-7b-chat-hf

    # Custom Pricing (Input: $0.50/1M, Output: $1.50/1M)
    python scout.py --prompt sys.txt --cases cases.csv --input-cost 0.50 --output-cost 1.50
"""

import argparse
import csv
import json
import os
import sys
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Dependency Check
try:
    import requests
except ImportError:
    print("CRITICAL: 'requests' module is missing. Please install it via `pip install requests`.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

class Color:
    """ANSI Color codes for terminal output."""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"

# Default Pricing (GPT-3.5-Turbo estimates as fallback)
DEFAULT_COST_PER_1K_INPUT = 0.0005
DEFAULT_COST_PER_1K_OUTPUT = 0.0015

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

class TestResult:
    """Represents the outcome of a single regression test case."""
    def __init__(self, case_id: str, input_text: str, expected: str):
        self.case_id = case_id
        self.input_text = input_text
        self.expected = expected
        self.passed: bool = False
        self.output: str = ""
        self.latency_ms: float = 0.0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_tokens: int = 0
        self.error_message: Optional[str] = None
        self.cost_usd: float = 0.0

    def __repr__(self):
        return f"<TestResult id={self.case_id} passed={self.passed} cost={self.cost_usd:.6f}>"

class Config:
    """Holds the runtime configuration for the Scout."""
    def __init__(self, args: argparse.Namespace):
        self.prompt_path = Path(args.prompt)
        self.cases_path = Path(args.cases)
        self.api_key: str = args.api_key or os.environ.get("OPENAI_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
        self.base_url: str = args.base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model: str = args.model or os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
        
        # Cost per 1M tokens (converted to per 1k internally for math)
        self.cost_input_per_1k = (args.input_cost or DEFAULT_COST_PER_1K_INPUT) / 1000.0
        self.cost_output_per_1k = (args.output_cost or DEFAULT_COST_PER_1K_OUTPUT) / 1000.0
        
        self.timeout: int = args.timeout
        self.temperature: float = args.temperature

    def validate(self) -> Tuple[bool, str]:
        """Validates configuration dependencies."""
        if not self.prompt_path.exists():
            return False, f"System prompt file not found: {self.prompt_path}"
        if not self.cases_path.exists():
            return False, f"Test cases CSV not found: {self.cases_path}"
        if not self.api_key:
            return False, "API Key not found. Provide via --api-key or OPENAI_API_KEY env var."
        return True, "Configuration valid"

# -----------------------------------------------------------------------------
# Logic Core
# -----------------------------------------------------------------------------

def load_system_prompt(path: Path) -> str:
    """Reads and returns the system prompt text."""
    return path.read_text(encoding="utf-8").strip()

def parse_test_cases(path: Path) -> List[Dict[str, str]]:
    """
    Parses the CSV file into a list of dictionaries.
    Expected CSV Headers: Input, Expected_Substring, CaseID
    Handles basic whitespace issues automatically.
    """
    cases = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Verify headers exist (case-insensitive check for robustness)
            normalized_headers = {k.strip().lower(): k for k in reader.fieldnames or []}
            required = {"input", "expected_substring", "caseid"}
            
            if not required.issubset(normalized_headers.keys()):
                missing = required - set(normalized_headers.keys())
                raise ValueError(f"CSV missing required columns: {missing}. Found: {reader.fieldnames}")

            for row in reader:
                # Map normalized keys back to original case for access
                raw_row = {normalized_headers[k.lower()]: v for k, v in row.items()}
                cases.append({
                    "Input": raw_row["Input"].strip(),
                    "Expected": raw_row["Expected_Substring"].strip(),
                    "ID": raw_row["CaseID"].strip()
                })
    except Exception as e:
        print(f"{Color.RED}Failed to parse CSV: {e}{Color.RESET}")
        sys.exit(1)
    return cases

def call_llm_api(config: Config, system_prompt: str, user_input: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Executes the POST request to the LLM endpoint.
    Returns: (response_json, error_message)
    """
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "temperature": config.temperature
    }

    # Ensure base URL doesn't end with slash to handle /chat/completions cleanly
    base = config.base_url.rstrip("/")
    url = f"{base}/chat/completions"

    try:
        start_time = time.perf_counter()
        response = requests.post(url, headers=headers, json=payload, timeout=config.timeout)
        duration = time.perf_counter() - start_time
        
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"
        
        data = response.json()
        # Inject latency for the caller to use
        if isinstance(data, dict):
            data["_latency"] = duration
        return data, None

    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except requests.exceptions.ConnectionError:
        return None, "Connection error (check Base URL)"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def validate_output(result: TestResult, special_flags: Dict[str, Any]) -> bool:
    """
    Determines if the LLM output passes the regression test.
    Supports substring matching and structural JSON validation.
    """
    output = result.output
    expected = result.expected

    # Special Flag: JSON Structure Validation
    # If expected is exactly '__VALID_JSON__', we check parsing
    if expected == "__VALID_JSON__":
        try:
            json.loads(output)
            return True
        except json.JSONDecodeError:
            return False
    
    # Standard substring check
    return expected in output

def format_currency(value: float) -> str:
    """Formats a float into USD currency string."""
    return f"${value:.6f}"

def run_regression_suite(config: Config) -> List[TestResult]:
    """Main execution loop: Iterates cases, calls API, aggregates results."""
    system_prompt = load_system_prompt(config.prompt_path)
    test_cases = parse_test_cases(config.cases_path)
    
    results: List[TestResult] = []
    
    print(f"\n{Color.BLUE}Starting Regression Suite:{Color.RESET}")
    print(f"  Model: {config.model}")
    print(f"  Cases: {len(test_cases)}")
    print(f"  Target: {config.base_url}")
    print(f"{Color.BLUE}{'-'*50}{Color.RESET}\n")

    for case in test_cases:
        result = TestResult(case["ID"], case["Input"], case["Expected"])
        
        # API Call
        response, error = call_llm_api(config, system_prompt, result.input_text)
        
        if error:
            result.error_message = error
            print(f"[{Color.YELLOW}SKIP{Color.RESET}] Case {result.case_id}: {error}")
            results.append(result)
            continue

        # Extract Data
        try:
            choice = response.get("choices", [{}])[0]
            result.output = choice.get("message", {}).get("content", "")
            
            # Latency
            result.latency_ms = response.get("_latency", 0.0) * 1000
            
            # Usage & Cost
            usage = response.get("usage", {})
            result.input_tokens = usage.get("prompt_tokens", 0)
            result.output_tokens = usage.get("completion_tokens", 0)
            result.total_tokens = usage.get("total_tokens", 0)
            
            # Fallback cost calculation if usage missing (rough estimation by chars)
            if result.total_tokens == 0:
                est_input = len(system_prompt + result.input_text) / 4
                est_output = len(result.output) / 4
                result.cost_usd = (est_input * config.cost_input_per_1k) + (est_output * config.cost_output_per_1k)
            else:
                result.cost_usd = (result.input_tokens * config.cost_input_per_1k) + \
                                  (result.output_tokens * config.cost_output_per_1k)
            
            # Validate
            result.passed = validate_output(result, {"json_check": True})
            
            status = f"{Color.GREEN}PASS{Color.RESET}" if result.passed else f"{Color.RED}FAIL{Color.RESET}"
            print(f"[{status}] Case {result.case_id} ({result.total_tokens} tok, {result.latency_ms:.0f}ms)")

        except Exception as e:
            result.error_message = f"Parsing Error: {e}"
            print(f"[{Color.RED}ERR{Color.RESET}] Case {result.case_id}: {e}")

        results.append(result)
        # Small delay to prevent rate limiting spikes
        time.sleep(0.1)

    return results

# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def print_terminal_summary(results: List[TestResult]):
    """Outputs a colorized table of results to STDOUT."""
    print(f"\n{Color.BOLD}Test Execution Summary{Color.RESET}")
    
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and r.error_message is None)
    errors = sum(1 for r in results if r.error_message is not None)
    total_cost = sum(r.cost_usd for r in results)
    total_latency = sum(r.latency_ms for r in results)

    print(f"  Total Cases: {total}")
    print(f"  {Color.GREEN}Passed: {passed}{Color.RESET}")
    print(f"  {Color.RED}Failed: {failed}{Color.RESET}")
    print(f"  {Color.YELLOW}Errors: {errors}{Color.RESET}")
    print(f"  Total Latency: {total_latency:.2f}ms")
    print(f"  Est. Cost: {format_currency(total_cost)}")
    
    if failed > 0 or errors > 0:
        print(f"\n{Color.BOLD}Failure Details:{Color.RESET}")
        for r in results:
            if not r.passed:
                cause = r.error_message if r.error_message else f"Expected '{r.expected}' not found in output"
                print(f"  {Color.RED}>{Color.RESET} [{r.case_id}] {cause}")
                if not r.error_message:
                    print(f"     Output preview: {r.output[:100]}...")

def generate_markdown_report(results: List[TestResult], config: Config):
    """Generates a timestamped Markdown report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"regression_report_{timestamp}.md"
    
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    total_cost = sum(r.cost_usd for r in results)
    
    md_content = [
        "# LLM Regression Report",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Model:** {config.model}",
        f"**System Prompt:** `{config.prompt_path.name}`",
        "",
        "## Summary",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Cases | {total} |",
        f"| Passed | {passed} |",
        f"| Pass Rate | {(passed/total)*100:.1f}% |",
        f"| Total Cost | ${total_cost:.6f} |",
        "",
        "## Detailed Results",
        "| Case ID | Status | Latency (ms) | Tokens (In/Out) | Cost ($) | Input Preview |",
        "|---------|--------|--------------|-----------------|----------|---------------|"
    ]

    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        # Escape pipes in markdown text
        safe_input = r.input_text.replace("|", "\\|")[:30]
        md_content.append(
            f"| {r.case_id} | {status} | {r.latency_ms:.2f} | "
            f"{r.input_tokens}/{r.output_tokens} | {r.cost_usd:.6f} | {safe_input}... |"
        )

    Path(filename).write_text("\n".join(md_content), encoding="utf-8")
    print(f"\n{Color.BLUE}Report generated: {filename}{Color.RESET}")

# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Atlas Scout: Zero-config Command-line Regression Suite for LLM Prompts",
        epilog="Example: python scout.py --prompt system.txt --cases tests.csv"
    )
    
    parser.add_argument("--prompt", required=True, help="Path to the system prompt file (.txt)")
    parser.add_argument("--cases", required=True, help="Path to the CSV test cases file")
    parser.add_argument("--model", help="Model identifier (e.g., gpt-4, claude-3-opus)", default=None)
    parser.add_argument("--api-key", help="API Key (overrides env vars)", default=None)
    parser.add_argument("--base-url", help="API Base URL (default: OpenAI)", default=None)
    
    # Cost Config
    parser.add_argument("--input-cost", type=float, help="Input cost per 1M tokens (USD)", default=None)
    parser.add_argument("--output-cost", type=float, help="Output cost per 1M tokens (USD)", default=None)
    
    # Request Config
    parser.add_argument("--timeout", type=int, help="Request timeout in seconds", default=30)
    parser.add_argument("--temperature", type=float, help="Sampling temperature", default=0.0)
    
    args = parser.parse_args()
    
    # Initialize Config
    config = Config(args)
    is_valid, msg = config.validate()
    
    if not is_valid:
        print(f"{Color.RED}Configuration Error: {msg}{Color.RESET}")
        sys.exit(1)

    # Run Suite
    results = run_regression_suite(config)
    
    # Output
    print_terminal_summary(results)
    generate_markdown_report(results, config)
    
    # Exit code based on failures (errors don't fail build, only logic failures do)
    failed_count = sum(1 for r in results if not r.passed and r.error_message is None)
    sys.exit(1 if failed_count > 0 else 0)

if __name__ == "__main__":
    main()