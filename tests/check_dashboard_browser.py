"""Optional real-browser smoke check; install playwright in a test environment.

Serve build/ccl-preview-demo on localhost:8081 before running this script.
Uses the locally installed Google Chrome rather than downloading Chromium.
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--screenshot", type=Path, default=Path("build/ccl-dashboard-demo.png"))
    args = parser.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function("typeof BENCHMARK_DATA !== 'undefined' && document.querySelectorAll('tbody tr').length > 0")
        text = page.inner_text("body")
        assert "DEMO — shortened request arrays" in text, text[:500]
        assert "DEMO-C1-gpu-c16-r0" in text, "demo row missing"
        assert "Client" in text, "client metric labels missing"
        assert "N/A" in text, "unavailable demo metrics should render as N/A"
        assert page.locator(".header-logo").evaluate("img => img.complete && img.naturalWidth > 0"), "logo did not load"
        assert not errors, errors
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(args.screenshot), full_page=True)
        print(f"Browser smoke check passed; screenshot: {args.screenshot}")
        browser.close()


if __name__ == "__main__":
    main()
