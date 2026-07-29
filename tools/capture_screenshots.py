"""Capture screenshots of the running Streamlit app for the README.

Start the app first, then run this:

    python -m streamlit run app.py          # terminal 1
    python tools/capture_screenshots.py     # terminal 2

Images are written to docs/screenshots/. Regenerate them whenever the UI changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
VIEWPORT = {"width": 1400, "height": 1000}

# Streamlit renders tabs as buttons; each entry is (tab label, button to press,
# output file). A None button means "just show the tab as it loads".
SHOTS = [
    ("Price a flight", "Estimate fare", "01_price_a_flight.png"),
    ("Cheapest options", "Find cheapest options", "02_cheapest_options.png"),
    ("Best travel dates", "Scan dates", "03_best_travel_dates.png"),
    ("Model performance", None, "04_model_performance.png"),
]


def settle(page, timeout: int = 180_000) -> None:
    """Wait until Streamlit stops running a script."""
    page.wait_for_timeout(1200)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached",
                               timeout=timeout)
    except PWTimeout:
        pass
    page.wait_for_timeout(2500)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        page.goto(URL, wait_until="networkidle", timeout=120_000)
        settle(page)

        for tab_label, button_label, filename in SHOTS:
            page.get_by_role("tab", name=tab_label).click()
            page.wait_for_timeout(1200)

            if button_label:
                page.get_by_role("button", name=button_label).click()
                settle(page)

            target = OUT_DIR / filename
            page.screenshot(path=str(target), full_page=True)
            print(f"wrote {target.relative_to(OUT_DIR.parents[1])}")

        browser.close()


if __name__ == "__main__":
    main()
