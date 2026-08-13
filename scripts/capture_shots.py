"""Capture the README screenshot of the deployed VerdictLab dashboard.

- dashboard.png: full-page shot of the default sample pair
  (Sample: same score, more tools) at 1440x3000 viewport — hero, summary
  cards, comparison matrix, and the trajectory report card are all in one
  shot (the trajectory section is redundant as a separate image).

The Streamlit Cloud app is wrapped in an iframe; the outer page's full_page
height tracks the viewport, so a tall viewport captures the app content.
"""
import time

from playwright.sync_api import sync_playwright

URL = "https://verdictlab-wmdbf6rtfxjzh668zugy9d.streamlit.app/"
OUT = "/home/rayyan/projects/verdictlab/assets"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 3000})
        page.goto(URL, wait_until="domcontentloaded", timeout=180_000)
        page.wait_for_timeout(5_000)

        # Streamlit Cloud wraps the app in an iframe — find the frame with our hero.
        frame = None
        deadline = time.time() + 150
        while time.time() < deadline:
            for f in page.frames:
                try:
                    if "The report card for your AI agents" in f.content() or "report card" in f.content():
                        frame = f
                        break
                except Exception:
                    continue
            if frame is not None:
                break
            page.wait_for_timeout(3_000)

        if frame is None:
            print("ERROR: app frame not found after 150s")
            browser.close()
            return

        # Wait for the default sample pair to render fully (header + matrix).
        print("frame found, waiting for content...")
        deadline = time.time() + 90
        while time.time() < deadline:
            txt = frame.inner_text("body")
            if "geo_baseline" in txt and "geo_more_tools" in txt:
                break
            page.wait_for_timeout(3_000)
        page.wait_for_timeout(8_000)  # let tables/plots settle

        txt = frame.inner_text("body")
        print("MARKERS: has geo_baseline:", "geo_baseline" in txt,
              "| has geo_more_tools:", "geo_more_tools" in txt,
              "| has REGRESSED:", "REGRESSED" in txt)

        # Shot: full page (hero + cards + matrix + trajectory section).
        page.screenshot(path=f"{OUT}/dashboard.png", full_page=True)
        print("saved dashboard.png")

        browser.close()


if __name__ == "__main__":
    main()
