"""Upload the two demo images to the running Gradio app and save screenshots of the results."""
import time
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
CASES = [("aat2/demo_alert_k0064.png", "report/figures/aat2_app_alert.png"),
         ("aat2/demo_drowsy_K0037.png", "report/figures/aat2_app_drowsy.png")]

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 860}, device_scale_factor=2)
    for img, out in CASES:
        page.goto(URL)
        page.wait_for_selector("input[type=file]", state="attached", timeout=60000)
        page.set_input_files("input[type=file]", img)
        time.sleep(1.5)
        page.get_by_role("button", name="Submit").click()
        page.wait_for_function("() => [...document.querySelectorAll('textarea')].some(t => t.value.includes('P(closed)'))",
                               timeout=60000)
        time.sleep(1.5)
        page.screenshot(path=out, full_page=True)
        print("saved", out)
    browser.close()
