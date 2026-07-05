"""
Capture web interface screenshots using Playwright and Flask.
"""
import subprocess
import time
import sys
import os

print("=" * 60)
print("CAPTURING WEB INTERFACE SCREENSHOTS WITH PLAYWRIGHT")
print("=" * 60)

# Start Flask app in background
print("\n[1] Starting Flask app...")
flask_process = subprocess.Popen(
    [sys.executable, 'app.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=os.getcwd()
)

# Wait for Flask to start
time.sleep(4)

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1280, 'height': 900})
    
    # Screenshot 1: Home page
    print("\n[2] Capturing home page...")
    page.goto('http://127.0.0.1:8080', wait_until='networkidle')
    page.wait_for_timeout(1000)
    page.screenshot(path='static/web_interface.png', full_page=True)
    print("  Saved: static/web_interface.png")
    
    # Fill form
    print("\n[3] Filling form for prediction...")
    page.fill('#age', '30')
    page.select_option('#sex', 'female')
    page.fill('#bmi', '28.5')
    page.select_option('#children', '2')
    page.select_option('#smoker', 'no')
    page.click('label[for="north"]')
    page.wait_for_timeout(500)
    
    # Submit
    print("[4] Submitting form...")
    page.click('.predict-btn')
    page.wait_for_timeout(2000)
    
    # Screenshot 2: Result page
    print("[5] Capturing prediction result...")
    page.screenshot(path='static/prediction_result.png', full_page=True)
    print("  Saved: static/prediction_result.png")
    
    browser.close()

# Stop Flask
print("\n[6] Stopping Flask app...")
flask_process.terminate()
flask_process.wait()

print("\n" + "=" * 60)
print("SCREENSHOT CAPTURE COMPLETE!")
print("=" * 60)
