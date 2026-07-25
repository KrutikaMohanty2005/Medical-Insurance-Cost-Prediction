"""
Capture app screenshots using Playwright.
Updated to match current interface with radio buttons for region.
"""
import subprocess
import time
import sys
import os

print("=" * 60)
print("CAPTURING APP SCREENSHOTS")
print("=" * 60)

# Start Flask app
print("\n[1] Starting Flask app...")
flask_process = subprocess.Popen(
    [sys.executable, 'app.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=os.getcwd()
)
time.sleep(5)

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1280, 'height': 900})

    # Screenshot 1: Flask login page
    print("\n[2] Capturing Flask login page...")
    page.goto('http://127.0.0.1:8080/login', wait_until='networkidle')
    page.wait_for_timeout(2000)
    page.screenshot(path='screenshots/flask_login.png', full_page=True)
    print("  Saved: screenshots/flask_login.png")

    # Register a test user
    print("[3] Registering test user...")
    page.goto('http://127.0.0.1:8080/register', wait_until='networkidle')
    page.wait_for_timeout(1000)
    page.fill('#username', 'testuser')
    page.fill('#email', 'test@example.com')
    page.fill('#password', 'password123')
    page.click('button[type="submit"]')
    page.wait_for_timeout(2000)

    # Login
    print("[4] Logging in...")
    page.goto('http://127.0.0.1:8080/login', wait_until='networkidle')
    page.wait_for_timeout(1000)
    page.fill('#username', 'testuser')
    page.fill('#password', 'password123')
    page.click('button[type="submit"]')
    page.wait_for_timeout(3000)

    # Screenshot 2: Prediction form
    print("[5] Capturing prediction form...")
    page.screenshot(path='screenshots/web_interface.png', full_page=True)
    print("  Saved: screenshots/web_interface.png")

    # Fill and submit prediction
    print("[6] Filling prediction form...")
    page.fill('input[name="age"]', '35')
    page.select_option('select[name="sex"]', 'female')
    page.fill('input[name="bmi"]', '27.5')
    page.select_option('select[name="children"]', '1')
    page.select_option('select[name="smoker"]', 'no')

    # Region - click the label (radio input is hidden by CSS)
    page.click('label[for="region-north"]')

    # Medical history
    page.select_option('select[name="hospitalizations"]', '0')
    page.select_option('select[name="chronic_diseases"]', '0')
    page.select_option('select[name="pre_existing"]', '0')
    page.select_option('select[name="family_history"]', '1')
    page.select_option('select[name="surgeries"]', '0')

    # Lifestyle
    page.select_option('select[name="exercise_freq"]', '3')
    page.select_option('select[name="alcohol"]', '1')
    page.fill('input[name="smoking_years"]', '0')
    page.select_option('select[name="diet_quality"]', '3')
    page.select_option('select[name="stress_level"]', '2')
    page.fill('input[name="sleep_hours"]', '7')
    page.fill('input[name="water_intake"]', '2.5')
    page.select_option('select[name="sugar_intake"]', '1')

    page.click('.predict-btn')
    page.wait_for_timeout(3000)

    # Screenshot 3: Prediction result
    print("[7] Capturing prediction result...")
    page.screenshot(path='screenshots/prediction_result.png', full_page=True)
    print("  Saved: screenshots/prediction_result.png")

    browser.close()

# Stop Flask
print("\n[8] Stopping Flask app...")
flask_process.terminate()
flask_process.wait()

# Capture Streamlit screenshots
print("\n[9] Starting Streamlit app...")
streamlit_process = subprocess.Popen(
    [sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py', '--server.port', '8501', '--server.headless', 'true'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=os.getcwd()
)
time.sleep(10)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1280, 'height': 900})

    # Screenshot 4: Streamlit main page
    print("[10] Capturing Streamlit main page...")
    page.goto('http://localhost:8501', wait_until='networkidle')
    page.wait_for_timeout(5000)
    page.screenshot(path='screenshots/streamlit_main.png', full_page=True)
    print("  Saved: screenshots/streamlit_main.png")

    browser.close()

# Stop Streamlit
print("\n[11] Stopping Streamlit app...")
streamlit_process.terminate()
streamlit_process.wait()

print("\n" + "=" * 60)
print("ALL SCREENSHOTS CAPTURED!")
print("=" * 60)
