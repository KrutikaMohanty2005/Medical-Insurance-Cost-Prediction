"""
Capture web interface screenshots using Flask test client and pyppeteer.
"""
import subprocess
import time
import sys
import os

print("=" * 60)
print("CAPTURING WEB INTERFACE SCREENSHOTS")
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

# Try to capture using pyppeteer
try:
    import asyncio
    from pyppeteer import launch
    
    async def capture_screenshots():
        browser = await launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        
        page = await browser.newPage()
        await page.setViewport({'width': 1280, 'height': 900})
        
        # Screenshot 1: Home/Form page
        print("\n[2] Capturing form page...")
        await page.goto('http://127.0.0.1:8080', wait_until='networkidle0')
        await asyncio.sleep(1)
        await page.screenshot({'path': 'static/web_interface.png', 'fullPage': True})
        print("  Saved: static/web_interface.png")
        
        # Fill form for prediction screenshot
        print("\n[3] Filling form for prediction...")
        await page.type('#age', '30')
        await page.select('#sex', 'female')
        await page.type('#bmi', '28.5')
        await page.select('#children', '2')
        await page.select('#smoker', 'no')
        await page.click('#north')
        
        # Submit form
        print("[4] Submitting form...")
        await page.click('.predict-btn')
        await asyncio.sleep(2)
        
        # Screenshot 2: Result page
        print("[5] Capturing result page...")
        await page.screenshot({'path': 'static/prediction_result.png', 'fullPage': True})
        print("  Saved: static/prediction_result.png")
        
        await browser.close()
    
    # Run async function
    asyncio.get_event_loop().run_until_complete(capture_screenshots())
    
except Exception as e:
    print(f"\n  Pyppeteer failed: {e}")
    print("  Trying alternative method with Flask test client...")
    
    # Alternative: Use Flask test client
    from app import app
    
    with app.test_client() as client:
        # Home page
        print("\n[2] Capturing form page via test client...")
        response = client.get('/')
        with open('static/web_interface.html', 'wb') as f:
            f.write(response.data)
        print("  Saved: static/web_interface.html")
        
        # Prediction page
        print("[3] Capturing prediction result...")
        response = client.post('/predict', data={
            'age': '30',
            'sex': 'female',
            'bmi': '28.5',
            'children': '2',
            'smoker': 'no',
            'region': 'north'
        })
        with open('static/prediction_result.html', 'wb') as f:
            f.write(response.data)
        print("  Saved: static/prediction_result.html")

# Stop Flask
print("\n[6] Stopping Flask app...")
flask_process.terminate()
flask_process.wait()

print("\n" + "=" * 60)
print("SCREENSHOT CAPTURE COMPLETE!")
print("=" * 60)
