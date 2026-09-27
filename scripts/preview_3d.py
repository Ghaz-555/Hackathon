from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    b=p.chromium.launch(headless=True,args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
    page=b.new_page(viewport={'width':1600,'height':1050})
    page.on('pageerror',lambda e:print('PAGE ERROR',e))
    page.on('console',lambda m: print('CONSOLE',m.text) if m.type=='error' else None)
    page.goto('http://127.0.0.1:8000',wait_until='networkidle')
    page.wait_for_timeout(4000)
    print('canvas',page.locator('canvas').count())
    page.screenshot(path=str(root/'artifacts/stage3-first-look.png'),full_page=True)
    b.close()
