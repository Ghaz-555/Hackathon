"""Stage 3 browser regression. Start scripts/serve.py first.

Uses software WebGL so the real Three.js scene is exercised in headless Chromium.
Install Playwright + Chromium in the invoking Python environment. All requests
are confined to the local application; no external assets or APIs are needed.
"""
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts'
BASE = os.environ.get('GLASSWORK_URL', 'http://127.0.0.1:8001')
checks = []

def check(name):
    checks.append(name)
    print('PASS', name, flush=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
    page = browser.new_page(viewport={'width': 1600, 'height': 1050})
    page.add_init_script("sessionStorage.setItem('glasswork-engine','team')")
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.route('**/*', lambda r: r.continue_() if r.request.url.startswith(BASE) else r.abort())
    page.goto(BASE, wait_until='networkidle')
    expect(page.get_by_role('button', name='Inspect prompt', exact=True)).to_be_enabled()
    expect(page.locator('canvas')).to_be_visible()
    page.get_by_role('button', name='Architecture', exact=True).click()
    expect(page.locator('.world-label')).to_have_count(6)
    page.wait_for_timeout(1000)
    check('Real WebGL scene and all six world labels')

    def state():
        return page.evaluate("async () => (await fetch('/api/sessions/'+sessionStorage.getItem('glasswork-session'))).json()")

    def stage(name):
        page.get_by_role('navigation', name='Model stages').get_by_role('button', name=name).click()
        expect(page.locator('.inspector h2')).to_have_text(name)

    baseline = state()
    assert (baseline['parameter_count'], baseline['trainable_parameter_count']) == (640, 40)
    assert baseline['engine_id'] == 'team-glassbox-v1'
    expect(page.get_by_label('Selected character')).to_have_value('6')
    expect(page.locator('.vector code')).to_have_count(4)
    page.screenshot(path=str(OUT/'expanded-original-model-desktop.png'), full_page=True)
    check('Correct active engine and selected final character')

    for name in ['Position', 'Q · K · V', 'Attention', 'Projection', 'Probabilities', 'Embedding']:
        stage(name)
    page.get_by_label('Selected character').select_option('0')
    stage('Attention')
    expect(page.locator('.attention-list > div')).to_have_count(1)
    expect(page.locator('.attention-list code')).to_have_text('100.00%')
    page.get_by_label('Selected character').select_option('6')
    expect(page.locator('.attention-list > div')).to_have_count(7)
    check('Stage navigation, character selection and causal attention')

    # Exercise camera input on the canvas, away from floating DOM controls.
    canvas = page.locator('canvas')
    bounds = canvas.bounding_box()
    x, y = bounds['x'] + bounds['width']*.5, bounds['y'] + bounds['height']*.6
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x+90, y+30, steps=10)
    page.mouse.up()
    page.mouse.wheel(0, -150)
    page.get_by_role('button', name='Focus selected stage', exact=True).click()
    expect(page.get_by_role('button', name='Focus selected stage', exact=True)).to_have_attribute('aria-pressed', 'true')
    page.wait_for_timeout(400)
    page.screenshot(path=str(OUT/'expanded-original-model-attention-focus.png'), full_page=True)
    page.get_by_role('button', name='Reset camera', exact=True).click()
    expect(page.get_by_role('button', name='Focus selected stage', exact=True)).to_have_attribute('aria-pressed', 'false')
    check('Orbit, zoom, focus and reset camera controls')

    stage('Embedding')
    page.get_by_role('button', name='Play walkthrough', exact=True).click()
    expect(page.locator('.inspector h2')).to_have_text('Position', timeout=10000)
    page.get_by_role('button', name='Pause walkthrough', exact=True).click()
    page.get_by_role('button', name='Architecture', exact=True).click()
    page.get_by_role('button', name='Reduce motion', exact=True).click()
    expect(page.get_by_role('button', name='Reduce motion', exact=True)).to_have_attribute('aria-pressed', 'true')
    check('Animated walkthrough, pause and reduced motion')

    page.get_by_role('button', name='Experiment', exact=True).click()
    page.get_by_label('Temperature', exact=False).fill('0')
    expect(page.get_by_role('button', name='Generate 64 characters')).to_be_enabled()
    expect(page.locator('.probability-list code').first).to_have_text('100.00%')
    assert page.locator('.probability-list code').all_text_contents()[1:] == ['0.00%']*9
    assert state()['fingerprint'] == baseline['fingerprint']
    page.get_by_label('Temperature', exact=False).fill('1.8')
    expect(page.get_by_role('button', name='Generate 64 characters')).to_be_enabled()
    check('Temperature preserves weights and greedy probabilities are exact')
    page.get_by_role('button', name='Generate 64 characters').click()
    expect(page.get_by_role('button', name='Generate 64 characters')).to_be_enabled()
    sample = page.locator('.sample-box pre').inner_text()
    assert len(sample) >= 64
    page.get_by_role('button', name='Generate 64 characters').click()
    expect(page.get_by_role('button', name='Generate 64 characters')).to_be_enabled()
    assert page.locator('.sample-box pre').inner_text() == sample
    check('Repeatable seeded generation')

    page.get_by_role('button', name='Learning lab', exact=True).click()
    page.get_by_role('button', name='Random weights', exact=True).click()
    expect(page.locator('.training-progress')).to_contain_text('0 total updates')
    random = state()
    assert random['fingerprint'] != baseline['fingerprint']
    page.get_by_role('button', name='Train 1 step', exact=True).click()
    expect(page.locator('.training-progress')).to_contain_text('1 total updates')
    assert state()['fingerprint'] != random['fingerprint']
    expect(page.locator('.weight-matrix .changed')).to_have_count(40)
    page.get_by_role('button', name='Train 25 steps', exact=True).click()
    expect(page.locator('.training-progress')).to_contain_text('25 steps completed this run', timeout=30000)
    assert state()['step'] == 26
    assert len(state()['history']) == 27
    expect(page.locator('.recharts-surface')).to_be_visible()
    page.screenshot(path=str(OUT/'expanded-original-model-learning.png'), full_page=True)
    check('Random reset, one-step and 25-step training, actual weight deltas, loss chart')

    # Keep a request in flight long enough to deterministically test stopping.
    page.route('**/train', lambda route: (page.wait_for_timeout(150), route.continue_()))
    page.get_by_role('button', name='Train 25 steps', exact=True).click()
    page.get_by_role('button', name='Stop', exact=True).click()
    expect(page.get_by_role('button', name='Train 25 steps', exact=True)).to_be_enabled(timeout=15000)
    assert 26 < state()['step'] < 51
    page.unroute('**/train')
    page.get_by_role('button', name='Restore checkpoint', exact=True).click()
    expect(page.locator('.training-progress')).to_contain_text(f"{baseline['step']} total updates")
    assert state()['fingerprint'] == baseline['fingerprint']
    expect(page.locator('.weight-matrix .changed')).to_have_count(0)
    page.keyboard.press('Escape')
    expect(page.get_by_role('region', name='Experiment controls')).to_have_count(0)
    check('Stop between updates, restore checkpoint and Escape closes drawer')

    prompt = page.get_by_label('Model prompt', exact=True)
    prompt.fill('HELLO')
    page.get_by_role('button', name='Inspect prompt', exact=True).click()
    expect(page.get_by_role('alert')).to_contain_text('Unknown characters')
    prompt.fill('THE CAT ')
    page.get_by_role('button', name='Inspect prompt', exact=True).click()
    expect(page.get_by_role('alert')).to_have_count(0)
    expect(prompt).to_have_value('the cat ')
    expect(page.get_by_label('Selected character').locator('option')).to_have_count(8)
    prompt.fill('the cat '*16)
    page.get_by_role('button', name='Inspect prompt', exact=True).click()
    expect(page.get_by_label('Selected character').locator('option')).to_have_count(128)
    stage('Attention')
    expect(page.locator('.attention-list > div')).to_have_count(128)
    page.get_by_label('Selected character').select_option('0')
    expect(page.locator('.attention-list > div')).to_have_count(1)
    check('Input validation, uppercase normalization and full 128-character trace')

    prompt.fill('the cat')
    page.get_by_role('button', name='Inspect prompt', exact=True).click()
    expect(page.get_by_label('Selected character').locator('option')).to_have_count(7)
    page.get_by_role('button', name='Diagram view', exact=True).click()
    expect(page.locator('.diagram-stations button')).to_have_count(6)
    page.locator('.diagram-stations').get_by_role('button', name='Attention').click()
    expect(page.locator('.inspector h2')).to_have_text('Attention')
    page.get_by_role('button', name='3D view', exact=True).click()
    expect(page.locator('canvas')).to_be_visible()
    page.get_by_role('button', name='Architecture', exact=True).click()
    expect(page.locator('.world-label')).to_have_count(6)
    # A lost GPU context must preserve the lesson via its diagram fallback.
    page.locator('canvas').evaluate("canvas => canvas.dispatchEvent(new Event('webglcontextlost', {cancelable:true}))")
    expect(page.locator('.diagram-stations button')).to_have_count(6)
    check('Accessible diagram and automatic context-loss fallback')

    page.get_by_role('button', name='3D view', exact=True).click()
    page.set_viewport_size({'width':390,'height':844})
    page.evaluate('window.scrollTo(0,0)')
    page.wait_for_timeout(600)
    page.screenshot(path=str(OUT/'expanded-original-model-mobile.png'), full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile horizontal overflow'
    page.get_by_role('button', name='Experiment', exact=True).click()
    expect(page.get_by_role('button', name='Generate 64 characters')).to_be_visible()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    check('Mobile 3D, inspector and experiment controls without horizontal overflow')
    assert not errors, errors
    check('No uncaught browser exceptions, external requests blocked')
    result = {'browser':'Chromium with software WebGL', 'result':'PASS', 'checks':checks, 'page_errors':errors}
    (OUT/'expanded-original-model-browser-results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    page.evaluate("async () => fetch('/api/sessions/'+sessionStorage.getItem('glasswork-session'), {method:'DELETE'})")
    browser.close()
