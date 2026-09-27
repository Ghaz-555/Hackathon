"""Real-browser integration smoke test. Start scripts/serve.py before running.

Requires Playwright and its Chromium browser in the invoking Python environment.
Artifacts are saved for visual review, including the mobile layout.
"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts'

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width':1440,'height':1100}, device_scale_factor=1)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    # No external services are needed by the built lesson.
    page.route('**/*', lambda route: route.continue_() if route.request.url.startswith('http://127.0.0.1:8000') else route.abort())
    page.goto('http://127.0.0.1:8000', wait_until='networkidle')
    expect(page.get_by_role('button', name='Generate 64 characters')).to_be_enabled()
    expect(page.locator('.prob-row')).to_have_count(8)
    def state():
        return page.evaluate("async () => (await fetch('/api/sessions/'+sessionStorage.getItem('glasswork-session'))).json()")
    baseline = state()
    assert baseline['engine_id'] == 'team-glassbox-v1'
    assert baseline['parameter_count'] == 640
    assert baseline['trainable_parameter_count'] == 40
    assert baseline['config']['context_length'] == 128
    assert len(baseline['characters']) == 10
    expect(page.get_by_label('Attention layer').locator('option')).to_have_count(1)
    expect(page.get_by_label('Attention head').locator('option')).to_have_count(1)
    page.screenshot(path=str(OUT/'team-integration-desktop.png'), full_page=True)
    page.get_by_label('Temperature', exact=True).fill('0')
    expect(page.get_by_role('button', name='Generate 64 characters')).to_be_enabled()
    expect(page.locator('.prob-value').first).to_have_text('100.00%')
    page.wait_for_function("document.querySelectorAll('.bar')[1].getBoundingClientRect().width === 0")
    page.get_by_role('button', name='Surprising', exact=True).click()
    expect(page.locator('.temperature-value')).to_contain_text('1.80')
    expect(page.get_by_role('button', name='Generate 64 characters')).to_be_enabled()
    assert state()['fingerprint'] == baseline['fingerprint']
    page.get_by_role('button', name='Generate 64 characters').click()
    expect(page.locator('.generated')).to_be_visible()
    sample = page.locator('.generated').inner_text()
    page.get_by_role('button', name='Generate 64 characters').click()
    expect(page.get_by_role('button', name='Generate 64 characters')).to_be_enabled()
    assert page.locator('.generated').inner_text() == sample
    page.get_by_role('button', name='No, it doesn’t').click()
    expect(page.get_by_role('status')).to_contain_text('Exactly.')
    page.get_by_role('button', name='Learning lab', exact=True).click()
    page.get_by_role('button', name='Reset to random weights').click()
    expect(page.locator('.training-state')).to_contain_text('Optimizer step 0')
    random_state = state()
    assert random_state['fingerprint'] != baseline['fingerprint']
    page.get_by_role('button', name='Train 1 step', exact=True).click()
    expect(page.locator('.training-state')).to_contain_text('Optimizer step 1')
    expect(page.get_by_role('button', name='Train 25 steps', exact=True)).to_be_enabled()
    assert state()['fingerprint'] != random_state['fingerprint']
    page.get_by_role('button', name='Train 25 steps', exact=True).click()
    expect(page.locator('.train-progress')).to_contain_text('25 steps complete.', timeout=30000)
    assert state()['step'] == 26
    assert len(state()['history']) == 27
    expect(page.locator('.before-bar')).to_have_count(8)
    page.evaluate('window.scrollTo(0,0)')
    page.screenshot(path=str(OUT/'team-integration-learning.png'), full_page=True)
    page.get_by_role('button', name='Train 25 steps', exact=True).click()
    page.get_by_role('button', name='Stop', exact=True).click()
    expect(page.get_by_role('button', name='Train 25 steps', exact=True)).to_be_enabled()
    assert 26 < state()['step'] < 51, 'Stop must prevent the remaining updates'
    page.get_by_role('button', name='Restore trained checkpoint').click()
    expect(page.locator('.training-state')).to_contain_text(f"Optimizer step {baseline['step']:,}")
    assert state()['fingerprint'] == baseline['fingerprint']
    page.get_by_label('Give the model a starting point').fill('HELLO')
    page.get_by_role('button', name='Inspect prompt', exact=True).click()
    expect(page.get_by_role('alert')).to_contain_text('lowercase')
    page.get_by_label('Give the model a starting point').fill('THE CAT ')
    page.get_by_role('button', name='Inspect prompt', exact=True).click()
    expect(page.get_by_role('alert')).to_have_count(0)
    expect(page.locator('.heat-row')).to_have_count(8)
    expect(page.get_by_label('Give the model a starting point')).to_have_value('the cat ')
    page.get_by_label('Give the model a starting point').fill('the cat ' * 5)
    page.get_by_role('button', name='Inspect prompt', exact=True).click()
    expect(page.locator('.heat-row')).to_have_count(40)
    page.get_by_label('Give the model a starting point').fill('the cat ')
    page.get_by_role('button', name='Inspect prompt', exact=True).click()
    expect(page.locator('.heat-row')).to_have_count(8)
    page.get_by_role('button', name='About this model').click()
    expect(page.get_by_role('dialog')).to_be_visible()
    expect(page.get_by_role('dialog')).to_contain_text('40 output-projection weights')
    page.keyboard.press('Escape')
    expect(page.get_by_role('dialog')).to_have_count(0)
    page.get_by_role('button', name='Temperature lab', exact=True).click()
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path=str(OUT/'team-integration-mobile.png'), full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile horizontal overflow'
    assert not errors, errors
    result = {'browser':'Chromium', 'result':'PASS', 'checks':['real inference','temperature preserves weights','seeded generation','quiz feedback','random reset','one training step','25 live updates','loss history','before/after comparison','checkpoint restore','input validation','attention trace','model notes','mobile overflow'], 'page_errors':errors}
    result['checks'] += ['stop between training steps', 'Escape closes dialog', 'external requests blocked', 'greedy zero-probability bars', 'team-engine identity and parameter counts', 'single-head selectors', 'uppercase normalization', '40-character prompt', 'accurate training scope']
    (OUT / 'team-integration-browser-results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    page.evaluate("async () => fetch('/api/sessions/'+sessionStorage.getItem('glasswork-session'), {method:'DELETE'})")
    browser.close()
