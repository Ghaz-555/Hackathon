"""Check the standalone demo against real exported vectors in Chromium."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts'
OUT.mkdir(exist_ok=True)
checks=[]
def passed(name):
    checks.append(name)
    print('PASS',name,flush=True)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
    page=browser.new_page(viewport={'width':1500,'height':1000})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.route('**/*',lambda r:r.continue_() if r.request.url.startswith('http://127.0.0.1:5174') else r.abort())
    page.goto('http://127.0.0.1:5174/brain.html',wait_until='networkidle')
    page.wait_for_function("document.body.dataset.ready === 'true'")
    expect(page.locator('canvas')).to_be_visible()
    expect(page.locator('#neighbors li')).to_have_count(8)
    expect(page.locator('#token-id')).to_contain_text('768 VALUES')
    page.screenshot(path=str(OUT/'brain-desktop.png'),full_page=True)
    passed('Real GPT-2 dataset, WebGL points and high-dimensional neighbors')
    page.get_by_role('button',name='Explore A − B + C').click()
    expect(page.locator('#neighbors strong').first).to_have_text('queen')
    expect(page.locator('#analogy-note')).to_contain_text('Queen ranks #1')
    expect(page.locator('.token-label').filter(has_text='A − B + C')).to_be_visible()
    page.screenshot(path=str(OUT/'brain-analogy.png'),full_page=True)
    passed('Actual king - man + woman result and projected arrows')
    page.get_by_label('Starting token A').fill('king')
    page.get_by_label('Subtract token B').fill('king')
    page.get_by_label('Add token C').fill('cat')
    page.get_by_role('button',name='Explore A − B + C').click()
    expect(page.locator('#analogy-note')).to_contain_text('king − king + cat')
    assert page.locator('#neighbors strong').first.inner_text()!='queen'
    passed('Editing the expression recomputes results')
    page.get_by_label('Find a token',exact=True).fill('nonexistenttokenzz')
    page.get_by_role('button',name='Find token',exact=True).click()
    expect(page.get_by_role('alert')).to_contain_text('not in this token slice')
    page.get_by_label('Find a token',exact=True).fill('woman')
    page.get_by_role('button',name='Find token',exact=True).click()
    expect(page.locator('#selected-name')).to_have_text('woman')
    expect(page.get_by_role('alert')).to_be_hidden()
    page.locator('#neighbors button').first.click()
    assert page.locator('#selected-name').inner_text()!='woman'
    passed('Search validation and neighbor navigation')
    page.get_by_label('Thickness').fill('2')
    expect(page.locator('#width-value')).to_have_text('2.0')
    visible=int(page.locator('#point-count').inner_text().split(' / ')[0].replace(',',''))
    assert 0<visible<2048
    page.get_by_label('Depth').fill('1.5')
    expect(page.locator('#depth-value')).to_have_text('1.5')
    passed('Depth slice and thickness filter actual projected positions')
    page.get_by_role('button',name='Focus token',exact=True).click()
    page.get_by_role('button',name='Reset view',exact=True).click()
    page.get_by_role('button',name='Rotate',exact=True).click()
    expect(page.get_by_role('button',name='Rotate',exact=True)).to_have_attribute('aria-pressed','true')
    page.get_by_role('button',name='Reduce motion',exact=True).click()
    expect(page.get_by_role('button',name='Rotate',exact=True)).to_have_attribute('aria-pressed','false')
    page.get_by_role('button',name='Labels',exact=True).click()
    assert page.locator('.token-label:not(.important)').count()==0
    passed('Camera controls, labels and reduced motion')
    page.set_viewport_size({'width':390,'height':844})
    page.evaluate('window.scrollTo(0,0)')
    page.screenshot(path=str(OUT/'brain-mobile.png'),full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    passed('Mobile layout without horizontal overflow')
    page.locator('canvas').evaluate("canvas=>canvas.dispatchEvent(new Event('webglcontextlost',{cancelable:true}))")
    expect(page.locator('.webgl-fallback')).to_be_visible()
    page.get_by_label('Find a token',exact=True).fill('king')
    page.get_by_role('button',name='Find token',exact=True).click()
    expect(page.locator('#neighbors li')).to_have_count(8)
    passed('GPU-loss fallback preserves real-vector search')
    assert not errors,errors
    result={'result':'PASS','checks':checks,'page_errors':errors}
    (OUT/'browser-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    browser.close()
