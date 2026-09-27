"""Check both pages on the same production server, including session return."""
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
OUT=Path(__file__).resolve().parents[1]/'artifacts'
BASE=os.environ.get('GLASSWORK_URL', 'http://127.0.0.1:8001')
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
    page=browser.new_page(viewport={'width':1500,'height':1000})
    page.add_init_script("sessionStorage.setItem('glasswork-engine','team')")
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(BASE,wait_until='networkidle')
    expect(page.get_by_role('button',name='Inspect prompt',exact=True)).to_be_enabled()
    page.get_by_role('button', name='Architecture', exact=True).click()
    expect(page.locator('.world-label')).to_have_count(6)
    first=page.locator('canvas').screenshot()
    page.wait_for_timeout(350)
    assert first!=page.locator('canvas').screenshot(), 'Live tensor animation must change the canvas'
    page.get_by_role('button',name='Reduce motion',exact=True).click()
    page.wait_for_timeout(600)
    frozen=page.locator('canvas').screenshot()
    page.wait_for_timeout(350)
    assert frozen==page.locator('canvas').screenshot(), 'Reduced motion must freeze the explanatory flow'
    page.get_by_role('button',name='Reduce motion',exact=True).click()
    sid=page.evaluate("sessionStorage.getItem('glasswork-session')")
    page.get_by_role('link',name='Embedding space').click()
    page.wait_for_function("document.body.dataset.ready === 'true'")
    expect(page.locator('canvas')).to_be_visible()
    expect(page.locator('#neighbors li')).to_have_count(8)
    page.get_by_role('button',name='Explore A − B + C').click()
    expect(page.locator('#neighbors strong').first).to_have_text('queen')
    expect(page.locator('#analogy-note')).to_contain_text('Queen ranks #1')
    page.screenshot(path=str(OUT/'expanded-integrated-embeddings.png'),full_page=True)
    page.get_by_label('Find a token',exact=True).fill('unknownzz')
    page.get_by_role('button',name='Find token',exact=True).click()
    expect(page.get_by_role('alert')).to_contain_text('not in this token slice')
    page.get_by_label('Find a token',exact=True).fill('cat')
    page.get_by_role('button',name='Find token',exact=True).click()
    expect(page.locator('#selected-name')).to_have_text('cat')
    page.get_by_label('Thickness').fill('2')
    assert int(page.locator('#point-count').inner_text().split(' / ')[0].replace(',',''))<2048
    page.get_by_label('Depth').fill('1.5')
    page.get_by_role('button',name='Focus token',exact=True).click()
    page.get_by_role('button',name='Reset view',exact=True).click()
    page.get_by_role('button',name='Rotate',exact=True).click()
    page.get_by_role('button',name='Reduce motion',exact=True).click()
    expect(page.get_by_role('button',name='Rotate',exact=True)).to_have_attribute('aria-pressed','false')
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path=str(OUT/'expanded-integrated-embeddings-mobile.png'),full_page=True)
    page.locator('canvas').evaluate("c=>c.dispatchEvent(new Event('webglcontextlost',{cancelable:true}))")
    expect(page.locator('.webgl-fallback')).to_be_visible()
    page.get_by_role('button',name='Explore A − B + C').click()
    expect(page.locator('#neighbors strong').first).to_have_text('queen')
    page.locator('.brain-header .brand').click()
    expect(page.get_by_role('button',name='Inspect prompt',exact=True)).to_be_enabled()
    assert page.evaluate("sessionStorage.getItem('glasswork-session')")==sid
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path=str(OUT/'expanded-integrated-model-mobile.png'),full_page=True)
    assert not errors,errors
    result={'result':'PASS','checks':['live animation changes frames','reduced motion freezes animation','same-origin navigation','live model session retained','real embedding scene','computed analogy','search validation','depth slicing','camera controls','reduced motion','mobile layouts','GPU fallback'],'page_errors':errors}
    (OUT/'expanded-integrated-integrated-browser.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    page.evaluate("async()=>fetch('/api/sessions/'+sessionStorage.getItem('glasswork-session'),{method:'DELETE'})")
    browser.close()
