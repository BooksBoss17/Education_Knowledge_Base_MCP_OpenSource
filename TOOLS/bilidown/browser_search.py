"""Bilibili-only webpage search. No login, generic browser commands, or caller URLs."""
import asyncio
import json
from pathlib import Path
import random
import re
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlencode
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bilidown.browser_mcp import HERE,session
from bilidown.backend import WindowsJob

EXTRACT=(HERE/"browser_extract.js").read_text("utf8")

class Incomplete(Exception):
    def __init__(self,code,site_code=None):
        self.reason={"code":code}
        if site_code is not None:self.reason["site_code"]=site_code

def canonical(row):
    bv=row.get("bvid")
    if not isinstance(bv,str) or not re.fullmatch(r"BV[a-zA-Z0-9]{10}",bv):raise Incomplete("INVALID_VIDEO_CARD")
    return {**row,"url":"https://www.bilibili.com/video/"+bv}

async def search_page(browser,kind,keyword,page):
    query=urlencode(dict(keyword=keyword,page=page))
    await browser.navigate("https://search.bilibili.com/"+("upuser" if kind=="authors" else "video")+"?"+query)
    for _ in range(14):
        state=await browser.evaluate(EXTRACT)
        if state["limited"]:raise Incomplete("SITE_LIMITED")
        if state[kind] or state["empty"]:
            if page>1 and state["active"]!=page:raise Incomplete("PAGE_NOT_CONFIRMED")
            return state
        await asyncio.sleep(.4)
    raise Incomplete("PAGE_NOT_READY")

async def turn_page(browser,page):
    # The pager uses the visible page number or its jump box for distant pages.
    return await browser.evaluate("""() => {
        const target=PAGE;
        const buttons=[...document.querySelectorAll('button')];
        const btn=buttons.find(e=>e.className.includes('pagenation')&&e.innerText.trim()===String(target));
        if(btn&&!btn.disabled){btn.click();return true}
        const input=document.querySelector('[class*="pagenation"] input');
        if(!input)return false;
        const setter=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;
        setter.call(input,String(target));input.dispatchEvent(new Event('input',{bubbles:true}));
        for(const event of ['keydown','keyup'])input.dispatchEvent(new KeyboardEvent(event,{key:'Enter',code:'Enter',keyCode:13,bubbles:true}));
        return true
    }""".replace("PAGE",str(int(page))))

async def posts(browser,uid,name,args):
    page=args.get("page",1);size=args.get("page_size",20);budget=args.get("max_pages",1)
    start=(page-1)*size;stop=start+size*budget;pos=start
    result=dict(mode="author_posts",author=dict(uid=uid,name=name),complete=False,total=None,first_page=page,page_size=size)
    seen={};stable=True;deadline=time.monotonic()+65
    native=1;width=None;total=None
    try:
        await browser.navigate("https://space.bilibili.com/"+uid+"/upload/video")
        while True:
            for _ in range(25):
                event=browser.response(uid,native)
                if event:break
                state=await browser.evaluate(EXTRACT)
                if state["limited"]:raise Incomplete("SITE_LIMITED")
                if state["total"] is not None and total is None:result["total"]=state["total"]
                await asyncio.sleep(.35)
            if not event:raise Incomplete("PAGE_NOT_CONFIRMED")
            code=event.get("code") or event["http"]
            if code!=200 or "rows" not in event:
                state=await browser.evaluate(EXTRACT)
                if total is None:result["total"]=state["total"]
            if code in (412,-412,-352,-509,429):raise Incomplete("SITE_LIMITED",code)
            if code in (403,-403):raise Incomplete("SITE_DENIED",code)
            if code!=200 or "rows" not in event:raise Incomplete("PAGE_LOAD_FAILED",code)
            reported=event["total"];reported_width=event["size"]
            if type(reported)!=int or reported<0 or type(reported_width)!=int or reported_width<1:raise Incomplete("PAGE_NOT_CONFIRMED")
            visible_total=(await browser.evaluate(EXTRACT))["total"]
            result["web_total"]=visible_total
            if visible_total!=reported or (total is not None and reported!=total):stable=False
            if width and reported_width!=width:raise Incomplete("PAGE_SIZE_CHANGED")
            total=result["total"]=reported;width=reported_width;rows=event["rows"]
            page_start=(native-1)*width
            expected=max(0,min(width,total-page_start))
            if len(rows)!=expected:raise Incomplete("COUNT_MISMATCH")
            for index,row in enumerate(rows,page_start):
                if pos<=index<stop:
                    item=canonical(row)
                    if item["uid"]!=uid:raise Incomplete("AUTHOR_MISMATCH")
                    seen[item["bvid"]]=item;pos=index+1
            if pos>=total:
                result["complete"]=start==0 and stable and len(seen)==total
                result.update(reason=None if result["complete"] else {"code":"WINDOW_ONLY" if start else "COUNT_MISMATCH"},next_page=None)
                break
            if pos>=stop:raise Incomplete("PAGE_LIMIT")
            delay = random.uniform(2, 4)
            if time.monotonic() + delay >= deadline:raise Incomplete("TIME_BUDGET")
            await asyncio.sleep(delay)
            if time.monotonic() >= deadline:raise Incomplete("TIME_BUDGET")
            native=pos//width+1
            if not await turn_page(browser,native):raise Incomplete("PAGER_UNAVAILABLE")
    except (Incomplete,RuntimeError,TimeoutError) as ex:
        result.update(reason=ex.reason if isinstance(ex,Incomplete) else {"code":"BROWSER_UNAVAILABLE"},next_page=pos//size+1)
    videos=list(seen.values());word=args.get("title","").strip().casefold()
    result.update(scanned=len(videos),pages_read=max(0,(pos-start+size-1)//size))
    if word:videos=[v for v in videos if word in v["title"].casefold()]
    result.update(returned=len(videos),videos=videos)
    return result

async def bounded_run(runtime,args):
    async with asyncio.timeout(110):
        return await run(runtime,args)

async def run(runtime,args):
    keyword=args.get("title","").strip();name=args.get("author","").strip();uid=args.get("uid")
    page=args.get("page",1);size=args.get("page_size",20)
    async with session(runtime) as browser:
        if not uid and not name:
            start=(page-1)*size;end=start+size;seen=set();out=[]
            physical=1;more=False;reason=None;deadline=time.monotonic()+65
            while True:
                try:state=await search_page(browser,"videos",keyword,physical)
                except Incomplete as exc:
                    reason=exc.reason;break
                before=len(seen)
                for raw in state["videos"]:
                    v=canonical(raw)
                    if v["bvid"] in seen:continue
                    seen.add(v["bvid"])
                    if start<len(seen)<=end:out.append(v)
                more=state["next"] or len(seen)>end
                if len(seen)>=end or not state["next"]:break
                if len(seen)==before or time.monotonic()>=deadline:
                    reason={"code":"PAGE_NOT_CONFIRMED" if len(seen)==before else "TIME_BUDGET"};break
                physical+=1
            if args.get("title_exact"):out=[v for v in out if keyword.casefold() in v["title"].casefold()]
            result=dict(mode="title",page=page,next_page=page+1 if more else None,videos=out,returned=len(out))
            if reason:result.update(complete=False,reason=reason,next_page=page)
            return result
        if name:
            choices=(await search_page(browser,"authors",name,1))["authors"]
            matches=[x for x in choices if x["name"]==name]
            if len(matches)!=1:return dict(mode="choose_author",complete=False,candidates=choices)
            uid=matches[0]["uid"]
        return await posts(browser,uid,name,args)

if __name__=="__main__":
    job=WindowsJob();job.assign(SimpleNamespace(_handle=-1))
    try:reply=asyncio.run(bounded_run(Path(sys.argv[1]).resolve(strict=True),json.loads(sys.stdin.buffer.readline(8192))))
    except (Incomplete,RuntimeError,TimeoutError,OSError) as exc:
        reply=dict(complete=False,reason=exc.reason if isinstance(exc,Incomplete) else {"code":"BROWSER_UNAVAILABLE","detail":type(exc).__name__})
    print(json.dumps(reply,ensure_ascii=False))
