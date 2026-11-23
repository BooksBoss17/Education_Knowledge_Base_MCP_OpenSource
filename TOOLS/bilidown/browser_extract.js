() => {
 const txt = document.body?.innerText || '';
 const clean = s => (s || '').trim();
 const videos = [];
 for(const card of document.querySelectorAll('.bili-video-card')) {
   const title = card.querySelector('h3'), link = title?.closest('a') || title?.querySelector('a');
   const bv=link?.href.match(/\/video\/(BV[a-zA-Z0-9]{10})/);
   if (!bv) continue;
   const owner=card.querySelector('.bili-video-card__info--owner');
   videos.push({bvid:bv[1],title:clean(title.textContent),author:clean(owner?.querySelector('.bili-video-card__info--author')?.textContent || owner?.innerText?.split('\n')[0]),uid:owner?.href.match(/space\.bilibili\.com\/(\d+)/)?.[1]||'',duration:clean(card.querySelector('.bili-video-card__stats__duration')?.textContent)});
 }
 const authors=[...document.querySelectorAll('.b-user-info-card .i_card_title a')].map(a=>({name:clean(a.textContent),uid:a.href.match(/space\.bilibili\.com\/(\d+)/)?.[1]})).filter(x=>x.uid);
 const next=[...document.querySelectorAll('button')].find(x=>x.getAttribute('aria-label')==='下一页'||x.innerText.trim()==='下一页');
 const total=txt.match(/共\s*\d+\s*页\s*\/\s*([\d,]+)\s*个/) || txt.match(/(?:^|\n)视频\n([\d,]+)(?:\n|$)/);
 const active=document.querySelector('button[aria-current="page"],.vui_pagenation--btn-num.vui_button--active');
 return {videos,authors,next:!!next&&!next.disabled&&!next.className.includes('disabled'),active:Number(active?.textContent)||null,total:total?Number(total[1].replaceAll(',','')):null,
 empty:/今天真是寂寞如雪|暂无相关|没有找到相关|没有搜索到|没有找到.*?用户|啥都没搜到|没有更多搜索结果|未找到相关|空间主人还没投过视频/.test(txt),
 limited:/访问异常|请求过于频繁|完成验证|错误号\s*[:：]\s*412|安全验证/.test(txt)};
}
