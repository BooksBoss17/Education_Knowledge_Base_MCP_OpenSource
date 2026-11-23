'use strict';
// Observe only the public page's own list response. Keep media URLs, headers and cookies out of the result.
const fs = require('node:fs');
module.exports.default = async ({page}) => {
  page.on('response', async response => {
    try {
      const u = new URL(response.url());
      if (u.hostname !== 'api.bilibili.com' || !['/x/space/wbi/arc/search','/x/space/arc/search'].includes(u.pathname)) return;
      const rec = {uid:u.searchParams.get('mid'),page:Number(u.searchParams.get('pn') || 1),http:response.status()};
      if (rec.http === 200) {
        const j=await response.json(); rec.code=j.code;
        if (j.code===0) {
          const d=j.data;rec.total=d.page.count;rec.size=d.page.ps;
          rec.rows=d.list.vlist.map(v=>({bvid:v.bvid,title:v.title,uid:String(v.mid),author:v.author,duration:v.length}));
        }
      }
      fs.appendFileSync(process.env.EKB_BILI_PAGE_EVENTS,JSON.stringify(rec)+'\n','utf8');
    } catch { /* The caller treats an unobservable page as incomplete. */ }
  });
};
