// Render normalized SVGs and the assembled figure using an isolated browser.
const fs = require('fs');
const path = require('path');
const puppeteer = require('/tmp/4wkq-figure-tools/node_modules/puppeteer');
const ROOT = path.resolve(__dirname, '..');
(async () => {
  const browser = await puppeteer.launch({executablePath:'/home/hwy/.cache/ms-playwright/chromium-1246/chrome-linux64/chrome',headless:true,args:['--no-sandbox']});
  const items = ['overview','key_loop','next_steps','workflow_publication'].map(n => ({file:path.join(ROOT,'assets/ppt_4wkq/mermaid',n+'.svg'),width:1400,scale:2,pdf:false}));
  items.push({file:path.join(ROOT,'deliverables/4WKQ_figure_sources/Figure_4WKQ_workflow.svg'),width:1600,scale:3,pdf:true});
  for (const item of items) {
    const svg=fs.readFileSync(item.file,'utf8').replace(/<\?xml[^>]*>/,'');
    const vb=svg.match(/viewBox="([^"]+)"/)[1].split(/\s+/).map(Number);
    const height=Math.ceil(item.width*vb[3]/vb[2]);
    const page=await browser.newPage();
    await page.setViewport({width:item.width,height,deviceScaleFactor:item.scale});
    await page.setContent(`<html><head><style>*{box-sizing:border-box}html,body{margin:0;background:white}body>svg{display:block;width:${item.width}px;height:${height}px;max-width:none!important;max-height:none!important}</style></head><body>${svg}</body></html>`);
    await page.evaluate(()=>document.fonts.ready);
    await page.screenshot({path:item.file.replace(/\.svg$/,'.png'),omitBackground:false});
    if(item.pdf) await page.pdf({path:item.file.replace(/\.svg$/,'.pdf'),width:item.width+'px',height:height+'px',printBackground:true,margin:{top:0,right:0,bottom:0,left:0}});
    await page.close();
    console.log('Rendered',path.basename(item.file));
  }
  await browser.close();
})();
