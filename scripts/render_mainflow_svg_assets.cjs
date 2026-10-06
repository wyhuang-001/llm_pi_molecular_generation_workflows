// Render the normalized main-flow SVG to a high-resolution PNG with an isolated browser.
const fs = require('fs');
const path = require('path');
const puppeteer = require('/tmp/4wkq-figure-tools/node_modules/puppeteer');
const ROOT = path.resolve(__dirname, '..');
(async () => {
  const file = path.join(ROOT, 'assets/ppt_mainflow/mermaid/main_flow.svg');
  const width = 2800;
  const svg = fs.readFileSync(file, 'utf8').replace(/<\?xml[^>]*>/, '');
  const vb = svg.match(/viewBox="([^"]+)"/)[1].split(/\s+/).map(Number);
  const height = Math.ceil(width * vb[3] / vb[2]);
  const browser = await puppeteer.launch({
    executablePath: '/home/hwy/.cache/ms-playwright/chromium-1246/chrome-linux64/chrome',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width, height, deviceScaleFactor: 1 });
  await page.setContent(`<html><head><style>*{box-sizing:border-box}html,body{margin:0;background:white}body>svg{display:block;width:${width}px;height:${height}px;max-width:none!important;max-height:none!important}</style></head><body>${svg}</body></html>`);
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: file.replace(/\.svg$/, '.png'), omitBackground: false });
  await page.close();
  await browser.close();
  console.log('Rendered', path.basename(file).replace(/\.svg$/, '.png'), width + 'x' + height);
})();
