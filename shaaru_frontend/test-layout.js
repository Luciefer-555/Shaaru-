const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('http://localhost:3000/tailor');
  
  // Wait for it to load
  await page.waitForTimeout(3000);
  
  // Find the max-w-3xl div
  const el = await page.locator('.max-w-3xl.mx-auto.space-y-4').first();
  
  const margins = await el.evaluate((node) => {
    const style = window.getComputedStyle(node);
    return {
      marginLeft: style.marginLeft,
      marginRight: style.marginRight,
      width: style.width
    };
  });
  
  console.log('COMPUTED_STYLES:', margins);
  
  // Get parent dimensions
  const parent = await el.evaluateHandle((node) => node.parentElement);
  const parentStyles = await parent.evaluate((node) => {
    const style = window.getComputedStyle(node);
    return {
      width: style.width,
      display: style.display
    };
  });
  
  console.log('PARENT_WIDTH:', parentStyles);
  
  await browser.close();
})();
