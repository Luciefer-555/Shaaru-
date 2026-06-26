const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1280, height: 800 });

  // Idle state
  await page.goto('http://localhost:3000/tailor');
  await page.waitForTimeout(2500);
  await page.screenshot({ path: 'C:/Users/saipr/.gemini/antigravity/brain/7bba96da-0b6d-471d-ad0a-4e3fb4ceea1c/v0-idle.png' });
  console.log('idle done');

  // Send message -> chat state
  const ta = await page.locator('textarea').first();
  await ta.fill('I want a silk kurta made for a wedding');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(4000);
  await page.screenshot({ path: 'C:/Users/saipr/.gemini/antigravity/brain/7bba96da-0b6d-471d-ad0a-4e3fb4ceea1c/v0-chat.png' });
  console.log('chat done');

  await browser.close();
})();
