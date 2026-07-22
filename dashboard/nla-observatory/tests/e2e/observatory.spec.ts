/**
 * End-to-end checks against the REAL dev server + real evidence shards at
 * http://localhost:5199 (see playwright.config.ts webServer). Runs in three
 * viewport projects: desktop 1440x900, tablet 1024x768, mobile 390x844.
 */

import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const STATIONS = ["channel", "trace", "bench", "audit"] as const;

/** Console errors that are environment noise, not app defects. */
function isIgnorableConsoleError(text: string): boolean {
  return /favicon/i.test(text) || /Download the React DevTools/i.test(text);
}

function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isIgnorableConsoleError(msg.text())) {
      errors.push(msg.text());
    }
  });
  page.on("pageerror", (err) => {
    errors.push(`pageerror: ${String(err)}`);
  });
  return errors;
}

async function expectNoSeriousAxeViolations(page: Page): Promise<void> {
  const axeResults = await new AxeBuilder({ page }).analyze();
  const severe = axeResults.violations.filter(
    (violation) => violation.impact === "serious" || violation.impact === "critical",
  );
  expect(
    severe.map((violation) =>
      `${violation.id}: ${violation.help} — ${violation.nodes
        .map((node) => node.target.join(" "))
        .join(", ")}`,
    ),
  ).toEqual([]);
}

async function gotoStation(page: Page, station: string): Promise<void> {
  await page.goto(`/#/${station}`);
  // Network-idleish: static shards fetch shortly after load.
  await page.waitForLoadState("networkidle");
  // The station tab must be selected and real evidence text rendered.
  const selectedTab = page.locator('[role="tab"][aria-selected="true"]');
  await expect(selectedTab).toHaveText(new RegExp(station, "i"));
  await page.waitForFunction(
    () => (document.body.innerText ?? "").length > 500,
    undefined,
    { timeout: 20_000 },
  );
}

for (const station of STATIONS) {
  test(`${station} station renders real evidence without errors`, async ({ page }, testInfo) => {
    const consoleErrors = collectConsoleErrors(page);

    await gotoStation(page, station);

    // No ErrorBox / manifest failure anywhere on the page.
    await expect(page.locator('[role="alert"]')).toHaveCount(0);

    await page.screenshot({
      path: `screenshots/${testInfo.project.name}/${station}.png`,
      fullPage: true,
    });

    await expectNoSeriousAxeViolations(page);

    expect(consoleErrors).toEqual([]);
  });
}

test("poetry planning lens renders its result summary", async ({ page }, testInfo) => {
  const consoleErrors = collectConsoleErrors(page);
  await page.goto("/#/trace?poem=carrot-rabbit");
  await page.waitForLoadState("networkidle");

  await expect(page.getByRole("heading", { name: /future rhyme/i })).toBeVisible();
  await expect(page.getByText(/weak signal · causal test negative/i)).toBeVisible();
  await expect(page.getByText(/Full original poetry prefix/i)).toBeVisible();
  await expect(page.getByText(/NLA samples for selected token/i)).toBeVisible();
  await expect(page.getByText(/real activation/i).first()).toBeVisible();
  await expect(page.getByText(/shuffled control/i).first()).toBeVisible();
  const poetryTokens = page.locator(".trc-poetry-split .trc-inline-source-token");
  expect(await poetryTokens.count()).toBeGreaterThanOrEqual(8);
  await poetryTokens.first().click();
  await expect(poetryTokens.first()).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText(/Original prefix shown to Nano30B/i)).toBeVisible();
  await expect(page.getByText(/Baseline model output/i)).toBeVisible();
  await expect(page.getByText("Step 1")).toBeVisible();
  await expect(page.getByText("Step 2")).toBeVisible();
  await expect(page.getByText("Step 3")).toBeVisible();
  await expectNoSeriousAxeViolations(page);
  await page.screenshot({
    path: `screenshots/${testInfo.project.name}/trace-poetry.png`,
    fullPage: true,
  });
  expect(consoleErrors).toEqual([]);
});

test("document trace links full-source token selections to NLA verbalizations", async ({ page }) => {
  await gotoStation(page, "trace");

  await expect(page.getByText(/Full original document/i).first()).toBeVisible();
  await expect(page.getByText(/NLA output for selected token/i)).toBeVisible();
  await expect(page.getByText(/activation at the selected source token/i)).toBeVisible();
  await expect(page.getByText("Plain text was not sampled in this dashboard.", { exact: true })).toBeVisible();

  const documentTokens = page.locator(".trc-document-split .trc-inline-source-token");
  await expect(documentTokens).toHaveCount(40);
  await documentTokens.nth(1).click();
  await expect(documentTokens.nth(1)).toHaveAttribute("aria-pressed", "true");

  await expectNoSeriousAxeViolations(page);
});

test("channel row reader links the full source to its NLA description", async ({ page }) => {
  await gotoStation(page, "channel");

  await expect(page.getByRole("heading", { name: /Selected row: source/i })).toBeVisible();
  await expect(page.getByText(/1 · Original source text/i)).toBeVisible();
  await expect(page.getByText(/2 · NLA learned description/i)).toBeVisible();
  await expect(page.getByText(/stored activation → AV text → AR reconstruction/i)).toBeVisible();

  const source = page.locator(".chan-row-source-text");
  const firstSource = await source.innerText();
  expect(firstSource.length).toBeGreaterThan(100);
  await expect(page.getByLabel("NLA learned description")).not.toBeEmpty();

  await page.getByRole("button", { name: "Next validation example" }).click();
  await expect(page.getByText("Example 2 of 50", { exact: true })).toBeVisible();
  await expect.poll(async () => source.innerText()).not.toBe(firstSource);

  await page.getByRole("button", { name: /Browse all 50 examples/i }).click();
  const browser = page.getByRole("dialog", { name: "Choose a source example" });
  await expect(browser).toBeVisible();
  await expect(browser.getByRole("searchbox", { name: "Search validation examples" })).toBeVisible();
  await browser.getByRole("button", { name: /Example 3:/i }).click();
  await expect(browser).not.toBeVisible();
  await expect(page.getByText("Example 3 of 50", { exact: true })).toBeVisible();

  await expectNoSeriousAxeViolations(page);
});

test("dark theme preserves the channel evidence hierarchy", async ({ page }, testInfo) => {
  const consoleErrors = collectConsoleErrors(page);
  await page.addInitScript(() => window.localStorage.setItem("nla-observatory-theme", "dark"));
  await gotoStation(page, "channel");

  await expect(page.getByRole("heading", { name: /can language preserve an activation/i })).toBeVisible();
  await expectNoSeriousAxeViolations(page);
  await page.screenshot({
    path: `screenshots/${testInfo.project.name}/channel-dark.png`,
    fullPage: true,
  });
  expect(consoleErrors).toEqual([]);
});

test("reader guidance explains the NLA pipeline and local terminology", async ({ page }) => {
  await gotoStation(page, "channel");

  await expect(page.getByRole("heading", { name: "What the NLA is doing" })).toBeVisible();
  await expect(page.getByText(/2,688 numbers from Nano30B layer R33/i)).toBeVisible();

  const glossary = page.locator("details.nla-glossary");
  await glossary.locator("summary").click();
  await expect(glossary.getByText(/activation-to-verbalization model/i)).toBeVisible();

  const waterfall = page.locator("#chan-waterfall");
  await expect(waterfall.getByText("What this shows")).toBeVisible();
  await expect(waterfall.getByText("How to read it")).toBeVisible();
  await waterfall.locator("details.panel-terms > summary").click();
  await expect(waterfall.getByText(/directional mean-squared error/i)).toBeVisible();

  await expectNoSeriousAxeViolations(page);
});

test("bench permalink round-trips through a reload", async ({ page }) => {
  const permalink = "/#/bench?row=validation-248348&variant=syntax&dose=1";
  await page.goto(permalink);
  await page.waitForLoadState("networkidle");

  const selectedTab = page.locator('[role="tab"][aria-selected="true"]');
  await expect(selectedTab).toHaveText(/bench/i);

  await page.reload();
  await page.waitForLoadState("networkidle");

  await expect(page.locator('[role="tab"][aria-selected="true"]')).toHaveText(/bench/i);
  const url = page.url();
  expect(url).toContain("#/bench");
  expect(url).toContain("row=validation-248348");
  expect(url).toContain("variant=syntax");
  expect(url).toContain("dose=1");
});

test("no horizontal page overflow on mobile", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "overflow gate runs on the mobile project");
  for (const station of STATIONS) {
    await gotoStation(page, station);
    const { scrollWidth, innerWidth } = await page.evaluate(() => ({
      scrollWidth: document.scrollingElement?.scrollWidth ?? document.body.scrollWidth,
      innerWidth: window.innerWidth,
    }));
    expect(
      scrollWidth,
      `station ${station}: page scrollWidth ${scrollWidth} exceeds viewport ${innerWidth}`,
    ).toBeLessThanOrEqual(innerWidth + 1);
  }
});
