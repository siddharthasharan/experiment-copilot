// Chat turn tests for the showroom (deterministic JS, no LLM, no DOM needed).
// Loads the <script> from docs/index.html into a VM with a stub document,
// then drives window.ChatTest.handle() through scripted conversations.
// Run: node tests/test_chat_showroom.js
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const html = fs.readFileSync(path.join(__dirname, "..", "docs", "index.html"), "utf8");
const m = html.match(/<script>([\s\S]*)<\/script>/);
if (!m) { console.error("no script found in docs/index.html"); process.exit(1); }

function el() {
  return {
    innerHTML: "", scrollTop: 0, value: "",
    addEventListener: function () {},
    querySelectorAll: function () { return []; },
    classList: { add: function () {}, remove: function () {}, toggle: function () {} },
  };
}
const sandbox = {
  console: console,
  document: {
    getElementById: function () { return el(); },
    querySelectorAll: function () { return []; },
    addEventListener: function () {},
    body: { classList: { add: function () {}, remove: function () {} } },
  },
  window: null,
};
sandbox.window = sandbox;
sandbox.window.scrollTo = function () {};
vm.createContext(sandbox);
vm.runInContext(m[1], sandbox);
const Chat = sandbox.ChatTest;
if (!Chat) { console.error("ChatTest not exposed"); process.exit(1); }

let failures = 0;
function check(name, cond, detail) {
  if (cond) console.log("  [OK ] " + name);
  else { console.log("  [FAIL] " + name + (detail ? " — " + detail : "")); failures += 1; }
}
function htmlOf(r) { return r.html; }

console.log("chat turn tests:");
// 1. Initial message lists all 14 scenarios
Chat.reset();
let init = Chat.initial();
const titles = ["Clean win", "SRM", "peeker", "Novelty", "Guardrail", "Whale", "True null",
  "Dilution", "Trigger imbalance", "Triggered clean", "Overlapping", "segment mirage", "Real heterogeneity", "Spillover"];
for (const t of titles) check("initial lists " + t, init.toLowerCase().indexOf(t.toLowerCase()) >= 0);

// 2. Whale conversation: review -> diagnosis -> recommendation -> brief
Chat.reset();
let r = Chat.handle("Review the whale trap");
check("whale review verdict INCONCLUSIVE", htmlOf(r).indexOf("INCONCLUSIVE") >= 0, htmlOf(r).slice(0, 200));
check("whale review shows metrics", htmlOf(r).indexOf("Revenue") >= 0);
r = Chat.handle("Why is it inconclusive?");
check("whale diagnosis mentions whales", /whale/i.test(htmlOf(r)), htmlOf(r).slice(0, 200));
r = Chat.handle("What should we do?");
check("whale recommendation has next steps", htmlOf(r).indexOf("Next steps") >= 0);
check("whale next steps mention median/winsorized", /median|winsorized/i.test(htmlOf(r)));
r = Chat.handle("PM/Eng brief");
check("whale brief present", htmlOf(r).indexOf("PM/Eng brief") >= 0);

// 3. Peeker conversation (the screenshot case)
Chat.reset();
r = Chat.handle("Review the peeker's mirage");
check("peeker review verdict INCONCLUSIVE", htmlOf(r).indexOf("INCONCLUSIVE") >= 0);
r = Chat.handle("Diagnosis");
check("peeker diagnosis mentions interim/early", /interim|early/i.test(htmlOf(r)));
r = Chat.handle("Recommendation and next steps");
check("peeker next steps mention pre-register", /pre-register/i.test(htmlOf(r)));

// 4. Clean win: launch call
Chat.reset();
r = Chat.handle("Review the clean win");
check("clean_win review verdict LAUNCH", htmlOf(r).indexOf("LAUNCH") >= 0);
r = Chat.handle("Recommendation and next steps");
check("clean_win recommendation says Ship it", htmlOf(r).indexOf("Ship it") >= 0);

// 5. Methodology RAG: SRM question retrieves the SRM note
Chat.reset();
r = Chat.handle("What is SRM?");
check("SRM answer names the note", htmlOf(r).indexOf("Sample Ratio Mismatch") >= 0, htmlOf(r).slice(0, 200));

// 6. Unmapped question clarifies, never invents
Chat.reset();
r = Chat.handle("What is the weather in Paris?");
check("unknown question clarifies", /can.t answer that/i.test(htmlOf(r)));

// 7. Another scenario returns to the list
Chat.reset();
Chat.handle("Review the whale trap");
r = Chat.handle("Another scenario");
check("another scenario lists scenarios", htmlOf(r).indexOf("scenarios you can ask about") >= 0);

console.log(failures === 0 ? "chat: ALL PASS" : "chat: " + failures + " FAILURES");
process.exit(failures === 0 ? 0 : 1);
