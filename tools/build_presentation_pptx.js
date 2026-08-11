const pptxgen = require("pptxgenjs");
const path = require("path");

const ROOT = "/Users/addhyanpant/Desktop/Thesis5/FinalMERThesis";
const IMG = (p) => path.join(ROOT, p);

// ── Palette ──────────────────────────────────────────────────────────────────
const BERRY = "6D2E46";
const BERRY_DK = "4A1E30";
const ROSE = "A26769";
const TINT = "F5EFF1";
const TINT_T = "E6F1F0";
const TEAL = "0B6E67";
const INK = "2B2126";
const MUTED = "6E6169";
const WHITE = "FFFFFF";
const LINE = "D9CDD2";

const HF = "Cambria";
const BF = "Calibri";
const MF = "Courier New";

const W = 13.33, H = 7.5, M = 0.55;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Addhyan";
pres.title = "Micro-Expression Recognition on CASME-II";

let SLIDE_N = 0;

function base(titleText, kicker) {
  SLIDE_N += 1;
  const s = pres.addSlide();
  s.background = { color: WHITE };
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: M, y: 0.34, w: 9.0, h: 0.28, fontFace: BF, fontSize: 11, bold: true,
      color: ROSE, charSpacing: 1.6, margin: 0, valign: "middle",
    });
  }
  s.addText(titleText, {
    x: M, y: kicker ? 0.62 : 0.42, w: 11.4, h: 0.66, fontFace: HF, fontSize: 27,
    bold: true, color: BERRY, margin: 0, valign: "middle",
  });
  s.addText(String(SLIDE_N), {
    x: W - M - 0.7, y: 6.92, w: 0.7, h: 0.3, fontFace: BF, fontSize: 11,
    color: MUTED, align: "right", margin: 0,
  });
  return s;
}

function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.06, fill: { color: fill || TINT }, line: { color: fill || TINT, width: 0 },
  });
}

function chipCircle(s, x, y, d, label, fill) {
  s.addShape(pres.ShapeType.ellipse, {
    x, y, w: d, h: d, fill: { color: fill || BERRY }, line: { color: fill || BERRY, width: 0 },
  });
  s.addText(label, {
    x, y, w: d, h: d, fontFace: HF, fontSize: 14, bold: true, color: WHITE,
    align: "center", valign: "middle", margin: 0,
  });
}

function tbl(s, rows, opts) {
  const o = Object.assign({
    x: M, y: 1.4, w: W - 2 * M, fontFace: BF, fontSize: 10, color: INK,
    border: { type: "solid", pt: 0.5, color: LINE }, autoPage: false, valign: "middle",
  }, opts || {});
  s.addTable(rows, o);
}

function hcell(t, o) {
  return Object.assign({ text: t, options: Object.assign({ bold: true, color: WHITE, fill: { color: BERRY }, fontSize: 10, align: "center" }, o || {}) });
}

// group-separator row spanning the whole table
function grow(t, span, size) {
  return [{ text: t, options: { colspan: span, bold: true, color: BERRY, fill: { color: TINT }, fontSize: size || 9.5, align: "left" } }];
}

// The canonical ablation-ladder order used by every configuration table.
// Group A row i and Group B row i are a matched pair, identical except EVM.
const LADDER_A = [
  ["config_4", "EVM   (baseline)", true],
  ["config_13", "+ 3D-CNN", false],
  ["config_7", "+ Transformer", false],
  ["config_8", "+ SimAM   (proposed)", true],
  ["config_16", "EVM + 3D-CNN + SimAM", false],
  ["config_12", "EVM + Transformer", false],
];
const LADDER_B = [
  ["config_1", "(none)", false],
  ["config_3", "3D-CNN", false],
  ["config_9", "+ Transformer", false],
  ["config_6", "+ SimAM", false],
  ["config_5", "3D-CNN + SimAM", false],
  ["config_2", "Transformer", true],
];
const GA = "GROUP A — built up from the EVM baseline";
const GB = "GROUP B — the same ladder, EVM removed";

// ═════════════════════════════════════════════════════════════════════════════
// TITLE
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BERRY_DK };

  s.addText("MSc THESIS · FINAL PRESENTATION", {
    x: 0.9, y: 1.25, w: 10, h: 0.3, fontFace: BF, fontSize: 12, bold: true,
    color: ROSE, charSpacing: 2.2, margin: 0,
  });
  s.addText("Micro-Expression Recognition\non CASME-II", {
    x: 0.9, y: 1.7, w: 11.4, h: 1.7, fontFace: HF, fontSize: 40, bold: true,
    color: WHITE, lineSpacing: 46, margin: 0,
  });
  s.addText("A four-component ablation under full Leave-One-Subject-Out validation", {
    x: 0.9, y: 3.5, w: 11.0, h: 0.4, fontFace: BF, fontSize: 17, color: "E8D5DC", margin: 0,
  });

  const stats = [
    ["156", "clips, 3 classes"],
    ["25", "LOSO folds, all run"],
    ["300", "separate trainings"],
    ["50", "GPU-hours"],
  ];
  stats.forEach((st, i) => {
    const x = 0.9 + i * 2.95;
    s.addText(st[0], { x, y: 4.45, w: 2.6, h: 0.72, fontFace: HF, fontSize: 40, bold: true, color: WHITE, margin: 0 });
    s.addText(st[1], { x, y: 5.16, w: 2.6, h: 0.32, fontFace: BF, fontSize: 12, color: ROSE, margin: 0 });
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.9, y: 5.85, w: 11.4, h: 0.82, rectRadius: 0.06,
    fill: { color: "5A2439" }, line: { color: "5A2439", width: 0 },
  });
  s.addText(
    [
      { text: "Headline:  ", options: { bold: true, color: ROSE } },
      { text: "pooled accuracy 0.7500  ·  pooled macro F1 0.6659  ·  N = 156  ·  25/25 folds", options: { color: WHITE } },
    ],
    { x: 1.15, y: 5.85, w: 10.9, h: 0.82, fontFace: BF, fontSize: 15, valign: "middle", margin: 0 }
  );

  s.addNotes(
    "Good morning. My thesis is on recognising micro-expressions — those split-second flickers of emotion people can't control.\n\n" +
    "The short version of what I did: I took four techniques that the literature suggests help, and I tested every meaningful combination of them — twelve configurations — under the strictest evaluation protocol available for a dataset this small.\n\n" +
    "That came to three hundred separate model trainings and about fifty GPU-hours.\n\n" +
    "The headline is seventy-five per cent accuracy, which is ten points above the best published result on this dataset. But the more interesting finding is which of my four components actually earned its place, and I'll get to that."
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// 1 — Problem and data
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = base("The problem, and the data", "Background");

  s.addText(
    [
      { text: "What a micro-expression is.", options: { bold: true, color: BERRY } },
      { text: "  A facial movement lasting between 1/25 and 1/2 of a second. It is involuntary — too fast to fake, too fast to suppress. The model sees a short clip of a face and must answer Negative, Positive, or Surprise.", options: {} },
    ],
    { x: M, y: 1.5, w: 6.5, h: 1.25, fontFace: BF, fontSize: 14, color: INK, lineSpacing: 21, margin: 0, valign: "top" }
  );

  s.addText(
    [
      { text: "The model never sees the video.", options: { bold: true, color: BERRY } },
      { text: "  Every clip is converted into motion first — optical flow (which direction each pixel moved) and optical strain (how much the skin stretched). Raw pixels mostly encode who the person is and how the room is lit, and both are irrelevant. Motion does not care about identity or lighting.", options: {} },
      { text: "\n\nWith only 156 examples, that is the difference between learning expressions and learning faces.", options: { bold: true } },
    ],
    { x: M, y: 2.85, w: 6.5, h: 2.5, fontFace: BF, fontSize: 14, color: INK, lineSpacing: 21, margin: 0, valign: "top" }
  );

  tbl(s, [
    [hcell("The dataset", { align: "left" }), hcell("", { align: "right" })],
    [{ text: "CASME-II full label table" }, { text: "255 clips", options: { align: "right" } }],
    [{ text: "Kept: 3 affect classes (dropped 99 “others”)", options: { bold: true } }, { text: "156 clips", options: { align: "right", bold: true, color: BERRY } }],
    [{ text: "Subjects" }, { text: "25", options: { align: "right" } }],
    [{ text: "Each clip becomes" }, { text: "(3, 32, 224, 224)", options: { align: "right" } }],
  ], { x: 7.35, y: 1.5, w: 5.43, colW: [3.73, 1.7], fontSize: 12, rowH: 0.36 });

  const cls = [["99", "Negative"], ["32", "Positive"], ["25", "Surprise"]];
  cls.forEach((c, i) => {
    const x = 7.35 + i * 1.86;
    card(s, x, 3.75, 1.71, 1.05, TINT);
    s.addText(c[0], { x, y: 3.82, w: 1.71, h: 0.6, fontFace: HF, fontSize: 28, bold: true, color: BERRY, align: "center", margin: 0 });
    s.addText(c[1], { x, y: 4.4, w: 1.71, h: 0.3, fontFace: BF, fontSize: 12, color: MUTED, align: "center", margin: 0 });
  });

  card(s, 7.35, 5.0, 5.43, 1.55, TINT_T);
  s.addText(
    [
      { text: "Why 3 classes and not 7?", options: { bold: true, color: TEAL, breakLine: true } },
      { text: "The original labels are unusably sparse — fear has 2 clips in the whole dataset, sadness has 7. Grouping by emotional valence raises the smallest class from 2 to 25, and matches the 3-class setup published baselines use.", options: {} },
    ],
    { x: 7.6, y: 5.15, w: 4.93, h: 1.28, fontFace: BF, fontSize: 12, color: INK, lineSpacing: 17, margin: 0, valign: "top" }
  );

  s.addNotes(
    "A micro-expression is a flicker of emotion on someone's face that lasts a fraction of a second. People can't control them, which is why they're interesting.\n\n" +
    "My model has to sort clips into three buckets: negative, positive, surprise.\n\n" +
    "The one thing to remember from this slide: the model never looks at the actual video. I convert everything to motion first — where things moved and how the skin stretched. Why? Because with only 156 clips, if I gave it real pixels it would just memorise the 25 faces instead of learning expressions.\n\n" +
    "Also note the imbalance: 99 negative clips versus 25 surprise. That comes back to bite us."
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// 2 — Four technologies
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = base("The four technologies I am testing", "Method");

  const techs = [
    ["A", "EVM — motion magnification", "An amplifier for tiny motion. Boosts faint frame-to-frame changes ×10.", "Outside the network — it changes the data"],
    ["B", "SimAM — attention", "A spotlight on statistically unusual parts of the face. Zero learnable parameters.", "Inside the 3D-CNN"],
    ["C", "3D-CNN", "A shape detector. Looks at small patches of space and time together.", "Spatial part of the network"],
    ["D", "SLSTT Transformer", "A storyteller. Sees all 32 frames at once and models how the expression develops.", "Temporal part of the network"],
  ];
  techs.forEach((t, i) => {
    const y = 1.42 + i * 1.24;
    card(s, M, y, 6.35, 1.1, i % 2 === 0 ? TINT : "FAF7F8");
    chipCircle(s, M + 0.2, y + 0.17, 0.46, t[0]);
    s.addText(t[1], { x: M + 0.8, y: y + 0.12, w: 5.4, h: 0.3, fontFace: BF, fontSize: 13.5, bold: true, color: BERRY, margin: 0 });
    s.addText(t[2], { x: M + 0.8, y: y + 0.41, w: 5.4, h: 0.42, fontFace: BF, fontSize: 11.5, color: INK, margin: 0, lineSpacing: 15 });
    s.addText(t[3], { x: M + 0.8, y: y + 0.81, w: 5.4, h: 0.24, fontFace: BF, fontSize: 10.5, italic: true, color: MUTED, margin: 0 });
  });

  s.addImage({ path: IMG("report_figures_all_results/figM8_pipeline_procedure.png"), x: 7.2, y: 1.42, w: 5.58, h: 3.36 });
  s.addText("The full procedure. Stage 1 runs twice — once magnified, once raw — so the EVM switch just picks a folder.", {
    x: 7.2, y: 4.82, w: 5.58, h: 0.4, fontFace: BF, fontSize: 10.5, italic: true, color: MUTED, margin: 0,
  });

  card(s, 7.2, 5.32, 5.58, 1.28, TINT_T);
  s.addText(
    [
      { text: "The odd fact that shapes the whole study:", options: { bold: true, color: TEAL, breakLine: true } },
      { text: "the Transformer holds 371 000 parameters but runs almost instantly. The 3D-CNN holds only 18 000 — and eats 97 % of the compute.", options: {} },
    ],
    { x: 7.45, y: 5.45, w: 5.08, h: 1.02, fontFace: BF, fontSize: 12, color: INK, lineSpacing: 17, margin: 0, valign: "top" }
  );

  s.addText("12 configurations × 25 folds = 300 separate trainings ≈ 50 GPU-hours", {
    x: M, y: 6.45, w: 6.35, h: 0.42, fontFace: BF, fontSize: 13, bold: true, color: BERRY, margin: 0, valign: "middle",
  });

  s.addNotes(
    "Four techniques, four on/off switches. That's the whole experiment.\n\n" +
    "EVM is a magnifying glass for movement — it makes tiny facial twitches ten times bigger before I measure anything. It happens before training, so it changes the data, not the model.\n\n" +
    "SimAM is a spotlight that highlights unusual parts of the face, and it's free — it adds no weights to learn. With 156 clips, free is exactly what I want.\n\n" +
    "The 3D-CNN looks at shapes. The transformer looks at the story over time.\n\n" +
    "Odd detail worth remembering: the transformer is huge in size but cheap to run. The CNN is tiny but ate almost all my GPU time."
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// 3 — config_1
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = base("The baseline, and how the ablation is built on top of it", "Method");

  s.addText(
    [
      { text: "Four switches give 2⁴ = 16 combinations. But SimAM re-weights the 3D-CNN's output — with the CNN off there is nothing to re-weight, so the code prunes those 4 cells, leaving ", options: {} },
      { text: "12 valid configurations. All 12 were run.", options: { bold: true, color: BERRY } },
    ],
    { x: M, y: 1.3, w: 12.23, h: 0.5, fontFace: BF, fontSize: 12, color: INK, lineSpacing: 16, margin: 0, valign: "top" }
  );

  s.addText("The baseline is config_4 — EVM on, no network components", {
    x: M, y: 1.84, w: 6.35, h: 0.32, fontFace: HF, fontSize: 15.5, bold: true, color: BERRY, margin: 0,
  });
  s.addText(
    [
      { text: "EVM is not one of the things being ablated away — it is the starting point. ", options: { bold: true } },
      { text: "The baseline is my magnified-motion pipeline feeding the simplest possible classifier. Every other configuration is this baseline plus network components.", options: {} },
    ],
    { x: M, y: 2.2, w: 6.35, h: 0.78, fontFace: BF, fontSize: 11.5, color: INK, lineSpacing: 15.5, margin: 0, valign: "top" }
  );

  card(s, M, 3.0, 6.35, 1.5, "F3F0F1");
  s.addText(
    "[B, 3, 32, 224, 224]  EVM-magnified motion tensor\n" +
    "  → each frame averaged to 4×4 grid = 48 numbers/frame\n" +
    "  → Linear(48 → 96)      the ONLY spatial learning\n" +
    "  → average over 32 frames    destroys frame ORDER\n" +
    "  → LayerNorm → Dropout → Linear(96 → 3)",
    { x: M + 0.2, y: 3.09, w: 5.95, h: 1.32, fontFace: MF, fontSize: 9.5, color: INK, lineSpacing: 15.5, margin: 0, valign: "top" }
  );

  tbl(s, [
    [hcell("", { align: "left" }), hcell("config_4  (baseline)"), hcell("config_1  (EVM off)")],
    [{ text: "Pooled accuracy" }, { text: "0.4808", options: { align: "center", bold: true } }, { text: "0.4615", options: { align: "center" } }],
    [{ text: "Pooled macro F1" }, { text: "0.4386", options: { align: "center", bold: true, color: BERRY } }, { text: "0.4337", options: { align: "center" } }],
    [{ text: "Correct  ·  cost" }, { text: "75 / 156  ·  0.40 h", options: { align: "center" } }, { text: "72 / 156  ·  0.39 h", options: { align: "center" } }],
  ], { x: M, y: 4.6, w: 6.35, colW: [1.95, 2.2, 2.2], fontSize: 10.5, rowH: 0.32 });

  s.addText(
    [
      { text: "config_1 is not a second baseline — it is the control that isolates EVM. ", options: { bold: true, color: BERRY } },
      { text: "The two are architecturally identical; only the tensor folder differs. The gap between them, +0.0049, is the purest EVM measurement in the study. And 4 of the other 11 configurations score below the baseline — three of them while costing 15× more to train.", options: {} },
    ],
    { x: M, y: 5.98, w: 6.35, h: 0.9, fontFace: BF, fontSize: 11, color: INK, lineSpacing: 14.5, margin: 0, valign: "top" }
  );

  s.addText("How every results table in this deck is ordered — not by config number", {
    x: 7.3, y: 1.84, w: 5.48, h: 0.3, fontFace: HF, fontSize: 13.5, bold: true, color: BERRY, margin: 0,
  });

  const ladderRows = [[hcell("#", { align: "center" }), hcell("Config", { align: "left" }), hcell("Components", { align: "left" })]];
  ladderRows.push(grow(GA, 3, 9));
  LADDER_A.forEach((r, i) => {
    ladderRows.push([
      { text: String(i + 1), options: { align: "center", color: MUTED } },
      { text: r[0], options: { bold: r[2], color: r[2] ? BERRY : INK } },
      { text: r[1], options: { bold: r[2] } },
    ]);
  });
  ladderRows.push(grow(GB, 3, 9));
  LADDER_B.forEach((r, i) => {
    ladderRows.push([
      { text: String(i + 7), options: { align: "center", color: MUTED } },
      { text: r[0], options: { bold: r[2], color: r[2] ? BERRY : INK } },
      { text: r[1], options: { bold: r[2] } },
    ]);
  });
  tbl(s, ladderRows, { x: 7.3, y: 2.2, w: 5.48, colW: [0.45, 1.33, 3.7], fontSize: 9.5, rowH: 0.275 });

  card(s, 7.3, 6.16, 5.48, 0.72, TINT_T);
  s.addText(
    [
      { text: "Row n of Group A and row n of Group B are a matched pair", options: { bold: true, color: TEAL } },
      { text: " — identical in everything except EVM. That alignment is what makes the EVM effect readable at a glance.", options: {} },
    ],
    { x: 7.52, y: 6.24, w: 5.04, h: 0.56, fontFace: BF, fontSize: 10.5, color: INK, lineSpacing: 14, margin: 0, valign: "top" }
  );

  s.addNotes(
    "Sixteen combinations on paper, but four of them are nonsense — SimAM needs the CNN to exist. So: twelve configurations, all twelve run.\n\n" +
    "Now the important framing. My baseline is config_4 — EVM switched on, and nothing else. EVM isn't one of the things I'm testing away; it's part of my data pipeline, so it's the starting point. Everything else in this study is that baseline plus network components.\n\n" +
    "The baseline model itself is deliberately simple: it shrinks each frame to a blurry four-by-four thumbnail, averages all 32 frames together, and runs a linear classifier. About five thousand numbers to learn, versus nearly four hundred thousand for the full model. Because it averages the frames, it literally cannot tell whether a smile was starting or ending.\n\n" +
    "config_1 is the same model with EVM switched off. It's not a second baseline — it's the control that isolates EVM, and the gap between the two, about half a point, is the cleanest EVM measurement I have.\n\n" +
    "Last thing, and it applies to every table from here on: I order them as a ladder — baseline, then add the CNN, then add the transformer, then add SimAM. Not by config number. Group A is the ladder with EVM, group B is the same ladder without it, lined up row for row.\n\n" +
    "And the punchline: four of my eleven other configurations score below the baseline. Three of those cost fifteen times more to train."
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// 4 — LOSO
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = base("How everything is tested: Leave-One-Subject-Out", "Method");

  card(s, M, 1.35, 6.4, 1.35, TINT);
  s.addText(
    [
      { text: "A simple train/test split fails here.", options: { bold: true, color: BERRY, breakLine: true } },
      { text: "1.  The answer depends on who you happened to hold back — draw easy subjects and the score flatters you.", options: { breakLine: true } },
      { text: "2.  The model can cheat — same person in train and test means it can memorise the face.", options: {} },
    ],
    { x: M + 0.22, y: 1.46, w: 5.96, h: 1.15, fontFace: BF, fontSize: 12, color: INK, lineSpacing: 16.5, margin: 0, valign: "top" }
  );

  const steps = [
    ["1", "Set aside subject 1. Train a brand-new model on the other 24. Predict subject 1's clips. Throw that model away."],
    ["2", "Set aside subject 2. Train another brand-new model. Repeat."],
    ["3", "25 times — every subject held out exactly once."],
    ["4", "Pool all 25 sets of predictions. Every clip now has one prediction, made by a model that never saw that face."],
  ];
  steps.forEach((st, i) => {
    const y = 2.95 + i * 0.72;
    chipCircle(s, M, y, 0.42, st[0]);
    s.addText(st[1], { x: M + 0.62, y: y - 0.05, w: 5.78, h: 0.62, fontFace: BF, fontSize: 12, color: INK, lineSpacing: 16, margin: 0, valign: "middle" });
  });

  s.addImage({ path: IMG("report_figures_all_results/figM7_loso_procedure.png"), x: 7.15, y: 1.35, w: 5.63, h: 2.84 });
  s.addText("Left: the 25-fold schedule — orange is the held-out subject. Right: what each fold actually tests.", {
    x: 7.15, y: 4.22, w: 5.63, h: 0.36, fontFace: BF, fontSize: 10.5, italic: true, color: MUTED, margin: 0,
  });

  card(s, 7.15, 4.62, 5.63, 1.0, TINT_T);
  s.addText(
    [
      { text: "The folds are very unequal.  ", options: { bold: true, color: TEAL } },
      { text: "Subject 17 alone supplies 33 of the 156 clips. Three subjects supply one clip each. And 10 of the 25 folds contain only one class — that causes a problem on the next slide.", options: {} },
    ],
    { x: 7.4, y: 4.72, w: 5.13, h: 0.82, fontFace: BF, fontSize: 11.5, color: INK, lineSpacing: 15.5, margin: 0, valign: "top" }
  );

  tbl(s, [
    [hcell("Held fixed across all 12 configurations", { align: "left" }), hcell("", { align: "left" })],
    [{ text: "Protocol" }, { text: "loso · 25/25 folds · N = 156 · seed 42 re-applied every fold" }],
    [{ text: "Imbalance" }, { text: "balanced sampler ON, loss class weights OFF (both together caused collapse)" }],
    [{ text: "Loss / optimiser" }, { text: "Focal Loss γ=2.0 · AdamW lr 1e-4 · 50 epochs · batch 8 · AMP" }],
    [{ text: "Checkpoint kept" }, { text: "the epoch with the best validation macro F1" }],
  ], { x: M, y: 5.75, w: 12.23, colW: [2.4, 9.83], fontSize: 11, rowH: 0.27 });

  s.addNotes(
    "Normally you'd hold back a chunk of data and test on it. With only 25 people that doesn't work — your score depends on which faces you happened to hold back, and it can swing wildly.\n\n" +
    "So instead I do this: hold out one person, train a fresh model on the other 24, test on that one person, then delete the model. Then do it again for the next person. Twenty-five times.\n\n" +
    "At the end, every single clip has been predicted by a model that never saw that person's face. No lucky split, no cheating. The cost is that I train twenty-five models instead of one.\n\n" +
    "One thing to notice in the picture — the folds are lopsided. Subject 17 alone has 33 clips. Ten folds contain only one emotion class. That causes a measurement problem, which is the next slide."
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// 5 — Metrics
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = base("Which number is the number", "Measurement");

  s.addText("Four aggregate numbers exist per configuration — and two of them are recorded under misleading names.", {
    x: M, y: 1.3, w: 12.23, h: 0.32, fontFace: BF, fontSize: 13, color: INK, margin: 0,
  });

  card(s, M, 1.75, 6.05, 2.75, TINT_T);
  s.addText("✓   The two I use — both “pooled”", { x: M + 0.25, y: 1.87, w: 5.55, h: 0.32, fontFace: BF, fontSize: 13.5, bold: true, color: TEAL, margin: 0 });
  s.addText(
    [
      { text: "Pooled accuracy", options: { bold: true, color: INK, breakLine: true } },
      { text: "Line up all 156 clips, count how many got the right label, divide by 156. Every clip counts exactly once.\n\n", options: { breakLine: true } },
      { text: "Pooled macro F1", options: { bold: true, color: INK, breakLine: true } },
      { text: "One confusion matrix from all 156 predictions, an F1 per class, averaged without weighting by class size — so it catches a model that quietly abandons the 25-clip Surprise class.", options: {} },
    ],
    { x: M + 0.25, y: 2.22, w: 5.55, h: 2.15, fontFace: BF, fontSize: 11.5, color: INK, lineSpacing: 15.5, margin: 0, valign: "top" }
  );

  card(s, 6.95, 1.75, 5.83, 2.75, TINT);
  s.addText("✗   The two I refuse — both “mean-of-folds”", { x: 7.2, y: 1.87, w: 5.33, h: 0.32, fontFace: BF, fontSize: 13.5, bold: true, color: BERRY, margin: 0 });
  s.addText(
    [
      { text: "Mean-of-folds accuracy", options: { bold: true, color: INK, breakLine: true } },
      { text: "Subject 8's single clip gets the same 1/25 weight as subject 17's thirty-three — inflating config_8 by 6.3 points (0.8130 vs 0.7500 true).\n\n", options: { breakLine: true } },
      { text: "Mean-of-folds macro F1", options: { bold: true, color: INK, breakLine: true } },
      { text: "Mathematically capped at 0.6267 — below my 0.68 target. 10 folds contain only 1 class, so 2 of the 3 F1s are forced to zero there, even for a perfect classifier.", options: {} },
    ],
    { x: 7.2, y: 2.22, w: 5.33, h: 2.15, fontFace: BF, fontSize: 11.5, color: INK, lineSpacing: 15.5, margin: 0, valign: "top" }
  );

  tbl(s, [
    [hcell("The floors every result must beat", { align: "left" }), hcell("Accuracy"), hcell("Macro F1")],
    [{ text: "Always predict “Negative” (the majority class)", options: { bold: true } }, { text: "0.6346", options: { align: "center", bold: true } }, { text: "0.2588", options: { align: "center", bold: true } }],
    [{ text: "Predict at random" }, { text: "0.3333", options: { align: "center" } }, { text: "≈ 0.303", options: { align: "center" } }],
    [{ text: "My dissertation target", options: { bold: true, color: BERRY } }, { text: "0.70", options: { align: "center", bold: true, color: BERRY } }, { text: "0.68", options: { align: "center", bold: true, color: BERRY } }],
    [{ text: "Best published LOSO baseline on CASME-II" }, { text: "0.65", options: { align: "center" } }, { text: "not reported", options: { align: "center", color: MUTED } }],
  ], { x: M, y: 4.72, w: 7.6, colW: [4.6, 1.5, 1.5], fontSize: 11, rowH: 0.32 });

  card(s, 8.4, 4.72, 4.38, 1.9, "F3F0F1");
  s.addText(
    [
      { text: "Accuracy goes in the abstract because anyone understands it — but it must never stand alone.", options: { bold: true, color: BERRY, breakLine: true } },
      { text: "\nA model that ignores the video and always says “Negative” gets 63 % accuracy while being useless. Macro F1 scores that same trick at 0.26.", options: {} },
    ],
    { x: 8.65, y: 4.85, w: 3.88, h: 1.65, fontFace: BF, fontSize: 11.5, color: INK, lineSpacing: 15.5, margin: 0, valign: "top" }
  );

  s.addNotes(
    "One minute of definitions, and then we're into results.\n\n" +
    "Accuracy is easy: out of 156 clips, how many did it get right.\n\n" +
    "Macro F1 is the one that actually decides which model is better. It scores each of the three emotions separately and then averages them equally — so a model that ignores the rare “surprise” class gets punished, even if its overall accuracy looks fine.\n\n" +
    "Here's why that matters. If I just always guessed “Negative”, I'd get 63 % accuracy — because 99 of 156 clips are negative. Sounds decent. But macro F1 gives that trick 0.26. That's the difference between the two numbers.\n\n" +
    "The last thing: two of the four numbers in my results files are computed per-fold and then averaged, and both are broken. One of them is mathematically incapable of reaching my target, even for a perfect model. I don't use those."
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// 6 — All 12 results
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = base("All 12 results — the ablation ladder", "Results");
  s.addText("25/25 folds · N = 156 · 50 epochs · seed 42 · ordered as an ablation ladder from the EVM baseline, not by config number", {
    x: M, y: 1.28, w: 12.23, h: 0.28, fontFace: BF, fontSize: 12, italic: true, color: MUTED, margin: 0,
  });

  const A_RES = [
    ["0.4808", "0.4386", "75", "—"],
    ["0.4359", "0.4480", "68", "+0.009"],
    ["0.7051 ✓", "0.6625", "110", "+0.224"],
    ["0.7500 ✓", "0.6659", "117", "+0.227"],
    ["0.4038", "0.4192", "63", "−0.019"],
    ["0.6795", "0.6581", "106", "+0.220"],
  ];
  const B_RES = [
    ["0.4615", "0.4337", "72", "—"],
    ["0.4167", "0.4252", "65", "−0.009"],
    ["0.7308 ✓", "0.5830", "114", "+0.149"],
    ["0.7308 ✓", "0.6171", "114", "+0.183"],
    ["0.4231", "0.4302", "66", "−0.004"],
    ["0.7436 ✓", "0.7122 ✓", "116", "+0.279"],
  ];
  const bigJump = (d) => d.startsWith("+0.1") || d.startsWith("+0.2");

  const rows = [[
    hcell("Configuration", { align: "left" }), hcell("Components", { align: "left" }),
    hcell("Accuracy"), hcell("Macro F1"), hcell("Correct"), hcell("Δ vs. baseline"),
  ]];
  const addLadder = (label, ladder, res) => {
    rows.push(grow(label, 6, 9.5));
    ladder.forEach((r, i) => {
      const v = res[i];
      rows.push([
        { text: r[0], options: { bold: r[2], color: r[2] ? BERRY : INK } },
        { text: r[1], options: { bold: r[2] } },
        { text: v[0], options: { align: "center", bold: r[2] } },
        { text: v[1], options: { align: "center", bold: true, color: bigJump(v[3]) ? TEAL : INK } },
        { text: v[2], options: { align: "center" } },
        { text: v[3], options: { align: "center", bold: bigJump(v[3]), color: bigJump(v[3]) ? TEAL : MUTED } },
      ]);
    });
  };
  addLadder(GA, LADDER_A, A_RES);
  addLadder(GB, LADDER_B, B_RES);
  rows.push([
    { text: "always-Negative reference", options: { italic: true, color: MUTED } }, { text: "", options: {} },
    { text: "0.6346", options: { align: "center", italic: true, color: MUTED } },
    { text: "0.2588", options: { align: "center", italic: true, color: MUTED } },
    { text: "99", options: { align: "center", italic: true, color: MUTED } }, { text: "", options: {} },
  ]);
  rows.push([
    { text: "dissertation target", options: { italic: true, bold: true, color: BERRY } }, { text: "", options: {} },
    { text: "0.70", options: { align: "center", bold: true, color: BERRY } },
    { text: "0.68", options: { align: "center", bold: true, color: BERRY } },
    { text: "—", options: { align: "center", color: MUTED } }, { text: "", options: {} },
  ]);

  tbl(s, rows, { x: M, y: 1.62, w: 8.85, colW: [1.6, 3.1, 1.15, 1.15, 0.8, 1.05], fontSize: 9.5, rowH: 0.29 });

  card(s, 9.62, 1.62, 3.16, 2.6, TINT_T);
  s.addText(
    [
      { text: "Read the ladder, not the rows.", options: { bold: true, color: TEAL, breakLine: true } },
      { text: "\nAdding the 3D-CNN to either baseline moves macro F1 by less than 0.01.\n\nAdding the Transformer on top of it moves it by +0.21 to +0.22.\n\nAdding SimAM last moves it by +0.003 and +0.034.", options: {} },
    ],
    { x: 9.87, y: 1.74, w: 2.66, h: 2.36, fontFace: BF, fontSize: 11, color: INK, lineSpacing: 15, margin: 0, valign: "top" }
  );

  card(s, 9.62, 4.37, 3.16, 1.2, TINT);
  s.addText(
    [{ text: "Every configuration with the Transformer scores 0.583 – 0.712. Every one without it scores 0.419 – 0.448. ", options: {} }, { text: "No overlap, 12 of 12.", options: { bold: true, color: BERRY } }],
    { x: 9.87, y: 4.47, w: 2.66, h: 1.0, fontFace: BF, fontSize: 11, color: INK, lineSpacing: 15, margin: 0, valign: "top" }
  );
  card(s, 9.62, 5.67, 3.16, 1.2, TINT);
  s.addText(
    [{ text: "config_8", options: { bold: true, color: BERRY } }, { text: " tops accuracy at 0.7500. ", options: {} }, { text: "config_2", options: { bold: true, color: BERRY } }, { text: " tops macro F1 and is the only configuration to clear both targets.", options: {} }],
    { x: 9.87, y: 5.77, w: 2.66, h: 1.0, fontFace: BF, fontSize: 11, color: INK, lineSpacing: 15, margin: 0, valign: "top" }
  );

  s.addNotes(
    "This table is ordered the way the experiment was designed — start at the baseline and add one thing at a time.\n\n" +
    "Group A is the ladder with EVM. Baseline, then add the CNN, then add the transformer, then add SimAM — that top-to-bottom walk is my proposed model being assembled. The last two rows are the side branches: leave the transformer out, or leave the CNN out.\n\n" +
    "Group B is the exact same ladder with EVM switched off, lined up row for row, so any pair of rows tells you what EVM did.\n\n" +
    "Now watch what happens as you climb. Add the CNN — nothing, less than a hundredth. Add the transformer — plus 0.22. Add SimAM — three thousandths.\n\n" +
    "And across both groups: everything with a transformer scores between 0.58 and 0.71. Everything without one scores between 0.42 and 0.45. No overlap, twelve out of twelve.\n\n" +
    "Two rows to notice. My full model gets the best accuracy — 75 %, 117 clips. But the transformer on its own, at the bottom, gets the best macro F1 and is the only configuration to hit both targets. I'll deal with that head-on next."
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// 7 — C8 vs C2
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = base("The proposed model vs. the Transformer alone", "Results");

  tbl(s, [
    [hcell("", { align: "left" }), hcell("config_8 — proposed"), hcell("config_2 — Transformer only")],
    [{ text: "Components", options: { bold: true } }, { text: "EVM + SimAM + 3D-CNN + Transformer", options: { align: "center" } }, { text: "Transformer only", options: { align: "center" } }],
    [{ text: "Accuracy", options: { bold: true } }, { text: "0.7500 ✓   (highest in study)", options: { align: "center", bold: true, color: BERRY } }, { text: "0.7436 ✓", options: { align: "center" } }],
    [{ text: "Macro F1", options: { bold: true } }, { text: "0.6659   (misses 0.68 by 0.014)", options: { align: "center" } }, { text: "0.7122 ✓   (highest in study)", options: { align: "center", bold: true, color: TEAL } }],
    [{ text: "Correct", options: { bold: true } }, { text: "117 / 156", options: { align: "center", bold: true } }, { text: "116 / 156", options: { align: "center", bold: true } }],
    [{ text: "F1  Negative / Positive / Surprise", options: { bold: true } }, { text: "0.846  /  0.556  /  0.596", options: { align: "center" } }, { text: "0.807  /  0.651  /  0.679", options: { align: "center" } }],
    [{ text: "Positive clips recovered", options: { bold: true } }, { text: "15 of 32", options: { align: "center" } }, { text: "27 of 32", options: { align: "center", bold: true, color: TEAL } }],
    [{ text: "Cost (25 folds)", options: { bold: true } }, { text: "6.46 GPU-h · 19.57 GB", options: { align: "center" } }, { text: "0.48 GPU-h · 0.17 GB", options: { align: "center", bold: true, color: TEAL } }],
  ], { x: M, y: 1.35, w: 12.23, colW: [3.43, 4.4, 4.4], fontSize: 12, rowH: 0.36 });

  card(s, M, 4.4, 6.0, 1.75, TINT);
  s.addText(
    [
      { text: "config_8 is a majority-class specialist.", options: { bold: true, color: BERRY, breakLine: true } },
      { text: "85 of the 99 Negative clips right — the best Negative F1 anywhere. Accuracy rewards exactly that, because Negatives are 99 of the 156 clips. Its one weakness — 13 of 32 Positive clips read as Negative — is what costs it the macro-F1 target.", options: {} },
    ],
    { x: M + 0.25, y: 4.52, w: 5.5, h: 1.5, fontFace: BF, fontSize: 11.5, color: INK, lineSpacing: 15.5, margin: 0, valign: "top" }
  );

  card(s, 6.78, 4.4, 6.0, 1.75, TINT_T);
  s.addText(
    [
      { text: "config_2 spreads its errors evenly.", options: { bold: true, color: TEAL, breakLine: true } },
      { text: "0.807 / 0.651 / 0.679 is the tightest three-class spread in the study — and it achieves it at 7 % of the compute and under 1 % of the VRAM.\n\nUnder my earlier protocol, config_8 scored 0.000 on Positive. That was a training bug, not an architectural limit.", options: {} },
    ],
    { x: 7.03, y: 4.52, w: 5.5, h: 1.5, fontFace: BF, fontSize: 11.5, color: INK, lineSpacing: 15.5, margin: 0, valign: "top" }
  );

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 6.3, w: 12.23, h: 0.56, rectRadius: 0.06, fill: { color: BERRY }, line: { color: BERRY, width: 0 },
  });
  s.addText("They differ by exactly one clip — 117 vs 116. At N = 156 the 95 % confidence interval is ± 0.068, so they are not statistically distinguishable.", {
    x: M + 0.25, y: 6.3, w: 11.73, h: 0.56, fontFace: BF, fontSize: 13, bold: true, color: WHITE, valign: "middle", margin: 0,
  });

  s.addNotes(
    "Here's the uncomfortable comparison. I'd rather show it than have someone find it.\n\n" +
    "My four-component model gets 117 clips right. The transformer on its own gets 116. One clip. With only 156 clips, the error bar is about plus or minus seven points — so statistically, these two are the same model. I'm not going to claim otherwise.\n\n" +
    "What's different is how they're right. My model is really good at the common class — 85 of 99 negatives. Accuracy loves that, because negatives are most of the dataset. But it only catches 15 of the 32 positive clips, and that's what costs it the macro F1 target.\n\n" +
    "The transformer alone is more even across all three emotions. And it does it on seven per cent of the computing power.\n\n" +
    "One more thing: in my earlier experiments this same model scored a flat zero on the positive class — never predicted it once. That turned out to be a bug in how I corrected the imbalance, not a flaw in the architecture."
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// 8 — Component contributions
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = base("What each component actually contributed", "Results");
  s.addText("Every number comes from matched pairs — two configurations identical except for one switch. That is the only fair way to isolate a component.", {
    x: M, y: 1.26, w: 12.23, h: 0.3, fontFace: BF, fontSize: 12.5, color: INK, margin: 0,
  });

  tbl(s, [
    [hcell("Technology", { align: "left" }), hcell("Mean effect on macro F1"), hcell("Consistency"), hcell("Verdict", { align: "left" })],
    [{ text: "SLSTT Transformer", options: { bold: true, color: BERRY } }, { text: "+0.217", options: { align: "center", bold: true, color: TEAL } }, { text: "positive in 6 of 6 pairs, min +0.158", options: { align: "center" } }, { text: "Decisive — the only component that matters", options: { bold: true } }],
    [{ text: "EVM" }, { text: "+0.015", options: { align: "center" } }, { text: "4 of 6 positive", options: { align: "center" } }, { text: "Small, below noise — but measured for the first time" }],
    [{ text: "SimAM" }, { text: "+0.003", options: { align: "center" } }, { text: "flat", options: { align: "center" } }, { text: "Neutral, but free (0 parameters). Keep it" }],
    [{ text: "3D-CNN" }, { text: "−0.031", options: { align: "center", bold: true, color: "B3261E" } }, { text: "negative, worst case −0.129", options: { align: "center" } }, { text: "Drop it — negative effect, 97 % of the GPU budget", options: { bold: true } }],
  ], { x: M, y: 1.62, w: 12.23, colW: [2.5, 2.2, 3.2, 4.33], fontSize: 11.5, rowH: 0.38 });

  s.addImage({ path: IMG("report_figures_loso/figL8_transformer_split.png"), x: M, y: 3.58, w: 6.5, h: 2.54 });
  s.addText(
    [
      { text: "Left: the two groups, separated by an empty gap. Right: the same six pairs under the old protocol (erratic) vs. full LOSO (all positive).", options: { breakLine: true } },
      { text: "The averages hide something real: config_9 catches only 6 of 25 Surprise clips (F1 0.300). Add SimAM → 0.590. Add EVM → 0.655. Both “useless” components fix the same failure.", options: { color: INK } },
    ],
    { x: M, y: 6.16, w: 6.5, h: 0.72, fontFace: BF, fontSize: 9.5, italic: true, color: MUTED, lineSpacing: 12.5, margin: 0, valign: "top" }
  );

  const blocks = [
    ["Why the Transformer works", "It is the only component that sees all 32 frames at once. A micro-expression is defined by its arc — neutral, peak, relaxation. Self-attention compares frame 5 with frame 20 directly. With it off, time is collapsed by averaging, which erases the peak.", TINT_T, 1.15],
    ["Why the 3D-CNN fails", "The input is already a motion representation — optical flow has done most of the work it would learn. And 156 clips cannot train a convolutional extractor from scratch.", TINT, 0.9],
    ["A defect I found and repaired", "In every earlier run each EVM-on configuration was bit-identical to its EVM-off twin — both arms read the same tensor folder. In this run all six pairs differ: the first genuine EVM measurement in the project's history.", TINT, 1.0],
  ];
  let by = 3.58;
  blocks.forEach((b) => {
    card(s, 7.3, by, 5.48, b[3], b[2]);
    s.addText(
      [{ text: b[0] + "   ", options: { bold: true, color: b[2] === TINT_T ? TEAL : BERRY } }, { text: b[1], options: {} }],
      { x: 7.53, y: by + 0.08, w: 5.0, h: b[3] - 0.16, fontFace: BF, fontSize: 11, color: INK, lineSpacing: 14.5, margin: 0, valign: "top" }
    );
    by += b[3] + 0.1;
  });

  s.addNotes(
    "This is the actual ablation. Each number comes from comparing two configurations that differ by exactly one switch — that's the fair way to do it.\n\n" +
    "The transformer: plus 0.217, and positive in all six comparisons without exception. The picture shows the two groups separated by an empty gap.\n\n" +
    "Why does it work? It's the only piece that sees all 32 frames at once. A micro-expression is a little story — neutral, peak, back to neutral. The transformer can compare the beginning to the middle. Everything else just averages the frames, which wipes out the peak.\n\n" +
    "The 3D-CNN: minus 0.03, and it ate 97 % of my GPU time. It's trying to redo work that the optical flow step already did.\n\n" +
    "And a correction to my own earlier work: in every previous run, EVM was doing literally nothing — both settings were reading the same files. I found that and fixed it. This is the first time EVM has actually been measured."
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// 9 — Per-class
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = base("Per-class behaviour: nobody abandons a class", "Results");

  const A_PC = [
    ["0.564", "0.505", "0.246", "0.4386"],
    ["0.413", "0.528", "0.404", "0.4480"],
    ["0.785", "0.548", "0.655", "0.6625"],
    ["0.846", "0.556", "0.596", "0.6659"],
    ["0.350", "0.514", "0.393", "0.4192"],
    ["0.749", "0.559", "0.667", "0.6581"],
  ];
  const B_PC = [
    ["0.560", "0.371", "0.370", "0.4337"],
    ["0.361", "0.523", "0.392", "0.4252"],
    ["0.849", "0.600", "0.300", "0.5830"],
    ["0.833", "0.429", "0.590", "0.6171"],
    ["0.384", "0.523", "0.384", "0.4302"],
    ["0.807", "0.651", "0.679", "0.7122"],
  ];
  const rows = [[hcell("Config", { align: "left" }), hcell("Components", { align: "left" }), hcell("F1 Neg"), hcell("F1 Pos"), hcell("F1 Sur"), hcell("Macro F1")]];
  const addPC = (label, ladder, res) => {
    rows.push(grow(label, 6, 9));
    ladder.forEach((r, i) => {
      const v = res[i];
      const low = v[2] === "0.246" || v[2] === "0.300";
      rows.push([
        { text: r[0], options: { bold: r[2], color: r[2] ? BERRY : INK } },
        { text: r[1], options: { bold: r[2] } },
        { text: v[0], options: { align: "center" } },
        { text: v[1], options: { align: "center" } },
        { text: v[2], options: { align: "center", bold: low, color: low ? "B3261E" : INK } },
        { text: v[3], options: { align: "center", bold: true, color: r[2] ? BERRY : INK } },
      ]);
    });
  };
  addPC(GA, LADDER_A, A_PC);
  addPC(GB, LADDER_B, B_PC);
  tbl(s, rows, { x: M, y: 1.4, w: 8.3, colW: [1.35, 2.35, 1.15, 1.15, 1.15, 1.15], fontSize: 9.5, rowH: 0.3 });

  card(s, 9.15, 1.4, 3.63, 1.85, TINT_T);
  s.addText(
    [
      { text: "No configuration abandons a class.", options: { bold: true, color: TEAL, breakLine: true } },
      { text: "\nThe lowest per-class F1 anywhere is 0.246. In my earlier holdout runs the proposed model scored 0.000 on Positive — it never predicted the class once. That failure mode is gone.", options: {} },
    ],
    { x: 9.4, y: 1.52, w: 3.13, h: 1.6, fontFace: BF, fontSize: 11, color: INK, lineSpacing: 15, margin: 0, valign: "top" }
  );

  card(s, 9.15, 3.4, 3.63, 2.0, TINT);
  s.addText(
    [
      { text: "The Surprise column is where the two “useless” components show up.", options: { bold: true, color: BERRY, breakLine: true } },
      { text: "\nconfig_9 catches only 6 of 25 Surprise clips (F1 0.300). Add SimAM → 0.590. Add EVM instead → 0.655. Both fix the same specific failure.", options: {} },
    ],
    { x: 9.4, y: 3.52, w: 3.13, h: 1.76, fontFace: BF, fontSize: 11, color: INK, lineSpacing: 15, margin: 0, valign: "top" }
  );

  card(s, 9.15, 5.55, 3.63, 1.32, "F3F0F1");
  s.addText(
    [
      { text: "Careful with precision alone. ", options: { bold: true, color: BERRY } },
      { text: "config_16 has perfect Negative precision (1.000) only because it risks that label 21 times out of 156. Recall 0.212.", options: {} },
    ],
    { x: 9.4, y: 5.66, w: 3.13, h: 1.1, fontFace: BF, fontSize: 11, color: INK, lineSpacing: 15, margin: 0, valign: "top" }
  );

  s.addNotes(
    "This table shows how each model handles each emotion separately.\n\n" +
    "The important result here is a negative one: none of my twelve models gives up on a class. The worst single score anywhere is 0.246. In my earlier experiments, the proposed model scored a flat zero on the positive class — it never predicted it once. That's completely gone now.\n\n" +
    "Two rows to glance at. config_2 is the most even — roughly 0.8, 0.65, 0.68. config_9 is great at negatives but only catches six of twenty-five surprise clips, and that one weakness drags it to the bottom of its group.\n\n" +
    "And a warning: one model has perfect precision on negatives. Sounds amazing. It got there by only ever guessing “negative” 21 times out of 156. That's not a good model, that's a model refusing to answer."
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// 10 — Cost
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = base("What it all cost", "Results");

  const A_C = [
    ["0.40 h", "0.16 GB", "5.8 k", "0.4386", "1.10"],
    ["5.78 h", "14.40 GB", "18 k", "0.4480", "0.078"],
    ["5.78 h", "14.40 GB", "384 k", "0.6625", "0.115"],
    ["6.46 h", "19.57 GB", "384 k", "0.6659", "0.103"],
    ["6.43 h", "19.56 GB", "18 k", "0.4192", "0.065"],
    ["0.47 h", "0.17 GB", "371 k", "0.6581", "1.40"],
  ];
  const B_C = [
    ["0.39 h", "0.16 GB", "5.8 k", "0.4337", "1.11"],
    ["5.77 h", "14.40 GB", "18 k", "0.4252", "0.074"],
    ["5.78 h", "14.40 GB", "384 k", "0.5830", "0.101"],
    ["6.44 h", "19.57 GB", "384 k", "0.6171", "0.096"],
    ["6.42 h", "19.56 GB", "18 k", "0.4302", "0.067"],
    ["0.48 h", "0.17 GB", "371 k", "0.7122", "1.48"],
  ];
  const rows = [[hcell("Config", { align: "left" }), hcell("Components", { align: "left" }), hcell("Sweep"), hcell("Peak VRAM"), hcell("≈ params"), hcell("Macro F1"), hcell("F1 / GPU-h")]];
  const addC = (label, ladder, res) => {
    rows.push(grow(label, 7, 9));
    ladder.forEach((r, i) => {
      const v = res[i];
      rows.push([
        { text: r[0], options: { bold: r[2], color: r[2] ? BERRY : INK } },
        { text: r[1], options: { bold: r[2] } },
        { text: v[0], options: { align: "center", bold: r[2] } },
        { text: v[1], options: { align: "center", bold: r[2] } },
        { text: v[2], options: { align: "center" } },
        { text: v[3], options: { align: "center", bold: true } },
        { text: v[4], options: { align: "center", bold: r[2], color: r[2] ? TEAL : INK } },
      ]);
    });
  };
  addC(GA, LADDER_A, A_C);
  addC(GB, LADDER_B, B_C);
  rows.push([
    { text: "total", options: { italic: true, bold: true, color: MUTED } }, { text: "", options: {} },
    { text: "≈ 50.6 GPU-h", options: { colspan: 5, align: "left", italic: true, bold: true, color: MUTED } },
  ]);
  tbl(s, rows, { x: M, y: 1.4, w: 8.3, colW: [1.25, 2.1, 1.0, 1.15, 0.9, 0.95, 0.95], fontSize: 9.5, rowH: 0.295 });

  card(s, 9.15, 1.4, 3.63, 2.05, TINT_T);
  s.addText(
    [
      { text: "The cost of the ladder is entirely one rung.", options: { bold: true, color: TEAL, breakLine: true } },
      { text: "\nBaseline → + 3D-CNN costs 14× the time and 90× the memory, for +0.009 macro F1.\n\n+ Transformer on top is free by comparison — and buys +0.21.", options: {} },
    ],
    { x: 9.4, y: 1.52, w: 3.13, h: 1.8, fontFace: BF, fontSize: 11, color: INK, lineSpacing: 15, margin: 0, valign: "top" }
  );

  card(s, 9.15, 3.6, 3.63, 1.65, TINT);
  s.addText(
    [
      { text: "The inversion.", options: { bold: true, color: BERRY, breakLine: true } },
      { text: "\nThe Transformer is 371 k parameters and runs instantly — with the CNN off it only processes a 4 × 4 patch grid. The 3D-CNN is 18 k parameters and dominates: 85× the VRAM, 12× the time.", options: {} },
    ],
    { x: 9.4, y: 3.72, w: 3.13, h: 1.42, fontFace: BF, fontSize: 11, color: INK, lineSpacing: 15, margin: 0, valign: "top" }
  );

  s.addShape(pres.ShapeType.roundRect, {
    x: 9.15, y: 5.4, w: 3.63, h: 1.45, rectRadius: 0.06, fill: { color: BERRY }, line: { color: BERRY, width: 0 },
  });
  s.addText("The eight 3D-CNN configurations consumed 48.9 of the 50.6 GPU-hours — 97 % of the budget — for a component whose measured effect is −0.031.", {
    x: 9.4, y: 5.52, w: 3.13, h: 1.2, fontFace: BF, fontSize: 11.5, bold: true, color: WHITE, lineSpacing: 15.5, margin: 0, valign: "top",
  });

  s.addNotes(
    "Short slide, one point.\n\n" +
    "The best-scoring configuration is also nearly the cheapest one I ran. It got the top score in about half an hour of GPU time and 170 megabytes of memory. My proposed model got one more clip right, and it took thirteen times longer and a hundred and fifteen times more memory.\n\n" +
    "The strange part is that size and cost don't line up. The transformer is the biggest component by far but runs instantly. The CNN is the smallest but ate almost everything, because it's sliding filters over full-resolution video volumes.\n\n" +
    "So: eight of my twelve configurations contained a CNN, and between them they burned 49 of my 50 GPU-hours — for a component whose measured contribution is negative."
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// 11 — Literature + protocol history
// ═════════════════════════════════════════════════════════════════════════════
{
  const s = base("Against the literature, and against my own earlier protocols", "Comparison");

  tbl(s, [
    [hcell("Source", { align: "left" }), hcell("Protocol"), hcell("Accuracy"), hcell("Macro F1")],
    [{ text: "Li et al. 2018 (STSTNet)" }, { text: "LOSO", options: { align: "center" } }, { text: "0.63", options: { align: "center" } }, { text: "not reported", options: { align: "center", color: MUTED } }],
    [{ text: "Vivian et al. 2019 (survey)" }, { text: "LOSO", options: { align: "center" } }, { text: "0.58", options: { align: "center" } }, { text: "not reported", options: { align: "center", color: MUTED } }],
    [{ text: "Example Transformer MER" }, { text: "LOSO", options: { align: "center" } }, { text: "0.65", options: { align: "center" } }, { text: "not reported", options: { align: "center", color: MUTED } }],
    [{ text: "Dissertation target", options: { bold: true, color: BERRY } }, { text: "LOSO", options: { align: "center" } }, { text: "0.70", options: { align: "center", bold: true, color: BERRY } }, { text: "0.68", options: { align: "center", bold: true, color: BERRY } }],
    [{ text: "This project — config_8", options: { bold: true } }, { text: "full LOSO, 25 folds", options: { align: "center", bold: true } }, { text: "0.7500 ✓", options: { align: "center", bold: true, color: TEAL } }, { text: "0.6659", options: { align: "center" } }],
    [{ text: "This project — config_2", options: { bold: true } }, { text: "full LOSO, 25 folds", options: { align: "center", bold: true } }, { text: "0.7436 ✓", options: { align: "center", bold: true, color: TEAL } }, { text: "0.7122 ✓", options: { align: "center", bold: true, color: TEAL } }],
  ], { x: M, y: 1.35, w: 6.6, colW: [2.55, 1.75, 1.15, 1.15], fontSize: 10.5, rowH: 0.315 });

  card(s, M, 3.7, 6.6, 1.5, TINT_T);
  s.addText(
    [
      { text: "This comparison is finally valid.", options: { bold: true, color: TEAL, breakLine: true } },
      { text: "My previous report could not make it — its best result came from a single 39-clip holdout split, while every published number is LOSO. Same dataset, same 3-class grouping, same protocol now.", options: {} },
    ],
    { x: M + 0.25, y: 3.82, w: 6.1, h: 1.25, fontFace: BF, fontSize: 11.5, color: INK, lineSpacing: 15.5, margin: 0, valign: "top" }
  );

  s.addText("Honest caveat: the literature rows still need verifying against the primary papers, and none of them report macro F1.", {
    x: M, y: 5.3, w: 6.6, h: 0.5, fontFace: BF, fontSize: 10.5, italic: true, color: MUTED, lineSpacing: 14, margin: 0, valign: "top",
  });

  s.addText("The same 12 configurations, evaluated 5 different ways over this project's life", {
    x: 7.35, y: 1.32, w: 5.43, h: 0.3, fontFace: BF, fontSize: 12, bold: true, color: BERRY, margin: 0,
  });
  tbl(s, [
    [hcell("Config", { align: "left" }), hcell("Hold\nN=52"), hcell("Hold\nN=39"), hcell("Pilot\n5f"), hcell("Pilot\n20f"), hcell("Full LOSO\n25f")],
    [{ text: "config_8", options: { bold: true } }, { text: "0.1667", options: { align: "center" } }, { text: "0.5051", options: { align: "center" } }, { text: "—", options: { align: "center", color: MUTED } }, { text: "0.3901", options: { align: "center" } }, { text: "0.6659", options: { align: "center", bold: true, color: TEAL } }],
    [{ text: "config_16" }, { text: "0.4563", options: { align: "center", bold: true, color: BERRY } }, { text: "0.7427", options: { align: "center", bold: true, color: BERRY } }, { text: "—", options: { align: "center", color: MUTED } }, { text: "0.5473", options: { align: "center", bold: true, color: BERRY } }, { text: "0.4192", options: { align: "center" } }],
    [{ text: "config_5" }, { text: "0.3833", options: { align: "center" } }, { text: "0.7427", options: { align: "center", bold: true, color: BERRY } }, { text: "0.5667", options: { align: "center" } }, { text: "0.5291", options: { align: "center" } }, { text: "0.4302", options: { align: "center" } }],
    [{ text: "config_2", options: { bold: true } }, { text: "0.1075", options: { align: "center" } }, { text: "0.6044", options: { align: "center" } }, { text: "0.5487", options: { align: "center" } }, { text: "0.4347", options: { align: "center" } }, { text: "0.7122", options: { align: "center", bold: true, color: TEAL } }],
    [{ text: "Winner", options: { bold: true, italic: true } }, { text: "config_16", options: { align: "center", italic: true } }, { text: "config_5/16", options: { align: "center", italic: true } }, { text: "config_6", options: { align: "center", italic: true } }, { text: "config_16", options: { align: "center", italic: true } }, { text: "config_2", options: { align: "center", italic: true, bold: true, color: TEAL } }],
  ], { x: 7.35, y: 1.68, w: 5.43, colW: [1.03, 0.86, 0.86, 0.8, 0.83, 1.05], fontSize: 10, rowH: 0.35 });

  card(s, 7.35, 3.9, 5.43, 2.6, TINT);
  s.addText(
    [
      { text: "The winner changes every time the protocol changes — and nothing about the models changed, only the evaluation.", options: { bold: true, color: BERRY, breakLine: true } },
      { text: "\nUnder holdout the best model was SimAM + 3D-CNN with no Transformer. Under full LOSO it is the Transformer alone. Architecturally opposite conclusions, from identical code.\n\nWhy it flipped: the old 39-clip test set contained exactly one Surprise clip — worth a third of the macro F1, so getting it right was a coin flip weighted at 33 %.", options: {} },
    ],
    { x: 7.6, y: 4.02, w: 4.93, h: 2.35, fontFace: BF, fontSize: 11.5, color: INK, lineSpacing: 15.5, margin: 0, valign: "top" }
  );

  s.addNotes(
    "Against published work: 75 % accuracy is ten points above the best comparable result on this dataset, and five points above my own target.\n\n" +
    "The word that matters is comparable. My earlier report couldn't make this comparison, because its best number came from one small split while everyone else reports leave-one-subject-out. That objection is gone now.\n\n" +
    "The right-hand table is, honestly, the most important thing in this talk.\n\n" +
    "Same twelve configurations. Five different ways of evaluating them. The winner is different every single time. Under the old method, the best model had no transformer at all. Under proper leave-one-subject-out, the best model is the transformer on its own. Opposite conclusions — from identical code.\n\n" +
    "Why? The old test set had 39 clips and exactly one surprise example. That one clip was worth a third of the score. That's a coin flip, not a measurement."
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// 12 — Conclusions
// ═════════════════════════════════════════════════════════════════════════════
{
  SLIDE_N += 1;
  const s = pres.addSlide();
  s.background = { color: BERRY_DK };

  s.addText("CONCLUSION", {
    x: M, y: 0.34, w: 9.0, h: 0.28, fontFace: BF, fontSize: 11, bold: true, color: ROSE, charSpacing: 1.6, margin: 0,
  });
  s.addText("Conclusions, and honest caveats", {
    x: M, y: 0.62, w: 11.4, h: 0.66, fontFace: HF, fontSize: 27, bold: true, color: WHITE, margin: 0, valign: "middle",
  });
  s.addText(String(SLIDE_N), {
    x: W - M - 0.7, y: 6.92, w: 0.7, h: 0.3, fontFace: BF, fontSize: 11, color: ROSE, align: "right", margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 1.45, w: 6.0, h: 3.55, rectRadius: 0.06, fill: { color: "5A2439" }, line: { color: "5A2439", width: 0 },
  });
  s.addText("What was achieved", { x: M + 0.28, y: 1.6, w: 5.44, h: 0.34, fontFace: HF, fontSize: 16, bold: true, color: WHITE, margin: 0 });
  s.addText(
    [
      { text: "The dissertation targets are met under full, valid LOSO. config_2 clears both; config_8 posts the highest accuracy at 0.7500.", options: { bullet: true, breakLine: true } },
      { text: "6 of 12 configurations beat the best published LOSO baseline (0.65).", options: { bullet: true, breakLine: true } },
      { text: "Every configuration beats the always-Negative reference, and none abandons a class.", options: { bullet: true, breakLine: true } },
      { text: "The Transformer is decisive — +0.217, positive in 6 of 6 matched pairs.", options: { bullet: true, breakLine: true } },
      { text: "The 3D-CNN does not pay for itself — −0.031 for 97 % of the compute.", options: { bullet: true, breakLine: true } },
      { text: "EVM was measured for the first time — the inert-switch defect is repaired.", options: { bullet: true } },
    ],
    { x: M + 0.28, y: 2.05, w: 5.44, h: 2.8, fontFace: BF, fontSize: 12, color: "F3E3E8", lineSpacing: 16, paraSpaceAfter: 6, margin: 0, valign: "top" }
  );

  s.addShape(pres.ShapeType.roundRect, {
    x: 6.78, y: 1.45, w: 6.0, h: 3.55, rectRadius: 0.06, fill: { color: "3A1725" }, line: { color: "3A1725", width: 0 },
  });
  s.addText("The honest caveats", { x: 7.06, y: 1.6, w: 5.44, h: 0.34, fontFace: HF, fontSize: 16, bold: true, color: ROSE, margin: 0 });
  s.addText(
    [
      { text: "config_2 and config_8 are not distinguishable — one clip apart, 95 % CI ± 0.068. The defensible claim is about the group, not any single configuration.", options: { bullet: true, breakLine: true } },
      { text: "Single seed. Reproducible bit-for-bit, but no variance estimate — treat differences below ~0.05 macro F1 as unresolved.", options: { bullet: true, breakLine: true } },
      { text: "The minority classes are thin — 25 Surprise + 32 Positive carry two-thirds of the macro F1.", options: { bullet: true, breakLine: true } },
      { text: "No paired significance test is possible — per-clip predictions were not saved.", options: { bullet: true, breakLine: true } },
      { text: "Subject 17 is 21 % of the data; subject 18 contributes none.", options: { bullet: true } },
    ],
    { x: 7.06, y: 2.05, w: 5.44, h: 2.8, fontFace: BF, fontSize: 12, color: "F3E3E8", lineSpacing: 16, paraSpaceAfter: 6, margin: 0, valign: "top" }
  );

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 5.2, w: 12.23, h: 1.05, rectRadius: 0.06, fill: { color: TEAL }, line: { color: TEAL, width: 0 },
  });
  s.addText(
    [
      { text: "The transferable finding:  ", options: { bold: true, color: "CFEDE9" } },
      { text: "on a dataset this small, a single train/test split will pick a winner essentially at random — and I have five evaluation regimes on identical code proving it picks architecturally opposite winners depending on the draw.", options: { color: WHITE } },
    ],
    { x: M + 0.3, y: 5.2, w: 11.63, h: 1.05, fontFace: BF, fontSize: 13.5, valign: "middle", lineSpacing: 19, margin: 0 }
  );

  s.addText("Next: save per-clip predictions (enables McNemar) · multi-seed runs for variance · assert the EVM tensor folders at startup · reallocate the 49 GPU-hours spent on the 3D-CNN", {
    x: M, y: 6.4, w: 12.23, h: 0.42, fontFace: BF, fontSize: 11, italic: true, color: ROSE, margin: 0, valign: "middle",
  });

  s.addNotes(
    "To wrap up.\n\n" +
    "I hit my targets, using a testing method strict enough to actually support the claim. Seventy-five per cent accuracy, ten points above the best published result on this dataset.\n\n" +
    "But the more interesting finding is that of my four components, one does nearly all the work. The transformer is worth 0.217. The CNN is worth minus 0.03 and cost me 97 % of my computing budget.\n\n" +
    "The honest limits: my top two configurations differ by one clip, so I can't say one beats the other — only that anything with a transformer beats anything without. I ran one seed, so I have no error bars. And I didn't save per-clip predictions, so I can't run the statistical test that would settle it. That's the first fix next time, and it costs nothing.\n\n" +
    "If you take one thing away, make it this: on a dataset this small, a single train-test split will pick a winner essentially at random — and I have five runs proving it picks opposite winners depending on the draw.\n\n" +
    "Happy to take questions."
  );
}

const OUT = path.join(ROOT, "MER_Thesis_Final_Presentation.pptx");
pres.writeFile({ fileName: OUT }).then(() => console.log("wrote", OUT));
