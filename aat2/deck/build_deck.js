// AAT 2 presentation: Driver Drowsiness Detection using CNN (7 slides)
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const path = require("path");
const fs = require("fs");
const { applyTheme } = require("./apply_theme.js");

const ROOT = path.resolve(__dirname, "../..");
const IMG = path.join(__dirname, "img");
const M = JSON.parse(fs.readFileSync(path.join(ROOT, "aat2/outputs/metrics.json")));
const SYS = JSON.parse(fs.readFileSync(path.join(ROOT, "aat2/outputs/ddd_system_eval.json")));
const FACE = JSON.parse(fs.readFileSync(path.join(ROOT, "aat2/outputs_face_cnn/metrics.json")));
const pct = (v, d = 2) => (v * 100).toFixed(d) + "%";

const THEME = {
  name: "Night Drive",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1A1F2E", lt1: "FFFFFF", dk2: "1E2B45", lt2: "EEF1F6",
    accent1: "F2A541", accent2: "C0392B", accent3: "2E8B57",
    accent4: "3B6EA8", accent5: "6B7385", accent6: "E07B39",
    hlink: "3B6EA8", folHlink: "6B7385",
  },
};
const HEX = THEME.colors;

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";            // 10 x 5.625 in
  pres.title = "Driver Drowsiness Detection using CNN";
  pres.author = "Arpit Srivastava, Khushi Agrawal, Himanshu, Harsh Kumar Gupta";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  // ---------- layouts ----------
  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.35, w: 6.4, h: 1.5, fontSize: 38, bold: true, align: "left",
                                  color: C.background1, valign: "top", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.95, w: 6.4, h: 0.9, fontSize: 16,
                                  color: C.accent1, valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 28, bold: true, align: "left",
                                  color: C.text2, valign: "middle", margin: 0 }, text: "" } },
      { text: { text: "Driver Drowsiness Detection using CNN   |   Dept. of ISE, DSCE",
                options: { x: 0.5, y: 5.2, w: 7.0, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
    ],
    slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
  });
  pres.defineSlideMaster({
    title: "CLOSING_DARK",
    background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 28, bold: true, align: "left",
                                  color: C.background1, valign: "middle", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
  });

  const I = {
    eye: await icon(fa.FaEye, HEX.dk2), eyeW: await icon(fa.FaEye, HEX.lt1),
    eyeSlash: await icon(fa.FaEyeSlash, HEX.lt1), crash: await icon(fa.FaCarCrash, HEX.lt1),
    brain: await icon(fa.FaBrain, HEX.lt1), bell: await icon(fa.FaBell, HEX.lt1),
    check: await icon(fa.FaCheck, HEX.lt1), times: await icon(fa.FaTimes, HEX.lt1),
    road: await icon(fa.FaRoad, HEX.lt1), github: await icon(fa.FaGithub, HEX.lt1),
  };

  // icon in a coloured circle, the deck's one visual motif
  function badge(slide, img, x, y, d, fill, name) {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" }, objectName: name + " circle" });
    const p = d * 0.25;
    slide.addImage({ data: img, x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: name + " icon" });
  }
  const txt = (slide, text, o) => slide.addText(text, { isTextBox: true, margin: 0, ...o });

  // ---------- 1. title ----------
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("Driver Drowsiness Detection using CNN", { placeholder: "title" });
  s.addText("AAT 2  |  Introduction to Deep Learning (22IS741)", { placeholder: "body" });
  txt(s, [
    { text: "Arpit Srivastava (1DS23IS023)   Khushi Agrawal (1DS23IS074)", options: { breakLine: true } },
    { text: "Himanshu (1DS23IS061)   Harsh Kumar Gupta (1DS23IS054)", options: { breakLine: true } },
    { text: "Dept. of ISE, Dayananda Sagar College of Engineering" },
  ], { x: 0.6, y: 3.95, w: 6.6, h: 0.95, fontSize: 13, color: C.background2, paraSpaceAfter: 4, objectName: "team" });
  s.addShape(pres.shapes.OVAL, { x: 7.15, y: 1.35, w: 2.3, h: 2.3, fill: { color: C.accent1 }, line: { type: "none" }, objectName: "eye circle" });
  s.addImage({ data: I.eye, x: 7.65, y: 1.85, w: 1.3, h: 1.3, objectName: "eye icon" });
  txt(s, "Eyes closed for 15 frames  →  alarm", { x: 6.8, y: 3.85, w: 3.0, h: 0.4, fontSize: 13, italic: true,
       color: C.accent1, align: "center", objectName: "tagline" });
  s.addNotes("DriveAlert watches the driver through a camera. A CNN decides in every frame whether the eyes are open or closed, and an alarm sounds when they stay closed for 15 frames in a row.");

  // ---------- 2. problem and approach ----------
  pres.addSection({ title: "Approach" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Approach" });
  s.addText("Long eye closure is the clearest sign of drowsiness", { placeholder: "title" });
  const rows = [
    [I.crash, C.accent2, "The problem", "Drowsy drivers react slowly and can fall asleep for seconds; there is no roadside test for it."],
    [I.eyeSlash, C.accent1, "The signal", "A normal blink lasts about 0.1 s. A drowsy driver keeps the eyes shut much longer."],
    [I.brain, C.accent4, "Our approach", "A CNN classifies each eye as Open or Closed; a frame counter turns that into an alert."],
  ];
  rows.forEach(([img, col, head, body], k) => {
    const x = 0.5 + k * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.25, w: 2.85, h: 1.75, rectRadius: 0.08, fill: { color: C.background2 },
                                                  line: { type: "none" }, objectName: head + " card" });
    badge(s, img, x + 0.2, 1.42, 0.6, col, head);
    txt(s, head, { x: x + 0.95, y: 1.5, w: 1.8, h: 0.45, fontSize: 16, bold: true, color: C.text2, valign: "middle", objectName: head + " head" });
    txt(s, body, { x: x + 0.2, y: 2.15, w: 2.5, h: 0.8, fontSize: 13, color: C.text1, valign: "top", objectName: head + " body" });
  });
  s.addImage({ path: path.join(IMG, "aat2_block_diagram.png"), x: 0.5, y: 3.45, w: 9.0, h: 9.0 * 359 / 2089, objectName: "pipeline" });
  s.addNotes("Problem, signal and approach. The pipeline below: camera, Haar face detection, two eye crops, eye-state CNN, frame counter, alarm.");

  // ---------- 3. design choice ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Approach" });
  s.addText("Whole-face models failed on unseen drivers", { placeholder: "title" });
  const cards = [
    [pct(FACE.accuracy, 1), "CNN on the whole face", `98% on training faces, AUC ${FACE.auc.toFixed(2)} on new people`, C.accent2, I.times],
    [pct(FACE.mobilenet_probe_accuracy, 1), "MobileNetV2 features", "Pre-trained ImageNet CNN on the whole face: chance level", C.accent2, I.times],
    [pct(M.test.accuracy), "CNN on eye state", "Final design: eye closure does not depend on who the driver is", C.accent3, I.check],
  ];
  cards.forEach(([big, head, body, col, ic], k) => {
    const x = 0.5 + k * 3.05, sel = k === 2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.3, w: 2.85, h: 3.1, rectRadius: 0.08,
      fill: { color: sel ? C.text2 : C.background2 }, line: { type: "none" }, objectName: head + " card" });
    badge(s, ic, x + 2.15, 1.45, 0.5, col, head);
    txt(s, big, { x: x + 0.25, y: 1.55, w: 2.4, h: 0.95, fontSize: 40, bold: true, color: sel ? C.accent1 : col, objectName: head + " value" });
    txt(s, "accuracy on unseen faces", { x: x + 0.25, y: 2.45, w: 2.4, h: 0.3, fontSize: 11, color: sel ? C.background2 : C.accent5, objectName: head + " unit" });
    txt(s, head, { x: x + 0.25, y: 2.95, w: 2.4, h: 0.4, fontSize: 16, bold: true, color: sel ? C.background1 : C.text2, objectName: head + " head" });
    txt(s, body, { x: x + 0.25, y: 3.35, w: 2.4, h: 0.95, fontSize: 13, color: sel ? C.background2 : C.text1, valign: "top", objectName: head + " body" });
  });
  txt(s, `Always answering "Drowsy" already scores ${pct(FACE.baseline_always_drowsy, 1)}: the whole-face models learned who the person is, not how tired they look.`,
      { x: 0.5, y: 4.55, w: 9.0, h: 0.5, fontSize: 13, italic: true, color: C.text1, objectName: "insight" });
  s.addNotes("We tested fairly by splitting the Driver Drowsiness Dataset by person. The whole-face CNN memorised identities. Switching to eye state fixed that.");

  // ---------- 4. data and model ----------
  pres.addSection({ title: "Method" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("A small CNN trained on 42,131 eye crops", { placeholder: "title" });
  s.addImage({ path: path.join(IMG, "aat2_dataset_samples.png"), x: 0.5, y: 1.25, w: 4.7, h: 4.7 * 1320 / 2617, objectName: "eye samples" });
  txt(s, "Training eyes (rows 1-2) and the separate real-world test photos (rows 3-4)",
      { x: 0.5, y: 3.68, w: 4.7, h: 0.3, fontSize: 10, color: C.accent5, objectName: "samples caption" });
  s.addImage({ path: path.join(IMG, "aat2_cnn_architecture.png"), x: 5.45, y: 1.25, w: 4.05, h: 4.05 * 595 / 2175, objectName: "architecture" });
  txt(s, [
    { text: "Input: 64×64 grayscale eye", options: { bullet: { indent: 14 }, breakLine: true } },
    { text: `3 conv blocks, ${M.params.toLocaleString("en-US")} parameters`, options: { bullet: { indent: 14 }, breakLine: true } },
    { text: "Split by face: 70 / 15 / 15", options: { bullet: { indent: 14 }, breakLine: true } },
    { text: "Augmentation: flip, rotate, blur, noise", options: { bullet: { indent: 14 }, breakLine: true } },
    { text: "Adam, binary cross-entropy, early stopping", options: { bullet: { indent: 14 } } },
  ], { x: 5.45, y: 2.55, w: 4.05, h: 1.9, fontSize: 14, color: C.text1, paraSpaceAfter: 5, valign: "top", objectName: "model facts" });
  txt(s, "BatchNorm momentum 0.9: the default 0.99 made the model predict Closed for every image",
      { x: 0.5, y: 4.2, w: 4.7, h: 0.6, fontSize: 12, italic: true, color: C.accent2, objectName: "bug note" });
  s.addNotes("Eyes are cropped as squares of 1.6 times the eye width so the crop size cannot leak the label. A BatchNorm momentum bug was found and fixed during training.");

  // ---------- 5. results ----------
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText(`${pct(M.test.accuracy)} on held-out eyes, ${pct(M.real.accuracy)} on real photos`, { placeholder: "title" });
  const stats = [[pct(M.test.accuracy), `held-out test, ${M.test.n.toLocaleString("en-US")} eyes`, C.text2],
                 [pct(M.real.accuracy), `real photos, ${M.real.n.toLocaleString("en-US")} eyes`, C.accent6],
                 [M.real.recall.toFixed(3), "recall on real closed eyes", C.accent3]];
  stats.forEach(([big, lab, col], k) => {
    txt(s, big, { x: 0.5, y: 1.25 + k * 1.2, w: 3.0, h: 0.7, fontSize: 40, bold: true, color: col, objectName: lab + " value" });
    txt(s, lab, { x: 0.5, y: 1.92 + k * 1.2, w: 3.0, h: 0.3, fontSize: 13, color: C.accent5, objectName: lab + " label" });
  });
  s.addImage({ path: path.join(IMG, "aat2_confusion_matrix.png"), x: 3.75, y: 1.3, w: 5.75, h: 5.75 * 983 / 2189, objectName: "confusion matrices" });
  txt(s, `Real photos come from a different source, so the drop to ${pct(M.real.accuracy)} (AUC ${M.real.auc.toFixed(3)}) is the realistic figure`,
      { x: 3.75, y: 4.0, w: 5.75, h: 0.6, fontSize: 12, italic: true, color: C.text1, objectName: "results note" });
  s.addNotes("The held-out set comes from the same source as training. The real-world set is the honest test. Most real errors are open eyes called closed; the 15-frame rule filters those single-frame errors.");

  // ---------- 6. real videos and demo ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText(`${SYS.drowsy_clip_higher} of ${SYS.people_with_both_clips} drowsy videos had more eye closure`, { placeholder: "title" });
  const people = SYS.per_person.map(r => r[0].toUpperCase());
  s.addChart(pres.charts.BAR, [
    { name: "Drowsy video", labels: people, values: SYS.per_person.map(r => +(r[1] * 100).toFixed(1)) },
    { name: "Alert video", labels: people, values: SYS.per_person.map(r => +(r[2] * 100).toFixed(1)) },
  ], { x: 0.4, y: 1.15, w: 5.9, h: 3.85, barDir: "col", barGrouping: "clustered", barGapWidthPct: 40,
       chartColors: [HEX.accent2, HEX.accent3],
       showTitle: true, title: "Frames with eyes closed (%) per driver", titleFontSize: 12, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
       showLegend: true, legendPos: "t", legendFontSize: 10, legendFontFace: "+mn-lt", legendColor: HEX.dk1,
       catAxisLabelFontSize: 8, catAxisLabelColor: HEX.accent5, catAxisLabelFontFace: "+mn-lt",
       valAxisLabelFontSize: 9, valAxisLabelColor: HEX.accent5, valAxisLabelFontFace: "+mn-lt",
       valAxisMaxVal: 100, valGridLine: { color: "DDE1E8", size: 0.5 }, catGridLine: { style: "none" }, objectName: "perclos chart" });
  [["app_alert_face.png", "ALERT  P(closed) = 0.00", C.accent3], ["app_drowsy_face.png", "DROWSY  P(closed) = 1.00", C.accent2]].forEach(([f, lab, col], k) => {
    const x = 6.55 + k * 1.5;
    s.addImage({ path: path.join(IMG, f), x, y: 1.3, w: 1.4, h: 1.4, objectName: lab + " image" });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.8, w: 1.4, h: 0.5, rectRadius: 0.06, fill: { color: col }, line: { type: "none" }, objectName: lab + " chip" });
    txt(s, lab.split("  ")[0], { x, y: 2.8, w: 1.4, h: 0.28, fontSize: 12, bold: true, color: C.background1, align: "center", objectName: lab + " state" });
    txt(s, lab.split("  ")[1], { x, y: 3.06, w: 1.4, h: 0.22, fontSize: 10, color: C.background1, align: "center", objectName: lab + " prob" });
  });
  txt(s, [
    { text: "Gradio app, same driver", options: { bold: true, breakLine: true } },
    { text: `Average closed-eye share: ${pct(SYS.mean_perclos_drowsy, 1)} in drowsy videos vs ${pct(SYS.mean_perclos_alert, 1)} in alert videos` },
  ], { x: 6.55, y: 3.5, w: 2.95, h: 1.4, fontSize: 13, color: C.text1, paraSpaceAfter: 4, valign: "top", objectName: "demo note" });
  s.addNotes("The whole pipeline was run on 13,924 frames of real driver videos the model never saw. On the right, the Gradio app on two real frames of the same driver.");

  // ---------- 7. conclusion ----------
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
  s.addText("Eye-state CNN gives a reliable drowsiness alarm", { placeholder: "title" });
  const take = [
    [I.eyeW, C.accent1, "Eye state, not identity", "Splitting the task into eye state plus a frame counter generalises to new drivers"],
    [I.check, C.accent3, "Tested honestly", `${pct(M.test.accuracy)} held-out, ${pct(M.real.accuracy)} on real photos from another source`],
    [I.bell, C.accent2, "Working system", "Gradio web app and a real-time webcam monitor with an alarm"],
  ];
  take.forEach(([img, col, head, body], k) => {
    const y = 1.3 + k * 0.95;
    badge(s, img, 0.5, y, 0.62, col, head);
    txt(s, head, { x: 1.35, y: y - 0.02, w: 4.0, h: 0.35, fontSize: 16, bold: true, color: C.background1, objectName: head + " head" });
    txt(s, body, { x: 1.35, y: y + 0.32, w: 4.2, h: 0.5, fontSize: 13, color: C.background2, valign: "top", objectName: head + " body" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.0, y: 1.3, w: 3.5, h: 2.15, rectRadius: 0.08, fill: { color: C.text1 }, line: { type: "none" }, objectName: "next steps card" });
  txt(s, "Next steps", { x: 6.25, y: 1.45, w: 3.0, h: 0.4, fontSize: 16, bold: true, color: C.accent1, objectName: "next head" });
  txt(s, [
    { text: "More real and infrared driver images", options: { bullet: { indent: 14 }, breakLine: true } },
    { text: "Add yawning and head nodding", options: { bullet: { indent: 14 }, breakLine: true } },
    { text: "Landmark-based eye localisation", options: { bullet: { indent: 14 }, breakLine: true } },
    { text: "Test in a real vehicle", options: { bullet: { indent: 14 } } },
  ], { x: 6.25, y: 1.9, w: 3.1, h: 1.85, fontSize: 13, color: C.background1, paraSpaceAfter: 5, valign: "top", objectName: "next list" });
  s.addImage({ data: I.github, x: 0.5, y: 4.45, w: 0.32, h: 0.32, objectName: "github icon" });
  txt(s, [{ text: "github.com/Arpit529Srivastava/Deep-Learning-AAT",
            options: { hyperlink: { url: "https://github.com/Arpit529Srivastava/Deep-Learning-AAT" }, color: C.background2 } }],
    { x: 0.95, y: 4.45, w: 5.0, h: 0.32, fontSize: 12, color: C.background2, valign: "middle", objectName: "repo link" });
  txt(s, "Thank you", { x: 6.0, y: 4.25, w: 3.5, h: 0.6, fontSize: 28, bold: true, color: C.accent1, align: "right", objectName: "thanks" });
  s.addNotes("Summary and next steps. Code, models and report are on GitHub.");

  const out = path.join(ROOT, "AAT2_Driver_Drowsiness_Detection.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("saved", out);
})();
