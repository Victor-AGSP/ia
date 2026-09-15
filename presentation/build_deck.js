// Builds the project presentation from results/report_data.json, results/tex/values.tex
// and the figures in results/. Every number shown comes from those generated files.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const RES = path.join(ROOT, "results");
const data = JSON.parse(fs.readFileSync(path.join(RES, "report_data.json"), "utf8"));
const M = {};
for (const m of fs.readFileSync(path.join(RES, "tex", "values.tex"), "utf8").matchAll(/\\newcommand\{\\(\w+)\}\{(.*)\}/g)) {
  M[m[1]] = m[2]
    .replace(/\s*\$\\pm\$\s*/g, " ± ")
    .replace(/\$\\bar (\w)\^\*\$/g, "$1*")
    .replace(/\$\\bar (\w)\$/g, "$1");
}

const C = { red: "B83227", green: "4F7A28", ink: "1F2933", muted: "5B6770", card: "F3F5F6", dark: "1E2A30", white: "FFFFFF", orange: "D9822B", grey: "AEB6BC" };
const HEAD = "Cambria", BODY = "Calibri";
const f3 = (x) => Number(x).toFixed(3);
const f2 = (x) => Number(x).toFixed(2);

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "Proyecto 1 — Segmentación y clasificación de madurez de tomates";

function pngSize(file) {
  const buf = fs.readFileSync(file);
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

function image(slide, name, x, y, w, h) {
  const file = path.join(RES, name);
  const { w: iw, h: ih } = pngSize(file);
  const scale = Math.min(w / iw, h / ih);
  const dw = iw * scale, dh = ih * scale;
  slide.addImage({ path: file, x: x + (w - dw) / 2, y: y + (h - dh) / 2, w: dw, h: dh });
}

function header(slide, n, title) {
  slide.background = { color: C.white };
  slide.addShape(pres.shapes.OVAL, { x: 0.45, y: 0.3, w: 0.5, h: 0.5, fill: { color: C.red }, line: { color: C.red } });
  slide.addText(String(n), { x: 0.45, y: 0.3, w: 0.5, h: 0.5, align: "center", valign: "middle", fontFace: BODY, fontSize: 14, bold: true, color: C.white, margin: 0, isTextBox: true });
  slide.addText(title, { x: 1.1, y: 0.25, w: 8.5, h: 0.6, fontFace: HEAD, fontSize: 26, bold: true, color: C.ink, valign: "middle", margin: 0, isTextBox: true });
}

function text(slide, content, x, y, w, h, opts = {}) {
  slide.addText(content, { x, y, w, h, fontFace: BODY, fontSize: 13, color: C.ink, valign: "top", margin: 0, isTextBox: true, ...opts });
}

function bullets(slide, items, x, y, w, h, fontSize = 13) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 } }));
  slide.addText(runs, { x, y, w, h, fontFace: BODY, fontSize, color: C.ink, valign: "top", margin: 0, isTextBox: true });
}

function card(slide, x, y, w, h, title, body, color = C.red, bodySize = 12) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.card }, line: { color: C.card }, rectRadius: 0.08 });
  slide.addText(title, { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: 0.32, fontFace: BODY, fontSize: 13, bold: true, color, margin: 0, isTextBox: true });
  slide.addText(body, { x: x + 0.15, y: y + 0.45, w: w - 0.3, h: h - 0.55, fontFace: BODY, fontSize: bodySize, color: C.ink, valign: "top", margin: 0, isTextBox: true });
}

function stat(slide, x, y, w, value, label, color = C.red) {
  slide.addText(value, { x, y, w, h: 0.6, fontFace: HEAD, fontSize: 30, bold: true, color, margin: 0, isTextBox: true });
  slide.addText(label, { x, y: y + 0.6, w, h: 0.45, fontFace: BODY, fontSize: 11, color: C.muted, margin: 0, isTextBox: true });
}

const cell = (t, o = {}) => ({ text: String(t), options: { fontFace: BODY, fontSize: 10.5, color: C.ink, valign: "middle", ...o } });
const hcell = (t) => cell(t, { bold: true, color: C.white, fill: { color: C.dark } });
const tableOpts = (x, y, w, colW) => ({ x, y, w, colW, border: { type: "solid", pt: 0.5, color: "D5DADD" }, rowH: 0.27, margin: 0.04 });

const strategies = data.strategies;
const seg = data.segmentation_summary;
const best = data.best_segmentation;
const bestSeg = seg.find((s) => s.combination === best);
const feat = (list) => list.join(", ");

// 1. Title -------------------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: C.dark };
  image(s, "segmentation_examples.png", 5.6, 0.6, 4.1, 4.4);
  s.addText("Proyecto 1", { x: 0.6, y: 1.0, w: 4.8, h: 0.4, fontFace: BODY, fontSize: 16, color: C.orange, bold: true, margin: 0, isTextBox: true });
  s.addText("Segmentación y clasificación del estado de madurez de tomates", { x: 0.6, y: 1.45, w: 4.8, h: 1.9, fontFace: HEAD, fontSize: 30, bold: true, color: C.white, margin: 0, isTextBox: true, valign: "top" });
  s.addText("K-Means · Bayes con razón de verosimilitudes · SFS · PCA", { x: 0.6, y: 3.5, w: 4.8, h: 0.4, fontFace: BODY, fontSize: 14, color: "D6DCE0", margin: 0, isTextBox: true });
  const authors = data.config.AUTHORS ? `${data.config.AUTHORS} · ` : "";
  s.addText(`${authors}INFO1185 Inteligencia Artificial · Septiembre 2026`, { x: 0.6, y: 4.6, w: 4.8, h: 0.35, fontFace: BODY, fontSize: 12, color: "9AA5AD", margin: 0, isTextBox: true });
  s.addNotes("Presentar el problema en una frase: decidir maduro/inmaduro usando solo los píxeles que K-Means identifica como fruto.");
}

// 2. Problem and objective ---------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 1, "Problema y objetivo");
  text(s, "Decidir si un tomate está maduro o inmaduro a partir de una imagen RGB, usando únicamente los píxeles que un método no supervisado identifica como fruto.", 0.5, 1.05, 9.0, 0.6, { fontSize: 15 });
  const steps = ["Imagen RGB", "K-Means k=2\n7 combinaciones", "Cluster fruto\n+ post-proceso", "Descriptores\nde color", "Análisis · SFS\n· PCA", "Bayes\nln Λ ≥ ln θ*"];
  const bw = 1.3, gap = 0.26, x0 = 0.5, y0 = 1.95;
  steps.forEach((t, i) => {
    const x = x0 + i * (bw + gap);
    const fill = i < 3 ? C.red : i < 5 ? C.orange : C.green;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: bw, h: 0.9, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
    s.addText(t, { x, y: y0, w: bw, h: 0.9, align: "center", valign: "middle", fontFace: BODY, fontSize: 11.5, bold: true, color: C.white, margin: 2, isTextBox: true });
    if (i < steps.length - 1) s.addShape(pres.shapes.LINE, { x: x + bw + 0.03, y: y0 + 0.45, w: gap - 0.06, h: 0, line: { color: C.muted, width: 1.5, endArrowType: "triangle" } });
  });
  card(s, 0.5, 3.2, 2.9, 1.9, "No supervisado", "K-Means separa fruto y fondo por color; se evalúa con Jaccard contra la máscara anotada.", C.red);
  card(s, 3.55, 3.2, 2.9, 1.9, "Supervisado", "Clasificador bayesiano gaussiano sobre descriptores del fruto; umbral por Youden.", C.green);
  card(s, 6.6, 3.2, 2.9, 1.9, "Preguntas", "¿Qué canales segmentan mejor? ¿Qué variables distinguen la madurez? ¿Ayuda PCA? ¿Cuánto pesa la segmentación?", C.orange);
  s.addNotes("Etapas: segmentación no supervisada, extracción de características, dos estrategias de selección, PCA y comparación final.");
}

// 3. Data ---------------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 2, "Descripción de los datos");
  stat(s, 0.5, 1.05, 2.1, M.NImages, `imágenes útiles (${M.NRipe} maduras / ${M.NUnripe} inmaduras)`);
  stat(s, 2.75, 1.05, 2.1, M.NDuplicates, "duplicados exactos eliminados (MD5)");
  stat(s, 5.0, 1.05, 2.1, `${M.NTrain}/${M.NVal}/${M.NTest}`, "entrenamiento / validación / test", C.green);
  stat(s, 7.25, 1.05, 2.3, "≤ 256 px", "lado máximo tras reducir (anchos originales 250–7792)", C.orange);
  image(s, "dataset_montage.png", 0.5, 2.35, 9.0, 2.3);
  text(s, `Partición estratificada por imagen, semilla ${M.Seed}. Antes de particionar se detectaron y eliminaron dos pares de imágenes idénticas. Fondos simples (blanco) y complejos (follaje, platos, madera); varios frutos por imagen.`, 0.5, 4.75, 9.0, 0.7, { fontSize: 11.5, color: C.muted });
  s.addNotes("Destacar el control de duplicados: sin él una misma imagen podía estar en entrenamiento y validación. Dos imágenes RGBA se convierten a RGB.");
}

// 4. Methodology ------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 3, "Metodología general y control de fuga");
  const rows = [
    [hcell("Decisión"), hcell("Datos usados")],
    [cell("Combinación de canales y post-procesado"), cell("Desarrollo (entrenamiento + validación)")],
    [cell("Selección por análisis, SFS, estandarización, PCA"), cell("Solo entrenamiento")],
    [cell("Umbral de la razón de verosimilitudes (Youden)"), cell("Validación, con el modelo de entrenamiento")],
    [cell("Métricas finales (AUC, exactitud, sens., espec.)"), cell("Test: no interviene en decisiones", { bold: true, color: C.red })],
    [cell("Análisis complementarios (máscara, robustez)"), cell(`Test y ${M.NRepeats} particiones; no cambian el sistema`)],
  ];
  s.addTable(rows, { ...tableOpts(0.5, 1.1, 5.6, [3.1, 2.5]), rowH: 0.5 });
  card(s, 6.4, 1.1, 3.1, 1.75, "Reproducible", `Semilla ${M.Seed} en partición, muestreo de píxeles, K-Means y validación cruzada. Parámetros en tomato/config.py.`, C.green);
  card(s, 6.4, 3.0, 3.1, 1.75, "Trazable", "Las tablas y cifras del informe y de esta presentación se generan desde los resultados del código.", C.orange);
  s.addNotes("Regla: el test nunca participa en decisiones. El umbral se aplica al mismo modelo con el que se eligió en validación.");
}

// 5. EDA fruit vs background ------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 4, "Análisis exploratorio: fruto vs fondo");
  image(s, "pixel_hist_fruit_background.png", 0.5, 0.95, 9.0, 3.8);
  card(s, 0.5, 4.8, 2.9, 0.7, "Fruto: S alto, B bajo", "", C.red);
  card(s, 3.55, 4.8, 2.9, 0.7, "Fondo muy variable", "", C.muted);
  card(s, 6.6, 4.8, 2.9, 0.7, "Tomate verde ≈ follaje", "", C.green);
  s.addNotes("Este análisis explica por qué R y B aparecen en las mejores combinaciones de K-Means.");
}

// 6. EDA ripe vs unripe ------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 5, "Análisis exploratorio: maduro vs inmaduro");
  image(s, "pixel_hist_ripe_unripe.png", 0.5, 0.95, 9.0, 3.8);
  card(s, 0.5, 4.8, 4.4, 0.7, "H y a* separan las clases; R sube y G baja", "", C.green);
  card(s, 5.05, 4.8, 4.45, 0.7, "H circular: media de 5° y 355° ≠ 180°", "", C.red);
  s.addNotes("La media circular del tono es clave: la media aritmética produce tonos espurios en tomates rojos (ver ablación).");
}

// 7. K-Means configuration --------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 6, "Segmentación con K-Means");
  const cfg = [
    ["Clusters", "k = 2 (fruto / fondo), un modelo por imagen y combinación"],
    ["Inicialización", "k-means++, 10 reinicios (menor inercia)"],
    ["Convergencia", `Δ centroides < 1e-4 o 300 iteraciones (máx. observado: ${M.IterMax})`],
    ["Semilla", `${M.Seed}; ajuste con ${Number(M.PixelSample).toLocaleString("es-CL")} píxeles, asignación de todos`],
    ["Cluster fruto", "puntaje de objeto: brillo, saturación, compacidad, centralidad, área, borde"],
  ];
  s.addTable(cfg.map(([a, b]) => [cell(a, { bold: true }), cell(b)]), { ...tableOpts(0.5, 1.05, 5.3, [1.35, 3.95]), rowH: 0.42 });
  const rows = [[hcell("Canales"), hcell("Puntaje"), hcell("Mín. borde"), hcell("Máx. sat."), hcell("Oráculo")]];
  for (const r of seg) rows.push([cell(r.combination, { bold: r.combination === best }), cell(f3(r.dev_heuristic)), cell(f3(r.dev_min_border)), cell(f3(r.dev_max_saturation)), cell(f3(r.dev_oracle_mean), { color: C.muted })]);
  s.addTable(rows, tableOpts(6.05, 1.05, 3.45, [0.65, 0.7, 0.7, 0.7, 0.7]));
  text(s, `Jaccard medio en desarrollo según la regla para elegir el cluster fruto. El puntaje de objeto gana en ${M.HeurBestCount} de 7 combinaciones y queda cerca del oráculo: el límite es K-Means, no la regla.`, 6.05, 3.35, 3.45, 1.2, { fontSize: 10.5, color: C.muted });
  text(s, "La máscara de referencia solo se usa para calcular J = |A∩B| / |A∪B|.", 0.5, 3.35, 5.3, 0.4, { fontSize: 12, italic: true });
  s.addNotes("El oráculo elige el cluster con mayor Jaccard: es la cota superior con esos clusters. Los pesos del puntaje y el prior de área se fijaron a priori a mano, sin usar máscaras de test (limitación).");
}

// 8. Seven combinations ------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 7, "Comparación de las siete combinaciones (Jaccard)");
  const labels = seg.map((r) => r.combination);
  s.addChart(pres.charts.BAR, [
    { name: "K-Means", labels, values: seg.map((r) => r.dev_mean) },
    { name: "+ post-procesado", labels, values: seg.map((r) => r.dev_post_mean) },
    { name: "Oráculo de cluster", labels, values: seg.map((r) => r.dev_oracle_mean) },
  ], {
    x: 0.4, y: 1.0, w: 6.2, h: 4.2, barDir: "col", barGrouping: "clustered", chartColors: [C.red, C.orange, C.grey],
    valAxisMinVal: 0.4, valAxisMaxVal: 0.65, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00",
    showLegend: true, legendPos: "t", legendFontSize: 10, catAxisLabelFontSize: 11, valAxisLabelFontSize: 9,
    valGridLine: { color: "E3E6E8", size: 0.5 }, catGridLine: { style: "none" },
    showTitle: true, title: "Jaccard medio en desarrollo", titleFontSize: 12, titleColor: C.ink,
    catAxisLabelColor: C.muted, valAxisLabelColor: C.muted,
  });
  stat(s, 6.9, 1.1, 2.7, best, `mejor combinación (J = ${M.BestDevJ})`);
  stat(s, 6.9, 2.2, 2.7, M.BestDevPostJ, "con post-procesado morfológico", C.orange);
  text(s, `Diferencia mejor–peor (${M.WorstCombo}): ${f3(bestSeg.dev_mean - Number(M.WorstDevJ))}, frente a DE entre imágenes ≈ ${f2(bestSeg.dev_std)}: ranking débil. R separa rojo de verde; B separa objetos saturados de fondos claros. Test: J = ${M.BestTestJ} (máscara final con post-proceso: ${M.BestTestPostJ}).`, 6.9, 3.35, 2.7, 1.9, { fontSize: 11 });
  s.addNotes("El eje empieza en 0.40 para ver diferencias; son pequeñas. G solo es la peor porque el tomate verde y el follaje tienen G parecido.");
}

// 9. Segmentation examples --------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 8, `Ejemplos de segmentación (${best})`);
  image(s, "segmentation_examples.png", 0.4, 0.95, 6.3, 4.5);
  card(s, 6.9, 1.05, 2.65, 2.0, "Fallo típico", "Fondo bimodal o follaje brillante: los dos clusters separan dos fondos y el tomate queda repartido.", C.red, 11.5);
  card(s, 6.9, 3.2, 2.65, 2.0, "Post-procesado", `Apertura + cierre (elipse 5×5), relleno de huecos y eliminación de componentes < 1 %. Mejora la media de desarrollo en ${M.PostDevBetterCount}/7; en test ${M.BestTestJ} → ${M.BestTestPostJ}.`, C.orange, 11.5);
  s.addNotes("Filas: peor, mediana y mejor imagen de desarrollo. Mostrar que Jaccard alto requiere fondos simples.");
}

// 10. Features ----------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 9, "Características extraídas del fruto segmentado");
  const rows = [
    [hcell("Descriptor"), hcell("Justificación")],
    [cell("R̄, Ḡ, B̄", { bold: true }), cell("Mínimo exigido; la madurez pasa energía de G a R")],
    [cell("H̄ (circular)", { bold: true }), cell("Cromaticidad independiente del brillo; media circular")],
    [cell("S̄, V̄", { bold: true }), cell("Saturación y brillo; separan color de iluminación")],
    [cell("ā*", { bold: true }), cell("Eje verde↔rojo CIELAB: índice clásico de madurez")],
    [cell("b̄*", { bold: true }), cell("Eje azul↔amarillo, variable de control")],
  ];
  s.addTable(rows, { ...tableOpts(0.5, 1.05, 3.9, [1.1, 2.8]), rowH: 0.55 });
  image(s, "feature_distributions.png", 4.6, 1.0, 5.0, 3.6);
  text(s, "Promedios calculados solo con los píxeles de la máscara K-Means final (entrenamiento).", 4.6, 4.7, 5.0, 0.5, { fontSize: 11, color: C.muted });
  s.addNotes("Todas las características se calculan con la máscara estimada, no con la de referencia: se evalúa el sistema real.");
}

// 11. Analysis-based selection ------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 10, "Selección basada en el análisis exploratorio");
  const rows = [[hcell("Descriptor"), hcell("d de Cohen"), hcell("AUC univ."), hcell("Elegida")]];
  const sel = new Set(data.analysis_features);
  const sorted = [...data.separability].sort((a, b) => Math.abs(b.cohen_d) - Math.abs(a.cohen_d));
  for (const r of sorted) {
    const o = sel.has(r.feature) ? { bold: true, color: C.red } : {};
    rows.push([cell(r.feature, o), cell(f2(r.cohen_d), o), cell(f3(r.auc_univariate), o), cell(sel.has(r.feature) ? "sí" : r.feature === "a*" ? "no (|r|=0.92 con H)" : "no", o)]);
  }
  s.addTable(rows, tableOpts(0.5, 1.05, 4.6, [1.0, 1.0, 1.0, 1.6]));
  text(s, `Regla: ordenar por |d| y aceptar si |r| ≤ 0.85 con las ya elegidas (máx. 3). Resultado: ${M.AnalysisFeatures}.`, 0.5, 3.65, 4.6, 0.7, { fontSize: 12, bold: true });
  text(s, "Evita variables casi colineales: Bayes ingenuo contaría dos veces la misma evidencia.", 0.5, 4.35, 4.6, 0.6, { fontSize: 11, color: C.muted });
  image(s, "feature_correlation.png", 5.3, 1.0, 4.3, 4.2);
  s.addNotes("H es la más separable (maduro ≈ 20°, inmaduro ≈ 80°). a* es la segunda pero redundante con H. G y R aportan intensidad por canal.");
}

// 12. SFS ----------------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 11, "Selección mediante SFS");
  image(s, "sfs_trace.png", 0.4, 1.0, 5.4, 3.3);
  text(s, "Wrapper: se añade la variable que maximiza el AUC de validación cruzada 5×10 del mismo Bayes (solo entrenamiento). Se elige el prefijo más corto a ≤ 0.001 del mejor.", 0.5, 4.4, 5.3, 0.9, { fontSize: 11.5 });
  card(s, 6.0, 1.05, 3.55, 1.3, "Seleccionadas", `SFS: ${M.SFSFeatures}\nAnálisis: ${M.AnalysisFeatures}`, C.green, 12.5);
  card(s, 6.0, 2.5, 3.55, 2.75, "¿Coinciden?", `Solo en H (primera elegida). SFS cambia G y R por S y V: optimiza el AUC conjunto, no la separación individual, y el criterio está saturado (AUC ≈ 0.98, DE ≈ 0.04). En ${M.NRepeats} particiones: SFS ${M.SfsDistinctSubsets} conjuntos distintos; análisis ${M.AnDistinctSubsets}.`, C.red, 11);
  s.addNotes("Mensaje: SFS confirma que H es la variable principal, pero su elección posterior es inestable con 36 imágenes. Incluso diferencias de menos de un nivel de gris entre entornos cambiaron el subconjunto (H,B,S,R en una versión previa) sin cambiar el desempeño en test.");
}

// 13. Bayes design ---------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 12, "Clasificador bayesiano y razón de verosimilitudes");
  card(s, 0.5, 1.05, 4.3, 1.55, "Regla de decisión", "ln Λ(x) = ln p(x|maduro) − ln p(x|inmaduro) ≥ ln θ → maduro\np(x|ω) = Π N(xⱼ; μ, σ²) (gaussiano ingenuo, MV en entrenamiento)", C.red, 12);
  card(s, 5.0, 1.05, 4.55, 1.55, "Relación con Bayes", "Mínimo error: θ = P(inmaduro)/P(maduro). Variar θ recorre la ROC. Con costos, θ incorpora C₁₀ y C₀₁.", C.green, 12);
  image(s, "threshold_sweep.png", 0.4, 2.75, 9.2, 2.5);
  s.addNotes("La salida del modelo es un log-cociente de verosimilitudes sin priors; el prior y los costos quedan en el umbral.");
}

// 14. Operating point and results ------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 13, "Punto de operación y evaluación");
  const rows = [[hcell("Estrategia"), hcell("Entradas"), hcell("ln θ*"), hcell("AUC val"), hcell("AUC test"), hcell("Exact."), hcell("Sens."), hcell("Espec.")]];
  for (const r of strategies) {
    const t = r.test_metrics;
    rows.push([cell(r.name, { bold: true }), cell(feat(r.features)), cell(f2(r.threshold)), cell(f3(r.val_metrics.auc)), cell(f3(t.auc)), cell(f3(t.accuracy)), cell(f3(t.sensitivity)), cell(f3(t.specificity))]);
  }
  s.addTable(rows, tableOpts(0.5, 1.0, 9.0, [1.6, 1.5, 0.8, 0.85, 0.85, 0.8, 0.8, 0.8]));
  card(s, 0.5, 2.3, 4.0, 1.75, "Criterio: índice de Youden en validación", "J = sens + espec − 1. Sin costos definidos y clases balanceadas: ambos errores pesan igual. Umbral = punto medio entre puntuaciones; empates → centro.", C.green, 11);
  text(s, `Test de ${M.NTest} imágenes sin errores: compatible con exactitud real ≥ 0.74 (Clopper–Pearson 95 %). No permite ordenar las estrategias.`, 0.5, 4.2, 4.0, 1.0, { fontSize: 11.5, bold: true, color: C.red });
  image(s, "roc_curves.png", 4.7, 2.3, 4.85, 3.1);
  s.addNotes(`Los tres umbrales quedan sobre ln θ = 0. Con θ = 1 habría errores de validación en ${M.ZeroThrErrors}; ${M.ZeroThrOk} sin errores. La validación es separable: Youden da J = 1 en un intervalo y se toma su punto medio.`);
}

// 15. PCA -----------------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 14, "PCA y número de componentes");
  image(s, "pca_variance.png", 0.4, 1.0, 5.3, 2.9);
  image(s, "pca_loadings.png", 0.4, 3.95, 5.3, 1.5);
  const pca = data.pca;
  const rows = [[hcell("PC"), hcell("λ"), hcell("Var. %"), hcell("Acum. %")]];
  pca.eigenvalues.slice(0, 5).forEach((ev, i) => rows.push([cell(`PC${i + 1}`, { bold: i < pca.n_components }), cell(f2(ev)), cell((100 * pca.ratio[i]).toFixed(1)), cell((100 * pca.cumulative[i]).toFixed(1))]));
  s.addTable(rows, tableOpts(5.95, 1.05, 3.6, [0.8, 0.9, 0.95, 0.95]));
  card(s, 5.95, 2.85, 3.6, 2.35, `p = ${M.PCAn} (${M.PCAcum} %)`, "Menor p con ≥ 95 % de varianza; coincide con el codo y con Kaiser (λ₄ ≪ 1). PCA en entrenamiento sobre 8 variables estandarizadas. Decorrelaciona globalmente, pero no garantiza independencia dentro de cada clase.", C.orange, 11.5);
  s.addNotes("Solo PC1 está alineada con la madurez; PC2 y PC3 capturan brillo y amarillo, que dependen del fondo y la iluminación.");
}

// 16. Comparison -----------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 15, "Comparación de los tres clasificadores");
  image(s, "repeated_splits.png", 0.4, 1.0, 6.4, 2.6);
  stat(s, 0.5, 3.75, 2.1, M.SfsRepKmAcc.split(" ± ")[0], `Bayes + SFS (${M.SfsRepKmErr} errores)`, C.ink);
  stat(s, 2.65, 3.75, 2.1, M.AnRepKmAcc.split(" ± ")[0], `Bayes + análisis (${M.AnRepKmErr} errores)`, C.red);
  stat(s, 4.8, 3.75, 2.1, M.PcaRepKmAcc.split(" ± ")[0], `PCA + Bayes (${M.PcaRepKmErr} errores)`, C.orange);
  text(s, `Exactitud media en test, ${M.NRepeats} particiones, máscaras K-Means (${M.NRepPredictions} predicciones).`, 0.5, 4.85, 6.3, 0.4, { fontSize: 10.5, color: C.muted });
  bullets(s, [
    "Partición principal: las tres empatan con AUC y exactitud 1.",
    "Las tres difieren en pocos errores, dentro de la variabilidad: no se pueden ordenar.",
    `PCA vs análisis (gana/empata/pierde): ${M.PcaVsAn}. Estudio exploratorio, no usado para elegir.`,
    `Cada partición repite todo, incluida la segmentación (${M.RepComboFreq}).`,
    `Si se rehace la selección con máscaras ideales, SFS comete ${M.SfsRepRefErr} errores: posible sobreajuste de la selección.`,
  ], 7.0, 1.05, 2.6, 4.2, 11);
  s.addNotes("Mismas condiciones: mismas particiones, mismo umbral de Youden, mismas métricas. Cada partición elige también la combinación de canales y el post-procesado con su propio desarrollo. Las particiones comparten imágenes (incluidas las del test principal): miden estabilidad, no significancia, y son 360 predicciones sobre 60 imágenes, no 360 casos nuevos.");
}

// 17. Integrated analysis -------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 16, "Análisis integrado: segmentación → clasificación");
  image(s, "integration.png", 0.4, 0.95, 6.2, 2.15);
  const errs = (S) => ["An", "Sfs", "Pca"].map((k) => cell(M[`${k}Rep${S}Err`]));
  const rows = [
    [hcell("Variante (errores)"), hcell("Análisis"), hcell("SFS"), hcell("PCA")],
    [cell("K-Means (sistema)", { bold: true }), ...errs("Km")],
    [cell("Ideal solo en test, mismo modelo"), ...errs("Swap")],
    [cell("Ideal, diseño fijo (reentrena)"), ...errs("Fix")],
    [cell("Ideal, rediseño completo", { color: C.muted }), ...errs("Ref")],
    [cell("K-Means, H aritmético", { color: C.muted }), ...errs("Lin")],
  ];
  s.addTable(rows, { ...tableOpts(0.5, 3.2, 6.0, [2.7, 1.1, 1.1, 1.1]), rowH: 0.27 });
  text(s, `Errores acumulados en ${M.NRepeats} particiones (${M.NRepPredictions} predicciones por celda). Solo las dos primeras variantes ideales mantienen fija la selección.`, 0.5, 4.9, 6.0, 0.5, { fontSize: 10, color: C.muted });
  stat(s, 6.9, 1.05, 2.7, `ρ = ${M.RhoShift}`, "Jaccard vs desplazamiento de descriptores");
  bullets(s, [
    "Peor máscara → descriptores más contaminados.",
    `${M.TestLowJCorrect}/${M.NTest} imágenes de test con J < 0.2 se clasifican bien: el fondo tiene color compatible con la clase.`,
    "Con la selección fija y máscara ideal: 0 errores. La segmentación es la principal fuente de errores.",
    "Sin media circular, PCA pierde exactitud.",
  ], 6.9, 2.3, 2.7, 3.0, 11);
  s.addNotes("Ablación controlada: se cambia una sola cosa. Ideal solo en test mantiene variables, modelo y umbral. Diseño fijo mantiene variables y reajusta modelo y umbral. El rediseño completo no es controlado y no se usa para concluir.");
}

// 18. Answers -------------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, 17, "Respuestas a las preguntas del proyecto");
  const qa = [
    ["1. Mejor combinación", `${best}: J = ${M.BestDevJ} (${M.BestDevPostJ} con post-proceso).`],
    ["2. ¿Por qué?", "R separa rojo/verde; B separa fruto saturado de fondos claros."],
    ["3. Variables (EDA)", "H circular y a* (|d| > 3), luego G y R."],
    ["4. SFS", `${M.SFSFeatures}. Coincide solo en H con el análisis.`],
    ["5. Mejor estrategia", "Equivalentes dentro de la variabilidad; análisis es menos inestable e interpretable."],
    ["6. Punto de operación", "Youden en validación: sin costos, clases balanceadas."],
    ["7. Componentes", `${M.PCAn} PCs: ≥ 95 % varianza (${M.PCAcum} %), codo y Kaiser.`],
    ["8. ¿PCA mejora?", `No: empata en el test principal y al repetir (${M.PcaRepKmAcc.split(" ± ")[0]} de exactitud media).`],
    ["9. Segmentación", `Principal fuente de errores: con máscara ideal ${M.AnRepKmErr}/${M.SfsRepKmErr}/${M.PcaRepKmErr} → 0. Pero el fondo correlaciona con la clase.`],
  ];
  const w = 2.9, h = 1.3, gx = 0.15, gy = 0.13;
  qa.forEach(([q, a], i) => {
    const x = 0.5 + (i % 3) * (w + gx), y = 1.0 + Math.floor(i / 3) * (h + gy);
    card(s, x, y, w, h, q, a, [C.red, C.green, C.orange][i % 3], 11);
  });
  s.addNotes("Tener cada respuesta lista en menos de un minuto para la entrevista.");
}

// 19. Conclusions ----------------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: C.dark };
  s.addText("Conclusiones", { x: 0.6, y: 0.4, w: 8.8, h: 0.7, fontFace: HEAD, fontSize: 32, bold: true, color: C.white, margin: 0, isTextBox: true });
  const items = [
    ["Segmentación", `K-Means k=2 con ${best} + post-proceso: J ≈ ${M.BestDevPostJ}. Es el eslabón débil del sistema.`],
    ["Madurez = tono", "Con H como media circular, un Bayes con 3 variables clasifica casi perfectamente."],
    ["Selección", "Análisis, SFS y PCA equivalentes en este experimento; el análisis es menos inestable. PCA no mejora."],
    ["Limitaciones", `${M.NImages} imágenes y test de ${M.NTest}: techo de desempeño. El fondo correlaciona con la clase; sin textura ni forma.`],
  ];
  items.forEach(([t, b], i) => {
    const x = 0.6 + (i % 2) * 4.5, y = 1.35 + Math.floor(i / 2) * 1.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 4.3, h: 1.75, fill: { color: "2B3A42" }, line: { color: "2B3A42" }, rectRadius: 0.08 });
    s.addText(t, { x: x + 0.2, y: y + 0.15, w: 3.9, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: C.orange, margin: 0, isTextBox: true });
    s.addText(b, { x: x + 0.2, y: y + 0.6, w: 3.9, h: 1.05, fontFace: BODY, fontSize: 13, color: C.white, margin: 0, isTextBox: true, valign: "top" });
  });
  s.addNotes("Trabajo futuro: k>2 con varios clusters de fruto, textura y forma, Bayes con covarianza completa, más datos.");
}

const out = path.join(__dirname, "Proyecto1_Tomates.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("written", out));
