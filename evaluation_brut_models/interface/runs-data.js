/* Données statiques de la première version : aucun chargement réseau. */
window.initialRuns = [
  {
    label: "Baseline · batches non regroupés",
    createdAt: "2026-09-22_21-12-57Z",
    model: "Qwen3.5-0.8B",
    mode: "zero-shot",
    environment: "qwen-h100-finetuning",
    groupedBatches: false,
    paddingRate: 0.3924,
    executionAccuracy: 0.11783884489986027,
    exactMatch: 0.012109920819748486,
    syntaxValid: 0.9590125756870052,
    generationEps: 3.512,
    generationDuration: 10.301833333333333,
    evaluationDuration: 18.54,
    gpuPeak: 16941.6,
    energy: 49.495628,
    truncationRate: 0.0796
  },
  {
    label: "Baseline · batches regroupés par longueur",
    createdAt: "2026-09-22_21-24-39Z",
    model: "Qwen3.5-0.8B",
    mode: "zero-shot",
    environment: "qwen-h100-finetuning",
    groupedBatches: true,
    paddingRate: 0.0774,
    executionAccuracy: 0.1159757801583605,
    exactMatch: 0.011644154634373545,
    syntaxValid: 0.959944108057755,
    generationEps: 3.884,
    generationDuration: 9.3675,
    evaluationDuration: 12.91,
    gpuPeak: 16286.98,
    energy: 41.337276,
    truncationRate: 0.0787
  }
];
