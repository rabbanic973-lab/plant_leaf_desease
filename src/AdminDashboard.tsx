import React, { useState, useEffect } from "react";
import { Upload, Play, CheckCircle, AlertCircle, Database, Eye, Tag, BarChart3, Loader2, ArrowRight } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { cn } from "./lib/utils";

type SegLeaf = { id: string; source_image: string; image_b64: string; box: number[]; confidence: number };
type EpochMetric = { epoch: number; train_loss: number; train_accuracy: number; train_confidence: number; val_accuracy: number; val_confidence: number };

const DISEASE_OPTIONS = ["Healthy","Early_Blight","Late_Blight","Powdery_Mildew","Rust","Bacterial_Spot","Leaf_Mold","Target_Spot","Mosaic_Virus","Leaf_Scorch","Black_Rot","Cedar_Rust","Septoria","Other_Disease"];

export default function AdminDashboard({ onBack }: { onBack: () => void }) {
  const [step, setStep] = useState<1|2|3|4>(1);
  // Step 1
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [csvFile, setCsvFile] = useState<File|null>(null);
  const [uploadStatus, setUploadStatus] = useState<"idle"|"uploading"|"done"|"error">("idle");
  const [uploadMsg, setUploadMsg] = useState("");
  // Step 2
  const [segLeaves, setSegLeaves] = useState<SegLeaf[]>([]);
  const [leafLabels, setLeafLabels] = useState<Record<string,string>>({});
  const [segStatus, setSegStatus] = useState<"idle"|"segmenting"|"done">("idle");
  const [saveStatus, setSaveStatus] = useState<"idle"|"saving"|"done">("idle");
  // Step 3
  const [trainStatus, setTrainStatus] = useState<"idle"|"training"|"done"|"error">("idle");
  const [metrics, setMetrics] = useState<EpochMetric[]>([]);
  const [trainMsg, setTrainMsg] = useState("");
  // Step 4
  const [modelMeta, setModelMeta] = useState<any>(null);

  // Poll training status
  useEffect(() => {
    if (trainStatus !== "training") return;
    const interval = setInterval(async () => {
      try {
        const r = await fetch("/api/admin/training-status");
        if (r.ok) {
          const d = await r.json();
          if (d.metrics) setMetrics(d.metrics);
          if (d.status === "done") { setTrainStatus("done"); setTrainMsg(d.message || "Done!"); setModelMeta(d.custom_model_meta); clearInterval(interval); }
          if (d.status === "error") { setTrainStatus("error"); setTrainMsg(d.message || "Error"); clearInterval(interval); }
        }
      } catch {}
    }, 2000);
    return () => clearInterval(interval);
  }, [trainStatus]);

  // Step 1: Upload images + CSV
  const handleUpload = async () => {
    if (imageFiles.length === 0 && !csvFile) return;
    setUploadStatus("uploading");
    setUploadMsg("Uploading & segmenting leaves...");
    const fd = new FormData();
    imageFiles.forEach(f => fd.append("images", f));
    if (csvFile) fd.append("dataset", csvFile);
    try {
      const r = await fetch("/api/admin/upload-training-data", { method: "POST", body: fd });
      if (r.ok) {
        const d = await r.json();
        setUploadStatus("done");
        setUploadMsg(`${d.message || "Done"} — ${JSON.stringify(d.class_distribution || {})}`);
      } else { throw new Error("Upload failed"); }
    } catch { setUploadStatus("error"); setUploadMsg("Upload failed."); }
  };

  // Step 2: Segment for review
  const handleSegment = async () => {
    if (imageFiles.length === 0) return;
    setSegStatus("segmenting");
    const fd = new FormData();
    imageFiles.slice(0, 20).forEach(f => fd.append("images", f));
    try {
      const r = await fetch("/api/admin/segment-for-training", { method: "POST", body: fd });
      if (r.ok) {
        const d = await r.json();
        setSegLeaves(d.leaves || []);
        const labels: Record<string,string> = {};
        (d.leaves || []).forEach((l: SegLeaf) => { labels[l.id] = "Healthy"; });
        setLeafLabels(labels);
        setSegStatus("done");
      } else { setSegStatus("idle"); }
    } catch { setSegStatus("idle"); }
  };

  // Step 2: Save labeled leaves
  const handleSaveLabels = async () => {
    setSaveStatus("saving");
    const payload = segLeaves.map(l => ({ image_b64: l.image_b64, label: leafLabels[l.id] || "Healthy" }));
    try {
      const r = await fetch("/api/admin/save-training-leaves", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ leaves: payload })
      });
      if (r.ok) setSaveStatus("done");
      else setSaveStatus("idle");
    } catch { setSaveStatus("idle"); }
  };

  // Step 3: Train
  const handleTrain = async () => {
    setTrainStatus("training"); setMetrics([]); setTrainMsg("Starting training...");
    try {
      const r = await fetch("/api/admin/train-model", { method: "POST" });
      if (r.ok) {
        const d = await r.json();
        setTrainStatus("done"); setMetrics(d.metrics || []);
        setTrainMsg(d.message || "Training complete!");
        setModelMeta({ class_names: d.classes, best_val_accuracy: d.best_accuracy });
      } else { throw new Error(); }
    } catch { setTrainStatus("error"); setTrainMsg("Training failed."); }
  };

  const lastMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-mono p-6 lg:p-12">
      <div className="max-w-5xl mx-auto">
        <header className="flex justify-between items-center mb-8 border-b border-slate-800 pb-6">
          <div className="flex items-center gap-4">
            <div className="bg-green-600 p-2 rounded-lg"><Database className="w-6 h-6 text-white" /></div>
            <div>
              <h1 className="text-2xl font-bold text-white">Training Pipeline</h1>
              <p className="text-slate-500 text-sm">Tree → Segment Leaves → Label → Train → Predict</p>
            </div>
          </div>
          <button onClick={onBack} className="text-slate-400 hover:text-white text-sm border border-slate-800 px-4 py-2 rounded-lg">Back</button>
        </header>

        {/* Step Indicator */}
        <div className="flex items-center gap-2 mb-10 overflow-x-auto pb-2">
          {[
            { n: 1, label: "Upload Data" },
            { n: 2, label: "Review Leaves" },
            { n: 3, label: "Train Model" },
            { n: 4, label: "Metrics" }
          ].map((s, i) => (
            <React.Fragment key={s.n}>
              <button onClick={() => setStep(s.n as 1|2|3|4)}
                className={cn("px-4 py-2 rounded-lg text-xs font-bold transition-all whitespace-nowrap",
                  step === s.n ? "bg-green-600 text-white" : "bg-slate-800 text-slate-400 hover:bg-slate-700")}>
                {`0${s.n}. ${s.label}`}
              </button>
              {i < 3 && <ArrowRight className="w-4 h-4 text-slate-700 shrink-0" />}
            </React.Fragment>
          ))}
        </div>

        {/* Step 1: Upload */}
        {step === 1 && (
          <motion.div initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Image Upload */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                <h3 className="font-bold text-white mb-3 flex items-center gap-2"><Upload className="w-4 h-4 text-green-500" /> Plant/Tree Images</h3>
                <p className="text-slate-500 text-xs mb-4">Upload whole plant/tree images. Leaves will be auto-segmented.</p>
                <div className={cn("border-2 border-dashed rounded-xl p-6 flex flex-col items-center cursor-pointer transition-all",
                  imageFiles.length > 0 ? "border-green-500/50 bg-green-500/5" : "border-slate-700 hover:bg-slate-800")}
                  onClick={() => document.getElementById("train-imgs")?.click()}>
                  <Upload className={cn("w-8 h-8 mb-2", imageFiles.length > 0 ? "text-green-500" : "text-slate-600")} />
                  <p className="text-sm">{imageFiles.length > 0 ? `${imageFiles.length} images selected` : "Select images"}</p>
                  <input id="train-imgs" type="file" accept="image/*" multiple className="hidden"
                    onChange={e => e.target.files && setImageFiles(Array.from(e.target.files))} />
                </div>
              </div>
              {/* CSV Upload */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                <h3 className="font-bold text-white mb-3 flex items-center gap-2"><Database className="w-4 h-4 text-blue-500" /> CSV Labels (Optional)</h3>
                <p className="text-slate-500 text-xs mb-4">CSV with columns: filename, disease_class. Maps images to disease labels.</p>
                <div className={cn("border-2 border-dashed rounded-xl p-6 flex flex-col items-center cursor-pointer transition-all",
                  csvFile ? "border-blue-500/50 bg-blue-500/5" : "border-slate-700 hover:bg-slate-800")}
                  onClick={() => document.getElementById("train-csv")?.click()}>
                  <Database className={cn("w-8 h-8 mb-2", csvFile ? "text-blue-500" : "text-slate-600")} />
                  <p className="text-sm">{csvFile ? csvFile.name : "Select CSV"}</p>
                  <input id="train-csv" type="file" accept=".csv" className="hidden"
                    onChange={e => e.target.files?.[0] && setCsvFile(e.target.files[0])} />
                </div>
              </div>
            </div>
            <div className="flex gap-4">
              <button onClick={handleUpload} disabled={imageFiles.length === 0 || uploadStatus === "uploading"}
                className="bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white py-3 px-8 rounded-xl font-bold flex items-center gap-2">
                {uploadStatus === "uploading" ? <><Loader2 className="w-4 h-4 animate-spin" /> Segmenting...</> : "Upload & Auto-Segment"}
              </button>
              <button onClick={() => { setStep(2); handleSegment(); }} disabled={imageFiles.length === 0}
                className="bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-white py-3 px-8 rounded-xl font-bold flex items-center gap-2">
                <Eye className="w-4 h-4" /> Preview Segmented Leaves
              </button>
            </div>
            {uploadMsg && (
              <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex items-center gap-3">
                {uploadStatus === "done" ? <CheckCircle className="w-5 h-5 text-green-500" /> : <AlertCircle className="w-5 h-5 text-red-500" />}
                <p className="text-sm text-slate-300">{uploadMsg}</p>
              </div>
            )}
          </motion.div>
        )}

        {/* Step 2: Review & Label Segmented Leaves */}
        {step === 2 && (
          <motion.div initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} className="space-y-6">
            {segStatus === "segmenting" && (
              <div className="flex items-center gap-3 bg-slate-900 p-6 rounded-2xl border border-slate-800">
                <Loader2 className="w-6 h-6 text-green-500 animate-spin" />
                <p>Segmenting leaves from uploaded images...</p>
              </div>
            )}
            {segLeaves.length > 0 && (
              <>
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-bold text-white">{segLeaves.length} Leaves Detected — Label Each</h3>
                  <button onClick={handleSaveLabels} disabled={saveStatus === "saving"}
                    className="bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white py-2 px-6 rounded-xl font-bold flex items-center gap-2">
                    {saveStatus === "saving" ? <Loader2 className="w-4 h-4 animate-spin" /> : <Tag className="w-4 h-4" />}
                    {saveStatus === "done" ? "Saved ✓" : "Save Labels & Proceed"}
                  </button>
                </div>
                <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3 max-h-[60vh] overflow-y-auto p-1">
                  {segLeaves.map(leaf => (
                    <div key={leaf.id} className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col">
                      <img src={`data:image/jpeg;base64,${leaf.image_b64}`} className="w-full aspect-square object-cover" />
                      <select value={leafLabels[leaf.id] || "Healthy"}
                        onChange={e => setLeafLabels(p => ({...p, [leaf.id]: e.target.value}))}
                        className="bg-slate-800 text-white text-[10px] p-1.5 border-0 w-full focus:ring-1 focus:ring-green-500">
                        {DISEASE_OPTIONS.map(d => <option key={d} value={d}>{d.replace(/_/g, " ")}</option>)}
                      </select>
                    </div>
                  ))}
                </div>
              </>
            )}
            {segLeaves.length === 0 && segStatus !== "segmenting" && (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center">
                <p className="text-slate-500">No segmented leaves yet. Go to Step 1 to upload images, then click "Preview Segmented Leaves".</p>
              </div>
            )}
          </motion.div>
        )}

        {/* Step 3: Train */}
        {step === 3 && (
          <motion.div initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 flex flex-col items-center gap-6">
              <div className="w-20 h-20 bg-slate-800 rounded-full flex items-center justify-center border border-slate-700">
                {trainStatus === "training" ? <Loader2 className="w-10 h-10 text-green-500 animate-spin" />
                  : trainStatus === "done" ? <CheckCircle className="w-10 h-10 text-green-500" />
                  : <Play className="w-10 h-10 text-slate-600 ml-1" />}
              </div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">
                {trainStatus === "training" ? `Training... ${lastMetric ? `Epoch ${lastMetric.epoch}` : ""}` : trainStatus === "done" ? "Complete" : "Ready"}
              </p>
              {lastMetric && trainStatus === "training" && (
                <div className="flex gap-6 text-center">
                  <div><p className="text-2xl font-bold text-green-400">{(lastMetric.train_accuracy*100).toFixed(1)}%</p><p className="text-[10px] text-slate-500">Train Acc</p></div>
                  <div><p className="text-2xl font-bold text-blue-400">{(lastMetric.val_accuracy*100).toFixed(1)}%</p><p className="text-[10px] text-slate-500">Val Acc</p></div>
                  <div><p className="text-2xl font-bold text-yellow-400">{(lastMetric.val_confidence*100).toFixed(1)}%</p><p className="text-[10px] text-slate-500">Val Conf</p></div>
                </div>
              )}
              <button onClick={handleTrain} disabled={trainStatus === "training"}
                className="bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white py-3 px-10 rounded-xl font-bold shadow-lg shadow-green-900/20">
                {trainStatus === "training" ? "Training..." : "Start Training (MobileNetV2)"}
              </button>
              {trainMsg && <p className="text-sm text-slate-300 bg-slate-800 px-4 py-2 rounded-lg">{trainMsg}</p>}
            </div>

            {/* Live Metrics Table */}
            {metrics.length > 0 && (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 overflow-x-auto">
                <h3 className="font-bold text-white mb-4 flex items-center gap-2"><BarChart3 className="w-4 h-4 text-green-500" /> Training Metrics</h3>
                <table className="w-full text-xs">
                  <thead><tr className="text-slate-500 border-b border-slate-800">
                    <th className="py-2 text-left">Epoch</th><th className="py-2 text-right">Loss</th>
                    <th className="py-2 text-right">Train Acc</th><th className="py-2 text-right">Train Conf</th>
                    <th className="py-2 text-right">Val Acc</th><th className="py-2 text-right">Val Conf</th>
                  </tr></thead>
                  <tbody>
                    {metrics.map(m => (
                      <tr key={m.epoch} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                        <td className="py-1.5">{m.epoch}</td>
                        <td className="py-1.5 text-right text-red-400">{m.train_loss.toFixed(4)}</td>
                        <td className="py-1.5 text-right text-green-400">{(m.train_accuracy*100).toFixed(1)}%</td>
                        <td className="py-1.5 text-right text-green-300">{(m.train_confidence*100).toFixed(1)}%</td>
                        <td className="py-1.5 text-right text-blue-400">{(m.val_accuracy*100).toFixed(1)}%</td>
                        <td className="py-1.5 text-right text-yellow-400">{(m.val_confidence*100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </motion.div>
        )}

        {/* Step 4: Results */}
        {step === 4 && (
          <motion.div initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} className="space-y-6">
            {modelMeta ? (
              <div className="bg-slate-900 border border-green-800/50 rounded-2xl p-8 space-y-6">
                <div className="flex items-center gap-3">
                  <CheckCircle className="w-8 h-8 text-green-500" />
                  <div>
                    <h3 className="text-xl font-bold text-white">Custom Model Active</h3>
                    <p className="text-slate-500 text-sm">Your fine-tuned model is now used for all predictions.</p>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-slate-800 p-4 rounded-xl text-center">
                    <p className="text-3xl font-bold text-green-400">{modelMeta.num_classes || modelMeta.class_names?.length || "?"}</p>
                    <p className="text-xs text-slate-500 mt-1">Classes</p>
                  </div>
                  <div className="bg-slate-800 p-4 rounded-xl text-center">
                    <p className="text-3xl font-bold text-blue-400">{modelMeta.best_val_accuracy ? (modelMeta.best_val_accuracy*100).toFixed(1)+"%" : "N/A"}</p>
                    <p className="text-xs text-slate-500 mt-1">Best Val Accuracy</p>
                  </div>
                  <div className="bg-slate-800 p-4 rounded-xl text-center">
                    <p className="text-3xl font-bold text-yellow-400">MobileNetV2</p>
                    <p className="text-xs text-slate-500 mt-1">Architecture</p>
                  </div>
                </div>
                {modelMeta.class_names && (
                  <div>
                    <h4 className="text-sm font-bold text-slate-400 mb-2">Trained Classes</h4>
                    <div className="flex flex-wrap gap-2">
                      {modelMeta.class_names.map((c: string) => (
                        <span key={c} className="bg-slate-800 text-slate-300 px-3 py-1 rounded-lg text-xs">{c.replace(/_/g, " ")}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center">
                <p className="text-slate-500">No custom model trained yet. Complete steps 1-3 first.</p>
              </div>
            )}

            {metrics.length > 0 && (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                <h3 className="font-bold text-white mb-4">Accuracy Curve</h3>
                <div className="h-40 flex items-end gap-1">
                  {metrics.map(m => (
                    <div key={m.epoch} className="flex-1 flex flex-col items-center gap-1">
                      <div className="w-full bg-green-500/80 rounded-t" style={{height: `${m.train_accuracy * 100}%`}} />
                      <span className="text-[8px] text-slate-500">{m.epoch}</span>
                    </div>
                  ))}
                </div>
                <div className="flex justify-between text-[10px] text-slate-500 mt-2">
                  <span>Epoch 1</span><span>Epoch {metrics.length}</span>
                </div>
              </div>
            )}
          </motion.div>
        )}

        <footer className="mt-12 text-center text-slate-600 text-xs">
          LEAFGUARD TRAINING PIPELINE • KAGGLE LEAF DETECTION DATASET COMPATIBLE
        </footer>
      </div>
    </div>
  );
}
