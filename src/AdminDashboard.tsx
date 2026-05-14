import React, { useState } from "react";
import { Lock, Upload, Play, CheckCircle, AlertCircle, Database, ShieldCheck, ChevronRight } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { cn } from "./lib/utils";

export default function AdminDashboard({ onBack }: { onBack: () => void }) {
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [status, setStatus] = useState<"IDLE" | "UPLOADING" | "TRAINING" | "DONE">("IDLE");
  const [message, setMessage] = useState("");

  const handleUpload = async () => {
    if (!csvFile) return;

    setStatus("UPLOADING");
    setMessage("Uploading dataset to vision engine...");

    const formData = new FormData();
    formData.append("dataset", csvFile);

    try {
      const response = await fetch("/api/admin/upload-training-data", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        setStatus("IDLE");
        setMessage("Data uploaded successfully. Ready to train.");
      } else {
        throw new Error("Upload failed");
      }
    } catch (err) {
      setStatus("IDLE");
      setMessage("Error during upload.");
    }
  };

  const handleTrain = async () => {
    setStatus("TRAINING");
    setMessage("Retraining PyTorch model on new leaf data... (Epochs 1-5)");

    try {
      const response = await fetch("/api/admin/train-model", {
        method: "POST",
      });

      if (response.ok) {
        setStatus("DONE");
        setMessage("Model successfully retrained and deployed to production.");
      } else {
        throw new Error("Training failed");
      }
    } catch (err) {
      setStatus("IDLE");
      setMessage("Error during training.");
    }
  };



  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-mono p-6 lg:p-12">
      <div className="max-w-4xl mx-auto">
        <header className="flex justify-between items-center mb-12 border-b border-slate-800 pb-8">
          <div className="flex items-center gap-4">
            <div className="bg-green-600 p-2 rounded-lg">
              <Database className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Model Management</h1>
              <p className="text-slate-500 text-sm">System ID: LEAFGUARD-V3-ALPHA</p>
            </div>
          </div>
          <button 
            onClick={onBack}
            className="text-slate-400 hover:text-white text-sm border border-slate-800 px-4 py-2 rounded-lg transition-all"
          >
            Back to Application
          </button>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Step 1: Upload */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 flex flex-col gap-6">
            <div className="flex items-center gap-3 text-white">
              <span className="bg-slate-800 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold">01</span>
              <h3 className="text-lg font-bold">Dataset Ingestion</h3>
            </div>
            
            <p className="text-slate-400 text-sm">Upload new leaf disease samples in CSV format for the training pipeline.</p>
            
            <div 
              className={cn(
                "border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center transition-all cursor-pointer",
                csvFile ? "border-green-500/50 bg-green-500/5" : "border-slate-700 bg-slate-800/50 hover:bg-slate-800"
              )}
              onClick={() => document.getElementById('admin-csv')?.click()}
            >
              <Upload className={cn("w-10 h-10 mb-4", csvFile ? "text-green-500" : "text-slate-600")} />
              <p className="text-sm font-medium text-center truncate w-full px-4">
                {csvFile ? csvFile.name : "Select Training CSV"}
              </p>
              <input 
                id="admin-csv"
                type="file" 
                accept=".csv"
                className="hidden"
                onChange={(e) => e.target.files && setCsvFile(e.target.files[0])}
              />
            </div>

            <button 
              onClick={handleUpload}
              disabled={!csvFile || status !== "IDLE"}
              className="w-full bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-white py-3 rounded-xl flex items-center justify-center gap-2 transition-all font-bold"
            >
              {status === "UPLOADING" ? "Uploading..." : "Upload Dataset"}
            </button>
          </div>

          {/* Step 2: Retrain */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 flex flex-col gap-6">
            <div className="flex items-center gap-3 text-white">
              <span className="bg-slate-800 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold">02</span>
              <h3 className="text-lg font-bold">Engine Retraining</h3>
            </div>
            
            <p className="text-slate-400 text-sm">Trigger the PyTorch training loop on the latest ingested leaf datasets.</p>
            
            <div className="flex-1 flex flex-col items-center justify-center bg-slate-800/50 rounded-xl p-6">
              <div className="w-16 h-16 bg-slate-900 rounded-full flex items-center justify-center border border-slate-700 mb-4">
                {status === "TRAINING" ? (
                   <div className="w-8 h-8 border-4 border-green-500 border-t-transparent rounded-full animate-spin" />
                ) : status === "DONE" ? (
                   <CheckCircle className="w-8 h-8 text-green-500" />
                ) : (
                   <Play className="w-8 h-8 text-slate-600 ml-1" />
                )}
              </div>
              <p className="text-xs text-slate-500 text-center uppercase tracking-tighter">Engine Status: {status}</p>
            </div>

            <button 
              onClick={handleTrain}
              disabled={status === "TRAINING" || status === "UPLOADING"}
              className="w-full bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white py-3 rounded-xl flex items-center justify-center gap-2 transition-all font-bold shadow-lg shadow-green-900/20"
            >
              {status === "TRAINING" ? "Training..." : "Start Retraining Loop"}
            </button>
          </div>
        </div>

        {/* System Message */}
        <AnimatePresence>
          {message && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-8 bg-slate-900 border border-slate-800 p-4 rounded-xl flex items-center gap-3"
            >
              <div className="bg-green-500/20 p-2 rounded-lg">
                <Database className="w-4 h-4 text-green-500" />
              </div>
              <p className="text-sm text-slate-300 font-mono tracking-tight">{message}</p>
            </motion.div>
          )}
        </AnimatePresence>

        <footer className="mt-12 text-center text-slate-600 text-xs">
          LEAFGUARD ENTERPRISE ADMIN PANEL • SECURE END-TO-END MODEL PIPELINE
        </footer>
      </div>
    </div>
  );
}
