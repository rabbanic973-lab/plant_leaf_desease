import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import multer from "multer";
import cors from "cors";

const upload = multer({ storage: multer.memoryStorage() });

async function startServer() {
  const app = express();
  const PORT = process.env.PORT || 7860;

  // Enable CORS so Vercel can talk to Render
  app.use(cors());
  app.use(express.json());

  // API endpoint for multiple image analysis
  app.post("/api/analyze-leaf", upload.array("images", 30), async (req, res) => {
    try {
      const files = req.files as Express.Multer.File[];
      if (!files || files.length === 0) {
        return res.status(400).json({ error: "No images uploaded" });
      }

      // Process each image independently
      const analysisPromises = files.map(async (file) => {
        const mimeType = file.mimetype;
        
        try {
          // Send to Python ML backend for 3-stage analysis
          const renderApiUrl = process.env.RENDER_VISION_API_URL || "http://localhost:8000";
          
          const formData = new FormData();
          const blob = new Blob([file.buffer], { type: mimeType });
          formData.append("file", blob, file.originalname);
          
          let visionData: any = { leaves: [], is_plant: true };
          try {
            const visionResponse = await fetch(`${renderApiUrl}/analyze-vision`, {
              method: "POST",
              body: formData
            });
            
            if (visionResponse.ok) {
              visionData = await visionResponse.json();
            } else {
              console.error("ML Backend Error:", await visionResponse.text());
            }
          } catch (e) {
            console.error("Failed to call ML Backend:", e);
          }

          // If the image was rejected by the CLIP gate
          if (visionData.is_plant === false) {
            return {
              filename: file.originalname,
              data: {
                is_plant: false,
                rejection_confidence: visionData.rejection_confidence || 0,
                message: visionData.message || "This image does not appear to contain a plant or leaf.",
                leaves: [],
                suggestions: [],
                prevention: [],
              }
            };
          }

          const leaves = Array.isArray(visionData.leaves) ? visionData.leaves : [];

          // Generate suggestions based on detected issues
          const detectedIssues = leaves
              .filter((l: any) => l.status === "Diseased" && l.diseaseName && l.diseaseName !== "null")
              .map((l: any) => l.diseaseName);

          let suggestions: string[] = [];
          let prevention: string[] = [];

          if (detectedIssues.length > 0) {
            const uniqueDiseases = [...new Set(detectedIssues)];
            suggestions = [
              `Detected: ${uniqueDiseases.join(", ")}`,
              "Apply appropriate fungicide or treatment for the identified disease(s).",
              "Prune and remove severely affected leaves to prevent spread.",
              "Consult a local agricultural extension office for targeted treatment."
            ];
            prevention = [
              "Improve air circulation around plants.",
              "Water at the base of the plant, avoiding leaf wetness.",
              "Rotate crops annually to break disease cycles.",
              "Use disease-resistant varieties when possible."
            ];
          } else {
            suggestions = [
              "Your plant appears healthy! Continue with regular care.",
              "Monitor regularly for any new symptoms."
            ];
            prevention = [
              "Ensure adequate light and air circulation.",
              "Maintain consistent watering schedule.",
              "Keep leaves dry when watering.",
              "Apply preventive organic treatments seasonally."
            ];
          }

          return {
            filename: file.originalname,
            data: {
              is_plant: true,
              plant_confidence: visionData.plant_confidence || 0,
              leaves,
              suggestions,
              prevention,
            }
          };
        } catch (err) {
          console.error(`Error processing ${file.originalname}:`, err);
          return { filename: file.originalname, error: "Failed to process this image" };
        }
      });

      const results = await Promise.all(analysisPromises);
      res.json({ results });
    } catch (error) {
      console.error("Error analyzing leaves:", error);
      res.status(500).json({ error: "Failed to analyze images." });
    }
  });

  // API endpoint for CSV data segregation
  app.post("/api/segregate-data", upload.single("dataset"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No CSV file uploaded" });
      }

      const renderApiUrl = process.env.RENDER_VISION_API_URL || "http://localhost:8000";
      const formData = new FormData();
      const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
      formData.append("dataset", blob, req.file.originalname);

      const pythonResponse = await fetch(`${renderApiUrl}/segregate-data`, {
        method: "POST",
        body: formData
      });

      if (pythonResponse.ok) {
        const data = await pythonResponse.json();
        res.setHeader('Content-Type', 'text/csv');
        res.setHeader('Content-Disposition', 'attachment; filename="segregated_data.csv"');
        res.send(data.csv_data);
      } else {
        res.status(pythonResponse.status).json({ error: "Python backend failed" });
      }
    } catch (error) {
      console.error("Error processing CSV:", error);
      res.status(500).json({ error: "Failed to process CSV data." });
    }
  });

  // API endpoint for leaf extraction from tree images
  app.post("/api/extract-leaves", upload.array("images", 10), async (req, res) => {
    try {
      const files = req.files as Express.Multer.File[];
      if (!files || files.length === 0) {
        return res.status(400).json({ error: "No images uploaded" });
      }

      const renderApiUrl = process.env.RENDER_VISION_API_URL || "http://localhost:8000";
      const formData = new FormData();
      files.forEach((file) => {
        const blob = new Blob([file.buffer], { type: file.mimetype });
        formData.append("images", blob, file.originalname);
      });

      const pythonResponse = await fetch(`${renderApiUrl}/extract-leaves`, {
        method: "POST",
        body: formData
      });

      if (pythonResponse.ok) {
        const data = await pythonResponse.json();
        res.json(data);
      } else {
        res.status(pythonResponse.status).json({ error: "Python backend failed" });
      }
    } catch (error) {
      console.error("Error extracting leaves:", error);
      res.status(500).json({ error: "Failed to extract leaves." });
    }
  });
  // Upload Training Data (images + optional CSV)
  app.post("/api/admin/upload-training-data", upload.fields([
    { name: "images", maxCount: 500 },
    { name: "dataset", maxCount: 1 }
  ]), async (req, res) => {
    try {
      const renderApiUrl = process.env.RENDER_VISION_API_URL || "http://localhost:8000";
      const formData = new FormData();

      const allFiles = req.files as { [fieldname: string]: Express.Multer.File[] };

      // Forward images
      if (allFiles?.images) {
        for (const file of allFiles.images) {
          const blob = new Blob([file.buffer], { type: file.mimetype });
          formData.append("images", blob, file.originalname);
        }
      }

      // Forward CSV
      if (allFiles?.dataset?.[0]) {
        const csv = allFiles.dataset[0];
        const blob = new Blob([csv.buffer], { type: csv.mimetype });
        formData.append("dataset", blob, csv.originalname);
      }

      const pythonResponse = await fetch(`${renderApiUrl}/upload-training-data`, {
        method: "POST",
        body: formData
      });

      if (pythonResponse.ok) {
        const data = await pythonResponse.json();
        res.json(data);
      } else {
        res.status(pythonResponse.status).json({ error: "Python backend failed" });
      }
    } catch (error) {
      console.error("Error uploading training data:", error);
      res.status(500).json({ error: "Failed to upload training data." });
    }
  });

  // Segment images for training (returns cropped leaf base64 images)
  app.post("/api/admin/segment-for-training", upload.array("images", 100), async (req, res) => {
    try {
      const files = req.files as Express.Multer.File[];
      if (!files || files.length === 0) {
        return res.status(400).json({ error: "No images uploaded" });
      }

      const renderApiUrl = process.env.RENDER_VISION_API_URL || "http://localhost:8000";
      const formData = new FormData();
      files.forEach((file) => {
        const blob = new Blob([file.buffer], { type: file.mimetype });
        formData.append("images", blob, file.originalname);
      });

      const pythonResponse = await fetch(`${renderApiUrl}/segment-for-training`, {
        method: "POST",
        body: formData
      });

      if (pythonResponse.ok) {
        const data = await pythonResponse.json();
        res.json(data);
      } else {
        res.status(pythonResponse.status).json({ error: "Segmentation failed" });
      }
    } catch (error) {
      console.error("Error segmenting for training:", error);
      res.status(500).json({ error: "Failed to segment images." });
    }
  });

  // Save labeled training leaves
  app.post("/api/admin/save-training-leaves", express.json({ limit: "200mb" }), async (req, res) => {
    try {
      const renderApiUrl = process.env.RENDER_VISION_API_URL || "http://localhost:8000";
      const pythonResponse = await fetch(`${renderApiUrl}/save-training-leaves`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body)
      });

      if (pythonResponse.ok) {
        const data = await pythonResponse.json();
        res.json(data);
      } else {
        res.status(pythonResponse.status).json({ error: "Save failed" });
      }
    } catch (error) {
      console.error("Error saving training leaves:", error);
      res.status(500).json({ error: "Failed to save training data." });
    }
  });

  // Trigger Model Training
  app.post("/api/admin/train-model", async (req, res) => {
    try {
      const renderApiUrl = process.env.RENDER_VISION_API_URL || "http://localhost:8000";
      const pythonResponse = await fetch(`${renderApiUrl}/train-model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" }
      });

      if (pythonResponse.ok) {
        const data = await pythonResponse.json();
        res.json(data);
      } else {
        res.status(pythonResponse.status).json({ error: "Python backend failed" });
      }
    } catch (error) {
      console.error("Error triggering training:", error);
      res.status(500).json({ error: "Failed to trigger training." });
    }
  });

  // Training Status
  app.get("/api/admin/training-status", async (req, res) => {
    try {
      const renderApiUrl = process.env.RENDER_VISION_API_URL || "http://localhost:8000";
      const pythonResponse = await fetch(`${renderApiUrl}/training-status`);

      if (pythonResponse.ok) {
        const data = await pythonResponse.json();
        res.json(data);
      } else {
        res.json({ status: "idle", has_custom_model: false });
      }
    } catch (error) {
      res.json({ status: "idle", has_custom_model: false });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    // Express 4 wildcard
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(Number(PORT), "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
