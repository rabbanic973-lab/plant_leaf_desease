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
        const base64Data = file.buffer.toString("base64");
        
        try {
          // STEP 1: Vision Model via external ML service (Render)
          const renderApiUrl = process.env.RENDER_VISION_API_URL || "http://localhost:8000";
          
          const formData = new FormData();
          const blob = new Blob([file.buffer], { type: mimeType });
          formData.append("file", blob, file.originalname);
          
          let leaves = [];
          try {
            const visionResponse = await fetch(`${renderApiUrl}/analyze-vision`, {
              method: "POST",
              body: formData
            });
            
            if (visionResponse.ok) {
              const visionData = await visionResponse.json();
              leaves = Array.isArray(visionData.leaves) ? visionData.leaves : [];
            } else {
              console.error("Render API Error:", await visionResponse.text());
            }
          } catch (e) {
            console.error("Failed to call Render API:", e);
          }

          // STEP 2: Suggestions based on detected issues
          const detectedIssues = leaves
              .filter((l: any) => l.status === "Diseased" && l.diseaseName && l.diseaseName !== "null")
              .map((l: any) => l.diseaseName);

          let suggestions: string[] = [];
          let prevention: string[] = [];

          if (detectedIssues.length > 0) {
            suggestions = ["Apply suitable fungicide.", "Prune affected areas."];
            prevention = ["Improve air circulation.", "Water the base of the plant."];
          } else {
            suggestions = ["Continue normal care and watering.", "Monitor the plant regularly for any new symptoms."];
            prevention = ["Ensure adequate light and air circulation.", "Do not overwater.", "Keep leaves dry when watering."];
          }

          const finalData = {
            leaves,
            suggestions,
            prevention
          };
          
          return { filename: file.originalname, data: finalData };
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
  // Upload Training Data
  app.post("/api/admin/upload-training-data", upload.single("dataset"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No CSV file uploaded" });
      }

      const renderApiUrl = process.env.RENDER_VISION_API_URL || "http://localhost:8000";
      const formData = new FormData();
      const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
      formData.append("dataset", blob, req.file.originalname);

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
