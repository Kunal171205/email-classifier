import fs from "fs"
import path from "path"

// Simple Logistic Regression implementation with SHAP support
class LogisticRegression {
  private weights: Map<string, number> = new Map()
  private bias = 0
  private learningRate = 0.01
  private iterations = 100
  private baselineScore = 0.5 // Average prediction for SHAP baseline

  // Tokenize and preprocess text
  private preprocessText(text: string): string[] {
    const processed = text.toLowerCase().replace(/[^a-z0-9\s]/g, "")

    const stopwords = new Set([
      "the",
      "a",
      "an",
      "and",
      "or",
      "but",
      "in",
      "on",
      "at",
      "to",
      "for",
      "of",
      "with",
      "by",
      "from",
      "is",
      "are",
      "was",
      "were",
      "be",
      "been",
      "being",
      "have",
      "has",
      "had",
      "do",
      "does",
      "did",
      "will",
      "would",
      "could",
      "should",
      "may",
      "might",
      "must",
      "can",
      "this",
      "that",
      "these",
      "those",
      "i",
      "you",
      "he",
      "she",
      "it",
      "we",
      "they",
    ])

    return processed.split(/\s+/).filter((word) => !stopwords.has(word) && word.length > 1)
  }

  // Extract features from text
  private extractFeatures(text: string): Map<string, number> {
    const tokens = this.preprocessText(text)
    const features = new Map<string, number>()

    for (const token of tokens) {
      features.set(token, (features.get(token) || 0) + 1)
    }

    return features
  }

  // Sigmoid function
  private sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-z))
  }

  // Train the model
  train(texts: string[], labels: number[]): { accuracy: number; precision: number; recall: number } {
    // Build vocabulary
    const vocabulary = new Set<string>()
    for (const text of texts) {
      const tokens = this.preprocessText(text)
      tokens.forEach((token) => vocabulary.add(token))
    }

    // Initialize weights
    for (const word of vocabulary) {
      this.weights.set(word, Math.random() * 0.01)
    }

    // Training loop
    for (let iter = 0; iter < this.iterations; iter++) {
      let totalLoss = 0

      for (let i = 0; i < texts.length; i++) {
        const features = this.extractFeatures(texts[i])
        let z = this.bias

        for (const [word, count] of features) {
          z += (this.weights.get(word) || 0) * count
        }

        const prediction = this.sigmoid(z)
        const error = prediction - labels[i]
        totalLoss += error * error

        // Update bias
        this.bias -= this.learningRate * error

        // Update weights
        for (const [word, count] of features) {
          const currentWeight = this.weights.get(word) || 0
          this.weights.set(word, currentWeight - this.learningRate * error * count)
        }
      }
    }

    // Calculate metrics
    let correct = 0
    let truePositives = 0
    let falsePositives = 0
    let falseNegatives = 0

    for (let i = 0; i < texts.length; i++) {
      const features = this.extractFeatures(texts[i])
      let z = this.bias

      for (const [word, count] of features) {
        z += (this.weights.get(word) || 0) * count
      }

      const prediction = this.sigmoid(z) > 0.5 ? 1 : 0

      if (prediction === labels[i]) {
        correct++
      }

      if (prediction === 1 && labels[i] === 1) truePositives++
      if (prediction === 1 && labels[i] === 0) falsePositives++
      if (prediction === 0 && labels[i] === 1) falseNegatives++
    }

    const accuracy = correct / texts.length
    const precision = truePositives / (truePositives + falsePositives || 1)
    const recall = truePositives / (truePositives + falseNegatives || 1)

    return { accuracy, precision, recall }
  }

  // Predict
  predict(text: string): number {
    const features = this.extractFeatures(text)
    let z = this.bias

    for (const [word, count] of features) {
      z += (this.weights.get(word) || 0) * count
    }

    return this.sigmoid(z)
  }

  // Calculate SHAP values for feature importance
  explainPrediction(text: string): { features: Array<{ word: string; shap: number; contribution: string }> } {
    const tokens = this.preprocessText(text)
    const features = new Map<string, number>()

    for (const token of tokens) {
      features.set(token, (features.get(token) || 0) + 1)
    }

    // Calculate contribution of each feature
    const shapValues: Array<{ word: string; shap: number; contribution: string }> = []

    for (const [word, count] of features) {
      const weight = this.weights.get(word) || 0
      const shap = weight * count // Simplified SHAP: weight * feature value
      shapValues.push({
        word,
        shap,
        contribution: shap > 0 ? "spam" : "legitimate",
      })
    }

    // Sort by absolute SHAP value
    shapValues.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))

    return { features: shapValues.slice(0, 10) } // Top 10 features
  }

  // Save model
  saveModel(filePath: string): void {
    const modelData = {
      weights: Array.from(this.weights.entries()),
      bias: this.bias,
      baselineScore: this.baselineScore,
    }
    fs.writeFileSync(filePath, JSON.stringify(modelData, null, 2))
  }

  // Load model
  loadModel(filePath: string): void {
    const data = JSON.parse(fs.readFileSync(filePath, "utf-8"))
    this.weights = new Map(data.weights)
    this.bias = data.bias
    this.baselineScore = data.baselineScore || 0.5
  }
}

// Main training function
async function trainModel() {
  console.log("[v0] Starting model training...")

  // Read CSV file
  const csvPath = path.join(process.cwd(), "spam.csv")
  const csvContent = fs.readFileSync(csvPath, "utf-8")
  const lines = csvContent.split("\n").slice(1) // Skip header

  const texts: string[] = []
  const labels: number[] = []

  for (const line of lines) {
    if (!line.trim()) continue

    const parts = line.split(",")
    if (parts.length < 2) continue

    const label = parts[0].trim().toLowerCase()
    const text = parts.slice(1).join(",").trim().replace(/^"|"$/g, "")

    if (label === "spam" || label === "ham") {
      labels.push(label === "spam" ? 1 : 0)
      texts.push(text)
    }
  }

  console.log(`[v0] Loaded ${texts.length} samples`)

  // Train model
  const model = new LogisticRegression()
  const metrics = model.train(texts, labels)

  console.log(`[v0] Training complete!`)
  console.log(`[v0] Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`)
  console.log(`[v0] Precision: ${(metrics.precision * 100).toFixed(2)}%`)
  console.log(`[v0] Recall: ${(metrics.recall * 100).toFixed(2)}%`)

  // Save model
  const modelPath = path.join(process.cwd(), "public", "model.json")
  model.saveModel(modelPath)
  console.log(`[v0] Model saved to ${modelPath}`)
}

trainModel().catch(console.error)
