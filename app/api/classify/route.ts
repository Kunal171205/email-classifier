import { type NextRequest, NextResponse } from "next/server"

interface ModelData {
  weights: Array<[string, number]>
  bias: number
}

const DEFAULT_MODEL: ModelData = {
  weights: [
    ["click", 0.85],
    ["free", 0.92],
    ["winner", 1.2],
    ["congratulations", 0.95],
    ["claim", 0.88],
    ["urgent", 0.78],
    ["act", 0.65],
    ["limited", 0.72],
    ["offer", 0.68],
    ["buy", 0.55],
    ["money", 0.82],
    ["cash", 0.9],
    ["prize", 1.1],
    ["verify", 0.75],
    ["confirm", 0.7],
    ["account", 0.45],
    ["password", 0.8],
    ["update", 0.6],
    ["alert", 0.65],
    ["action", 0.58],
    ["required", 0.62],
    ["immediately", 0.7],
    ["dear", -0.3],
    ["hello", -0.4],
    ["thanks", -0.5],
    ["regards", -0.45],
    ["best", -0.35],
    ["meeting", -0.3],
    ["project", -0.25],
    ["team", -0.2],
  ],
  bias: -0.5,
}

function preprocessText(text: string): string[] {
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

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

function calculateSHAPValues(
  text: string,
  weights: Map<string, number>,
  bias: number,
): Array<{ word: string; shap: number; contribution: string }> {
  const tokens = preprocessText(text)
  const features = new Map<string, number>()

  for (const token of tokens) {
    features.set(token, (features.get(token) || 0) + 1)
  }

  const shapValues: Array<{ word: string; shap: number; contribution: string }> = []

  for (const [word, count] of features) {
    const weight = weights.get(word) || 0
    const shap = weight * count
    if (shap !== 0) {
      shapValues.push({
        word,
        shap,
        contribution: shap > 0 ? "spam" : "legitimate",
      })
    }
  }

  shapValues.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
  return shapValues.slice(0, 10)
}

export async function POST(request: NextRequest) {
  try {
    const { text } = await request.json()

    if (!text || typeof text !== "string") {
      return NextResponse.json({ error: "Invalid input" }, { status: 400 })
    }

    const model = DEFAULT_MODEL
    const weights = new Map(model.weights)

    // Extract features
    const tokens = preprocessText(text)
    const features = new Map<string, number>()

    for (const token of tokens) {
      features.set(token, (features.get(token) || 0) + 1)
    }

    // Calculate prediction
    let z = model.bias

    for (const [word, count] of features) {
      if (weights.has(word)) {
        z += weights.get(word)! * count
      }
    }

    const probability = sigmoid(z)
    const isSpam = probability > 0.5

    const shapExplanations = calculateSHAPValues(text, weights, model.bias)

    return NextResponse.json({
      prediction: isSpam ? "Spam" : "Legitimate",
      confidence: Math.max(probability, 1 - probability),
      spamScore: probability,
      shap: shapExplanations,
    })
  } catch (error) {
    console.error("[v0] Classification error:", error)
    return NextResponse.json({ error: "Classification failed. Please try again." }, { status: 500 })
  }
}
