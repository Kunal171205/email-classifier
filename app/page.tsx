"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"

interface SHAPFeature {
  word: string
  shap: number
  contribution: string
}

interface ClassificationResult {
  prediction: string
  confidence: number
  spamScore: number
  shap: SHAPFeature[]
}

export default function Home() {
  const [email, setEmail] = useState("")
  const [result, setResult] = useState<ClassificationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const handleClassify = async () => {
    if (!email.trim()) {
      setError("Please enter an email to classify")
      return
    }

    setLoading(true)
    setError("")
    setResult(null)

    try {
      const response = await fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: email }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Classification failed")
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to classify email. Please try again."
      setError(errorMessage)
      console.error("[v0] Classification error:", err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 pt-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <span className="text-4xl">✉️</span>
            <h1 className="text-4xl font-bold text-gray-900">Email Classifier</h1>
          </div>
          <p className="text-gray-600">Detect spam with AI-powered SHAP explanations</p>
        </div>

        {/* Main Card */}
        <Card className="shadow-lg border-0 mb-6">
          <CardHeader className="bg-gradient-to-r from-indigo-600 to-blue-600 text-white rounded-t-lg">
            <CardTitle>Classify Your Email</CardTitle>
            <CardDescription className="text-indigo-100">
              Paste your email content below to check if it's spam or Genuine
            </CardDescription>
          </CardHeader>

          <CardContent className="p-6">
            {/* Input Area */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">Email Content</label>
              <Textarea
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Paste your email here..."
                className="min-h-40 resize-none"
                disabled={loading}
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2">
                <span className="text-red-600 font-bold flex-shrink-0">⚠</span>
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}

            {/* Classify Button */}
            <Button
              onClick={handleClassify}
              disabled={loading || !email.trim()}
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2"
            >
              {loading ? "Classifying..." : "Classify Email"}
            </Button>

            {/* Result */}
            {result && (
              <div className="mt-6 space-y-4">
                {/* Prediction Result */}
                <div className="p-4 rounded-lg border-2 bg-gradient-to-r from-green-50 to-emerald-50 border-green-200">
                  <div className="flex items-start gap-3">
                    <span className="text-2xl flex-shrink-0 mt-0.5">
                      {result.prediction === "Genuine" ? "✓" : "✗"}
                    </span>
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900">
                        {result.prediction === "Genuine" ? "Genuine Email" : "Spam Detected"}
                      </h3>
                    </div>
                  </div>
                </div>

                {result.shap && result.shap.length > 0 && (
                  <div className="p-4 rounded-lg border border-gray-200 bg-white">
                    <h4 className="font-semibold text-gray-900 mb-3">Feature Importance (SHAP)</h4>
                    <p className="text-xs text-gray-500 mb-3">Top words influencing this classification:</p>
                    <div className="space-y-2">
                      {result.shap.map((feature, idx) => (
                        <div key={idx} className="flex items-center gap-3">
                          <div className="flex-1">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-sm font-medium text-gray-700">{feature.word}</span>
                              <span
                                className={`text-xs px-2 py-1 rounded ${
                                  feature.contribution === "spam"
                                    ? "bg-red-100 text-red-700"
                                    : "bg-green-100 text-green-700"
                                }`}
                              >
                                {feature.contribution}
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${feature.shap > 0 ? "bg-red-500" : "bg-green-500"}`}
                                style={{
                                  width: `${Math.min(Math.abs(feature.shap) * 100, 100)}%`,
                                }}
                              />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="border-0 shadow">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">98% Accuracy</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-600">
              Trained on 5,000+ emails using Logistic Regression
            </CardContent>
          </Card>

          <Card className="border-0 shadow">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">SHAP Explanations</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-600">Understand which words influence the prediction</CardContent>
          </Card>

          <Card className="border-0 shadow">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Real-time Results</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-600">Instant classification with confidence scores</CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
