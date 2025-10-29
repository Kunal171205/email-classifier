"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts"
import Link from "next/link"

export default function Dashboard() {
  const modelMetrics = [
    { metric: "Accuracy", value: "98.07%" },
    { metric: "Precision", value: "99.17%" },
    { metric: "Recall", value: "86.23%" },
    { metric: "F1-Score", value: "92.25%" },
  ]

  const confusionData = [
    { name: "True Negatives", value: 895, fill: "#10b981" },
    { name: "False Positives", value: 1, fill: "#f59e0b" },
    { name: "False Negatives", value: 19, fill: "#ef4444" },
    { name: "True Positives", value: 119, fill: "#3b82f6" },
  ]

  const modelComparison = [
    { name: "Logistic Regression", accuracy: 98.07 },
    { name: "Multinomial NB", accuracy: 97.49 },
    { name: "Random Forest", accuracy: 97.39 },
    { name: "SVC", accuracy: 97.0 },
    { name: "Decision Tree", accuracy: 92.46 },
    { name: "KNN", accuracy: 91.3 },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8 pt-8">
          <Link href="/" className="text-indigo-600 hover:text-indigo-700 mb-4 inline-block">
            ← Back to Classifier
          </Link>
          <h1 className="text-4xl font-bold text-gray-900">Model Performance Dashboard</h1>
          <p className="text-gray-600 mt-2">Logistic Regression Email Classifier Metrics</p>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          {modelMetrics.map((item) => (
            <Card key={item.metric} className="border-0 shadow">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-600">{item.metric}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-indigo-600">{item.value}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Model Comparison */}
          <Card className="border-0 shadow">
            <CardHeader>
              <CardTitle>Model Comparison</CardTitle>
              <CardDescription>Accuracy across different classifiers</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={modelComparison}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="accuracy" fill="#4f46e5" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Confusion Matrix */}
          <Card className="border-0 shadow">
            <CardHeader>
              <CardTitle>Confusion Matrix</CardTitle>
              <CardDescription>Test set predictions breakdown</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={confusionData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {confusionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Details */}
        <Card className="border-0 shadow">
          <CardHeader>
            <CardTitle>Model Details</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Preprocessing Steps</h3>
                <ul className="space-y-1 text-sm text-gray-600">
                  <li>• Lowercase conversion</li>
                  <li>• Tokenization</li>
                  <li>• Alphanumeric filtering</li>
                  <li>• Stopword removal</li>
                  <li>• Porter Stemming</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Model Configuration</h3>
                <ul className="space-y-1 text-sm text-gray-600">
                  <li>• Algorithm: Logistic Regression</li>
                  <li>• Vectorizer: CountVectorizer</li>
                  <li>• Max Features: 3000</li>
                  <li>• Training Samples: 4135</li>
                  <li>• Test Samples: 1034</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
