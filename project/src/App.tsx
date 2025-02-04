import React, { useState, useCallback } from "react";
import {
  Shield,
  Upload,
  Zap,
  CheckCircle,
  AlertTriangle,
  Github,
  Loader2,
} from "lucide-react";

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<{
    isDeepfake: boolean;
    confidence: number;
  } | null>(null);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    if (
      droppedFile &&
      (droppedFile.type.startsWith("image/") ||
        droppedFile.type.startsWith("video/"))
    ) {
      handleFile(droppedFile);
    }
  }, []);

  const handleFile = (file: File) => {
    setFile(file);
    if (file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else if (file.type.startsWith("video/")) {
      setPreview(URL.createObjectURL(file));
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setIsLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      setResult(result);
    } catch (error) {
      console.error("Error analyzing file:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white">
      {/* Hero Section */}
      <header className="container mx-auto px-6 py-16">
        <nav className="flex justify-between items-center mb-16">
          <div className="flex items-center space-x-2">
            <Shield className="w-8 h-8 text-blue-400" />
            <span className="text-xl font-bold">DeepGuard</span>
          </div>
          <div className="flex space-x-6">
            <a
              href="#features"
              className="hover:text-blue-400 transition-colors"
            >
              Features
            </a>
            <a
              href="#how-it-works"
              className="hover:text-blue-400 transition-colors"
            >
              How it Works
            </a>
            <a href="#demo" className="hover:text-blue-400 transition-colors">
              Demo
            </a>
          </div>
        </nav>

        <div className="flex flex-col lg:flex-row items-center justify-between gap-12">
          <div className="lg:w-1/2">
            <h1 className="text-5xl font-bold leading-tight mb-6">
              Detect Deepfakes with
              <span className="text-blue-400"> AI-Powered Precision</span>
            </h1>
            <p className="text-xl text-gray-300 mb-8">
              Protect yourself from digital deception. Our advanced AI
              technology detects manipulated media with 99.9% accuracy.
            </p>
            <div className="flex space-x-4">
              <a
                href="#demo"
                className="bg-blue-500 hover:bg-blue-600 px-8 py-3 rounded-lg font-semibold transition-colors"
              >
                Try Demo
              </a>
              <button className="border border-gray-500 hover:border-blue-400 px-8 py-3 rounded-lg font-semibold transition-colors">
                Learn More
              </button>
            </div>
          </div>
          <div className="lg:w-1/2">
            <img
              src="https://images.unsplash.com/photo-1633412802994-5c058f151b66?auto=format&fit=crop&w=800&q=80"
              alt="AI Technology Visualization"
              className="rounded-lg shadow-2xl"
            />
          </div>
        </div>
      </header>

      {/* Features Section */}
      <section id="features" className="bg-gray-800 py-20">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-16">
            Why Choose DeepGuard?
          </h2>
          <div className="grid md:grid-cols-3 gap-12">
            <FeatureCard
              icon={<Zap className="w-12 h-12 text-blue-400" />}
              title="Real-Time Detection"
              description="Get instant results with our lightning-fast detection engine"
            />
            <FeatureCard
              icon={<CheckCircle className="w-12 h-12 text-blue-400" />}
              title="99.9% Accuracy"
              description="Industry-leading precision in detecting manipulated media"
            />
            <FeatureCard
              icon={<AlertTriangle className="w-12 h-12 text-blue-400" />}
              title="Early Warning System"
              description="Proactive alerts for potential deepfake threats"
            />
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section id="demo" className="py-20">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-16">
            Try It Yourself
          </h2>
          <div className="max-w-4xl mx-auto">
            <div
              className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                isDragging
                  ? "border-blue-400 bg-blue-400/10"
                  : "border-gray-600"
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              {!file && (
                <>
                  <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <p className="text-xl mb-4">
                    Drag and drop your media file here
                  </p>
                  <p className="text-gray-400 mb-6">
                    Supported formats: JPG, PNG, MP4
                  </p>
                  <input
                    type="file"
                    id="file-upload"
                    className="hidden"
                    accept="image/*,video/*"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleFile(file);
                    }}
                  />
                  <label
                    htmlFor="file-upload"
                    className="bg-blue-500 hover:bg-blue-600 px-8 py-3 rounded-lg font-semibold transition-colors cursor-pointer inline-block"
                  >
                    Browse Files
                  </label>
                </>
              )}

              {file && (
                <div className="space-y-6">
                  <div className="max-w-lg mx-auto">
                    {file.type.startsWith("image/") && (
                      <img
                        src={preview!}
                        alt="Preview"
                        className="rounded-lg shadow-lg max-h-[400px] mx-auto"
                      />
                    )}
                    {file.type.startsWith("video/") && (
                      <video
                        src={preview!}
                        controls
                        className="rounded-lg shadow-lg max-h-[400px] mx-auto"
                      />
                    )}
                  </div>

                  <div className="flex justify-center gap-4">
                    <button
                      onClick={handleAnalyze}
                      disabled={isLoading}
                      className="bg-blue-500 hover:bg-blue-600 px-8 py-3 rounded-lg font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      {isLoading && (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      )}
                      {isLoading ? "Analyzing..." : "Analyze"}
                    </button>
                    <button
                      onClick={() => {
                        setFile(null);
                        setPreview(null);
                        setResult(null);
                      }}
                      className="border border-gray-500 hover:border-blue-400 px-8 py-3 rounded-lg font-semibold transition-colors"
                    >
                      Clear
                    </button>
                  </div>

                  {result && (
                    <div
                      className={`mt-6 p-6 rounded-lg ${
                        result.isDeepfake === true
                          ? "bg-red-500/20"
                          : "bg-green-500/20"
                      }`}
                    >
                      <h3 className="text-2xl font-bold mb-2">
                        {result.isDeepfake === true
                          ? "Deepfake Detected!"
                          : "Authentic Media"}
                      </h3>
                      <p className="text-lg">
                        Confidence: {Number(result.confidence).toFixed(2)}%
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 py-12">
        <div className="container mx-auto px-6">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <Shield className="w-6 h-6 text-blue-400" />
              <span className="font-bold">DeepGuard</span>
            </div>
            <div className="flex items-center space-x-6">
              <a
                href="https://github.com"
                className="text-gray-400 hover:text-white transition-colors"
              >
                <Github className="w-6 h-6" />
              </a>
              <span className="text-gray-400">
                © 2024 DeepGuard. All rights reserved.
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="bg-gray-900 p-8 rounded-lg text-center">
      <div className="mb-6">{icon}</div>
      <h3 className="text-xl font-semibold mb-4">{title}</h3>
      <p className="text-gray-400">{description}</p>
    </div>
  );
}

export default App;
