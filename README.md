Exoplanet Hunter AI

Overview

Exoplanet Hunter AI is an intelligent web application that harnesses the power of machine learning to detect and classify exoplanets from NASA's Kepler Space Telescope mission data. By analyzing planetary and stellar parameters, the system predicts whether a celestial object is a confirmed exoplanet or a false positive, helping astronomers prioritize candidates for follow-up observations.

Problem Statement

The search for exoplanets generates massive amounts of data that require extensive manual analysis. NASA's Kepler mission alone monitored 150,000+ stars and identified thousands of potential planetary candidates. However, distinguishing genuine exoplanets from false positives (binary stars, starspots, or instrumental noise) is time-consuming and requires expert knowledge. Many promising candidates remain unverified due to limited resources and analysis capacity.

Our Solution

Exoplanet Hunter AI democratizes exoplanet detection by:

Automating Classification: AI models analyze 15+ planetary parameters in seconds, achieving 87-89% accuracy
Prioritizing Candidates: Confidence scores help researchers focus on high-probability exoplanets
Batch Processing: Analyze hundreds of candidates simultaneously via CSV upload
Educational Insights: Interactive AI assistant explains predictions, compares to our solar system, and teaches astronomy concepts
Transparency: Feature importance analysis shows which parameters drove each classification decision

Key Features

1. Multi-Model AI Engine

Three advanced ML models: Random Forest, XGBoost, and Neural Network
Users can compare predictions across models
Best model achieves 89%+ accuracy with 0.97 ROC AUC score

2. Single Prediction Analysis

Input 15 planetary/stellar parameters through intuitive web interface
Instant predictions with confidence scores
Detailed insights: planet type classification, temperature analysis, habitability assessment
Solar system comparisons (e.g., "Similar size to Neptune")
Top 5 most influential features visualization

3. Batch CSV Upload

Upload entire datasets for mass analysis
Automatic validation of required columns
Summary statistics: confirmed vs. false positive counts, average confidence
Downloadable results as CSV for further research

4. Conversational AI Assistant 

Real-time Q&A about exoplanets, predictions, and space science
Context-aware responses based on current prediction results
Suggested follow-up questions
Voice input support (speech-to-text)
Educational explanations in friendly language with emojis

5. Prediction History & Export

Tracks last 100 predictions locally
Export history as CSV for record-keeping
Save individual predictions as JSON files
Clear history option

6. Interactive Visualizations

Model performance comparison charts
Confusion matrices for all models
Feature importance bar graphs
Dark/light theme toggle

7. Educational Content

"About" section explaining AI models and methodology
Feature explanations (what each parameter measures)
Exoplanet facts and discovery insights

Technologies Used

Backend

Python 3.x: Core programming language
Flask: Web framework for REST API
NumPy & Pandas: Data processing and manipulation
scikit-learn: Random Forest, Neural Network, data preprocessing
XGBoost: Gradient boosting classifier
Joblib: Model serialization

Machine Learning

Random Forest Classifier: Ensemble learning (300 trees, depth 25)
XGBoost: Gradient boosting (300 estimators, learning rate 0.05)
Neural Network (MLP): Deep learning (4 hidden layers: 256-128-64-32 neurons)
StandardScaler: Feature normalization
Train-Test Split: 80-20 validation

AI Chat Assistant

Hugging Face API: Microsoft Phi-3-mini-4k-instruct model
Free inference endpoint: No API costs
Context-aware prompting: Incorporates prediction results

Frontend

HTML5 & CSS3: Responsive design with modern gradients
Vanilla JavaScript: Dynamic interactions, no frameworks needed
Web Speech API: Voice input for chat
Fetch API: Asynchronous server communication

Visualization

Matplotlib: Chart generation
Seaborn: Statistical visualizations
Custom CSS: Animated UI components

Data Management

NASA Kepler Dataset: 7,326 labeled samples
Feature Engineering: 15 parameters including derived features
CSV Processing: Batch upload handler with validation

Dataset & Performance

Source: NASA Kepler Space Telescope Mission Data
Training Samples: 7,326 confirmed exoplanets and false positives
Features: 15 parameters (orbital period, transit depth, planetary radius, temperature, stellar properties, signal-to-noise ratio, etc.)
Best Model Accuracy: 89.2%
ROC AUC Score: 0.97
Processing Speed: <1 second per prediction

Educational Impact
The conversational AI makes space science accessible to:

Students: Learn about exoplanet detection methods interactively
Educators: Demonstrate real-world ML applications in astronomy
Amateur Astronomers: Understand what makes planets detectable
Researchers: Quick validation tool for candidate screening

Use Cases

Astronomical Research: Pre-screen Kepler candidates before expensive telescope time
Citizen Science: Enable non-experts to contribute to exoplanet discovery
Education: Interactive teaching tool for ML and astronomy courses
Data Exploration: Analyze historical Kepler data with modern AI techniques

Innovation Highlights

Hybrid Approach: Combines three ML algorithms for robust predictions
Explainable AI: Feature importance and confidence scores build trust
Conversational Interface: Natural language explanations bridge technical gap
Real-Time Processing: Instant feedback encourages experimentation
Open Science: Transparent methodology supports reproducible research


Built for: NASA Space Apps Challenge 2025
