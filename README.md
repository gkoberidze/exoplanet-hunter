This project is an AI-powered exoplanet detection system that analyzes NASA Kepler mission data to classify planetary candidates as either confirmed exoplanets or false positives. The system uses three advanced machine learning models (Random Forest, XGBoost, and Neural Network) trained on 7,326 data samples from the Kepler mission. Users can input planetary parameters (orbital period, transit depth, planetary radius, temperature, etc.) through a web interface, and the AI predicts whether the candidate is a genuine exoplanet with confidence scores and detailed explanations. The system also supports batch CSV uploads for analyzing multiple candidates simultaneously.

Benefits:

1) High Accuracy: Achieves 87-89% accuracy in classifying exoplanets, helping researchers prioritize which candidates warrant follow-up observations.
2) Time Efficiency: Analyzes hundreds of candidates in seconds, dramatically reducing manual analysis time.
3) Educational Value: Provides detailed insights about planetary characteristics, comparisons to our solar system, and explanations of features.
4) Accessibility: Makes advanced ML analysis available through a simple web interface requiring no coding knowledge.
5) Transparency: Shows feature importance and confidence scores, allowing scientists to understand the reasoning behind classifications.

Machine Learning Libraries:

scikit-learn (Random Forest, Neural Network)
XGBoost (Gradient Boosting)
TensorFlow (Deep Learning)
pandas & NumPy (Data Processing)

This project demonstrates how modern machine learning can be applied to real astronomical challenges, making cutting-edge technology accessible and useful for the scientific community while maintaining transparency and educational value.
