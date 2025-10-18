import os
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

model = genai.GenerativeModel('gemini-pro')


def get_system_prompt():
    """System prompt that tells the AI how to behave"""
    return """You are ExoHunter AI, an expert assistant for exoplanet detection and analysis.

Your role:
- Help users understand exoplanet detection and NASA Kepler mission data
- Explain predictions made by machine learning models
- Answer questions about planetary characteristics, orbital mechanics, and astronomy
- Be enthusiastic about space science and discovery
- Use emojis occasionally (🪐 🌟 🔭 🚀) to make conversations engaging
- Keep responses concise but informative (2-4 sentences usually)
- If asked about specific predictions, reference the data provided in context

Your personality:
- Friendly and educational
- Excited about space exploration
- Patient with beginners
- Scientific but not overly technical

When explaining predictions:
- Focus on the key features that influenced the classification
- Compare to familiar concepts (Earth, our solar system)
- Explain scientific concepts in simple terms
- Suggest what makes exoplanets interesting or unique
"""


def generate_chat_response(user_message, context=None):
    """
    Generate AI response to user message using FREE Google Gemini

    Args:
        user_message: The user's question/message
        context: Optional dict with prediction data, features, etc.

    Returns:
        AI response string
    """

    # Build the full prompt
    full_prompt = get_system_prompt() + "\n\n"

    # Add context if provided (e.g., current prediction results)
    if context:
        full_prompt += "Current analysis context:\n"

        if 'prediction' in context:
            full_prompt += f"- Prediction: {context['prediction']}\n"
        if 'confidence' in context:
            full_prompt += f"- Confidence: {context['confidence']}\n"
        if 'model_used' in context:
            full_prompt += f"- Model: {context['model_used']}\n"
        if 'input_values' in context:
            # Show key features
            key_features = ['koi_period', 'koi_prad', 'koi_teq']
            feature_str = {
                k: v for k, v in context['input_values'].items() if k in key_features}
            full_prompt += f"- Key features: {feature_str}\n"
        if 'insights' in context and 'key_indicators' in context['insights']:
            full_prompt += f"- Key insights: {', '.join(context['insights']['key_indicators'][:2])}\n"

        full_prompt += "\n"

    full_prompt += f"User question: {user_message}\n\nProvide a helpful, concise response (2-4 sentences):"

    try:
        # Call Gemini API (FREE!)
        response = model.generate_content(full_prompt)
        return response.text

    except Exception as e:
        # Fallback to pre-written responses if API fails
        fallback = get_fallback_response(user_message)
        if fallback:
            return fallback
        return f"I'm having trouble connecting right now. Try asking: 'What makes a good exoplanet?' or 'How does transit detection work?'"


def get_suggested_questions(context=None):
    """Generate suggested questions based on context"""

    base_questions = [
        "What makes a good exoplanet candidate?",
        "How does the transit method work?",
        "Tell me about the Kepler mission",
        "What's the habitable zone?"
    ]

    if context and 'prediction' in context:
        if 'Exoplanet' in context.get('prediction', ''):
            base_questions = [
                "Why was this classified as an exoplanet?",
                "What are the most important features?",
                "Could this planet support life?",
                "How does this compare to Earth?"
            ] + base_questions[:2]
        else:
            base_questions = [
                "Why is this a false positive?",
                "What would make this a real exoplanet?",
                "What causes false positives?",
                "How can we improve detection?"
            ] + base_questions[:2]

    return base_questions[:4]


# Fallback responses for common questions (no API needed)
def get_fallback_response(message):
    """Pre-written responses for common questions"""

    message_lower = message.lower()

    # Greetings
    if any(word in message_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! 👋 I'm ExoHunter AI, your guide to exoplanet discovery. Ask me anything about the predictions, planets, or space science!"

    # Help
    if 'help' in message_lower:
        return "I can help you with:\n🔭 Understanding predictions\n🪐 Explaining exoplanet features\n📊 Interpreting model results\n🌟 General astronomy questions\n\nJust ask me anything!"

    # Transit method
    if 'transit' in message_lower and 'method' in message_lower:
        return "The transit method detects exoplanets by measuring the tiny dip in a star's brightness when a planet passes in front of it! 🔭 It's like a mini-eclipse. The size and frequency of these dips tell us about the planet's size and orbit. This is how Kepler discovered thousands of exoplanets!"

    # Kepler mission
    if 'kepler' in message_lower and 'mission' in message_lower:
        return "The Kepler Space Telescope (2009-2018) was NASA's exoplanet-hunting spacecraft! 🚀 It stared at 150,000 stars simultaneously, discovering over 2,700 confirmed exoplanets. Our AI models are trained on this amazing dataset to help identify new candidates!"

    # Habitable zone
    if 'habitable' in message_lower or 'goldilocks' in message_lower:
        return "The habitable zone (or 'Goldilocks zone') is the region around a star where temperatures allow liquid water to exist! 🌊 Not too hot, not too cold - just right for potential life. Earth is in our Sun's habitable zone, and we're finding more candidates around other stars!"

    # False positive
    if 'false positive' in message_lower and 'cause' in message_lower:
        return "False positives happen when something mimics a planetary transit! 🎭 Common causes include: binary star systems eclipsing each other, starspots crossing the star's face, or instrumental noise. That's why we use AI to help distinguish real planets from imposters!"

    # Features/importance
    if any(word in message_lower for word in ['feature', 'important', 'parameter']):
        return "Key features for exoplanet detection include: orbital period (how long to orbit), transit depth (brightness dip), planetary radius, and signal-to-noise ratio! 📊 Our AI weighs these factors to determine if a signal is a real planet or a false positive."

    # Life
    if 'life' in message_lower or 'habitable' in message_lower:
        return "For potential life, we look for: Earth-like size, location in the habitable zone, and appropriate temperature! 🌍 Planets receiving similar radiation to Earth are especially interesting. However, detecting life requires much more detailed observations than just finding the planet!"

    return None
