import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")

# Using Microsoft's Phi-3 model (fast, smart, FREE)
API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"

headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}


def get_system_prompt():
    """System prompt that tells the AI how to behave"""
    return """You are ExoHunter AI, an expert assistant for exoplanet detection.

Your role:
- Help users understand exoplanet detection and NASA Kepler data
- Explain machine learning predictions clearly
- Answer astronomy and space science questions
- Be enthusiastic about discovery (use emojis: 🪐 🌟 🔭 🚀)
- Keep responses SHORT (2-4 sentences max)
- Be friendly and educational

When explaining predictions:
- Focus on key features that matter most
- Compare to Earth and our solar system
- Use simple, non-technical language
- Highlight what makes discoveries exciting"""


def generate_chat_response(user_message, context=None):
    """
    Generate AI response using FREE Hugging Face API

    Args:
        user_message: User's question
        context: Optional prediction data

    Returns:
        AI response string
    """

    # Build the prompt
    full_prompt = get_system_prompt() + "\n\n"

    # Add context if provided
    if context:
        full_prompt += "Current analysis:\n"
        if 'prediction' in context:
            full_prompt += f"- Result: {context['prediction']}\n"
        if 'confidence' in context:
            full_prompt += f"- Confidence: {context['confidence']}\n"
        if 'insights' in context and 'key_indicators' in context['insights']:
            key_points = context['insights']['key_indicators'][:2]
            full_prompt += f"- Key features: {', '.join(key_points)}\n"
        full_prompt += "\n"

    full_prompt += f"User: {user_message}\nAssistant:"

    # Try API call
    try:
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.9,
                "return_full_text": False
            }
        }

        response = requests.post(
            API_URL, headers=headers, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0]['generated_text'].strip()
            elif isinstance(result, dict) and 'generated_text' in result:
                return result['generated_text'].strip()

        # If model is loading, wait and retry once
        elif response.status_code == 503:
            time.sleep(2)
            response = requests.post(
                API_URL, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0]['generated_text'].strip()

    except Exception as e:
        print(f"API Error: {e}")

    # Fallback to smart pre-written responses
    return get_fallback_response(user_message)


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
            return [
                "Why was this classified as an exoplanet?",
                "What are the most important features?",
                "Could this planet support life?",
                "How does this compare to Earth?"
            ]
        else:
            return [
                "Why is this a false positive?",
                "What would make it a real exoplanet?",
                "What causes false positives?",
                "How can we improve detection?"
            ]

    return base_questions[:4]


def get_fallback_response(message):
    """Smart fallback responses"""

    message_lower = message.lower()

    # Greetings
    if any(word in message_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! 👋 I'm ExoHunter AI. Ask me about exoplanets, predictions, or space science!"

    # Help
    if 'help' in message_lower:
        return "I can explain predictions, features, and astronomy concepts! Try asking: 'How does transit detection work?' or 'What makes a planet habitable?' 🔭"

    # Transit method
    if 'transit' in message_lower:
        return "The transit method detects planets by measuring tiny brightness dips when a planet passes in front of its star! 🔭 Like a mini-eclipse. This is how Kepler found thousands of exoplanets!"

    # Kepler mission
    if 'kepler' in message_lower:
        return "Kepler Space Telescope (2009-2018) discovered over 2,700 exoplanets! 🚀 It watched 150,000 stars continuously. Our AI uses this amazing dataset to identify new candidates!"

    # Habitable zone
    if 'habitable' in message_lower or 'life' in message_lower:
        return "The habitable zone is where liquid water can exist! 🌊 Not too hot, not too cold. Earth is in our Sun's habitable zone, and we're finding similar worlds around other stars!"

    # False positives
    if 'false positive' in message_lower:
        return "False positives happen when something mimics a planet's signal! 🎭 Could be binary stars, starspots, or noise. That's why we use AI to distinguish real planets from imposters!"

    # Features
    if 'feature' in message_lower or 'important' in message_lower:
        return "Key features include: orbital period, transit depth, planet radius, and signal strength! 📊 Our AI analyzes these to predict if a signal is a real planet or false alarm."

    # Default
    return "That's a great question! 🌟 For exoplanet detection, we look at how stars dim when planets pass in front. The Kepler mission data helps our AI learn patterns. What else would you like to know?"
