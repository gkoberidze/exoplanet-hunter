def get_feature_explanation(feature_name):
    """Return explanation for each feature"""
    explanations = {
        'koi_period': 'Orbital Period: Time it takes for the planet to complete one orbit around its star (in days)',
        'koi_depth': 'Transit Depth: How much the star dims when the planet passes in front of it (in parts per million)',
        'koi_duration': 'Transit Duration: How long the planet blocks the star during transit (in hours)',
        'koi_prad': 'Planetary Radius: Size of the planet compared to Earth (in Earth radii)',
        'koi_teq': 'Equilibrium Temperature: Estimated temperature of the planet (in Kelvin)',
        'koi_steff': 'Stellar Effective Temperature: Surface temperature of the host star (in Kelvin)',
        'koi_srad': 'Stellar Radius: Size of the host star compared to our Sun (in Solar radii)',
        'koi_slogg': 'Stellar Surface Gravity: Gravity at the surface of the host star',
        'koi_insol': 'Insolation Flux: Amount of radiation the planet receives from its star',
        'koi_model_snr': 'Signal-to-Noise Ratio: Strength of the transit signal',
        'koi_impact': 'Impact Parameter: How close the planet passes to the center of the star during transit',
        'koi_kepmag': 'Kepler Magnitude: Brightness of the host star',
        'koi_score': 'Disposition Score: Confidence score for the classification',
        'planet_star_radius_ratio': 'Planet-to-Star Radius Ratio: Relative size comparison',
        'duration_period_ratio': 'Transit Duration to Period Ratio: How long the transit lasts relative to orbit'
    }
    return explanations.get(feature_name, 'No explanation available')


def get_prediction_insight(prediction, probability, feature_values, feature_importance=None):
    """Generate detailed insights about the prediction"""

    is_exoplanet = prediction == 1
    confidence = max(probability) * 100

    insights = {
        'classification': 'Confirmed Exoplanet' if is_exoplanet else 'False Positive',
        'confidence_level': get_confidence_level(confidence),
        'key_indicators': [],
        'similar_exoplanets': [],
        'recommendations': []
    }

    # Analyze key indicators
    if 'koi_period' in feature_values:
        period = feature_values['koi_period']
        if period < 10:
            insights['key_indicators'].append(
                f"Short orbital period ({period:.2f} days) - suggests close proximity to star")
        elif period > 365:
            insights['key_indicators'].append(
                f"Long orbital period ({period:.2f} days) - similar to outer planets in our solar system")

    if 'koi_prad' in feature_values:
        radius = feature_values['koi_prad']
        planet_type = classify_planet_type(radius)
        insights['key_indicators'].append(
            f"Planet type: {planet_type} (radius: {radius:.2f} Earth radii)")

    if 'koi_teq' in feature_values:
        temp = feature_values['koi_teq']
        if 273 <= temp <= 373:
            insights['key_indicators'].append(
                f"Temperature ({temp:.0f}K) in liquid water range - potentially habitable zone!")
        elif temp > 1000:
            insights['key_indicators'].append(
                f"Very hot ({temp:.0f}K) - likely a hot Jupiter or similar")
        else:
            insights['key_indicators'].append(f"Temperature: {temp:.0f}K")

    # Add recommendations
    if is_exoplanet:
        insights['recommendations'].append(
            "This candidate shows strong exoplanet characteristics")
        insights['recommendations'].append(
            "Follow-up observations recommended for confirmation")
        if confidence < 80:
            insights['recommendations'].append(
                "Moderate confidence - additional data would strengthen classification")
    else:
        insights['recommendations'].append(
            "Signal likely caused by stellar variability or instrumental noise")
        if confidence < 80:
            insights['recommendations'].append(
                "Consider re-analysis with additional data")

    return insights


def get_confidence_level(confidence):
    """Categorize confidence level"""
    if confidence >= 95:
        return "Very High"
    elif confidence >= 85:
        return "High"
    elif confidence >= 70:
        return "Moderate"
    else:
        return "Low"


def classify_planet_type(radius):
    """Classify planet based on radius"""
    if radius < 1.25:
        return "Rocky (Earth-like)"
    elif radius < 2.0:
        return "Super-Earth"
    elif radius < 6.0:
        return "Neptune-like"
    else:
        return "Jupiter-like (Gas Giant)"


def get_exoplanet_facts():
    """Return interesting facts about exoplanets"""
    return [
        "Over 5,500 exoplanets have been confirmed as of 2024",
        "The Kepler mission discovered over 2,700 confirmed exoplanets",
        "Some exoplanets orbit two stars (circumbinary planets)",
        "Hot Jupiters are gas giants that orbit very close to their stars",
        "The habitable zone is where liquid water could exist on a planet's surface",
        "Transit method detects planets by measuring the dimming of starlight",
        "Many exoplanets have been found in the Goldilocks zone",
        "Some exoplanets are made of diamond or have oceans of lava"
    ]


def compare_to_solar_system(feature_values):
    """Compare exoplanet characteristics to solar system planets"""
    comparisons = []

    if 'koi_period' in feature_values:
        period = feature_values['koi_period']
        if period < 1:
            comparisons.append(
                "Orbital period shorter than any planet in our solar system")
        elif period < 88:
            comparisons.append("Similar orbital period to Mercury (88 days)")
        elif period < 225:
            comparisons.append("Similar orbital period to Venus (225 days)")
        elif period < 365:
            comparisons.append("Similar orbital period to Earth (365 days)")
        elif period < 687:
            comparisons.append("Similar orbital period to Mars (687 days)")

    if 'koi_prad' in feature_values:
        radius = feature_values['koi_prad']
        if radius < 0.5:
            comparisons.append("Smaller than Earth (possibly Mars-sized)")
        elif 0.9 <= radius <= 1.1:
            comparisons.append("Similar size to Earth")
        elif 3.5 <= radius <= 4.5:
            comparisons.append("Similar size to Neptune")
        elif radius > 10:
            comparisons.append(
                "Larger than Jupiter (our solar system's largest planet)")

    return comparisons
