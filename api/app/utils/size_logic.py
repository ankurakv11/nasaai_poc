import json
from typing import Dict, Any, Tuple, List
import pandas as pd

def get_fit_details(user_measurements: Dict[str, float], garment_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any], float]:
    """
    Analyzes the fit of a garment against user measurements (in cm) and returns a rating,
    detailed analysis, and an average score.
    """
    analysis = {}
    scores = []
    descriptions = []
    
    print(f"Analyzing fit with user measurements: {user_measurements}")
    print(f"Garment data: {garment_data}")

    # Maps user measurement keys to garment data column names
    measurement_mapping = {
        'shoulder': 'Shoulder (Inches)',
        'chest': 'Chest (Inches)',
        'waist': 'Waist (Inches)',
        'hip': 'Hip (Inches)',  # User hip measurement
        'Butt': 'Hip (Inches)',  # Alternative name for hip
        'body_length': 'Length (Inches)',
        'inseam': 'Inseam Length (Inches)',
    }

    # Maps user measurement keys to the 'type' for scoring logic
    scoring_type_mapping = {
        'shoulder': 'shoulder',
        'chest': 'chest',
        'waist': 'waist',
        'hip': 'hip',
        'Butt': 'hip',
        'body_length': 'length',
        'inseam': 'inseam',
    }

    for user_key, garment_key in measurement_mapping.items():
        user_val_cm = user_measurements.get(user_key)
        garment_val_cm = garment_data.get(garment_key)
        
        print(f"Checking measurement: {user_key} -> {garment_key}")
        print(f"  User value: {user_val_cm}, Garment value: {garment_val_cm}")

        # Ensure both values are valid numbers
        if user_val_cm is not None and garment_val_cm is not None and pd.notna(user_val_cm) and pd.notna(garment_val_cm) and garment_val_cm > 0:
            diff_cm = garment_val_cm - user_val_cm
            # No need to convert diff_inches to cm again, as both values are already in cm
            
            measurement_type = scoring_type_mapping.get(user_key)
            score, description = get_fit_score_and_description(measurement_type, diff_cm)
            
            print(f"  Valid comparison: diff={diff_cm}, score={score}, description={description}")
            
            analysis[measurement_type] = {
                "user_measurement": user_val_cm,
                "garment_measurement": garment_val_cm,
                "difference": diff_cm,
                "score": score,
                "description": description
            }
            scores.append(score)
            descriptions.append(description)
        else:
            print(f"  Invalid comparison: user_val={user_val_cm}, garment_val={garment_val_cm}")

    if not scores:
        print("No valid measurements found for scoring")
        return "Not Enough Data", analysis, 0.0

    avg_score = sum(scores) / len(scores)
    overall_rating = get_overall_rating(avg_score, descriptions)
    
    print(f"Final result: rating={overall_rating}, score={avg_score}, analyses={len(analysis)}")
    return overall_rating, analysis, avg_score


def get_fit_score_and_description(measurement_type: str, diff_cm: float) -> Tuple[float, str]:
    """
    Calculates a fit score from 1-10 and provides a descriptive string.
    Difference (`diff`) is expected in centimeters.
    """
    if measurement_type == 'shoulder':
        if -1.3 <= diff_cm <= 1.3: score = 9.5; desc = "Perfect fit over Shoulder"
        elif 1.3 < diff_cm <= 3.8: score = 8.5; desc = "The product is regular fit of the shoulder"
        elif -3.8 <= diff_cm < -1.3: score = 6.3; desc = "The product is tight in the shoulder."
        elif diff_cm > 3.8: score = 5.5; desc = "The product is too loose in the shoulder."
        else: score = 4.0; desc = "The product is very tight in the shoulder."

    elif measurement_type == 'chest':
        if 3.8 <= diff_cm <= 6.4: score = 9.8; desc = "Perfect fit over bust"
        elif 6.4 < diff_cm <= 10.2: score = 8.9; desc = "The product is regular fit of the chest/ bust area"
        elif 1.3 <= diff_cm < 3.8: score = 6.9; desc = "The product is tight in the chest/bust area."
        elif diff_cm > 10.2: score = 5.9; desc = "The product is too loose in the chest/bust area."
        else: score = 3.5; desc = "The product is very tight in the chest/bust area."

    elif measurement_type == 'waist':
        if 2.5 <= diff_cm <= 5.1: score = 9.7; desc = "Perfect fit over waist"
        elif 5.1 < diff_cm <= 8.9: score = 9.3; desc = "The product is regular fit in the waistline."
        elif 0.0 <= diff_cm < 2.5: score = 7.5; desc = "The product is tight in the waistline."
        elif diff_cm > 8.9: score = 6.3; desc = "The product is too loose in the waistline."
        else: score = 3.0; desc = "The product is very tight in the waistline."

    elif measurement_type == 'length':
        # For length, longer is generally okay, but too short is not.
        if 0 <= diff_cm <= 5: score = 9.5; desc = "Perfect length."
        elif diff_cm > 5: score = 8.5; desc = "Slightly long, but acceptable."
        elif -5 <= diff_cm < 0: score = 6.0; desc = "A bit short."
        else: score = 3.0; desc = "Too short."

    elif measurement_type == 'inseam':
        # For inseam, precision is more important.
        if -1.5 <= diff_cm <= 1.5: score = 9.8; desc = "Perfect inseam length."
        elif 1.5 < diff_cm <= 4: score = 8.0; desc = "A bit long."
        elif -4 <= diff_cm < -1.5: score = 6.0; desc = "A bit short."
        else: score = 3.0; desc = "Length is significantly off."

    else:  # Default for hip or others
        if 3.8 <= diff_cm <= 6.4: score = 9.6; desc = "Perfect fit."
        elif 6.4 < diff_cm <= 10.2: score = 8.8; desc = "Regular fit."
        elif 1.3 <= diff_cm < 3.8: score = 7.2; desc = "Snug fit."
        elif diff_cm > 10.2: score = 6.1; desc = "Loose fit."
        else: score = 2.5; desc = "Very tight."

    return score, desc

def get_overall_rating(score: float, descriptions: List[str]) -> str:
    """
    Determines the overall rating based on fit descriptions, which is more reliable
    than averaging scores that can be low for both "tight" and "loose" fits.
    """
    lower_descriptions = [d.lower() for d in descriptions]

    is_very_tight = any("very tight" in d or "too short" in d for d in lower_descriptions)
    if is_very_tight:
        return "Very Tight"

    is_tight = any("tight" in d or "snug" in d or "a bit short" in d for d in lower_descriptions)
    is_loose = any("loose" in d or "long" in d for d in lower_descriptions)

    # A mix of tight and loose measurements indicates a poor fit overall.
    if is_tight and is_loose:
        return "Poor Fit"

    if is_tight:
        return "Tight Fit"
    
    if is_loose:
        return "Too Loose Fit"

    # If no major issues are found in descriptions, use the average score for fits
    # that are generally good.
    if score >= 9.0:
        return "Best Fit!"
    if score >= 8.0:
        return "Regular Fit"
    
    # Fallback for scores that are not perfect but have no glaring issues.
    return "Good Fit"