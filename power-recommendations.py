"""
Power Recommendation Module for Cognitive Science Experiments
------------------------------------------------------------
This module provides guidance on recommended statistical power levels
based on effect size magnitude and test type, along with functions to
check if planned studies meet these recommendations.
"""

import math
import numpy as np
from scipy.stats import norm, chi2


# Recommended power levels by test type and effect size
power_recommendations = {
    't-test': {
        'independent': {
            'small': 0.90,    # d = 0.2
            'medium': 0.85,   # d = 0.5
            'large': 0.80     # d = 0.8
        },
        'paired': {
            'small': 0.90,
            'medium': 0.80,
            'large': 0.80
        }
    },
    'correlation': {
        'simple': {
            'small': 0.90,    # r = 0.1
            'medium': 0.80,   # r = 0.3
            'large': 0.80     # r = 0.5
        },
        'partial': {
            'small': 0.95,
            'medium': 0.85,
            'large': 0.80
        }
    },
    'regression': {
        'simple': {
            'small': 0.90,    # f² = 0.02
            'medium': 0.85,   # f² = 0.15
            'large': 0.80     # f² = 0.35
        },
        'multiple': {
            'small': 0.95,
            'medium': 0.85,
            'large': 0.85
        },
        'interaction': {
            'small': 0.95,
            'medium': 0.90,
            'large': 0.85
        }
    },
    'anova': {
        'between': {
            'small': 0.95,    # f = 0.1
            'medium': 0.85,   # f = 0.25
            'large': 0.80     # f = 0.4
        },
        'within': {
            'small': 0.90,
            'medium': 0.80,
            'large': 0.80
        },
        'mixed': {
            'small': 0.95,
            'medium': 0.85,
            'large': 0.80
        }
    },
    'chi-square': {
        'goodness-of-fit': {
            'small': 0.90,    # w = 0.1
            'medium': 0.85,   # w = 0.3
            'large': 0.80     # w = 0.5
        },
        'independence': {
            'small': 0.90,
            'medium': 0.85,
            'large': 0.80
        }
    },
    'sem': {
        'path-analysis': {
            'small': 0.95,
            'medium': 0.90,
            'large': 0.85
        },
        'full-model': {
            'small': 0.95,
            'medium': 0.90,
            'large': 0.90
        }
    }
}

# Effect size benchmarks by test type
effect_size_benchmarks = {
    't-test': {'small': 0.2, 'medium': 0.5, 'large': 0.8},
    'correlation': {'small': 0.1, 'medium': 0.3, 'large': 0.5},
    'regression': {'small': 0.02, 'medium': 0.15, 'large': 0.35},
    'anova': {'small': 0.1, 'medium': 0.25, 'large': 0.4},
    'chi-square': {'small': 0.1, 'medium': 0.3, 'large': 0.5},
    'sem': {'small': 0.1, 'medium': 0.3, 'large': 0.5}
}

# Adjustment factors for study characteristics
power_adjustment_factors = {
    # Multiple tests requiring correction
    'multiple_tests': {
        '2-4 tests': 0.05,      # Add 5% power
        '5-10 tests': 0.10,     # Add 10% power
        '10+ tests': 0.15       # Add 15% power
    },
    # Noisy measurements
    'measurement_reliability': {
        'high (>0.90)': 0.00,
        'moderate (0.70-0.90)': 0.05,
        'low (<0.70)': 0.10
    },
    # Expected attrition
    'expected_attrition': {
        'minimal (<5%)': 0.00,
        'moderate (5-15%)': 0.05,
        'high (>15%)': 0.10
    },
    # Publication goals
    'publication_goals': {
        'exploratory': 0.00,
        'confirmatory': 0.05,
        'pivotal study': 0.10
    }
}


def get_recommended_power(test_type, test_subtype, effect_size_category):
    """
    Get the recommended power level for a given test type and effect size.
    
    Parameters:
        test_type: Primary test type (e.g., 't-test', 'anova')
        test_subtype: Specific test subtype (e.g., 'independent', 'within')
        effect_size_category: Size of expected effect ('small', 'medium', 'large')
    
    Returns:
        Recommended power level
    """
    try:
        return power_recommendations[test_type][test_subtype][effect_size_category]
    except KeyError:
        raise ValueError(f"Invalid test type, subtype, or effect size category. "
                         f"Valid options are: {list(power_recommendations.keys())}")


def get_effect_size_value(test_type, effect_size_category):
    """
    Get the numeric effect size value for a given test type and category.
    
    Parameters:
        test_type: Primary test type (e.g., 't-test', 'anova')
        effect_size_category: Size of effect ('small', 'medium', 'large')
    
    Returns:
        Numeric effect size value
    """
    try:
        return effect_size_benchmarks[test_type][effect_size_category]
    except KeyError:
        raise ValueError(f"Invalid test type or effect size category. "
                         f"Valid options are: {list(effect_size_benchmarks.keys())}")


def calculate_adjusted_power_target(base_power, adjustments=None):
    """
    Calculate power target with adjustments for study characteristics.
    
    Parameters:
        base_power: Base recommended power from test type and effect size
        adjustments: Dict of adjustment factors to apply
    
    Returns:
        Adjusted power target (capped at 0.95)
    """
    if adjustments is None:
        return base_power
    
    total_adjustment = 0
    for factor, level in adjustments.items():
        try:
            category = power_adjustment_factors.get(factor, {})
            adjustment = category.get(level, 0)
            total_adjustment += adjustment
        except (KeyError, TypeError):
            continue
    
    # Cap power at 0.95 as higher values require extremely large samples
    return min(base_power + total_adjustment, 0.95)


def evaluate_power_sufficiency(planned_power, recommended_power, tolerance=0.05):
    """
    Evaluate if the planned power is sufficient based on recommendations.
    
    Parameters:
        planned_power: The power level of the planned study
        recommended_power: The recommended power level
        tolerance: Acceptable deviation from recommendation
    
    Returns:
        Dictionary with evaluation results
    """
    difference = planned_power - recommended_power
    
    if difference >= 0:
        status = "sufficient"
        message = "Planned power meets or exceeds recommendations."
    elif difference >= -tolerance:
        status = "borderline"
        message = f"Planned power is slightly below recommendations (within {tolerance*100}% tolerance)."
    else:
        status = "insufficient"
        message = "Planned power is below recommended levels."
    
    return {
        "status": status,
        "message": message,
        "planned_power": planned_power,
        "recommended_power": recommended_power,
        "difference": difference
    }


def recommend_sample_size(test_type, test_subtype, effect_size_category, 
                         significance_level=0.05, adjustments=None):
    """
    Recommend sample size based on test type, effect size, and recommended power.
    
    Parameters:
        test_type: Primary test type (e.g., 't-test', 'anova')
        test_subtype: Specific test subtype (e.g., 'independent', 'within')
        effect_size_category: Size of expected effect ('small', 'medium', 'large')
        significance_level: Alpha level for the test
        adjustments: Dict of adjustment factors to apply
    
    Returns:
        Dictionary with recommended sample size and power
    """
    # Get numeric effect size
    effect_size = get_effect_size_value(test_type, effect_size_category)
    
    # Get base recommended power
    base_power = get_recommended_power(test_type, test_subtype, effect_size_category)
    
    # Apply any adjustments
    target_power = calculate_adjusted_power_target(base_power, adjustments)
    
    # Calculate critical values
    alpha = significance_level / 2  # Two-tailed by default
    critical_z = abs(norm.ppf(alpha))
    power_z = norm.ppf(target_power)
    
    # Calculate sample size based on test type
    if test_type == 't-test':
        if test_subtype == 'independent':
            # For independent samples t-test
            n_per_group = ((critical_z + power_z) ** 2) / ((effect_size ** 2) / 2)
            sample_size = math.ceil(n_per_group * 2)
        else:  # paired
            # For paired samples t-test
            correlation = 0.5  # Assumed correlation between measures
            sample_size = math.ceil(((critical_z + power_z) ** 2) / 
                                   (effect_size ** 2 * (1 - correlation)))
    
    elif test_type == 'correlation':
        # For correlation
        fisher_z = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
        sample_size = math.ceil(((critical_z + power_z) / fisher_z) ** 2 + 3)
    
    elif test_type == 'regression':
        # For regression (simplified)
        predictors = 1  # Default
        if test_subtype == 'multiple':
            predictors = 3  # Assume moderate number of predictors
        elif test_subtype == 'interaction':
            predictors = 3  # Main effects + interaction
            
        sample_size = math.ceil((8 / effect_size) * 
                               ((critical_z + power_z) ** 2) + predictors + 1)
    
    elif test_type == 'anova':
        if test_subtype == 'between':
            # Between-subjects ANOVA
            groups = 3  # Assume 3 groups by default
            lambda_val = (critical_z + power_z) ** 2
            n_per_group = lambda_val / (effect_size ** 2)
            sample_size = math.ceil(n_per_group * groups)
        
        elif test_subtype == 'within':
            # Within-subjects ANOVA
            conditions = 3  # Assume 3 conditions by default
            lambda_val = (critical_z + power_z) ** 2
            base_n = lambda_val / (effect_size ** 2)
            correlation = 0.5  # Assumed correlation between measures
            correction_factor = 1 - correlation
            sample_size = math.ceil(base_n * correction_factor)
        
        else:  # mixed
            # Mixed ANOVA (simplified)
            groups = 2  # Assume 2 groups by default
            conditions = 3  # Assume 3 conditions by default
            # Use the more demanding component (typically between-subjects)
            lambda_val = (critical_z + power_z) ** 2
            n_per_group = lambda_val / (effect_size ** 2)
            sample_size = math.ceil(n_per_group * groups)
    
    elif test_type == 'chi-square':
        # Chi-square test (simplified)
        df = 3  # Assume moderate degrees of freedom
        critical_chi = chi2.ppf(1 - significance_level, df)
        sample_size = math.ceil(critical_chi / (effect_size ** 2))
    
    else:  # sem or other complex designs
        # Very simplified approach for SEM
        # Typically uses rules of thumb based on parameters/paths
        if effect_size_category == 'small':
            sample_size = 200
        elif effect_size_category == 'medium':
            sample_size = 100
        else:  # large
            sample_size = 60
    
    return {
        "recommended_sample_size": sample_size,
        "target_power": target_power,
        "effect_size_category": effect_size_category,
        "numeric_effect_size": effect_size,
        "test_type": test_type,
        "test_subtype": test_subtype
    }


def get_power_for_sample_size(sample_size, test_type, test_subtype, effect_size_category, 
                            significance_level=0.05):
    """
    Calculate expected power for a given sample size, test type, and effect size.
    
    Parameters:
        sample_size: Number of participants
        test_type: Primary test type (e.g., 't-test', 'anova')
        test_subtype: Specific test subtype (e.g., 'independent', 'within')
        effect_size_category: Size of expected effect ('small', 'medium', 'large')
        significance_level: Alpha level for the test
    
    Returns:
        Dictionary with expected power and evaluation
    """
    # Get numeric effect size
    effect_size = get_effect_size_value(test_type, effect_size_category)
    
    # Get recommended power
    recommended_power = get_recommended_power(test_type, test_subtype, effect_size_category)
    
    # Calculate critical value
    alpha = significance_level / 2  # Two-tailed by default
    critical_z = abs(norm.ppf(alpha))
    
    # Calculate power based on test type and sample size
    if test_type == 't-test':
        if test_subtype == 'independent':
            # For independent samples t-test
            n_per_group = sample_size / 2
            ncp = effect_size * np.sqrt(n_per_group / 2)
            power = 1 - norm.cdf(critical_z - ncp)
        else:  # paired
            # For paired samples t-test
            correlation = 0.5  # Assumed correlation between measures
            effective_n = sample_size * (1 - correlation)
            ncp = effect_size * np.sqrt(effective_n)
            power = 1 - norm.cdf(critical_z - ncp)
    
    elif test_type == 'correlation':
        # For correlation
        fisher_z = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
        se = 1 / np.sqrt(sample_size - 3)
        ncp = fisher_z / se
        power = 1 - norm.cdf(critical_z - ncp)
    
    elif test_type == 'regression':
        # For regression
        predictors = 1  # Default
        if test_subtype == 'multiple':
            predictors = 3
        elif test_subtype == 'interaction':
            predictors = 3
            
        ncp = effect_size * (sample_size - predictors - 1)
        power = 1 - norm.cdf(critical_z - np.sqrt(ncp))
    
    elif test_type == 'anova':
        if test_subtype == 'between':
            # Between-subjects ANOVA
            groups = 3  # Assume 3 groups by default
            n_per_group = sample_size / groups
            ncp = np.sqrt(n_per_group) * effect_size
            power = 1 - norm.cdf(critical_z - ncp)
        
        elif test_subtype == 'within':
            # Within-subjects ANOVA
            correlation = 0.5  # Assumed correlation between measures
            adjusted_n = sample_size * (1 - correlation)
            ncp = np.sqrt(adjusted_n) * effect_size
            power = 1 - norm.cdf(critical_z - ncp)
        
        else:  # mixed
            # Mixed ANOVA (simplified, focus on between component)
            groups = 2  # Assume 2 groups by default
            n_per_group = sample_size / groups
            ncp = np.sqrt(n_per_group) * effect_size
            power = 1 - norm.cdf(critical_z - ncp)
    
    elif test_type == 'chi-square':
        # Chi-square test (simplified)
        df = 3  # Assume moderate degrees of freedom
        ncp = sample_size * (effect_size ** 2)
        critical_chi = chi2.ppf(1 - significance_level, df)
        mean = df + ncp
        sd = np.sqrt(2 * (df + 2 * ncp))
        power = 1 - norm.cdf((critical_chi - mean) / sd)
    
    else:  # sem or other complex designs
        # Very simplified approach for SEM
        if sample_size < 60:
            power = 0.6
        elif sample_size < 100:
            power = 0.7
        elif sample_size < 200:
            power = 0.8
        elif sample_size < 400:
            power = 0.9
        else:
            power = 0.95
    
    # Evaluate if planned power is sufficient
    evaluation = evaluate_power_sufficiency(power, recommended_power)
    
    return {
        "expected_power": power,
        "recommended_power": recommended_power,
        "sample_size": sample_size,
        "evaluation": evaluation["status"],
        "message": evaluation["message"],
        "effect_size_category": effect_size_category,
        "numeric_effect_size": effect_size,
        "test_type": test_type,
        "test_subtype": test_subtype
    }


def power_analysis_summary(test_type, test_subtype, effect_size_category, sample_size, 
                         significance_level=0.05, adjustments=None):
    """
    Provide a comprehensive summary of power analysis for planned study.
    
    Parameters:
        test_type: Primary test type (e.g., 't-test', 'anova')
        test_subtype: Specific test subtype (e.g., 'independent', 'within')
        effect_size_category: Size of expected effect ('small', 'medium', 'large')
        sample_size: Planned number of participants
        significance_level: Alpha level for the test
        adjustments: Dict of adjustment factors to apply
    
    Returns:
        Dictionary with comprehensive power analysis summary
    """
    # Get recommended power (adjusted if needed)
    base_power = get_recommended_power(test_type, test_subtype, effect_size_category)
    adjusted_power = calculate_adjusted_power_target(base_power, adjustments)
    
    # Get recommended sample size based on adjusted power
    sample_recommendation = recommend_sample_size(
        test_type, test_subtype, effect_size_category, 
        significance_level, adjustments
    )
    
    # Calculate expected power for planned sample size
    power_expectation = get_power_for_sample_size(
        sample_size, test_type, test_subtype, 
        effect_size_category, significance_level
    )
    
    # Provide sample size recommendations for range of effect sizes
    effect_sizes = ['small', 'medium', 'large']
    sample_size_by_effect = {}
    for effect in effect_sizes:
        sample_size_by_effect[effect] = recommend_sample_size(
            test_type, test_subtype, effect, 
            significance_level, adjustments
        )["recommended_sample_size"]
    
    # Provide supplementary recommendations
    if power_expectation["evaluation"] == "insufficient":
        if test_subtype in ['independent', 'between']:
            alternative_designs = ["Consider a within-subjects design if feasible",
                                 "Add covariates to reduce unexplained variance",
                                 "Focus on more specific hypotheses with larger expected effects"]
        else:
            alternative_designs = ["Add more measurements per participant",
                                 "Use more reliable measures to reduce error variance",
                                 "Focus analysis on specific contrasts rather than omnibus tests"]
    else:
        alternative_designs = []
    
    # Create summary
    summary = {
        "planned_study": {
            "test_type": test_type,
            "test_subtype": test_subtype,
            "effect_size_category": effect_size_category,
            "planned_sample_size": sample_size,
            "significance_level": significance_level,
            "applied_adjustments": adjustments or {}
        },
        "power_analysis": {
            "base_recommended_power": base_power,
            "adjusted_recommended_power": adjusted_power,
            "expected_power": power_expectation["expected_power"],
            "evaluation_status": power_expectation["evaluation"],  # Fixed key name
            "message": power_expectation["message"]
        },
        "sample_size": {
            "recommended_sample_size": sample_recommendation["recommended_sample_size"],
            "difference_from_planned": sample_recommendation["recommended_sample_size"] - sample_size,
            "sample_sizes_by_effect_size": sample_size_by_effect
        },
        "recommendations": {
            "alternative_designs": alternative_designs,
            "minimum_detectable_effect_size": "Would need to be calculated specifically"
        }
    }
    
    return summary


# Example usage
if __name__ == "__main__":
    # Example 1: Get recommended power for a test type and effect size
    power = get_recommended_power('anova', 'within', 'medium')
    print(f"Recommended power for within-subjects ANOVA with medium effect: {power}")
    
    # Example 2: Get recommended sample size
    sample_rec = recommend_sample_size('t-test', 'independent', 'medium')
    print(f"Recommended sample size for independent t-test (medium effect): {sample_rec['recommended_sample_size']}")
    
    # Example 3: Check if planned study has sufficient power
    power_check = get_power_for_sample_size(30, 'anova', 'within', 'medium')
    print(f"Power check for N=30 in within-ANOVA: {power_check['evaluation']} (Power = {power_check['expected_power']:.2f})")
    
    # Example 4: Jazz musician study with limited sample
    jazz_analysis = power_analysis_summary(
        'anova', 'within', 'medium', 15,  # N=15 for jazz study
        adjustments={'measurement_reliability': 'moderate (0.70-0.90)'}
    )
    print("\nJazz Musician Study Power Analysis:")
    print(f"Expected power: {jazz_analysis['power_analysis']['expected_power']:.2f}")
    print(f"Evaluation: {jazz_analysis['power_analysis']['evaluation_status']}")  # Fixed key name
    print(f"Recommended sample size: {jazz_analysis['sample_size']['recommended_sample_size']}")
    
    # Example 5: With multiple tests requiring correction
    corrected_analysis = power_analysis_summary(
        'anova', 'within', 'medium', 30,
        adjustments={'multiple_tests': '5-10 tests'}
    )
    print("\nAnalysis with Multiple Comparison Correction:")
    print(f"Adjusted recommended power: {corrected_analysis['power_analysis']['adjusted_recommended_power']:.2f}")
    print(f"Recommended sample size: {corrected_analysis['sample_size']['recommended_sample_size']}")
