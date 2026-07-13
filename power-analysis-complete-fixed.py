"""
Power Analysis Functions for Cognitive Science Experiments
---------------------------------------------------------
This module provides functions to calculate sample size requirements and statistical power
for various test types commonly used in cognitive science research, including mixed models.
"""

import math
import numpy as np
from scipy.stats import norm, chi2


def calculate_sample_size(
    desired_power: float,
    test_type: str,
    desired_effect_size: float,
    significance_level: float = 0.05,
    group_number: int = 2,
    condition_number: int = 2,
    subject_design: str = 'between',
    measurements_per_subject: int = 1,
    direction: str = 'two-tail',
    expected_dependent_variance: float = 1.0,
    predictors: int = None
) -> int:
    """
    Calculate required sample size for a given statistical test.
    
    Parameters:
        desired_power: Probability of detecting a true effect if it exists (1-β)
        test_type: Type of statistical test ('t-test', 'correlation', 'regression', 'anova', etc.)
        desired_effect_size: Expected magnitude of the effect
        significance_level: Alpha value (Type I error rate)
        group_number: Number of groups
        condition_number: Number of conditions
        subject_design: 'between' or 'within' for repeated measures
        measurements_per_subject: Number of measurements per subject for within-designs
        direction: 'one-tail' or 'two-tail'
        expected_dependent_variance: Variance in dependent measure
        predictors: Number of predictors for regression analyses
    
    Returns:
        Required sample size (rounded up to the nearest integer)
    """
    # Determine critical value based on test direction
    alpha = significance_level if direction == 'one-tail' else significance_level / 2
    critical_z = abs(norm.ppf(alpha))
    power_z = norm.ppf(desired_power)
    
    required_n = None
    
    # Calculate sample size based on test type
    if test_type.lower() == 't-test':
        if subject_design == 'between':
            # For independent samples t-test
            required_n = 2 * ((critical_z + power_z) ** 2) / (desired_effect_size ** 2)
        else:
            # For paired samples t-test (within-subjects)
            correlation = 0.5  # Assumed correlation between measures
            required_n = ((critical_z + power_z) ** 2) / (desired_effect_size ** 2 * (1 - correlation))
    
    elif test_type.lower() == 'correlation':
        # For correlation analysis
        fisher_z = 0.5 * np.log((1 + desired_effect_size) / (1 - desired_effect_size))
        required_n = ((critical_z + power_z) / fisher_z) ** 2 + 3
    
    elif test_type.lower() == 'regression':
        # For regression analysis
        if not predictors:
            predictors = 1
        required_n = (8 / (desired_effect_size ** 2)) * ((critical_z + power_z) ** 2) + predictors + 1
    
    elif test_type.lower() == 'anova':
        if subject_design == 'between':
            # Between-subjects ANOVA
            lambda_val = (critical_z + power_z) ** 2
            n_per_group = lambda_val / (desired_effect_size ** 2)
            required_n = n_per_group * group_number
        else:
            # Within-subjects ANOVA
            lambda_val = (critical_z + power_z) ** 2
            base_n = lambda_val / (desired_effect_size ** 2)
            correlation = 0.5
            correction_factor = 1 - correlation
            required_n = base_n * correction_factor
    
    elif test_type.lower() == 'chi-square':
        # For chi-square test
        df = (group_number - 1) * (condition_number - 1)
        critical_chi = chi2.ppf(1 - significance_level, df)
        required_n = (critical_chi / (desired_effect_size ** 2))
    
    else:
        raise ValueError(f"Test type '{test_type}' not implemented")
    
    # Account for measurement variance
    if expected_dependent_variance > 1:
        required_n = required_n * expected_dependent_variance
    
    # Account for repeated measurements in within-subject designs
    if subject_design == 'within' and measurements_per_subject > 1:
        required_n = required_n / math.sqrt(measurements_per_subject)
    
    # Return ceiling of the number to ensure integer sample size
    return math.ceil(required_n)


def calculate_power(
    sample_size: int,
    test_type: str,
    desired_effect_size: float,
    significance_level: float = 0.05,
    group_number: int = 2,
    condition_number: int = 2,
    subject_design: str = 'between',
    measurements_per_subject: int = 1,
    direction: str = 'two-tail',
    expected_dependent_variance: float = 1.0,
    predictors: int = None
) -> float:
    """
    Calculate statistical power for a given sample size and effect size.
    
    Parameters:
        sample_size: Number of participants in the study
        test_type: Type of statistical test
        desired_effect_size: Expected magnitude of the effect
        significance_level: Alpha value (Type I error rate)
        group_number: Number of groups
        condition_number: Number of conditions
        subject_design: 'between' or 'within' for repeated measures
        measurements_per_subject: Number of measurements per subject for within-designs
        direction: 'one-tail' or 'two-tail'
        expected_dependent_variance: Variance in dependent measure
        predictors: Number of predictors for regression analyses
    
    Returns:
        Calculated statistical power (probability of detecting a true effect)
    """
    # Input validation
    if sample_size <= 0:
        raise ValueError("Sample size must be positive")
    if significance_level <= 0 or significance_level >= 1:
        raise ValueError("Significance level must be between 0 and 1 exclusive")
    if desired_effect_size <= 0:
        raise ValueError("Effect size must be positive")
    
    # Determine critical value based on test direction
    alpha = significance_level if direction == 'one-tail' else significance_level / 2
    critical_z = abs(norm.ppf(alpha))
    
    # Adjust effective effect size for measurement variance and repeated measurements
    effective_size = desired_effect_size
    if expected_dependent_variance > 1:
        effective_size = effective_size / math.sqrt(expected_dependent_variance)
    if subject_design == 'within' and measurements_per_subject > 1:
        effective_size = effective_size * math.sqrt(measurements_per_subject)
    
    non_centrality = None
    
    if test_type.lower() == 't-test':
        if subject_design == 'between':
            # Independent samples t-test
            n_per_group = sample_size / 2
            non_centrality = effective_size * math.sqrt(n_per_group / 2)
        else:
            # Paired samples t-test
            correlation = 0.5
            effective_n = sample_size * (1 - correlation)
            non_centrality = effective_size * math.sqrt(effective_n)
        
        # Calculate power
        power = 1 - norm.cdf(critical_z - non_centrality)
    
    elif test_type.lower() == 'correlation':
        # For correlation
        fisher_z = 0.5 * np.log((1 + effective_size) / (1 - effective_size))
        se = 1 / math.sqrt(sample_size - 3)
        non_centrality = fisher_z / se
        power = 1 - norm.cdf(critical_z - non_centrality)
    
    elif test_type.lower() == 'regression':
        # For regression
        k = predictors or 1
        non_centrality = effective_size * (sample_size - k - 1)
        power = 1 - norm.cdf(critical_z - math.sqrt(non_centrality))
    
    elif test_type.lower() == 'anova':
        if subject_design == 'between':
            # Between-subjects ANOVA
            n_per_group = sample_size / group_number
            non_centrality = math.sqrt(n_per_group) * effective_size
        else:
            # Within-subjects ANOVA
            correlation = 0.5
            adjusted_n = sample_size * (1 - correlation)
            non_centrality = math.sqrt(adjusted_n) * effective_size
        
        power = 1 - norm.cdf(critical_z - non_centrality)
    
    elif test_type.lower() == 'chi-square':
        # For chi-square test
        df = (group_number - 1) * (condition_number - 1)
        non_centrality = sample_size * (effective_size ** 2)
        critical_chi = chi2.ppf(1 - significance_level, df)
        mean = df + non_centrality
        sd = math.sqrt(2 * (df + 2 * non_centrality))
        power = 1 - norm.cdf((critical_chi - mean) / sd)
    
    else:
        raise ValueError(f"Test type '{test_type}' not implemented")
    
    return power


def calculate_sample_size_mixed_model(
    desired_power: float,
    within_effect_size: float,
    between_effect_size: float,
    n_within_factors: int,
    n_within_levels: list,
    n_between_factors: int,
    n_between_levels: list,
    significance_level: float = 0.05,
    expected_correlation: float = 0.5,
    direction: str = 'two-tail'
) -> int:
    """
    Calculate required sample size for a mixed design with both within and between subject factors.
    
    Parameters:
        desired_power: Target statistical power (1-β), typically 0.8
        within_effect_size: Expected effect size for within-subject factors (f)
        between_effect_size: Expected effect size for between-subject factors (f)
        n_within_factors: Number of within-subject factors
        n_within_levels: List containing number of levels for each within-subject factor
        n_between_factors: Number of between-subject factors
        n_between_levels: List containing number of levels for each between-subject factor
        significance_level: Alpha level, typically 0.05
        expected_correlation: Expected correlation between repeated measurements
        direction: 'one-tail' or 'two-tail'
    
    Returns:
        Required total sample size
    """
    # Validate inputs
    if len(n_within_levels) != n_within_factors:
        raise ValueError("Length of n_within_levels must match n_within_factors")
    if len(n_between_levels) != n_between_factors:
        raise ValueError("Length of n_between_levels must match n_between_factors")
    
    # Calculate total number of within-subject conditions
    total_within_conditions = 1
    for levels in n_within_levels:
        total_within_conditions *= levels
    
    # Calculate total number of between-subject groups
    total_between_groups = 1
    for levels in n_between_levels:
        total_between_groups *= levels
    
    # Calculate degrees of freedom for interactions
    df_within = total_within_conditions - 1
    df_between = total_between_groups - 1
    df_interaction = df_within * df_between
    
    # Determine critical value based on test direction
    alpha = significance_level if direction == 'one-tail' else significance_level / 2
    critical_z = abs(norm.ppf(alpha))
    power_z = norm.ppf(desired_power)
    
    # Calculate lambda (non-centrality parameter)
    lambda_val = (critical_z + power_z) ** 2
    
    # Calculate required sample size per group for within effects
    # Adjust for correlation between measurements
    correction_factor = 1 - expected_correlation
    n_per_group_within = (lambda_val / (within_effect_size ** 2)) * correction_factor
    
    # Calculate required sample size per group for between effects
    n_per_group_between = lambda_val / (between_effect_size ** 2)
    
    # For interaction effects, we take the more conservative approach
    n_per_group_interaction = max(n_per_group_within, n_per_group_between)
    
    # Final sample size calculation (per group)
    n_per_group = max(n_per_group_within, n_per_group_between, n_per_group_interaction)
    
    # Calculate total sample size
    total_sample_size = n_per_group * total_between_groups
    
    return math.ceil(total_sample_size)


def calculate_power_mixed_model(
    sample_size: int,
    within_effect_size: float,
    between_effect_size: float,
    n_within_factors: int,
    n_within_levels: list,
    n_between_factors: int,
    n_between_levels: list,
    significance_level: float = 0.05,
    expected_correlation: float = 0.5,
    direction: str = 'two-tail'
) -> dict:
    """
    Calculate power for a mixed design with both within and between subject factors.
    
    Parameters:
        sample_size: Total number of participants in the study
        within_effect_size: Expected effect size for within-subject factors (f)
        between_effect_size: Expected effect size for between-subject factors (f)
        n_within_factors: Number of within-subject factors
        n_within_levels: List containing number of levels for each within-subject factor
        n_between_factors: Number of between-subject factors
        n_between_levels: List containing number of levels for each between-subject factor
        significance_level: Alpha level, typically 0.05
        expected_correlation: Expected correlation between repeated measurements
        direction: 'one-tail' or 'two-tail'
    
    Returns:
        Dictionary containing power estimates for within effects, between effects, and overall
    """
    # Validate inputs
    if len(n_within_levels) != n_within_factors:
        raise ValueError("Length of n_within_levels must match n_within_factors")
    if len(n_between_levels) != n_between_factors:
        raise ValueError("Length of n_between_levels must match n_between_factors")
    
    # Calculate total number of between-subject groups
    total_between_groups = 1
    for levels in n_between_levels:
        total_between_groups *= levels
        
    # Sample size per group
    n_per_group = sample_size / total_between_groups
    
    # Determine critical value based on test direction
    alpha = significance_level if direction == 'one-tail' else significance_level / 2
    critical_z = abs(norm.ppf(alpha))
    
    # Calculate non-centrality parameters
    # For within effects (adjusted for correlation)
    correction_factor = 1 - expected_correlation
    lambda_within = n_per_group * (within_effect_size ** 2) / correction_factor
    
    # For between effects
    lambda_between = n_per_group * (between_effect_size ** 2)
    
    # Calculate power for within and between effects
    power_within = 1 - norm.cdf(critical_z - math.sqrt(lambda_within))
    power_between = 1 - norm.cdf(critical_z - math.sqrt(lambda_between))
    
    # Return both power estimates
    return {
        'within_effects': power_within,
        'between_effects': power_between,
        'overall': min(power_within, power_between)  # Conservative estimate
    }


# Effect size guidance by test type
effect_size_guidance = {
    't-test': {'small': 0.2, 'medium': 0.5, 'large': 0.8},
    'correlation': {'small': 0.1, 'medium': 0.3, 'large': 0.5},
    'regression': {'small': 0.02, 'medium': 0.15, 'large': 0.35},
    'anova': {'small': 0.1, 'medium': 0.25, 'large': 0.4},
    'chi-square': {'small': 0.1, 'medium': 0.3, 'large': 0.5},
    'mixed-model': {'small': 0.1, 'medium': 0.25, 'large': 0.4}
}

# Power size guidance



if __name__ == "__main__":
    # Examples
    print("Sample Size Examples:")
    
    # T-test
    print(f"Independent t-test (medium effect): {calculate_sample_size(0.8, 't-test', 0.5)}")
    print(f"Paired t-test (medium effect): {calculate_sample_size(0.8, 't-test', 0.5, subject_design='within')}")
    
    # Correlation
    print(f"Correlation (medium effect): {calculate_sample_size(0.8, 'correlation', 0.3)}")
    
    # ANOVA
    print(f"Between-subjects ANOVA (3 groups): {calculate_sample_size(0.8, 'anova', 0.25, group_number=3)}")
    
    # One-tailed vs Two-tailed
    print(f"Two-tailed t-test: {calculate_sample_size(0.8, 't-test', 0.5, direction='two-tail')}")
    print(f"One-tailed t-test: {calculate_sample_size(0.8, 't-test', 0.5, direction='one-tail')}")
    
    # Effect of measurements
    print(f"Within-subjects (1 measurement): {calculate_sample_size(0.8, 't-test', 0.5, subject_design='within', measurements_per_subject=1)}")
    print(f"Within-subjects (3 measurements): {calculate_sample_size(0.8, 't-test', 0.5, subject_design='within', measurements_per_subject=3)}")
    
    # Example for 3×2 Mixed Model with 3 within and 2 between levels
    print("\nMixed Model Example (3×2 design):")
    sample_size_mixed = calculate_sample_size_mixed_model(
        desired_power=0.8,
        within_effect_size=0.25,  # Medium effect for within factors
        between_effect_size=0.25,  # Medium effect for between factors
        n_within_factors=1,
        n_within_levels=[3],  # 3 levels of the within-subject factor
        n_between_factors=1,
        n_between_levels=[2],  # 2 levels of the between-subject factor
        significance_level=0.05,
        expected_correlation=0.5
    )
    print(f"Required sample size for 3×2 mixed design: {sample_size_mixed}")
    print(f"That's {sample_size_mixed // 2} participants per between-subject group")
    
    # Example for the jazz musician study (2×2 within-subject design)
    print("\nJazz Musician Study Example:")
    sample_size_jazz = calculate_sample_size(
        desired_power=0.8,
        test_type='anova',
        desired_effect_size=0.25,  # Medium effect size
        significance_level=0.05,
        group_number=4,  # 2 VR scenarios × 2 smells = 4 conditions
        subject_design='within',
        measurements_per_subject=4  # 4 performances per musician
    )
    print(f"Required sample size for the jazz musician study: {sample_size_jazz} musicians")
    
    # Power calculation examples
    print("\nPower Calculation Examples:")
    
    # T-test power
    print(f"Power for independent t-test (n=64, medium effect): {calculate_power(64, 't-test', 0.5):.4f}")
    print(f"Power for paired t-test (n=30, medium effect): {calculate_power(30, 't-test', 0.5, subject_design='within'):.4f}")
    
    # Correlation power
    print(f"Power for correlation (n=85, r=0.3): {calculate_power(85, 'correlation', 0.3):.4f}")
    
    # ANOVA power
    print(f"Power for between-subjects ANOVA (n=90, 3 groups): {calculate_power(90, 'anova', 0.25, group_number=3):.4f}")
    print(f"Power for within-subjects ANOVA (n=30, 4 conditions): {calculate_power(30, 'anova', 0.25, group_number=4, subject_design='within'):.4f}")
    
    # Effect of direction on power
    print(f"Power for two-tailed t-test (n=50): {calculate_power(50, 't-test', 0.5, direction='two-tail'):.4f}")
    print(f"Power for one-tailed t-test (n=50): {calculate_power(50, 't-test', 0.5, direction='one-tail'):.4f}")
    
    # Small vs large effect sizes
    print(f"Power for small effect (d=0.2, n=64): {calculate_power(64, 't-test', 0.2):.4f}")
    print(f"Power for large effect (d=0.8, n=64): {calculate_power(64, 't-test', 0.8):.4f}")
    
    # Example of power calculation for mixed model
    print("\nMixed Model Power Analysis:")
    mixed_power = calculate_power_mixed_model(
        sample_size=60,  # Total sample size
        within_effect_size=0.25,
        between_effect_size=0.25,
        n_within_factors=1,
        n_within_levels=[3],
        n_between_factors=1,
        n_between_levels=[2],
        expected_correlation=0.5
    )
    print(f"Power for within-subject effects: {mixed_power['within_effects']:.4f}")
    print(f"Power for between-subject effects: {mixed_power['between_effects']:.4f}")
    print(f"Overall estimated power: {mixed_power['overall']:.4f}")
    
    # Example of the jazz musician study power calculation
    print("\nJazz Musician Study Power Analysis (with n=20):")
    jazz_power = calculate_power(
        sample_size=20,
        test_type='anova',
        desired_effect_size=0.25,  # Medium effect size
        significance_level=0.05,
        group_number=4,  # 2 VR scenarios × 2 smells = 4 conditions
        subject_design='within',
        measurements_per_subject=4  # 4 performances per musician
    )
    print(f"Power for jazz study with 20 musicians: {jazz_power:.4f}")
    
    # Calculate power at different sample sizes for jazz study
    print("\nJazz Study Power at Different Sample Sizes:")
    sample_sizes = [10, 15, 20, 25, 30, 35, 40]
    powers = []
    
    for n in sample_sizes:
        power = calculate_power(
            sample_size=n,
            test_type='anova',
            desired_effect_size=0.25,
            significance_level=0.05,
            group_number=4,
            subject_design='within',
            measurements_per_subject=4
        )
        powers.append(power)
        print(f"n={n}: Power = {power:.4f}")
        


    # Check if desired power of 0.8 is achieved
    for i, power in enumerate(powers):
        if power >= 0.8:
            print(f"\nA sample size of {sample_sizes[i]} musicians achieves the desired power of 0.8")
            break
