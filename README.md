# Credit Scoring Business Understanding

## 1. How does the Basel II Accord's emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord introduces three pillars that fundamentally reshape how banks approach credit risk:

Pillar 1 (Minimum Capital Requirements) emphasizes that banks must hold capital proportional to the credit risk they undertake. This requires:

Risk sensitivity: Models must accurately quantify risk levels
Backtesting: Regular validation against actual outcomes
Stress testing: Assessment under adverse conditions

Pillar 2 (Supervisory Review) mandates that supervisors evaluate banks' internal capital adequacy assessments. This necessitates:

Transparency: Clear documentation of model assumptions and limitations
Explainability: Regulators must understand how decisions are made
Governance: Formal model validation frameworks

Pillar 3 (Market Discipline) promotes disclosure, requiring:

Auditability: External stakeholders must be able to assess model quality
Comparability: Standardized risk metrics
Implications for our model:
Interpretability is non-negotiable: Regulators require understanding of feature contributions
Comprehensive documentation: Every modeling choice must be justified
Robust validation: Extensive testing of model stability and performance
Risk-weighted approach: Capital allocation depends directly on model outputs

## 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

Why a proxy is necessary:

No historical loan data: The eCommerce platform hasn't offered credit before
Behavioral patterns as predictors: Customer engagement patterns (RFM) correlate with creditworthiness
Regulatory acceptance: Basel II allows internal ratings-based approaches with proper justification
Innovation requirement: Alternative data sources are increasingly accepted in credit scoring

Potential business risks:

Model Risk:

Proxy-target misalignment: RFM patterns may not perfectly correlate with actual credit default
Concept drift: ECommerce behavior ≠ credit behavior under financial stress
Selection bias: Current customers may differ from future credit applicants

Regulatory Risk:

Validation challenges: Proxies are harder to validate than actual defaults
Compliance uncertainty: Regulators may question proxy suitability
Capital adequacy: Misestimated risk could lead to insufficient capital buffers

Business Risk:

False positives: Rejecting creditworthy customers → lost revenue
False negatives: Approving high-risk customers → credit losses
Reputation damage: Poor credit decisions harm brand reputation
Financial loss: Direct losses from defaults and indirect costs of capital misallocation

Mitigation strategies:

Conservative thresholds: Higher approval standards initially
Phased rollout: Start with small credit limits
Continuous monitoring: Track actual performance vs. predictions

Model recalibration: Update as real default data becomes available

## 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

Simple, Interpretable Models (Logistic Regression with WoE)
Advantages for regulated context:

Regulatory compliance: Easier to explain to regulators and satisfy Pillar 2 requirements
Transparency: Clear feature contributions via coefficients
Stability: Less prone to overfitting on small datasets
Auditability: Simple to validate and backtest
Documentation: Easier to document assumptions and limitations
Basel II alignment: Supports the "use test" requirement that models be understandable to management

Disadvantages:

Performance ceiling: May not capture complex non-linear relationships
Feature engineering dependency: Heavy reliance on WoE transformation quality
Interaction limitations: Manual specification of interactions required
Complex, High-Performance Models (Gradient Boosting)

Advantages:

Predictive power: Often achieves superior discrimination
Automatic feature interactions: Captures complex patterns
Robustness: Handles non-linear relationships and missing data well
Ensemble benefits: Reduces variance through boosting

Disadvantages for regulated context:

Black box problem: Difficult to explain individual predictions (contradicts Basel II's transparency requirements)
Regulatory scrutiny: May face resistance from supervisors under Pillar 2
Validation complexity: Harder to comprehensively validate
Instability risk: Sensitive to hyperparameters and data shifts
Documentation burden: Challenging to fully document model behavior
