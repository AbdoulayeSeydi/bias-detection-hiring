# ML Bias Detection in Hiring: Education Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/bias-detection-hiring/blob/main/bias_detection_hiring.ipynb)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-complete-success.svg)

## Overview
This project measures systematic bias in ML-based resume screening by analyzing how education level affects matching scores, even when qualifications are equivalent.

**What this project does:**
- Analyzes job postings and candidate profiles using NLP embeddings
- Measures education-based disparities in hiring signals
- Tests whether predictive modeling amplifies or reduces bias
- Computes fairness metrics within equivalent qualification strata

**What this project does NOT do:**
- Predict who should be hired
- Build a production hiring tool
- Make causal claims about discrimination

## Key Findings

### Overall Bias Metrics (Baseline - Similarity Only)
- **Statistical Parity Difference (SPD):** -0.0430 (slight favor toward low-education candidates)
- **Disparate Impact Ratio:** 1.1481 ✓ **PASS** (exceeds 0.8 threshold)
- **Status:** No significant systematic bias detected in similarity scoring

### After Modeling (Logistic Regression)
- **Statistical Parity Difference (SPD):** -0.0285
- **Disparate Impact Ratio:** 1.1054 ✓ **PASS**
- **Change:** ΔSPD = +0.0145 | ΔDI = -0.0428

### Bias Amplification Analysis
**Finding:** **Modeling has MINIMAL EFFECT on education bias**
- Primary bias appears in the similarity scoring stage
- Education-based disparities are baked into the embeddings
- Logistic regression does not significantly amplify or reduce bias
- Both baseline and model pass the 80% rule for fairness

### Within-Stratum Analysis
- **Strata analyzed:** 1,960 candidate-job matches across multiple experience-role groups
- **Sample size:** 1,372 training samples, 588 test samples
- Equivalence enforced through experience-role stratification

## Project Structure
```
bias-detection-hiring/
├── bias_detection_hiring.ipynb    # Complete analysis pipeline
└── README.md                       # This file
```

## Notebook Sections

### Part 0: Environment Setup
- Install required packages (sentence-transformers, aif360)
- Import libraries and configure visualization settings

### Part 1: Data Acquisition & Processing
- Load 1,000 job postings (5 role types: Software Engineer, Data Scientist, Backend Developer, Frontend Developer, DevOps Engineer)
- Create 2,000 candidate profiles with education, experience, and skills
- Define education tiers:
  - **High:** PhD, Master's degree
  - **Mid:** Bachelor's degree
  - **Low:** Associate degree, Bootcamp, Self-taught
- Create experience-role strata for equivalence enforcement
- Generate text embeddings using Sentence-BERT (all-MiniLM-L6-v2)
- Compute candidate-job matching scores using cosine similarity

### Part 2: Exploratory Analysis
- Education distribution analysis across tiers
- Equivalence checks within strata (experience ranges < 2 years)
- Match score distributions by education tier (KDE plots)
- Correlation analysis between match scores, experience, and education
- Within-stratum visualization of score distributions

### Part 3: Bias Measurement
- Define outcome variable (high match = top 30% of similarity scores)
- Compute fairness metrics manually:
  - Statistical Parity Difference (SPD)
  - Disparate Impact Ratio (DI)
- Within-stratum fairness analysis
- Threshold sensitivity testing (20%, 30%, 40%, 50%)
- Comprehensive visualization dashboard:
  - Selection rates by education tier
  - Score distributions (overlaid KDE)
  - Stratum-level SPD distribution
  - Disparate Impact Ratio across strata

### Part 4: Modeling & Bias Amplification Test
- **Research Question:** Does predictive modeling amplify education-based bias?
- Train logistic regression model on:
  - Match score
  - Years of experience
  - Role category (one-hot encoded)
  - **Education NOT included as input** (discrimination by design if included)
- Model performance: AUC = 1.0000 (note: performance is NOT the goal)
- Compare baseline vs model fairness metrics
- Bias amplification analysis (ΔSPD, ΔDI)
- Final interpretation and visualization

## Methodology

### 1. Data Sources
- **Job Postings:** Synthetic dataset (1,000 postings across 5 tech roles)
- **Candidates:** Synthetic profiles (2,000 candidates with realistic distributions)
- **Note:** Clearly labeled as synthetic; results demonstrate methodology, not real-world findings

### 2. Education Tiers (Proxy for Socioeconomic Status)
- **High-tier:** Master's degree, PhD (25% of sample)
- **Mid-tier:** Bachelor's degree (40% of sample)
- **Low-tier:** Bootcamp, self-taught, associate degree (35% of sample)

### 3. Equivalence Enforcement (Critical for Valid Bias Measurement)
Created experience-role strata combining:
- **Role categories:** Software Engineer, Data Scientist, etc.
- **Experience bands:** Junior (0-2 yrs), Mid (2-5 yrs), Senior (5-10 yrs), Lead (10+ yrs)
- **Example stratum:** "Backend Developer_Mid"
- **Purpose:** Ensure comparisons are made within equivalent qualification groups
- **Result:** Filtered to strata with ≥10 candidates per education tier

### 4. Matching Score (Core Signal)
- Embedded job descriptions and candidate skills using Sentence-BERT
- Computed cosine similarity between resume and job embeddings
- Maximum similarity score used as "qualification match"
- This score simulates automated resume screening tools

### 5. Fairness Metrics
**Statistical Parity Difference (SPD):**
```
SPD = P(selected | high_ed) - P(selected | low_ed)
```
- Positive = high-ed favored
- Negative = low-ed favored
- Target: close to 0

**Disparate Impact Ratio (DI):**
```
DI = P(selected | low_ed) / P(selected | high_ed)
```
- < 0.8 = problematic (fails 80% rule)
- ≥ 0.8 = acceptable
- Target: close to 1.0

**Computed:**
- Overall (pooled across all candidates)
- Within-stratum (to control for qualification differences)
- Across multiple selection thresholds (20%, 30%, 40%, 50%)

### 6. Bias Amplification Test
**Hypothesis:** Does adding predictive modeling change bias?

**Baseline:** Similarity score only (NLP embeddings)
**Model:** Logistic regression (similarity + experience + role)

**Key Design Choice:** Education is NOT a model input
- Including protected attributes = discrimination by design
- Model must predict from legitimate signals only

**Evaluation:** Compare SPD and DI before and after modeling

## Technologies Used
- **Python 3.x**
- **Sentence Transformers:** all-MiniLM-L6-v2 for text embeddings
- **scikit-learn:** Logistic regression, preprocessing, metrics
- **pandas, numpy:** Data manipulation
- **matplotlib, seaborn:** Visualization
- **Google Colab:** Development environment

## Model Details

### Feature Engineering
- **Text embeddings:** 384-dimensional Sentence-BERT vectors
- **Structured features:** Years of experience (normalized), role categories (one-hot)
- **Matching score:** Cosine similarity between candidate and job embeddings

### Logistic Regression Model
```
Features: [match_score, years_experience, role_Data Scientist, 
           role_DevOps Engineer, role_Frontend Developer, role_Software Engineer]
Target: high_match (binary: top 30% of similarity scores)
```

**Top Coefficients:**
1. `match_score`: +9.45 (dominant predictor)
2. `years_experience`: -0.09
3. `role_DevOps Engineer`: -0.65

**Interpretation:** Match score overwhelmingly drives predictions; modeling adds minimal bias beyond embeddings.

## How to Run
1. Open `bias_detection_hiring.ipynb` in Google Colab
2. Run cells sequentially (Runtime → Run all)
3. Total runtime: ~30-40 minutes
   - Embedding generation: ~10 minutes
   - Analysis: ~5 minutes per section
4. All visualizations and results appear inline

## Results Summary

### Statistical Tests
- **ANOVA (score differences across education tiers):** p < 0.05
- **Conclusion:** Scores differ significantly, but effect size is small

### Threshold Sensitivity
Tested at 20%, 30%, 40%, 50% selection rates:
- **DI Ratio std:** < 0.1 (stable across thresholds)
- **Conclusion:** Bias findings are robust, not threshold-dependent

### Model Performance (For Reference Only)
- **AUC:** 1.0000 (perfect separation on test set)
- **Note:** This is NOT the evaluation criterion; we're measuring fairness, not accuracy

### Interpretation
✓ **No strong systematic bias detected** in either baseline or model
✓ **Minimal bias amplification** from predictive modeling
✓ **Education-based disparities exist but are small** and within acceptable thresholds
✓ **Primary bias source:** Similarity scoring (embeddings), not the predictive layer
✓ **Both approaches pass the 80% rule** for disparate impact

## Limitations

### Data Limitations
- **Synthetic data:** Results demonstrate methodology, not real-world hiring bias
- Education is a **proxy** for socioeconomic status, not a direct measure
- Sample size limited to 2,000 candidates, 1,000 job postings
- Simplified skill representations (text strings, not detailed taxonomies)

### Methodological Limitations
- **No causal claims:** This is correlation/measurement, not causation
- **No employer intent analysis:** Cannot determine if disparities are intentional
- **Simplified roles:** Real hiring involves more nuanced job categories
- **Binary education tiers:** Actual education backgrounds are more complex
- **Static analysis:** No temporal dimension or hiring pipeline stages

### Generalizability
- Results may not transfer to:
  - Real hiring data with actual outcomes
  - Different industries or job markets
  - Non-English job postings
  - Different embedding models or similarity metrics

### What This Project Does NOT Claim
❌ Evidence of discrimination in real hiring
❌ Causal relationship between education and hiring outcomes
❌ That this tool should be used for actual hiring decisions
❌ That all resume screening tools are biased
❌ Perfect measurement of socioeconomic bias

## Key Takeaways

### Technical Learnings
1. **NLP embeddings can encode subtle biases** from training data
2. **Fairness metrics must be computed within equivalent groups** (strata) to be valid
3. **Bias can exist at multiple pipeline stages:** text representation, scoring, modeling
4. **Threshold choices matter:** Always test sensitivity
5. **Manual metric computation matters:** Don't rely solely on libraries

### Fairness Insights
1. **Small biases in inputs can persist through pipelines** even without protected attributes as features
2. **Modeling may amplify, reduce, or maintain existing bias** depending on feature engineering
3. **Passing the 80% rule doesn't mean "no bias"**—just that bias is within legal/statistical thresholds
4. **Equivalence enforcement is critical** for credible bias measurement

### Project Management
1. **Clear scope boundaries prevent scope creep** (this is measurement, not prediction)
2. **Synthetic data is valid for methodology demonstration** when clearly labeled
3. **Limitations section is as important as results** for research credibility
4. **Visualizations are non-negotiable** for communicating bias findings

## Future Extensions

### Immediate Next Steps
- **Keyword analysis:** Identify language patterns driving small disparities
- **Real data validation:** Apply methodology to Stack Overflow Developer Survey
- **Institution prestige:** If data available, test whether school tier amplifies bias

### Advanced Extensions
- **Debiasing algorithms:** Test fairness-aware threshold adjustment
- **Temporal analysis:** Track how bias changes over time or hiring stages
- **Intersectional analysis:** Examine education + gender or education + race
- **Causal inference:** Use methods like regression discontinuity if outcome data available
- **Production deployment:** Build real-time bias monitoring dashboard

### Research Directions
- Compare different embedding models (BERT vs. RoBERTa vs. domain-specific)
- Test whether adversarial debiasing reduces disparities
- Validate against audit studies (correspondence testing)
- Examine bias in different job markets (tech vs. healthcare vs. finance)

## Academic Context
This project demonstrates techniques from:
- **ML Fairness:** Disparate impact, statistical parity, equalized odds
- **NLP Ethics:** Bias in language models and embeddings
- **Labor Economics:** Resume screening and hiring discrimination
- **Causal Inference:** Stratification for equivalence enforcement

## Contact
Questions or collaboration? Feel free to reach out!

abdoulayeaseydi@gmail.com


**Note:** This project was built as a learning exercise in ML fairness and bias detection. All limitations are clearly documented, and no claims are made about real-world hiring discrimination. The methodology is rigorous, the scope is intentionally narrow, and the findings are interpreted with appropriate caution.

## License
MIT License - Feel free to use this methodology for educational purposes.

---

**Built with:** Python, Sentence Transformers, scikit-learn, Google Colab  
**Project Type:** ML Fairness, Bias Detection, NLP Ethics  
**Status:** Complete ✓
