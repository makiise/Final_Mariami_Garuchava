Project Overview: Heart Failure Mortality Prediction
Author: Mariami Garuchava

This project is a comprehensive clinical data science study designed to predict mortality risk in patients with heart failure. By integrating medical domain knowledge with robust mathematical modeling, this study evaluates how specific physiological markers influence patient survival.

1. The Clinical Objective

Analyzing features:

trying to figure out about medical terms:

Age -> older hearts can't recover from stress as well as younger hearts

Anaemia - Less hemoglobin means the blood carries less oxygen. This makes the already-weak heart work twice as hard to oxygenate the body.
Creatinine Phosphokinase -> When it's high, it means muscle tissue (likely the heart) is damaged. 
Diabetes -> It causes "Diabetic Cardiomyopathy." High sugar levels damage the small blood vessels and nerves that control the heart, making heart failure more likely to be fatal.
Ejection fraction --->A normal ejection fraction is usually 50-70%, 41-49% - may signal early damage or risk, below 40%can indicate heart failure

High_blood_pressure --->significantly  increases heart disease risk by damaging arteries 
Platelets ---> If these are too high or low, it affects blood clotting. In heart failure, this can lead to strokes or internal bleeding.

serum_creatinine --> high means that kidneys are not working well

Serum Sodium: Low levels (Hyponatremia) in heart failure patients often mean the body is retaining too much water because the heart isn't pumping well. It’s a huge "danger" signal.

Sex -> Research often shows men are diagnosed earlier with heart failure, but women often have different types (like Heart Failure with Preserved Ejection Fraction).

Smoking_time ->  directly increases heart disease risk by damaging blood vessels

The goal is to move beyond simple data classification and understand the "vicious cycles" of heart failure. Each feature was analyzed through a clinical lens:

Mechanical Markers: Analysis of Ejection Fraction revealed a "critical threshold" at 35%; below this, mortality density surpasses survival density, cutoff for heart failure.

Biochemical Indicators: I identified Serum Creatinine as a right-skewed variable where even small increases indicate kidney dysfunction.

Metabolic Stressors: Features like Diabetes (Diabetic Cardiomyopathy) and Anaemia were treated as risk multipliers that force a weakened heart to work twice as hard.

Acute Damage: Creatinine Phosphokinase (CPK) was identified as a high-variance marker of muscle damage, exhibiting a massive right-skew (Mean 581 vs. Median 250).

age distribution:
on this KDE plot, we can definetely see the "Distribution shift", which means, based on the dataset,  older patients have a higher probability of the death event.
The "Dead" distribution is stochastically larger than the "Survivors" distribution.
"Survivors" distributions are more normally distributed, while "Dead" distribution has high mean and high variance.

ejection fraction distribution: visualization shows that patients who had low ejection fraction level (below 30%) most likely to be dead. Curve of  "Dead" distribution peaks at approximately 25%. "Survivors" distribution has bimodal nature, it has a huge peak at around 40% (which means early heart failure, but stable), and second much smaller peak at around 60% (that means healthy). Based on this visualization, we can see "critical threshold" at approximately 35%.
below that, density of red curve which is "Dead" distribution, is much higher than blue curve of "Survivors" distribution. That means 35% of ejection fraction is mathematically significant cutoff for survival prediction.

Serum creatinine: 
Based on the visualization, "Survivors" distribution has low variance, is super narrow and peaked at approximately 1. 
However, "Dead" distribution has higher variance, long tail on right, heavily right skewed, tail  even reaches 6.0 and more on scale. That means that even small increase of serum creatinine can be a reason for a death event to occur.

platelets:
Based on the boxplot showing how platelet concentration in blood relates to fatal heart disease, the median values for the “Survival” and “Dead” groups are almost the same. Additionally, the heights of the boxes are very close to each other, which can be considered essentially identical. This suggests that platelet concentration does not have a strong effect on mortality due to heart disease in this dataset.
We also observe many data points above 400,000, which cannot be considered noise and removed, since this is medical data. Such high platelet counts are indicators of thrombocytosis, which increases the risk of blood clots and can be dangerous for heart stability.

creatine phosphokinase:
The same pattern is observed here. The medians and box heights are almost identical, indicating that creatine phosphokinase does not play a significant role in heart disease mortality in this dataset. Both boxplots are concentrated near zero, meaning that most patients have low creatine phosphokinase levels and the distribution is highly skewed.
However, we also observe several extreme values (especially among survived patients) that can reach up to 6000. Some dead patients also had higher than 6000.

Based on the data, non-smoker patients show a higher number of survivors compared to smoker patients.
same goes with patients with high blood pressure, diabetes and anaemia.

The bottom "tail" of the Red (Dead) violin is much longer and thinner than the Blue one. This plot reveals that dead patients are prone to extreme Hyponatremia (low sodium levels below 125 mEq/L). Survivors exhibit a much tighter, distribution around the healthy mean of 137 mEq/L.
by this, we successfully identified the most important 3 factors of a heart disease, which are:
1. Serum Creatinine - with a strong positive correlation (0.37) 
 --> The kidney-heart link, low Serum Creatinine causes  kidney dysfunction, which strains the heart by causing fluid buildup and hypertension, increasing risks for heart failure.
2. Ejection Fraction - with a strong negative correlation (-0.29) 
 --> heart's main pumping chamber (left ventricle) isn't effectively pumping enough blood out to body = High risk of a heart disease.
3.  Age - with a solid positive correlation (0.22)

Time has strongest negative correlation on heatmap (-0.54), that's because, when as long as time increases, there is less probability that the patient will die, because the time describes the days the patient was observed. Patient either died or when the observation and collecting the data has ended, the patient was still alive. that variable might cause "data leakage", because model might assume that if the time is short, that's because death_event occured, which is not something that can be happen in real world during medical observation.  
"time" variable will be cause of model's very high accuracy, but it will fail on a real world scenario.

2. Mathematical Methodology & Integrity

Data Leakage Prevention: The time variable (follow-up period) was intentionally excluded from the training process. While including it yields artificially high accuracy (95%+), it represents retrospective observation rather than prospective prediction. Excluding it ensures the model is a valid tool for real-world medical intake.

Robust Preprocessing:

Median Imputation: Utilized for skewed numerical variables to remain robust against the extreme outliers identified in CPK and Platelets.

Z-Score Normalization: Applied to ensure all clinical units (e.g., Age 60 vs. Platelets 400,000) are treated with equal mathematical weight during model optimization.

Advanced Visualization: I utilized Kernel Density Estimates (KDE) to visualize "Distribution Shifts," proving that the "Dead" distribution is stochastically larger than the "Survivors" distribution.

3. Machine Learning Paradigm Comparison

I compared three distinct mathematical approaches:

Logistic Regression: A baseline linear classifier chosen for its interpretability and ability to calculate clinical Odds Ratios.

Random Forest: A bagging ensemble of 100 decision trees, ideal for capturing non-linear interactions (e.g., the combined risk of Age and High Creatinine).

XGBoost: A sequentially optimized gradient boosting model with built-in L1/L2 regularization to prevent overfitting on small clinical cohorts.

4. Evaluation Philosophy: Prioritizing Recall

In clinical medicine, the cost of a False Negative (missing a high-risk patient) is life-critical. Consequently, this project prioritizes Recall (Sensitivity) over global Accuracy.

The Finding: The Random Forest model was selected as the superior clinical tool, achieving the highest Recall (63%) and the fewest False Negatives in the confusion matrix analysis.


