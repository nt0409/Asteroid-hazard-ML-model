# Asteroid-hazard-ML-model
I created the following model for National Space day event held at my college.
The following model uses NEO satellite dataset for predicting if an asteroid is hazardous or not.
There are two models one is using Logistic regression however the overall accuracy of the model is 78% because I used SMOTE to generate synthetic sample to balance the dataset and increase the F-1 score(0.45) for True.
Second model is made using XGBOOST classifier and it has 82% percent accuracy with 0.49 F-1 score.
I am providing the streamlit code for XGBOOST classifier only.
The model takes 5 inputs '**est_diameter_min', 'est_diameter_max', 'relative_velocity', 'absolute_magnitude', 'miss_distance'**. and returns the output as true or false.
You can use **Apophis asteroid** data(available on google) which was said to be very dangerous with the above 5 parameters to check the model reliability.
