### This my submission to the Kaggle Titatnic competition.

I made the first attempts with random forests with the ranger R-package, implemented in the caret-package.

The first submission included "Pclass + Sex + Fare + SibSp + Embarked + Parch" as predictors and reached .7655 accuracy - room for improvement.

The second submission included the same variables + Age (with median imputation for the NAs), this tiny change improved the accuracy 
to .77990 and led to an advancement of 2,426 places on the Kaggle leaderboard.

This shows how mininal changes can improve the machine learning model and thus the overall prediction.

I'm also going to try logistic regression, gradient boosted modeling, and maybe an ensemble of all of them.
