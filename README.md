# European Call Options Pricing Prediction
This project utilized regression and classification modeling techniques to predict option pricing data on the S&P 500 independently of the Black-Scholes formula.

This project is a final project of DSO 530: Applied Modern Machine Learning Methods class at USC Marshall School of Business (Spring 2023)

Group members: Prae Kongchan, Ninh Nguyen, Jacqueline Guerra, Chollada Unggurarat, Van Le, and Joyce Xinyi Jiang

# Data Preparation
The training dataset contains 1,680 records, each with six columns. The columns are as follows:
1. Value: the current option value
2. S: the current asset value
3. K: the strike price of the option
4. r: the annual interest rate
5. tau: the time to maturity (in years)
6. BS: the Black-Scholes formula was applied to this data to get the predicted option value
   
To conduct model explorations, the data underwent three necessary processes: formatting revision
and imputation of missing values, outlier removal, and data standardization.

## Revising Format and Imputing Missing Values
In order to explore the binary classification model using the BS column as the predictor, it was
necessary to convert the words 'under' and 'over' to '0' and '1', respectively. This was achieved by
creating a new column called BS_dummy to hold the transformed values.
It was observed that some records contained blank values in either the Value, S, or K column. To
address this issue, missing values were imputed by taking the average amount after grouping by
tau and r. For instance, in cases where a row had a blank value in the S column, it was found that
the asset would share the same value if it had the same time to maturity and the risk-free rate.
However, some rows had blank values in all three of the S, K, and tau columns. As these missing
predictors were deemed significant, such rows were dropped to prevent the machine learning
algorithm from learning from incomplete data.

## Dropping Outliers
During the data preparation process, outliers were identified and dropped to improve the integrity
of the dataset. In particular, a single maximum value of $40,333 was dropped from the S column,
as the average value for this column was approximately $441. Additionally, two rows were
dropped from the dataset where the tau was over one year. It is worth noting that options for normal
companies typically have a contract period of around 3 months, and long-term options contracts
such as Long-Term Equity Anticipation Securities, which have an expiration date more than one
year away from the current date, were not considered for this case. After data cleaning, the training
dataset has been reduced to 1,676 records.

## Standardizing the Data
The ultimate goal to perform standardization is to bring down all the features to a common scale
without distorting the differences in the range of the values. Our final list of predictors has a mean
of 0 and a standard deviation of 1.

# Feature Engineering
In order to further derive further insights and potential underlying relationships amongst the
variables, we conducted feature engineering to create two new variables.

## Moneyness
The first feature created is moneyness, a measure that describes the relationship between the S
(current asset value) and K (option's strike price). If the ratio is greater than 1, the option is in the
money, meaning that the option holder can buy the underlying asset at a lower price than the
current market value. This predictor helps us in gauging potential gain of an option.

## Time to Maturity
The second feature created is time to maturity, which is formulated by multiplying Tau (Time to
maturity) by 365 days; illustrating the time value of an option. While this feature does not affect
the outcomes of our models, it allows a deeper and granular understanding of the maturity period
using the time unit of 'day' compared to 'year'.

# Model | Regression
In order to make the most accurate prediction of the Current Option Value, we tested various
regression machine learning models. However, before implementing these models, we performed
feature selection on the 6 standardized variables. Using best subset selection, we determined that
4 variables - Current Asset Value, Strike Price, Days to Expiration, and Moneyness - would
achieve the optimal RSS and AIC/BIC. As a result, we removed Years to Maturity and Annual
Interest Rate from consideration. We felt this was logical given Years to Maturity was replaced by
Days to Maturity, and Annual Interest Rate had the least variability of the features given it would
be the same for any options within the same time period.

## Exploration
The baseline model was a Linear Regression model which utilized 10-fold cross-validation. The
result of this model was an out-of-sample R2 of 0.983. To ensure that the model was not overfitting,
we also assessed the model using the Root Mean Squared Error Ratio, which was the ratio of the
training RMSE to the testing RMSE. The Linear Regression model achieved a healthy RMSE ratio
of 0.998, indicating similar performance across the training and test set.
To further explore performance, we then ran a Decision Tree model with 10-fold cross-validation
and a cost complexity pruning alpha of 0.00002. This CCP alpha was the optimal value as
determined by GridSearchCV. While this model achieved a high out-of-sample R2 (0.996), the
RMSE ratio was very low at 0.08, indicating a significantly different Root Mean Squared Error
for the training and test set. Because the negative business implications of an over-fitted model are significant, we decided to utilize a higher pruning parameter to err on the side of caution. The
Decision Tree model with a CCP alpha of 0.1 achieved an out-of-sample R2 of 0.988 and an RMSE
ratio of 0.85. This CCP alpha value was chosen because it maximized out-of-sample R2 and the
RMSE ratio.

We additionally tested multiple Random Forest models with 10-fold cross-validation and CCP
alphas of 0.1. The number of estimators varied within the range of 50 to 200 in order to determine
the most effective model to optimize out-of-sample R2

## Final Model
Out of all models tested, the Random Forest model with 200 estimators demonstrated the best
performance. The out-of-sample R2 score of 0.994 was the highest, indicating that it could account
for 99.4% of the variation in the Current Option Value. Moreover, the model's RMSE ratio of 0.86
was healthy, implying a low risk of over-fitting. Considering these impressive performance
metrics, we felt confident using this model for predicting Current Option Values.

# Model | Classification
To address the classification problem of option valuation, namely whether an option will be
overvalued or undervalued, we conducted an exploratory analysis of different classification
models. Our objective was to balance model accuracy with the risk of overfitting, and we tried
various approaches, including splitting the data using the validation set approach and cross-
validation, incorporating shrinkage methods of Ridge and Lasso, and tuning hyperparameters in random forest models.

## Exploration
For our baseline model, we used a Logistic Regression model with no penalty. We applied the
validation set approach to split the data into 70% for training and 30% for testing. We selected
standardized current asset value, strike price, annual interest rate, days to expiration, and
moneyness as predictors based on the results of best subset selection, which minimized the mean
classification error. To ensure convergence, we set the maximum number of iterations to 10,000,
and for reproducibility, we set the random state to 0. The model's classification error was 8.15%.
To improve the model's performance, we incorporated a penalty, starting with the L1 or Lasso
penalty, for Logistic Regression. We used 10-fold cross-validation to select the best penalty level
and used the same set of predictors as the baseline model. This improved model showed an increase
in prediction accuracy by 0.199%. The L2 penalty, or Ridge, did not show a significant difference
in accuracy from Lasso, with a mean classification error of 7.952%. In order to increase the model complexity, the Random Forest models were experimented, by setting a maximum tree depth of 10 and iterating through different numbers of estimators to
determine the optimal value. Specifically, the values of 10, 50, 100, 150, and 200 were tried.

# Final Model
Among all the models we tried, the Random Forest model with 150 trees performed the best. We
used the same set of predictors as the previous models. The classification error of this model was
6.56%, showing an improvement of 1.391% compared to the previous model.

# Business Understanding & Conclusion
Random forest was found to outperform other models in predicting European call option prices
due to its ability to handle complex relationships and non-linearities, and its ensemble nature. It
provides a balance between performance and complexity, for predicting call option values.

Four business understandings need to be considered when predicting option values:

1. Accurately predicting European call option values is essential, but the interpretation is
equally important for decision-making. Understanding the relationships between predictor
variables and response variables can provide valuable insights to guide investment
strategies, risk management, or policy decisions.
2. Machine learning models can outperform Black-Scholes in predicting option prices due to
their flexibility and adaptability. They can capture complex patterns, nonlinearities,
varying volatility, changing interest rates, and non-continuous trading scenarios, resulting
in higher accuracy and practicality in real-world trading scenarios.
3. Our classification model, including all predictor variables plus moneyness, showed the
highest accuracy, while the regression model was not significantly affected by the number
of variables. Therefore, we recommend including all four predictor variables to better
understand the factors that influence option pricing from a business perspective.
4. Tesla's uniqueness compared to other S&P 500 options, due to high volatility, emerging
industry dynamics, founder sentiments, and significant growth expectations, makes it
difficult to predict using existing machine learning patterns.
