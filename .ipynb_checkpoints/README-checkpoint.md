

# Counting Calories with Math!
Project for DSC80 @ UCSD

**Before you read, you should check out our exploratory data analysis of the dataset!**
<a href="https://jpuray.github.io/Recipe-Reviews-Analysis/">Exploratory data analysis of Recipes dataset</a>

## Framing the Problem

***Our problem: Can we predict the calories of a certain recipe based on other information from our dataset?***

In this context, we are creating a Regression model to predict our response variable, the caloric value of the recipe. We figured that the value of calories would be interesting to predict for our project because there's plenty of interesting features that could share relationships with 'calories'. For example, cuisine could be a good indication of calories because of certain countries'/regions' tendencies to use oily/greasier ingredients vs. others' tendencies to use more vegetables and lighter ingredients, but this is just one example (and not one that we necessarily used in our model!) 

To evaluate our model's success, we chose the **Root Mean Squared Error (RMSE)**. 

MSE is one of the conventional metrics for computing the quality of the Model's predictions, because of its ability to measure differences in actual vs. predicted values. However, it gives us a value that is hard to interpret – squared calories. To mitigate this, we use RMSE instead. This will give us the error in terms of our response variable's original value – calories. 

As a reminder, RMSE measures the difference between the actual data and the predictions of the data. A characteristic of RMSE is that for larger differences in actual vs. predicted values, the effect on RMSE will be disproportionately large. That is, RMSE *is* sensitive to outliers, but the subset of the recipes data that we use for predictions has outliers removed.

In terms of data we'd have at the time of prediction, we believe that all data from the subset of data we're using is available at the time of prediction. This includes:
- Name
- Minutes
- ID
- Tags
- Nutrition, which contains:
    - Calories
    - Total Fat
    - Sugar
    - Sodium
    - Protein
    - Saturated Fat
    - Carbohydrates
- Number of Steps
- Steps
- Description
- Ingredients
- Number of Ingredients

We also acknowledge that it may seem redundant to predict calories when the values are already there in the nutrition column. However, a real life scenario where our prediction model might be helpful would be if there was no calorie value input into a recipe or if food.com suddenly had all calorie values erased from its recipes. In that case, a high-quality model would be able to fill in those values for people who value the caloric value of a recipe (e.g. dietary, health, etc.)

## Baseline Model

***Getting Started***

To begin, we will establish a baseline model to compare with our other models later on. Our baseline model will use linear regression, trained on the nutrition features of our data. Since all the nutrition features are quantitative, implementing them into our model will be straightforward, without requiring any transformations. However, before starting the baseline model, we must determine which nutrition features are most highly correlated with calories by exploring the data.

***Exploring Features***

To determine which features are most correlated with calories we'll plot a correlation matrix. 

```py
sns.set(rc={"figure.figsize":(12, 10)})
sns.heatmap(nutrition.corr(), vmin=-1, vmax=1, annot=True)
```

<iframe src="data_viz/corr_matrix.jpg" width=800 height=600 frameBorder=0></iframe>

Based on our analysis, we have identified the most highly correlated features with calories in our dataset, which are `total fat (PDV)`, `sugar (PDV)`, `protein (PDV)`, `saturated fat (PDV)`, and `carbohydrates`. However, we must be cautious of highly correlated features, as they may lead to redundancy and not significantly reduce model error. For instance, since total fat and saturated fat have a high r-correlation of 0.86, we will exclude saturated fat from our model as it has a smaller correlation with calories. Similarly, since sugar and carbohydrates are highly correlated with an r-correlation of 0.88, we will exclude sugar from our analysis. 

As a result, our final set of features for the model will include `total fat (PDV)`, `protein (PDV)`, and `carbohydrates (PDV)`.

***Creating Model***

Now that we've chosen the features for our model we can now create it. 

```py
features = recipes[['total fat (PDV)','protein (PDV)','carbohydrates (PDV)']]
predictor = recipes['calories(#)']
X_train, X_test, y_train, y_test = train_test_split(features, predictor, 
                                                    test_size = .2, random_state=1)
base_model = Pipeline([
                ('lin-reg', LinearRegression())])
base_model.fit(X_train, y_train)
```

After testing our model on the training data, we obtained a residual mean squared error (RMSE) of **693.43 calories squared**. We then evaluated the model on the testing data, where we obtained an RMSE of **637.96 calories squared**. This suggests that our model performed better on the testing data, which is encouraging, as it indicates that the model generalizes well to new, unseen data, and does not overfit to the training data.

To better illustrate the performance of our model we can visualize it by creating a residual plot with the **true calories** on the x-axis and our **predicted calories** on the y-axis. 

```py
ax = sns.regplot(x=y_test, y=predictions,
       scatter_kws = {'color': '#75bbfd', 'alpha': 0.35}, line_kws = {'color': '#0339f8', 'alpha': 1.0})

#increase font size of all elements
sns.set(font_scale=1.4)

ax.set(xlabel='True Calories',
       ylabel='Predicted Calories',
       title='Baseline Model')
```
           
<iframe src="data_viz/baseline.jpg" width=800 height=600 frameBorder=0></iframe>

As you can see, many of our predictions are relatively close to the true calorie values. However, there is still room for improvement, and we aim to do so by incorporating other features from our dataset, such as the categorical variables.

---

## Final Model

***Choosing Categorical Features*** 

For our final model we will be assessing the categorical features of our dataset and see which one we can add in order to improve our model.

```py
categorical = recipes[['name','tags','description','ingredients']]
categorical.head()

```

|    | name                                 | tags                                                                                                                                                                                                                                                                                               | description                                                                                                                                                                                                                                                                                                                                                                       | ingredients                                                                                                                                                                                                                             |
|---:|:-------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | 1 brownies in the world    best ever | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings']                                                                        | these are the most; chocolatey, moist, rich, dense, fudgy, delicious brownies that you'll ever make.....sereiously! there's no doubt that these will be your fav brownies ever for you can add things to them or make them plain.....either way they're pure heaven!                                                                                                              | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour']                                                          |
|  1 | 1 in canada chocolate chip cookies   | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                                                                                                      | this is the recipe that we use at my school cafeteria for chocolate chip cookies. they must be the best chocolate chip cookies i have ever had! if you don't have margarine or don't like it, then just use butter (softened) instead.                                                                                                                                            | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips']                                                                             |
|  2 | 412 broccoli casserole               | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                                                                                               | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']                                                                   |
|  3 | millionaire pound cake               | ['time-to-make', 'course', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'american', 'southern-united-states', 'dinner-party', 'holiday-event', 'cakes', 'dietary', 'christmas', 'thanksgiving', 'low-sodium', 'low-in-something', 'taste-mood', 'sweet', '4-hours-or-less'] | why a millionaire pound cake?  because it's super rich!  this scrumptious cake is the pride of an elderly belle from jackson, mississippi.  the recipe comes from "the glory of southern cooking" by james villas.                                                                                                                                                                | ['butter', 'sugar', 'eggs', 'all-purpose flour', 'whole milk', 'pure vanilla extract', 'almond extract']                                                                                                                                |
|  4 | 2000 meatloaf                        | ['time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'potatoes', 'vegetables', '4-hours-or-less', 'meatloaf', 'simply-potatoes2']                                                                                                                                             | ready, set, cook! special edition contest entry: a mediterranean flavor inspired meatloaf dish. featuring: simply potatoes - shredded hash browns, egg, bacon, spinach, red bell pepper, and goat cheese.                                                                                                                                                                         | ['meatloaf mixture', 'unsmoked bacon', 'goat cheese', 'unsalted butter', 'eggs', 'baby spinach', 'yellow onion', 'red bell pepper', 'simply potatoes shredded hash browns', 'fresh garlic', 'kosher salt', 'white pepper', 'olive oil'] |

Upon analyzing the categorical features, we have identified two variables, namely `tags` and `ingredients`, that have the potential to enhance our model's performance. The tags associated with each recipe could provide valuable insights into its calorie content, while the ingredients used in a recipe are known to significantly impact its overall calorie count. 

***Implementation*** 

Upon closer inspection, it becomes apparent that these features are presented in the form of a list of strings. Therefore, to integrate them into our model, we must first convert them into a more suitable format for our model. To achieve this, we will utilize the CountVectorizer tool from SKLearn. This will enable us to transform the tags and ingredients variables into tokens, which we will then fit in our model.

```py
vectorizer = CountVectorizer(analyzer=lambda x: x)
final_transformer = ColumnTransformer([
                        ('tag-vec', vectorizer, 'tags'),
                        ('ingredient-vec', vectorizer, 'ingredients')],
                        remainder = 'passthrough') # leave all other columns as it 

lin = Pipeline([
                ('vectorizer',final_transformer),
                ('lin-reg', LinearRegression())])

lin.fit(X_train, y_train)
```

After fitting our linear regression model with the 2 new features we recieved a a RMSE of **209.33 calories squared** on the training data and a RMSE of ***336.62 calories squared***. Reducing the RMSE by double compared to our previous model. 

<iframe src="data_viz/final_model.png" width=800 height=600 frameBorder=0></iframe>

As you can see the data points are much tighter and are more accurate predictions of the true calories of recipes. However, let's see if we can improve our model even more by using another regression technique, Ridge Regression.

***Ridge Regression***

Ridge regression which is a model tuning method that helps data reduce error by shrinking coefficients and reducing the effects of multicolinearity.
To determine the best hyperparameters for ridge regression model we will use a function from SKLearn called `GridSearchCV`, a cross-validation technique which tests all possible combinations of the hyperparameters we listed below. We have 4 possible values for `alpha` and 4 possible values of `max_iter` which will test k=5 times. Which means we will be testing 80 different possible combinations and keeping the hyperparameteres that had the best average validation score. 

```py
# Possible hyperparameter combination
hyperparameters = {
    'alpha': [1,2,6,9], 
    'max_iter': [2,5,6,8],
}

# Determing the best HyperParameters out of the ones we chose
r_lin_searcher = GridSearchCV(r_lin, param_grid=hyperparameters)
r_lin_searcher.fit(X_train, y_train) # Fitting model
r_lin_searcher.best_params_
}
```

After testing various parameters, we found that an `alpha` of 9 and a `max_iter` of 5 produced the best results. However, when calculating the RMSE on the training and testing data, we discovered that the Ridge Regression model did not perform as well as our linear regression model. With a **training RMSE of 693.19** and **testing RMSE of 637.73** our new model performed significantly worse than our previous one. Although it was more complex, the linear regression model with added features still outperformed it. Nonetheless, our final model was a significant improvement over the baseline as the RMSE was almost halved. 

## Fairness Analysis
For our Fairness Analysis, we decided to measure the quality of predictions for more ingredients vs. less ingredients. We chose these two groups because a recipe with more ingredients might be a better feature for predicting calories because adding another ingredient to a recipe will inherently add more calories to the recipe, since an ingredient cannot have negative calories itself. However, a recipe with less ingredients might not give us as clear of a picture for prediction. A recipe could have 3 ingredients and still have a much higher caloric value than a recipe of 100 ingredients if the 3-ingredient recipe contains an *extremely* high-calorie ingredient. Thus, we wanted to assess if our model was able to account for these edge cases.

More specifically, wewe decided to compare the following groups of recipes:

>Number of Ingredients > 9

>Number of Ingredients <= 9

We chose 9 as our threshold because 9 ingredients is the mean value of ingredients in our dataset. Thus, ingredients greater than 9 will be classified as more ingredients and any recipe with less than 9 ingredients will classified as having less ingredients. 

***Null and Alternative Hypotheses***

 - **Null Hypothesis**:  Our model’s accuracy was the same for both recipes with <= 9 ingredients and >9 ingredients and any differences are due to random chance.
 - **Alternative Hypothesis**: The classifier’s accuracy is higher for more ingredients (>9)



For computation purposes, we added a column to our dataframe, 'more ingredients' that identifies whether the recipe has 'more' or 'less' ingredients. The first five rows of the dataframe with the inserted column appears as so:

|    | name                                 |     id | more ingredients   |
|---:|:-------------------------------------|-------:|:-------------------|
|  0 | 1 brownies in the world    best ever | 333281 | less               |
|  1 | 1 in canada chocolate chip cookies   | 453467 | more               |
|  2 | 412 broccoli casserole               | 306168 | less               |
|  3 | millionaire pound cake               | 286009 | less               |
|  4 | 2000 meatloaf                        | 475785 | more               |

### Evaluation Metric

Our evaluation metric for the Fairness Analysis is the Difference in RMSE. Similarly to evaluating our prediction model, using difference of RMSE will give us an idea of the difference between the quality of our prediction model between the group of recipes with less ingredients vs. the recipes with more ingredients. To calculate the observed difference in RMSE, we use the following code:

```py
obs = (
    results
    .groupby('more ingredients')
    .apply(lambda x: rmse(x['prediction'], x['calories_test']))
    .rename('rmse')
    .to_frame()
).diff().iloc[-1].iloc[0]
```
Giving us a p-value of: **-6.149033363119015** 

***Performing the test***
We are using a **5% significance level** for our test, since it is the conventional significance level.

```py
diff_in_acc = []
for _ in range(100):
    s = (
        results[['more ingredients', 'prediction', 'calories_test']].reset_index()
        .assign(more=results['more ingredients'].sample(frac=1.0, replace=False).reset_index(drop=True))
        .groupby('more')
        .apply(lambda x: rmse(x['prediction'], x['calories_test']))
        .diff()
        .iloc[-1]
    )
    
    diff_in_acc.append(s)
```

Calculating the p-value:
```py
p_val = (np.array(diff_in_acc)<= obs).mean()
0.0
```
Our p-value was 0.0, for which we reject the null hypothesis at the 5% significance level. 
Visualizing the distribution of test statistics:

<iframe src="data_viz/diff-rmse.html" width=800 height=600 frameBorder=0></iframe>

### Conclusion

We did find that there was a statistically significant difference in RMSE for the two groups. That is, our prediction tends to be more accurate for recipes with more ingredients. However, we cannot definitively conclude that our model is better for recipes with more ingredients.