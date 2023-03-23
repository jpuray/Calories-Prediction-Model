# Counting Calories with MATH!
Project for DSC80 

## Framing The Problem


## Baseline Model

***Getting Started***

To begin, we will establish a baseline model to compare with our other models later on. Our baseline model will use linear regression, trained on the nutrition features of our data. Since all the nutrition features are quantitative, implementing them into our model will be straightforward, without requiring any transformations. However, before starting the baseline model, we must determine which nutrition features are most highly correlated with calories by exploring the data.

***Exploring Features***

To determine which features are most correlated with calories we'll plot a correlation matrix. 

    sns.set(rc={"figure.figsize":(12, 10)})
    sns.heatmap(nutrition.corr(), vmin=-1, vmax=1, annot=True)
    
<iframe src="data_viz/corr_matrix.jpg" width=800 height=600 frameBorder=0></iframe>

Based on our analysis, we have identified the most highly correlated features with calories in our dataset, which are `total fat (PDV)`, `sugar (PDV)`, `protein (PDV)`, `saturated fat (PDV)`, and `carbohydrates`. However, we must be cautious of highly correlated features, as they may lead to redundancy and not significantly reduce model error. For instance, since total fat and saturated fat have a high r-correlation of 0.86, we will exclude saturated fat from our model as it has a smaller correlation with calories. Similarly, since sugar and carbohydrates are highly correlated with an r-correlation of 0.88, we will exclude sugar from our analysis. 

As a result, our final set of features for the model will include `total fat (PDV)`, `protein (PDV)`, and `carbohydrates (PDV)`.

***Creating Model***

Now that we've chosen the features for our model we can now create it. 

    features = recipes[['total fat (PDV)','protein (PDV)','carbohydrates (PDV)']]
    predictor = recipes['calories(#)']
    X_train, X_test, y_train, y_test = train_test_split(features, predictor, 
                                                        test_size = .2, random_state=1)
    base_model = Pipeline([
                    ('lin-reg', LinearRegression())])
    base_model.fit(X_train, y_train)

After testing our model on the training data, we obtained a residual mean squared error (RMSE) of 693.43 calories squared. We then evaluated the model on the testing data, where we obtained an RMSE of 637.96 calories squared. This suggests that our model performed better on the testing data, which is encouraging, as it indicates that the model generalizes well to new, unseen data, and does not overfit to the training data.

To better illustrate the performance of our model we can visualize it by creating a residual plot with the **true calories** on the x-axis and our **predicted calories** on the y-axis. 

    ax = sns.regplot(x=y_test, y=predictions,
           scatter_kws = {'color': '#75bbfd', 'alpha': 0.35}, line_kws = {'color': '#0339f8', 'alpha': 1.0})

    #increase font size of all elements
    sns.set(font_scale=1.4)

    ax.set(xlabel='True Calories',
           ylabel='Predicted Calories',
           title='Baseline Model')
           
<iframe src="data_viz/baseline_lin.jpg" width=800 height=600 frameBorder=0></iframe>

As you can see, many of our predictions are relatively close to the true calorie values. However, there is still room for improvement, and we aim to do so by incorporating other features from our dataset, such as the categorical variables.

## Final Model
