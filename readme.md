# Predicting the Value of Houses (working title)

## Project Description
    
The main goal here is to, using the Zillow dataset, discover some drivers behind the prices of single family homes, and using that, build multiple Regression models and refine them to create a model that accurately predicts single family home prices. The goals will be to make useful business recommendations based on my findings.

    
## Project Goals
    
- Discover drivers behind housing prices.

- Use those drivers to build a model that accurately predicts housing prices

- Deliver a report that a non-technical person can read and understand the steps taken, why they wre taken, and the outcomes.

    
## Initial Hypotheses and Questions
    
My initial hypothesis is that the biggest drivers of price will the house's location, as well as the square footage of the home. 

Some questions I have are:

- Do bedroom and bathroom counts have a relationship?

- Does year built have an effect on the number of bedrooms?

- Does year build have a relationship with the square feet of the house?
 
- Does fips have a relationship with the square feet?

- Is there a relationship between bathrooms and fips?
    
## Data Dictionary

|Feature| Definition|
|:--------|:-----------|
|parcel_id|  Unique identifier for the parcel (lots)|
|bedrooms| The number of bedrooms in the home|
|bathrooms| The number of bathrooms in the home|
|sqft| The calculated finished square feet of the living area in the home| 
|tax_amount| The total tax assessed value of the parcel (lot)|
|built| The year the home was built|
|fips|  Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details|

## My Plan
    
- Acquire and prepare my data using functions built stored in my acquire and prepare .py files.

- Verify my functions performed properly, and that my data is in the form I need it to be. 

- Explore my data, with the help of visual aids and statistical tests to discover pricing drivers.

- Answer my initial questions.

- Develop a model to determine the price of a home by:

    - Using drivers identified by my initial questions 
    
    - Evaluate models based upon my train and validate data subsets
    
    - Select my top three performing models for evaluation
    
    - Of the three, determine my best model to test on my test data subset
    
- Draw conclusions based on my results.

## Steps to Reproduce
    
- Clone this repository

- Acquire this dataset, either through Codeup or another source

- Store the data within the clone repo so it's accessible

- Run the final report
    
## Takeaways and Conclusions
    
- As time progresses, houses are being built with more bedrooms.

- In addition to the above, as time progresses, larger houses are also being built.

- I found that there is not a significant relationship between the home's fips code and size.

- There is also a relationship between fips code and number of bathrooms, which I found interesting.

- I did discover that the zip code and the year the house was assessed were not helpful in modeling.

- My models actually performed better without using the rfe or SelectKBest methods to narrow down features, but I believe that's because I started with relatively few features to begin with. I think those will be more helpful as more and more features are added to the dataset.

- My top performer was a Lasso-Lars model with an alpha value of 0.1.

- My final model did beat my baseline RMSE, by almost 9000. Not vasst amount, considering we are talking about near 200,000 here, but it **is** still an improvement over the baseline.
    
## Recommendations
    
- My biggest recommendation going forward would be to survery recent home-buyers to determine their biggest deciding factors in choosing their new homes, and gathering any valuable insights from that to create new metrics in data acquisition going forward.
