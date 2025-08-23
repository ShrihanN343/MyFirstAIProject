
# Evaluating Prediction Models and their Business Impact

### Motivation 

Since my freshmen year, I’ve been passionate about the worlds of business and computer science - a passion that stems from my love of problem-solving and deriving my own solutions. This project represents one example of how I leverage the resources available to me to enhance my programming skills while learning more about business intelligence through stock forecasting. Forecasting stock prices using sophisticated machine learning algorithms is standard practice in quantitative investing. Based on the literature (find and add citation!!!), performing stock forecasting using machine learning models significantly outperforms traditional statistical models. We observe that the LSTM (Long Short Term Memory) model outperformed the baseline and ARIMA models when tested across five distinct tickers. 
 


### Background

Deep learning models are able to outperform traditional methods of statistical analysis in capturing non-linear patterns as well as volatility. However, there continue to exist challenges with AI in financial markets such as data privacy, ethical concerns, and regulatory compliance.^1 For example, due to AI's black box nature, regulators may find it difficult to defend decisions, and this raises concerns about transparency.^5 There is a growing need for the adaption of AI skills in financial professions, as well as upgrades to infrastructure in order to integrate new technologies to take full advantage of opportunities.^3

### Dependencies

Our experiment utilized the following Python packages: 

- yfinance 
- Numpy
- Pandas
- Matplotlib
- Datetime
- OS
- Sklearn
- Pytorch


## Data Ingestion and Preprocessing

We source all of our data from the yfinance Python package, an open source Python package with the latest market data. The program checks for empty datasets and invalid tickers. We sample stock data of five different stocks between the years 2010 and 2025. The data was ingested as a Pandas dataframe, then export as a csv file with labelled columns. The preprocessing phase ensures that the model will be able to analyze and work with clean data. We make our repository open to the public, allowing anyone to replicate our results. 

In order to enable effective data analysis, machine learning algorithms utilize data normalization, which is the process of compressing data into values between 0 and 1. This compression causes all data points to be relative to one another, a crucial tenet for data analysis. 


## Feature Engineering

We create a time series feature from a dataframe of price data to generalize both short and long term trends. We engineer standard metrics used by financial professionals for technical analysis.(add citation here) These metrics include three Simple Moving Averages (SMA) which provide an unweighted mean over five, ten, and twenty days respectively. The next metric we leverage is the Exponential Moving Average (EMA), which gives more weight to recent prices making it more responsive to new information, also over five, ten, and twenty days respectively. The last metric we invoke is Volatility, which measures the standard deviation of daily log returns, helping the model understand periods of high and low price fluctuation within the past twenty days.


## Splitting Data

In machine learning, training a model requires us to work with finite data. In doing so, we must decide how best to split the data. Before training a model, the processed dataset must be split into training and testing subsets. We select a 70/30 split of training and testing data respectively. 


## Model Architecture

The LSTM (Long Short Term Model) is a type of stock predicting algorithm that is able to make feature associations across long sequences of time series data.  LSTM is a type of RNN (recurrent neural network) structure. RNNs are formed from feedforward networks (anticipation of sequential data is a specific strength). RNN structures’ “hidden state” enables them to remember specific info about a sequence. RNNs can also simultaneously update their internal memory as they process new data, allowing them to adapt patterns within a sequence.^4 This model was preferred in stock prediction for a period of time, but has since been replaced with more accurate and complex models. 


## Model Training


The model analyzes and learns patterns in the features from the training subset and : training, which the model analyzes and learns from, and testing, which the model's performance is evaluated against.

Strengths: statistical foundation (auto regression/moving avg), interpretability, effectiveness on linear trends

- The baseline model has a variety of strenghts, such as its effectiveness on linear trends, a string statistical foundation, and its interpretability. 

Weaknesses: Assumption of linearity (tries to draw straight lines through complex data), stationarity requirement (its statistical properties like mean and variance are constant over time), struggles with complexity

- However, the baseline model also has several drawbacks. The model struggles with complexity and assumes linearity through non-linear data. Another drawback to be noted is its stationarity requirements. 

    - IS STRENGHTS AND WEAKNESSES FOR BASELINE OR CLASSICAL??

- LSTM Network: What makes the LSTM model "standard"? Think about its core capability: memory. How does an LSTM's ability to "remember" long-term patterns in the data make it theoretically better suited for complex systems like stock markets compared to ARIMA?
    - LSTM is a type of a recurrent neural network and it has an internal structure called the memory cell. This cell structure allows the network to learn over time what information is important to keep versus what important is irrelevant to forget. 

### Results

3.1. Quantitative Comparison
Action: Run the evaluation scripts for all models to populate the RMSE table.
Brainstorming Questions:
- Before you write, just look at the completed table. What is the most obvious story the numbers are telling? Is there a clear winner? Are there any results that surprise you? 
    - The numbers point to the LSTM being the most accurate model of the three due to it consistently having the lowest RMSE across all tickers. I was surprised to see that despite the differences in the baseline and ARIMA models’ performances, they performed very close when predicting AMZN. In analyzing other tickers, however, the difference was sharp. For example, the ARIMA model had an RMSE of approximately 63 while the baseline model’s RMSE was around 24 for the ticker MSFT. 
- How can you introduce this table to the reader? Frame it as the primary evidence from your comparative study.
    - This table provides us with statistical data that compares the performances of all three models across five different ticker symbols of companies that are in a variety of sectors. Each model scores an RMSE value for each ticker. RMSE (root-mean squared error) refers to the average difference in predicted price versus observed price that the stock predicting algorithm makes when forecasting a stock price. The lower the RMSE, the better the model, and the more accurate the forecast.     

| Stock Ticker | Baseline RMSE | ARIMA RMSE | LSTM RMSE |
|--------------|---------------|------------|-----------|
| AAPL         | 13.6648       | 29.4999    | 4.8387    |
| AMZN         | 14.1469       | 14.5540    | 3.7320    |
| MSFT         | 24.2329       | 62.6977    | 12.3385   |
| UNH          | 46.1457       | 25.1849    | 15.6239   |
| XOM          | 6.5515        | 14.4534    | 2.4972    |

3.2. Interpretation of Results
Brainstorming Questions:
- Tell the Main Story: What's the headline? Start with the main conclusion from the table. Which model was generally the most accurate?
    - The LSTM model was the most accurate model across all stocks tested. The baseline model was the second most accurate in predicting all stock values except for UNH. The ARIMA model had the worst overall performance relative to the LSTM and baseline models. 
- Make it Tangible: Pick one stock (e.g., Apple). What does its RMSE number (4.8387) actually mean in plain English? 
    - RMSE (root-mean squared error) refers to the average difference in predicted price versus observed price that the stock predicting algorithm makes when forecasting a stock price. The lower the RMSE, the better the model, and the more accurate the forecast. 
- Go Beyond the Numbers: Now look at the graphs in your images/ folder. What do the visuals show you that the RMSE numbers alone don't? Do the models behave differently during periods of stability versus periods of high volatility? Does one model capture turning points better than another? This is where you can show deep analytical insight.
    - The LSTM model most accurately predicts shifting market trends and turning points in prices out of all three models. The classical model assumes a “price wall” when predicting prices of a stock has reached a relative maximum in the analyzed time period. 

### Business Implications

Brainstorming Questions:
- Think like a consultant. If a company could use your tool, how would it change their business? Move beyond "predicting prices." Could it be used for risk management? Could it help an analyst check their own biases? Could it automatically flag stocks that are behaving unusually?
    - Beyond predicting stock prices, machine learning algorithms can also be used to track a business’ inventory and supply chain. Expanding on the Covid-19 example from earlier in the document, an ML application could forecast rising demands for certain goods in times of international emergencies and provide businesses with crucial time to react and be prepared to meet these demands. An ML algorithm can be trained to optimize for a function of social good like reducing scarcity in essential supplies such as toilet paper and cleaning supplies, which were goods affected by scarcity during Covid-19. 
Future Work
Brainstorming Questions:
- No project is perfect. What were the limitations you observed? (e.g., the lag during volatility you saw in the charts). How could you address these limitations in a future version?
    - Future revisions on the LSTM model would include adding hyperparameters to the model to enable more accurate predictions.
- What's the next logical step to make this tool even more powerful? Think about adding more data (like news sentiment), improving the model architecture, or changing the prediction goal (e.g., predicting volatility instead of price).

### Conclusion

Guidance: A strong conclusion briefly mirrors the introduction.
Restate the core problem you set out to solve.
Briefly summarize the approach you took (your comparative analysis).
State your main, conclusive finding. End on a confident note about the value and potential of your work.

### Acknowledgements

Mr. Zach + citations (at least 10)

### Citations: 

1. Saberironaghi, Mohammadreza, Jing Ren, and Alireza Saberironaghi. “Stock Market Prediction Using Machine Learning and Deep Learning Techniques: A Review.” AppliedMath, vol. 5, no. 3, 2025, p. 76. MDPI

2. El Hajj, M., & Hammoud, J. “Unveiling the Influence of Artificial Intelligence and Machine Learning on Financial Markets.” Journal of Risk and Financial Management, vol. 16, no. 10, MDPI, 2023.

3. International Organization of Securities Commissions (IOSCO). Artificial Intelligence in Capital Markets: Use Cases, Risks, and Challenges. IOSCO Consultation Report CR/01/2025, Mar. 2025, IOSCO.

4. Analytics Vidhya citation here***

5. Full stack black box




USE THIS SOMEWHERE

Machine learning algorithms can also make predictions on variables that help increase business profits or reduce business expenses. The Covid-19 pandemic highlighted the fragility of domestic supply chains, which could have been mitigated through predictive models of inventory management.

