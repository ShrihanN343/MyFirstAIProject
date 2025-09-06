
# Evaluating Prediction Models and their Business Impact

### Motivation 

Since my freshman year, I’ve been passionate about the worlds of business and computer science - a passion that stems from my love of problem-solving and deriving my own solutions. This project represents one example of how I leverage the resources available to me to enhance my programming skills while learning more about business intelligence through stock forecasting. I set out to better understand how to leverage machine learning in predictive analytics, starting with exploring the differences between machine learning models and traditional statistical methods of predictive analytics as it pertains to stock forecasting. Forecasting stock prices using sophisticated machine learning algorithms is standard practice in quantitative investing. Based on the literature, performing stock forecasting using machine learning models significantly outperforms traditional statistical models.[6] We observe that the LSTM (Long Short Term Memory) model outperformed the baseline and ARIMA models when tested across five distinct tickers. 
 

### Background

Deep learning models are able to outperform traditional methods of statistical analysis in capturing non-linear feature associations as well as volatility. However, there continue to exist challenges with AI in financial markets such as data privacy, ethical concerns, and regulatory compliance.[1] For example, due to AI's black box nature, regulators may find it difficult to defend decisions, and this raises concerns about transparency.[5] There is a growing need for the adoption of AI skills in financial professions, as well as upgrades to infrastructure in order to integrate new technologies to take full advantage of opportunities.[3]


### Definitions

- **Feature Associations** refers to patterns learned by the model from the dataset.

- **Root-Mean Squared Error** (RMSE) refers to the average difference in predicted price versus observed price stock predicting algorithm makes when forecasting a stock price. Lower RMSE scores translate to more accurate predictions. 

- **Deep Learning** refers to a subfield of machine learning that uses multi-layered neural networks to extract complex patterns and features from big data. 

- **AI Black Box** refers to the lack of interpretability in complex neural networks.

- **Recurrent Neural Networks** refers to a recurrent form of feedforward neural networks.

- **Overfitting** refers to accuracy degradation due to a model being overly trained on the training dataset. 

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

We source all of our data from the yfinance Python package, an open source Python package with the latest market data. The program checks for empty datasets and invalid tickers. We sample ticker data of five different stocks between the years 2010 and 2025. The data was ingested as a Pandas dataframe, then exported as a csv file with labelled columns. The preprocessing phase ensures that the model will be able to analyze and work with clean data. We make our repository open to the public, allowing anyone to replicate our results. 

In order to enable effective data analysis, machine learning algorithms utilize data normalization, which is the process of compressing data into values between 0 and 1. This compression causes all data points to be relative to one another, a crucial tenet for data analysis. 


## Feature Engineering

We create time series features from a dataframe of price data to generalize both short and long term trends. We engineer standard metrics used by financial professionals for technical analysis.[7] These metrics include three Simple Moving Averages (SMA) which provide an unweighted mean over five, ten, and twenty days respectively. The next metric we leverage is the Exponential Moving Average (EMA), which gives more weight to recent prices making it more responsive to new information, also over five, ten, and twenty days respectively. The last metric we invoke is Volatility, which measures the standard deviation of daily log returns, helping the model understand periods of high and low price fluctuation within the past twenty days.


## Splitting Data

In machine learning, training a model requires us to work with finite data. In doing so, we must decide how best to split the data. Before training a model, the processed dataset must be split into training and testing subsets. We selected a 70/30 split of training and testing data respectively. This helps prevent the model from overfitting to the training data set. 


## Model Architecture

The Long-Short Term Memory (LSTM) model is a type of machine learning model that is able to make feature associations across long sequences of time series data. The LSTM model is a type of a recurrent neural network (RNN) which has an internal memory cell structure. This cell contains a "hidden state", which captures feature associations within a sequence. The cell is able to prioritize what feature associations are most closely correlated with changes in the label (price) data. LSTMs update their internal memory as they process new data, allowing them to adapt feature associations within a sequence.[4] This model was preferred in stock prediction for a period of time, but has since been replaced with more accurate and complex models. 


## Model Training

The model analyzes and learns feature associations derived from standard metrics used by financial professionals. LSTM models specialize in capturing linear trends, which have been approximated based on the non-linear features from our training subset. The following parameters were used to train the model: 

```python
# training lstm model
    lstm_trainer = LSTMTrainer(
        input_size = 1, 
        hidden_size = 50, 
        num_layers = 2,
        output_size = 1,
        epochs = 25,
        batch_size = 32, 
        learning_rate = 0.01,
        random_state = 42
    )
```


### Results

| Stock Ticker | Baseline RMSE | ARIMA RMSE | LSTM RMSE |
|--------------|---------------|------------|-----------|
| AAPL         | 13.6648       | 29.4999    | 7.6253    |
| AMZN         | 14.1469       | 14.5540    | 4.1601    |
| MSFT         | 24.2329       | 62.6977    | 16.8311   |
| UNH          | 46.1457       | 25.1849    | 25.2305   |
| XOM          | 6.5515        | 14.4534    | 5.3857    |
- **Fig. 1:** This table provides us with statistical data comparing the performances of all three models across five stock tickers in a variety of sectors. Each model scores an RMSE value for each ticker.

The LSTM model accurately predicts shifting market trends and turning points in prices when markets are stable. However, when volatility metrics rise beyond some threshold, we see a collapse of prediction accuracy relative to the models tested. This is evident by the marginally worse performance of the LSTM model relative to classical methods on the UNH stock ticker. The Baseline model uses a naive approach that assumes the price will remain the same in 30 days. 


### Business Implications

Beyond predicting stock prices, machine learning algorithms can also be used to track business inventory and supply chains. During future pandemics,  machine learning models could forecast rising demands for certain goods in times of international emergencies and provide businesses with crucial information to react and be prepared to meet these demands. A machine learning algorithm can be trained to optimize for a function of social good like reducing scarcity in essential supplies such as toilet paper and cleaning supplies, which were goods affected by scarcity during the Covid-19 pandemic. 


### Findings

The most unexpected finding of this research was the inconsistency in RMSE values for the UNH ticker across all three models tested. This anomaly in our findings is directly attributed to an extreme price drop in the testing dataset. Across the 4 other tickers, the LSTM model outperformed the Baseline and ARIMA models. In my research, I found that machine learning algorithms are able to capture a richness of feature associations attributable to changes in the label (price) data that traditional statistical methods are not. The LSTM model we tested demonstrated the lowest RMSE value across 4 of the 5 tickers analyzed. The LSTM model's success can be attributed to the memory cell's ability to capture the feature associations most closely correlated with changes in the price.


### Acknowledgements

**Zachariah Rodriguez-Mcdonough**: https://www.linkedin.com/in/zachariah-rodriguez-mcdonough-597b7b119  

### Citations: 

1. Saberironaghi, Mohammadreza, Jing Ren, and Alireza Saberironaghi. “Stock Market Prediction Using Machine Learning and Deep Learning Techniques: A Review.” AppliedMath, vol. 5, no. 3, 2025, p. 76. MDPI

2. El Hajj, M., & Hammoud, J. “Unveiling the Influence of Artificial Intelligence and Machine Learning on Financial Markets.” Journal of Risk and Financial Management, vol. 16, no. 10, MDPI, 2023.

3. International Organization of Securities Commissions (IOSCO). Artificial Intelligence in Capital Markets: Use Cases, Risks, and Challenges. IOSCO Consultation Report CR/01/2025, Mar. 2025, IOSCO.

4. Siddharth M. “Stock Price Prediction Using LSTM and Its Implementation.” Analytics Vidhya, 1 May 2025, analyticsvidhya.com/blog/2021/12/stock-price-prediction-using-lstm/.

5. FullStack Team. “How Machine Learning Is Transforming Predictive Analytics.” FullStack Blog, last updated 30 June 2025, fullstack.com/labs/resources/blog/how-machine-learning-is-revolutionizing-predictive-analytics?utm_source=chatgpt.com. Accessed 30 Aug. 2025.

6. FasterCapital. “AI vs Traditional Forecasting Models—Which Is Superior.” FasterCapital, [date not specified], https://fastercapital.com/articles/AI-vs-traditional-forecasting-models--Which-is-superior.html?utm_source=chatgpt.com#toc-1-4-raditional-vs-orecasting-ey-onsiderations.

7. Tradelink.pro. “Key Technical Analysis Indicators: SMA, EMA, and Bollinger Bands.” Tradelink.pro, n.d., tradelink.pro/blog/sma-ema-bollinger-bands-indicators/. 