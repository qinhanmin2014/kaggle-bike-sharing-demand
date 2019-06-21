# kaggle-bike-sharing-demand

https://www.kaggle.com/c/bike-sharing-demand/overview

- data/ - the datasets
- submission/ - output of the solution
- [Data Visualization.ipynb](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-bike-sharing-demand/blob/master/Data%20Visualization.ipynb) - data visualization
- GBDT_RF.py - solution based on Gradient Boost and Random Forest, rank 24/3251 (0.36535) within 60 lines of code
- [GBDT_RF.ipynb](https://nbviewer.jupyter.org/github/qinhanmin2014/kaggle-bike-sharing-demand/blob/master/GBDT_RF.ipynb) - detailed solution with some explanations 

### Data Fields
- datetime - hourly date + timestamp  
- season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
- holiday - whether the day is considered a holiday
- workingday - whether the day is neither a weekend nor holiday
- weather -
  * 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
  * 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
  * 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
  * 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
- temp - temperature in Celsius
- atemp - "feels like" temperature in Celsius
- humidity - relative humidity
- windspeed - wind speed
- casual - number of non-registered user rentals initiated
- registered - number of registered user rentals initiated
- count - number of total rentals

### Acknowledgment

- Inspired by https://github.com/logicalguess/kaggle-bike-sharing-demand
