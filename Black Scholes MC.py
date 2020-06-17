import numpy as np
import math
import time
import os
import pandas_datareader.data as web 
import datetime as dt 

class OptionPricing:

    def __init__(self, currentPrice, strikePrice, timeUntilExpiration, riskFreeRate, stockVol, iterations):
        self.currentPrice = currentPrice
        self.strikePrice = strikePrice
        self.timeUntilExpiration = timeUntilExpiration
        self.riskFreeRate = riskFreeRate
        self.stockVol = stockVol
        self.iterations = iterations

    def callOptionSimulation(self):

        #Create a table with 0s and the payoff. This is required since the payoff function is in the format of max(0,currentPrice-strikePrice)
        optionData = np.zeros([self.iterations, 2])

        #Create a 1 dimensional array as long as the number of iterations
        rand = np.random.normal(0, 1, [1, self.iterations])

        #Equation for the stock price 
        stockPrice = self.currentPrice*np.exp(self.timeUntilExpiration*(self.riskFreeRate - 0.5*self.stockVol**2) + self.stockVol*np.sqrt(self.timeUntilExpiration)*rand)

        #Calculate maximum payout for currentPrice-strikePrice
        optionData[:,1] = stockPrice - self.strikePrice

        #Calculate the average for the Monte-Carlo Simulation, where np.amax() will return the payoff function for the formula
        average = np.sum(np.amax(optionData, axis=1))/float(self.iterations)

        #Since the thsi is projecting the money sometime in the future, it has to be discounted back with the exp(-riskFreeRate*timeUntilExpiration) formula
        return np.exp(-1.0*self.riskFreeRate*self.timeUntilExpiration) * average

    def putOptionSimulation(self):

        #Create a table with 0s and the payoff. This is required since the payoff function is in the format of max(0,currentPrice-strikePrice)
        optionData = np.zeros([self.iterations, 2])

        #Create a 1 dimensional array as long as the number of iterations
        rand = np.random.normal(0, 1, [1, self.iterations])

        #Equation for the stock price 
        stockPrice = self.currentPrice*np.exp(self.timeUntilExpiration*(self.riskFreeRate - 0.5*self.stockVol**2) + self.stockVol*np.sqrt(self.timeUntilExpiration)*rand)

        #Calculate maximum payout for currentPrice-strikePrice
        optionData[:,1] = self.strikePrice - stockPrice

        #Calculate the average for the Monte-Carlo Simulation, where np.amax() will return the payoff function for the formula
        average = np.sum(np.amax(optionData, axis=1))/float(self.iterations)

        #Since the thsi is projecting the money sometime in the future, it has to be discounted back with the exp(-riskFreeRate*timeUntilExpiration) formula
        return np.exp(-1.0*self.riskFreeRate*self.timeUntilExpiration) * average

if __name__ == "__main__":

    currentPrice = 304.34
    strikePrice = 280
    daysUntilExpiration = 45
    timeUntilExpiration = daysUntilExpiration/365
    riskFreeRate = 0.30
    stockVol = 0.342
    iterations = 10000000

    model = OptionPricing(currentPrice, strikePrice, timeUntilExpiration, riskFreeRate, stockVol, iterations)
    print("Call option price with Monte Carlo simulation in 45 days: ", model.callOptionSimulation())
    print("Put option price with Monte Carlo simulation in 45 days: ", model.putOptionSimulation())
    