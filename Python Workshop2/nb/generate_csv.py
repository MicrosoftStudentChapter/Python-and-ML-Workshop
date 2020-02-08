# Importing the libs
import pandas as pd
from random import randint

class GenerateCsv:
    def generateRandomSalary(self, digitCount):
         rangeStart = 10**(digitCount-1)
         rangEnd = (10**digitCount)-1
         return randint(rangeStart, rangEnd)
     
    def createCsv(self):
        userid = []
        ageCounter = []
        salaryCounter = []
        boughtCounter = []
        highestSalary = 0;
        for x in range(400):
            bought = 0
            randomSalary = 0    
            age = randint(18, 60)
            randomSalary = self.generateRandomSalary(randint(4, 5))            
            if highestSalary < randomSalary:
                highestSalary = randomSalary                
            if age > 40:
                bought = randint(0,1)             
            if age > 40 and highestSalary/2 < randomSalary:
                bought = 1
            userid.append(x + 1)
            ageCounter.append(age)
            salaryCounter.append(randomSalary)
            boughtCounter.append(bought)
            
        finalUserData = {
            'id': userid, 
            'age': ageCounter,
            'salary': salaryCounter,
            'bought': boughtCounter
        }
        df = pd.DataFrame(finalUserData, columns = ['id', 'age', 'salary', 'bought'])
        df.to_csv('supermall.csv')
    
GenerateCsv().createCsv()
