import requests

# url = 'http://localhost:9696/predict'

url = 'https://autumn-waterfall-3694.fly.dev/predict'

customer = {
 "gender": "male",
 "seniorcitizen": 0,
 "partner": "no",
 "dependents": "yes",
 "phoneservice": "no",
 "multiplelines": "no_phone_service",
 "internetservice": "dsl",
 "onlinesecurity": "no",
 "onlinebackup": "yes",
 "deviceprotection": "no",
 "techsupport": "no",
 "streamingtv": "no",
 "streamingmovies": "no",
 "contract": "month-to-month",
 "paperlessbilling": "yes",
 "paymentmethod": "electronic_check",
 "tenure": 6,
 "monthlycharges": 29.85,
 "totalcharges": 129.85, 
 # "whatever": 31337
 }

response = requests.post(url, json=customer)
churn = response.json()

print('response:', churn)

print('prob of churning = ', churn)

if churn['churn_probability'] >= 0.5:
    print('send email with promo')
else: 
    print('dont do anything')