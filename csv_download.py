##pip install requests

import requests

Url = "https://raw.githubusercontent.com/PyThaiNLP/thai-sentiment-analysis-dataset/master/review_shopping.csv"

## request from Url variable
req = requests.get(Url)

# capture content in Url file (csv)
url_content = req.content

#open file
csv_file = open('sentiment.csv','wb')

#write csv file
csv_file.write(url_content)