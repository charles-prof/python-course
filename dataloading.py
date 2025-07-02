import csv

# Load the CSV file
with open('data/simple-weather.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        print(row)