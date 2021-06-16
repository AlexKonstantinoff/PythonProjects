import csv

res = []
with open("Test_data.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        res.append(row)
    csvfile.close()

print(res)

file = open('Output_test.csv', 'w', newline='')
for row in res:
    csv.writer(file).writerow(row)
file.close()