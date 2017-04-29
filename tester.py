file1 = open("reference_output.txt", 'r')
file2 = open("sampler_output.txt", 'r')

lines1 = [line.replace("\n", "").split(",") for line in file1.readlines()]

data1 = []

for line in lines1:
    data1 += line

lines2 = [line.replace("\n", "").split(",") for line in file2.readlines()]

data2 = []

for line in lines2:
    data2 += line

data1 = [float(item.strip()) for item in data1]
data2 = [float(item.strip()) for item in data2]

sum = 0
for i in range(len(data1)):
    sum += abs(data1[i]-data2[i])

print (sum/len(data1))
