# yes is 1, undecided is 0, no is -1
# data (example) shows the votes of first 7 members of senate alphabetically for bills 393-397
data = open("dataset.txt","r")
datas = data.read().split(";;")
for i in range(len(datas)):
    dataset.append(datas[i].split("/"))

print(dataset)
