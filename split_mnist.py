import pandas as pd
import random

df = pd.read_csv("mnist_train.csv")

# for col in df.columns:
# 	print(col)

# print(df["label"])

dfA = pd.DataFrame(columns = df.columns)
dfB = pd.DataFrame(columns = df.columns)
dfC = pd.DataFrame(columns = df.columns)

dA_zero = 0.2
dB_zero = 0.5
dC_zero = 1-dA_zero-dB_zero

dA_one = 0.5
dB_one = 0.3
dC_one = 1-dA_one-dB_one

dA_two = 0.3
dB_two = 0.2
dC_two = 1-dA_two-dB_two

dA_three = 0.4
dB_three = 0.4
dC_three = 1-dA_three-dB_three

dA_four = 0.2
dB_four = 0.4
dC_four = 1-dA_four-dB_four

dA_five = 0.4
dB_five = 0.2
dC_five = 1-dA_five-dB_five

dA_six = 0.3
dB_six = 0.3
dC_six = 1-dA_six-dB_six

dA_seven = 0.3
dB_seven = 0.4
dC_seven = 1-dA_seven-dB_seven

dA_eight = 0.4
dB_eight = 0.3
dC_eight = 1-dA_eight-dB_eight

dA_nine = 0.33
dB_nine = 0.33
dC_nine = 1-dA_nine-dB_nine

zeros = df.loc[df['label'] == 0]
zeros = zeros.reset_index(drop=True)
ones = df.loc[df['label'] == 1]
ones = ones.reset_index(drop=True)
twos = df.loc[df['label'] == 2]
twos = twos.reset_index(drop=True)
threes = df.loc[df['label'] == 3]
threes = threes.reset_index(drop=True)
fours = df.loc[df['label'] == 4]
fours = fours.reset_index(drop=True)
fives = df.loc[df['label'] == 5]
fives = fives.reset_index(drop=True)
sixes = df.loc[df['label'] == 6]
sixes = sixes.reset_index(drop=True)
sevens = df.loc[df['label'] == 7]
sevens = sevens.reset_index(drop=True)
eights = df.loc[df['label'] == 8]
eights = eights.reset_index(drop=True)
nines = df.loc[df['label'] == 9]
nines = nines.reset_index(drop=True)
if not zeros.empty:
	for index in range(len(zeros)):
		x = random.uniform(0,1)
		if(x < dA_zero):
			dfA = pd.concat([dfA, zeros.loc[index, :]], ignore_index=True)
		elif(x < dA_zero + dB_zero):
			dfB = pd.concat([dfB, zeros.loc[index, :]], ignore_index=True)
		else:
			dfC = pd.concat([dfC, zeros.loc[index, :]], ignore_index=True)

if not ones.empty:
	for index in range(len(ones)):
		x = random.uniform(0,1)
		if(x < dA_one):
			dfA = pd.concat([dfA, ones.loc[index, :]], ignore_index=True)
		elif(x < dA_one + dB_one):
			dfB = pd.concat([dfB, ones.loc[index, :]], ignore_index=True)
		else:
			dfC = pd.concat([dfC, ones.loc[index, :]], ignore_index=True)

if not twos.empty:
	for index in range(len(twos)):
		x = random.uniform(0,1)
		if(x < dA_two):
			dfA = pd.concat([dfA, twos.loc[index, :]], ignore_index=True)
		elif(x < dA_two + dB_two):
			dfB = pd.concat([dfB, twos.loc[index, :]], ignore_index=True)
		else:
			dfC = pd.concat([dfC, twos.loc[index, :]], ignore_index=True)

if not threes.empty:
	for index in range(len(threes)):
		x = random.uniform(0,1)
		if(x < dA_three):
			dfA = pd.concat([dfA, threes.loc[index, :]], ignore_index=True)
		elif(x < dA_three + dB_three):
			dfB = pd.concat([dfB, threes.loc[index, :]], ignore_index=True)
		else:
			dfC = pd.concat([dfC, threes.loc[index, :]], ignore_index=True)

if not fours.empty:
	for index in range(len(fours)):
		x = random.uniform(0,1)
		if(x < dA_four):
			dfA = pd.concat([dfA, fours.loc[index, :]], ignore_index=True)
		elif(x < dA_four + dB_four):
			dfB = pd.concat([dfB, fours.loc[index, :]], ignore_index=True)
		else:
			dfC = pd.concat([dfC, fours.loc[index, :]], ignore_index=True)

if not fives.empty:
	for index in range(len(fives)):
		x = random.uniform(0,1)
		if(x < dA_five):
			dfA = pd.concat([dfA, fives.loc[index, :]], ignore_index=True)
		elif(x < dA_five + dB_five):
			dfB = pd.concat([dfB, fives.loc[index, :]], ignore_index=True)
		else:
			dfC = pd.concat([dfC, fives.loc[index, :]], ignore_index=True)

if not sixes.empty:
	for index in range(len(sixes)):
		x = random.uniform(0,1)
		if(x < dA_six):
			dfA = pd.concat([dfA, sixes.loc[index, :]], ignore_index=True)
		elif(x < dA_six + dB_six):
			dfB = pd.concat([dfB, sixes.loc[index, :]], ignore_index=True)
		else:
			dfC = pd.concat([dfC, sixes.loc[index, :]], ignore_index=True)

if not sevens.empty:
	for index in range(len(sevens)):
		x = random.uniform(0,1)
		if(x < dA_seven):
			dfA = pd.concat([dfA, sevens.loc[index, :]], ignore_index=True)
		elif(x < dA_seven + dB_seven):
			dfB = pd.concat([dfB, sevens.loc[index, :]], ignore_index=True)
		else:
			dfC = pd.concat([dfC, sevens.loc[index, :]], ignore_index=True)

if not eights.empty:
	for index in range(len(eights)):
		x = random.uniform(0,1)
		if(x < dA_eight):
			dfA = pd.concat([dfA, eights.loc[index, :]], ignore_index=True)
		elif(x < dA_eight + dB_eight):
			dfB = pd.concat([dfB, eights.loc[index, :]], ignore_index=True)
		else:
			dfC = pd.concat([dfC, eights.loc[index, :]], ignore_index=True)

if not nines.empty:
	for index in range(len(nines)):
		x = random.uniform(0,1)
		if(x < dA_nine):
			dfA = pd.concat([dfA, nines.loc[index, :]], ignore_index=True)
		elif(x < dA_nine + dB_nine):
			dfB = pd.concat([dfB, nines.loc[index, :]], ignore_index=True)
		else:
			dfC = pd.concat([dfC, nines.loc[index, :]], ignore_index=True)


dfA.to_csv("split1.csv")
dfB.to_csv("split2.csv")
dfC.to_csv("split3.csv")