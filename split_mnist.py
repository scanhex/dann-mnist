import pandas as pd
import random

df = pd.read_csv("mnist_train.csv")

dfs = [pd.DataFrame(columns=df.columns) for i in range(3)]

probs = [
	[0.2, 0.5],
	[0.5, 0.3],
	[0.3, 0.2],
	[0.4, 0.4],
	[0.2, 0.4],
	[0.4, 0.2],
	[0.3, 0.3],
	[0.3, 0.4],
	[0.4, 0.3],
	[0.33, 0.33]
]

for ps in probs:
	ps.append(1-ps[0]-ps[1])

for label in range(10):
	label_df = df.loc[df['label'] == label]
	label_df = label_df.reset_index(drop=True)
	if not label_df.empty:
		for index in range(len(label_df)):
			x = random.uniform(0,1)
			if x < probs[label][0]:
				dfs[0] = pd.concat([dfs[0], label_df.loc[index, :]], ignore_index=True)
			elif x < probs[label][0] + probs[label][1]:
				dfs[1] = pd.concat([dfs[1], label_df.loc[index, :]], ignore_index=True)
			else:
				dfs[2] = pd.concat([dfs[2], label_df.loc[index, :]], ignore_index=True)


df[0].to_csv("split1.csv")
df[1].to_csv("split2.csv")
df[2].to_csv("split3.csv")