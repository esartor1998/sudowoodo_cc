import pandas as pd
import polars as pl
import argparse
import random
import re
from tqdm import tqdm
from serializer import restring_lists

ROWS=0
COLS=1
SEED=42
pl.set_random_seed(SEED)

PID = 'life_id'
RID = 'recordid'

parser = argparse.ArgumentParser(description='Join two dataframes, one with _right and the other without. Drop duplicate pairs.')
parser.add_argument('-t', '--target', type=str, required=True, help='The target file to self-join on.')
parser.add_argument('-n', '--negative-magnitude', type=float, default=9.0, help='How many more times negative pairs than positive pairs to generate. Default is 9.0.')
parser.add_argument('-o', '--output', type=str, required=True, help='The name of the output file.')
args = parser.parse_args()
#TODO: add other options, specifying the colnames to target for joining and for match identification etc

df = pd.read_csv(args.target)
df['label'] = restring_lists(df['label'])
df = df.astype(str)
df = df.fillna("")
df = df.astype(str)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print(df.dtypes)
print(df)
df = pl.from_pandas(df)
lf = df.lazy()
lf_pairs = (
	lf.join(lf, on=PID, how="inner", coalesce=False)
	.filter(pl.col(RID) < pl.col(f"{RID}_right")) # Unique pairs (A, B) where A < B
).with_columns(pl.lit(1).alias("match")) # these are all the positive pairs. we can generate negative pairs up to some ratio by looking with-out the RID_L-RID_R
lf_pairs.sink_csv(f'{args.output}_matches.csv', engine='streaming')

# Step 2: generate non-matching pairs randomly
total_negatives = int(len(df) * args.negative_magnitude)
negative_pairs = pl.DataFrame()
last_len = 0
max_consec_fails = 100_000
consec_fails = 0
pbar = tqdm(total=total_negatives, desc="Generating negative pairs")

while len(negative_pairs) < total_negatives and consec_fails < max_consec_fails:
	a = df.sample(n=total_negatives, with_replacement=True, shuffle=True)
	b = df.sample(n=total_negatives, with_replacement=True, shuffle=True)
	b = b.rename({col: f'{col}_right' for col in b.columns})
	joined = pl.concat([a, b], how='horizontal')
	negative_pairs = (
		pl.concat([negative_pairs, joined])
		.filter(pl.col(RID) < pl.col(f'{RID}_right'))
		.filter(pl.col(PID) != pl.col(f'{PID}_right'))
	).unique()
	if len(negative_pairs) == last_len:
		consec_fails += 1
		if consec_fails == max_consec_fails:
			print(f"Failed to generate new pairs after {consec_fails} consecutive attempts.\nProbably reaching the limit of pairs that can be added. Stopping.")
			break
	else:
		pbar.update(len(negative_pairs) - last_len)
		last_len = len(negative_pairs)
		consec_fails = 0

pbar.close()
print('negatives:', len(negative_pairs), '\ntotal in original df:', len(df))
negative_pairs = negative_pairs.with_columns(pl.lit(0).alias("match"))

#NOTE: this is all written in this way so that an extra "match" column can be added or not added depending on needs :)
pl.DataFrame(negative_pairs).lazy().sink_csv(f'{args.output}_negatives.csv', engine='streaming')
final_df = pl.concat([lf_pairs.collect(), negative_pairs]).sample(fraction=1.0, shuffle=True)
final_df.lazy().sink_csv(args.output, engine='streaming')
