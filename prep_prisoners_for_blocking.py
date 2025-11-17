import os
import ast
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from serializer import restring_list_col, binarize_string_binary_col, serialize_df_to_df, deserialize_df, split_df
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

random_state = 42

def stratified_kfold_with_val(X, y, n_splits=10, val_size=0.1):
	outer_kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

	for train_val_idx, test_idx in outer_kf.split(X, y):
		y_train_val = np.array(y)[train_val_idx]

		inner_sss = StratifiedShuffleSplit(
			n_splits=1, test_size=val_size, random_state=random_state
		)

		train_idx_inner, val_idx_inner = next(
			inner_sss.split(np.zeros(len(train_val_idx)), y_train_val)
		)

		train_idx = train_val_idx[train_idx_inner]
		val_idx   = train_val_idx[val_idx_inner]

		yield train_idx, val_idx, test_idx

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", type=str, required=True, default="prisoners_F_M_bodypart_cleaned_new.csv_bc_july.csv_mc.csv")
	parser.add_argument("--remove-binary", action='store_true', help="Remove binary columns indicating presence of tattoos.")
	parser.add_argument("--remove-multi", action='store_true', help="Remove multi-valued columns indicating tattoo descriptions.")
	parser.add_argument("--output-folder", type=str, default="data/em/prisoner_pairs")
	args = parser.parse_args()

	os.makedirs(args.output_folder, exist_ok=True)
	cache_location = f'{args.input}_serialized_removebin={args.remove_binary}_removemulti={args.remove_multi}_cached.csv'
	n_splits = 10

	df = pd.read_csv(args.input)
	y = np.array([str(int(item)) for item in df['PersonID'] == df['PersonID_right']])
	if not os.path.isfile(cache_location):
		print(f'Generating and cacheing the serialized version of the pairs file {args.input}...')
		df = binarize_string_binary_col(df, 'is_tattoo')
		df = restring_list_col(df, 'label')
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
		tableA_str, tableB_str, _ = split_df(df, '_right', 'match', verbose=False)
		tableA_str = '\r\n'.join([' '.join([f"COL {col} VAL {row[1][col]}" for col in tableA_str.columns]) for row in tqdm(tableA_str.iterrows(), total=len(tableA_str))])
		tableB_str = '\r\n'.join([' '.join([f"COL {col} VAL {row[1][col]}" for col in tableB_str.columns]) for row in tqdm(tableB_str.iterrows(), total=len(tableB_str))])
		df = df.drop(columns=['PersonID', 'PersonID_right'])
		if args.remove_binary:
			binary_cols = [col for col in df.columns if col.startswith('is_tattoo')]
			print(f'Removing binary columns: {binary_cols}')
			df = df.drop(columns=binary_cols)
		if args.remove_multi:
			multi_cols = [col for col in df.columns if col.startswith('label')]
			print(f'Removing multi-valued columns: {multi_cols}')
			df = df.drop(columns=multi_cols)
		df = serialize_df_to_df(df, suffix='_right', label_col='match', include_labels=False)
		df['label'] = y
		df.to_csv(cache_location)
	else:
		print(f'Reading cached {cache_location}')
		df = pd.read_csv(cache_location)

	X = np.array(df['text'].values)

	for fold, (train_idx, val_idx, test_idx) in tqdm(enumerate(stratified_kfold_with_val(X, y, n_splits=n_splits)), total=n_splits, desc="Processing folds..."):
		#print(f"\"TRAIN\" {fold}:", train_idx, f"TEST {fold}:", test_idx, f"VAL {fold}:", val_idx)
		#print(f"For fold {fold}\n\tLenght of the set is {len(train_idx)+len(test_idx)+len(val_idx)} items.")
		fold_name_string = f"{args.output_folder}/fold{fold}"
		os.makedirs(fold_name_string, exist_ok=True)

		X_train, X_test, X_valid = X[train_idx], X[test_idx], X[val_idx]
		y_train, y_test, y_valid = y[train_idx], y[test_idx], y[val_idx]

		# (re)serialize to Ditto format
		train_str_no_label = '\n'.join(X_train)
		train_str = '\r\n'.join(['\t'.join(row_elements) for row_elements in zip(X_train, y_train)])
		test_str  = '\r\n'.join(['\t'.join(row_elements) for row_elements in zip(X_test, y_test)])
		valid_str = '\r\n'.join(['\t'.join(row_elements) for row_elements in zip(X_valid, y_valid)])

		with open(os.path.join(fold_name_string, 'train.txt'), 'w', encoding='utf-8') as f:
			f.write(
				train_str
			)
		with open(os.path.join(fold_name_string, 'train_no_label.txt'), 'w', encoding='utf-8') as f:
			f.write(
				train_str_no_label
			)
		with open(os.path.join(fold_name_string, 'test.txt'), 'w', encoding='utf-8') as f:
			f.write(
				test_str
			)
		with open(os.path.join(fold_name_string, 'valid.txt'), 'w', encoding='utf-8') as f:
			f.write(
				valid_str
			)
		with open(os.path.join(fold_name_string, 'tableA.txt'), 'w', encoding='utf-8') as f:
			f.write(
				tableA_str
			)
		with open(os.path.join(fold_name_string, 'tableB.txt'), 'w', encoding='utf-8') as f:
			f.write(
				tableB_str
			)
		#deserialize_df(train_df).to_csv(os.path.join(fold_name_string, 'train.csv'), index=False)
		#deserialize_df(test_df).to_csv(os.path.join(fold_name_string, 'test.csv'), index=False)
		#deserialize_df(valid_df).to_csv(os.path.join(fold_name_string, 'valid.csv'), index=False)
		#print(f"Fold {fold} done.")
