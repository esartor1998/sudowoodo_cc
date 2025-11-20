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

def serialize_tableAandB(tableA_str, tableB_str, ignore=None):
	tableA_str = '\r\n'.join([' '.join([f"COL {col} VAL {row[1][col]}" for col in tableA_str.columns]) for row in tqdm(tableA_str.iterrows(), total=len(tableA_str))])
	tableB_str = '\r\n'.join([' '.join([f"COL {col} VAL {row[1][col]}" for col in tableB_str.columns]) for row in tqdm(tableB_str.iterrows(), total=len(tableB_str))])
	return tableA_str, tableB_str

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", type=str, required=True)
	parser.add_argument("--remove-binary", action='store_true')
	parser.add_argument("--remove-multi", action='store_true')
	parser.add_argument("--output-folder", type=str, default="data/em/prisoner_pairs")
	parser.add_argument("--label-col", type=str, default="match")
	parser.add_argument("--id_col", type=str, default="PersonID")
	args = parser.parse_args()

	os.makedirs(args.output_folder, exist_ok=True)
	cache_location = f'{args.input}_serialized_removebin={args.remove_binary}_removemulti={args.remove_multi}_cached.csv'
	n_splits = 10

	df = pd.read_csv(args.input)
	# Keep original IDs before serialization / dropping columns
	y = np.array([str(int(item)) for item in df[f'{args.id_col}'] == df[f'{args.id_col}_right']])
	if not os.path.isfile(cache_location):
		df = df.drop(columns=[f'{args.id_col}', f'{args.id_col}_right'])
		print(f'Generating and cacheing the serialized version of the pairs file {args.input}...')
		df = binarize_string_binary_col(df, 'is_tattoo')
		print(df.columns)
		try:
			df = restring_list_col(df, 'label')
			df = restring_list_col(df, 'label_right')
		except Exception as e:
			print(f'Could not re-stringify list columns: {e}')
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
		tableA_str, tableB_str, _ = split_df(df, '_right', args.label_col, verbose=False)
		tableA_str, tableB_str = serialize_tableAandB(tableA_str, tableB_str)
		if args.remove_binary:
			binary_cols = [col for col in df.columns if col.startswith('is_tattoo')]
			print(f'Removing binary columns: {binary_cols}')
			df = df.drop(columns=binary_cols)
		if args.remove_multi:
			multi_cols = [col for col in df.columns if col.startswith('label')]
			print(f'Removing multi-valued columns: {multi_cols}')
			df = df.drop(columns=multi_cols)
		df = serialize_df_to_df(df, suffix='_right', label_col=args.label_col, include_labels=False)
		df['label'] = y
		df.to_csv(cache_location)
	else:
		print(f'Reading cached {cache_location}')
		df = pd.read_csv(cache_location)
		tableA_str, tableB_str = split_df(df, '_right', args.label_col, verbose=False)
		tableA_str, tableB_str = serialize_tableAandB(tableA_str, tableB_str)

	#this is just X
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
		del train_str
		with open(os.path.join(fold_name_string, 'train_no_label.txt'), 'w', encoding='utf-8') as f:
			f.write(
				train_str_no_label
			)
		del train_str_no_label
		with open(os.path.join(fold_name_string, 'test.txt'), 'w', encoding='utf-8') as f:
			f.write(
				test_str
			)
		del test_str
		with open(os.path.join(fold_name_string, 'valid.txt'), 'w', encoding='utf-8') as f:
			f.write(
				valid_str
			)
		del valid_str
		with open(os.path.join(fold_name_string, 'tableA.txt'), 'w', encoding='utf-8') as f:
			f.write(
				tableA_str
			)
		with open(os.path.join(fold_name_string, 'tableB.txt'), 'w', encoding='utf-8') as f:
			f.write(
				tableB_str
			)
			
		train_ids_df = pd.DataFrame({
			"ltable_id": train_idx,
			"rtable_id": train_idx,
			"label": y[train_idx]
		})
		valid_ids_df = pd.DataFrame({
			"ltable_id": val_idx,
			"rtable_id": val_idx,
			"label": y[val_idx]
		})
		test_ids_df = pd.DataFrame({
			"ltable_id": test_idx,
			"rtable_id": test_idx,
			"label": y[test_idx]
		}) #because our pairs are premade, this is trivial; lindex and rindex are going to be = in tableA and tableB
		
		train_ids_df.to_csv(os.path.join(fold_name_string, "train.csv"), index=False)
		valid_ids_df.to_csv(os.path.join(fold_name_string, "valid.csv"), index=False)
		test_ids_df.to_csv(os.path.join(fold_name_string, "test.csv"), index=False)
		#deserialize_df(train_df).to_csv(os.path.join(fold_name_string, 'train.csv'), index=False)
		#deserialize_df(test_df).to_csv(os.path.join(fold_name_string, 'test.csv'), index=False)
		#deserialize_df(valid_df).to_csv(os.path.join(fold_name_string, 'valid.csv'), index=False)
		#print(f"Fold {fold} done.")
