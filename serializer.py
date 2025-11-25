import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# constants so i can write in normal words instead of magic numbers in case axis= comes up
ROWS = 0
COLS = 1


# the cases not covered by this converter are just out of the scope of this script
# so we can just assume that the label is either a 0 or a 1 already
def convert_label(label):
	if isinstance(label, str):
		if label.lower() in ["1", "true", "y", "label_1"]:
			return 1
		elif label.lower() in ["0", "false", "n", "label_0", pd.NA, np.nan, None]:
			return 0
	return int(label)


def serialize_df_to_df(df, suffix, label_col, include_labels=True, descr="Converting to Ditto format"):
	# split_df is a function you supply
	left_df, right_df, labels = split_df(df, suffix, label_col)

	left_cols = left_df.columns
	right_cols = right_df.columns

	texts = []
	out_labels = []

	for left_row, right_row, label in tqdm(
		zip(left_df.iterrows(), right_df.iterrows(), labels),
		total=len(left_df),
		desc=descr,
		unit="rows",
	):
		left_serialized = " ".join(
			f"COL {col} VAL {left_row[1][col]}" for col in left_cols
		)

		right_serialized = " ".join(
			f"COL {col} VAL {right_row[1][col]}" for col in right_cols
		)

		if include_labels:
			text = "\t".join([left_serialized, right_serialized, str(label)])
			out_labels.append(label)
		else:
			text = "\t".join([left_serialized, right_serialized])
			out_labels.append(None)

		texts.append(text)

	return pd.DataFrame({"text": texts, "labels": out_labels})


def serialize_dfs(left_df, right_df, labels, include_labels=True, descr="Converting to Ditto format"):
	left_cols = left_df.columns
	right_cols = right_df.columns
	return "\r\n".join(
		[
			"\t".join(
				[
					" ".join(
						[f"COL {col} VAL {left_row[1][col]}" for col in left_cols]
					),
					" ".join(
						[f"COL {col} VAL {right_row[1][col]}" for col in right_cols]
					),
					str(label),
				]
				if include_labels
				else [
					" ".join(
						[f"COL {col} VAL {left_row[1][col]}" for col in left_cols]
					),
					" ".join(
						[f"COL {col} VAL {right_row[1][col]}" for col in right_cols]
					),
				]
			)
			for left_row, right_row, label in tqdm(
				zip(left_df.iterrows(), right_df.iterrows(), labels),
				total=len(left_df),
				desc=descr,
				unit="rows",
			)
		]
	)


def split_df(df, suffix, label_col, verbose=False):
	# a switch to polars would make this work when things dont fit in memory, but... the rest of the program wont work if this
	# doesnt fit in memory so we can use pandas anyways lmoa
	labels = df[label_col].apply(convert_label)
	right_df = df.loc[:, df.columns.str.endswith(suffix)].copy()
	left_df = df.drop(columns=right_df.columns, axis=COLS).drop(
		columns=[label_col], axis=COLS
	)
	if verbose:
		print("left_df:")
		print(left_df)
		print("----------" * 4, "\n", "right_df:")
		print(right_df)
		print("----------" * 4, "\n", "labels:")
		print(labels)
	return left_df, right_df, labels


def serialize_df(
	df,
	suffix,
	label_col,
	verbose=False,
	include_labels=True,
	descr="Converting to Ditto format",
):
	left_df, right_df, labels = split_df(df, suffix, label_col, verbose)
	return serialize_dfs(left_df, right_df, labels, include_labels, descr)


def restring_list_col(df, col_name):
	print("Re-stringifying the list columns...")
	print(df)
	df[col_name] = [
		"[]"
		if item == []
		else ', '.join([subitem['label'] for subitem in ast.literal_eval(item)])
		for item in tqdm(df[col_name])
	]
	return df

def restring_lists(column):
	return [
		"[]"
		if item == []
		else str([subitem['label'] for subitem in ast.literal_eval(item)])
		for item in tqdm(column)
	]

def destring_list_col(df, col_name):
	print("De-stringifying the list columns...")
	print(df)
	df[col_name] = [
		[]
		if item == "[]"
		else [subitem['label'] for subitem in ast.literal_eval(item)]
		for item in tqdm(df[col_name])
	]
	return df


def binarize_string_binary_col(df, col_name):
	print("Binarizing the string binary columns...")
	print(df)
	df[col_name] = [
		1 if item.lower() in ["true", "1", "y", "yes", "label_1"] else 0
		for item in tqdm(df[col_name])
	]
	return df


def deserialize_df(df_raw):
	def parse_side(text, suffix=""):
		#slow regex :(
		pattern = re.compile(r"COL\s+(\S+)\s+VAL\s+(.*?)(?=\s+COL\s+\S+\s+VAL\s+|$)")
		matches = pattern.findall(text.strip())
		return {key.strip() + suffix: val.strip() for key, val in matches}

	def parse_rawer(text, suffix=""):
		left = parse_side(text[0])
		right = parse_side(text[1], suffix="_right")
		out = {**left, **right}
		out[match] = text[2]
		return out

	def parse_row(row):
		left = parse_side(row["left"])
		right = parse_side(row["right"], suffix="_right")
		out = {**left, **right}
		out["match"] = int(row["match"])
		out["match_confidence"] = float(row["match_confidence"])
		out["y_true"] = int(row["y_true"])
		return out

	# Parse each row into structured format
	lc = sum(1 for _ in df_raw)
	df_raw.seek(0)
	parsed_rows = [
		parse_side(line)
		for line in tqdm(df_raw, desc="Parsing raw data...", total=lc)
	]
	# Create a clean DataFrame
	df = pd.DataFrame(parsed_rows)
	print(df.columns)
	return df


# -------------------------------------------------
# convert the dataframes to the Ditto format
# -------------------------------------------------

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Serialize a .csv file into the way Ditto expects it. Writes output as UTF-8."
	)
	parser.add_argument("target", type=str, help="The target file to process.")
	parser.add_argument(
		"--label-col",
		type=str,
		default="label",
		help='The column that holds the match/nonmatch label. Defaults to "label". Should be a 0 or a 1, but the script will convert bools as well.',
	)
	parser.add_argument(
		"--suffix",
		type=str,
		default="_right",
		help='The suffix on the right side of the pair. Defaults to "_right".',
	)
	parser.add_argument(
		"-v", "--verbose", action="store_true", help="Enable verbose output."
	)
	parser.add_argument(
		"-f",
		"--force",
		action="store_true",
		help="Force overwrite of the output file if it already exists.",
	)
	args = parser.parse_args()

	target_path = Path(args.target)

	df = pd.read_csv(target_path).fillna("")

	# out location calculated as "in the parent of args.target, with the same name but .txt extension"
	out_location = target_path.parent.joinpath(target_path.stem).with_suffix(".txt")

	if out_location.exists() and not args.force:
		print(f"Output file {out_location} already exists. Use --force to overwrite.")
		# exit code 73 is "cannot create a file" which is self-inflicted here but we don't want to overwrite files by default
		exit(73)

	if args.verbose:
		print("Writing to", out_location, "...")

	with open(out_location, "w", encoding="utf-8") as f:
		f.write(
			serialize_df(pd.read_csv(target_path), args.suffix, args.label_col, args.verbose)
		)
	exit(0)
