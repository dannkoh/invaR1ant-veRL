from datasets import load_dataset

dataset = load_dataset("dannkoh/invaR1ant-easy")

dataset["base.train"].to_parquet("./train.parquet")
dataset["base.test"].to_parquet("./test.parquet")