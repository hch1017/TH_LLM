from datasets import load_dataset
import raw_datasets

dataset = load_dataset(path='/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Local/TH_V1', data_files='prompt.json')
print(dataset)


output_path = '/'
seed = 1234
local_rank = -1
dataset_name = 'Local/TH_V1'

d = raw_datasets.LocalTH_V1(output_path, seed, local_rank, dataset_name)