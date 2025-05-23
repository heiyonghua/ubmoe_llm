import json
import os
import datasets


if __name__ == '__main__':
    dataset_folder_path="/home/heiyonghua/fast_disk/QA_baseqween_BiFinetune/cache/winogrande"
    final_data=[]
    for file_name in os.listdir(dataset_folder_path):
        print(file_name)

        with open(f"{dataset_folder_path}/{file_name}",'r') as file:
            dataset=[json.loads(d) for d in file]
            print(dataset[0])
            print(dataset[0].keys())
            final_data+=[{"messages":[{"content":d['arguments']['gen_args_0']['arg_0'],"role":"user"}]} for d in dataset]

    with open("/home/heiyonghua/fast_disk/QA_baseqween_BiFinetune/data/for_dp/winogrande.json",'w') as file:
        json.dump(final_data,file,ensure_ascii=False,indent=4)