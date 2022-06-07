import os
import pickle
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split



def aggreate_func_data(input_list_path, output_path, file_name = "function_datas", split=10, min_size=3):
    print("Collcet functions ...")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    func_datas = []
    not_found = 0
    with open(input_list_path,"r") as f:
        binary_list = f.readlines()
        for file in tqdm(binary_list):
            pickle_file = file.strip(os.linesep)+".pickle"
            if not os.path.exists(pickle_file):
                print("{} not found".format(pickle_file))
                not_found += 1
                continue
            with open(pickle_file,"rb") as pf:
                cur_func_datas = pickle.load(pf)
                for func_data in cur_func_datas:
                    if func_data["cfg_size"] < min_size:
                        continue
                    else:
                        func_datas.append(func_data)
                        print(func_data["abstract_args_type"])

    print("total %d functions" % (len(func_datas)))
    print("%d not found" % not_found)
    print("Aggreate functions done.")

    data_path = os.path.join(output_path, file_name+".pickle")
    with open(data_path, "wb") as f:
        print("Saving func datas...")
        pickle.dump(func_datas, f)
        print("Saving done.")
    data_size = len(func_datas)//split
    chunk = 0
    for i in range(0, len(func_datas), data_size):
        chunk += 1
        data_path = os.path.join(output_path, str(chunk)+"_"+file_name+".pickle")
        tmp_func_datas = func_datas[i:min(i+data_size, len(func_datas))]
        with open(data_path, "wb") as f:
            print("Saving func datas...")
            pickle.dump(tmp_func_datas, f)
            print( "Saving {} done.".format(str(chunk)) )
        pass


if __name__ == "__main__":
    input_list_path = ""
    output_path = ""
    aggreate_func_data(input_list_path, output_path)
    pass