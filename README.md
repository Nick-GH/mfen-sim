MFEN-Sim

# Prerequiste
1. IDA pro (>=7.5);
2. ctags;
3. sqlite;
3. python packages in requirements.txt;


# Train the model
## FeatureExtractor
Reverse binary files and extract features from binary functions;
1. Configure the config options in the config directory to generate an input_list.txt containing the path of the file to be analyzed based on the input_list_generator directory;
2. Configure the input_list.txt path, IDA path, and analysis script path in the reverse_commands directory.
3. The scripts in the reverse_commands directory are executed to parse the binary files, and the parsing results are in the same directory as the binary files.

## MFEN_Sim
Model building and training;
1. Use the aggerate_func_data.py and split_data.py in the dataset directory to aggregate the reverse results and divide the dataset.
2. Pretain.py and type.py scripts in pretrain directory were used to train Bert model and fine-tune function signature prediction model for subsequent MFEN-Sim training. You need to set parameters in the config directory.
3. In the similarity directory, function data is first filtered through graph_filter.py, followed by generating the function signature prediction embeddings and function code literal embeddings of through embedding_generator.py. Training the MFEN model through the smfen-sim.py script. You need to set parameters in the config directory.

# Application
Binary similarity detection;
1. Parse the binary file using cmd_feature_generator.py and the results are stored in the database;
2. Generate the function's embedded vectors using application.py; 
3. Calculate the similarity between two binary functions using main_app.py. For details, see application.sh.






