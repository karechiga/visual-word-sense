# SemEval-2023 Task 1 - Visual Word Sense Disambiguation

## About this project

I present an approach for SemEval-2023 Task 1 which entails the classification of ambiguous word senses with limited 
context by identifying a representative image from a list of 10 images. The Training Data and Test Data is downloaded from [the SemEval-2023 Task 1 webpage](https://raganato.github.io/vwsd/). This repository explains the usage of the project allowing others to train a similar model. See the attached paper for more info on the methods and results.

## Dependencies

This project was developed using Python v. 3.9.15 with the following packages:

* numpy 1.23.5
* pytorch 1.13.1
* torchvision 0.14.1
* sentence-transformers 2.2.2
* wandb 0.13.10
* scikit-learn 1.2.0
* nltk 3.7
* pillow 9.3.0
* matplotlib 3.6.2

Along with other standard Python libraries. 

It is recommended to use a CUDA device to train a model. I used a device with 32 GB RAM which can handle a batch size of 32 samples.

## Usage

### Setting up the directories

1. The directory should contain a "code" directory (which is the root directory that the program should run out of) and a "data" directory.

2. The "data" directory should contain three folders: "train", "test", and "embeddings".

3. The "train" folder should feature the downloaded training data from the SemEval-2023 Task 1 website - train.data.v1.txt, train.gold.v1.txt, and a folder containing all of the images that are part of the training.

4. Likewise, the "test" folder will include the downloaded test data from the SemEval-2023 webpage and the folder of test images.

5. The "embeddings" folder is only needed if using GloVe embeddings. The GloVe embeddings can be downloaded [here](https://nlp.stanford.edu/projects/glove/) and placed in the folder. It is recommended to use Sentence-Transformer instead of GloVe.

Once these directories are created, we are ready to run the code.

### Training

We can train the model using the following command:

`$ python main.py train [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE] [--i_dropout I_DROPOUT] [--w_dropout W_DROPOUT] [--train_size TRAIN_SIZE] [--dev_size DEV_SIZE] [--word_linears WORD_LINEARS] [--word_activations WORD_ACTIVATIONS] [--out_linears OUT_LINEARS] [--out_activations OUT_ACTIVATIONS] [--seed SEED] [--early_stop EARLY_STOP] [--es_threshold ES_THRESHOLD] [--model MODEL] [--vector_combine VECTOR_COMBINE] [--shuffle_options] [--save] model_dir data_dir`

Required arguments:
  * `model_dir` - The directory path where the best models should be saved to. (Example: `../models_data`)
  * `data_dir` - The directory path where the "data" directory is located. (Example: `../data`)

Optional arguments:
  * `--epochs` - Maximum number of epochs to train the model on.
  * `--batch-size` - The number of training samples in each batch.
  * `--learning-rate` - The learning rate for the Adam optimizer
  * `--i_dropout` - the dropout percentage for the output of the image model.
  * `--w_dropout` - the dropout percentage for the output of the word model.
  * `--train_size` - The number of training samples to use.
  * `--dev_size` - The number of validation samples to use.
  * `--word_linears` - The number of linear layers in the feed-forward part of the word model.
  * `--word_activations` - The activation function for the word model layers. (`tanh` or `relu`)
  * `--out_linears` - Number of layers in used in the output feedforward network.
  * `--out_activations` - The activation function for the output layers. (`tanh` or `relu`)
  * `--seed` - The numpy random seed.
  * `--early_stop` - The number of epochs allowed to be below the `es_threshold` plus the validation accuray before stopping the training loop and saving the best model.
  * `--es_threshold` The threshold to be surpassed for the early stopping counter to be reset.
  * `--model` The pre-trained word embedding model to be used. (`glove`, `mpnet_base_v2`, `MiniLM-L12-v2`).
  * `--vector_combine` - The method of combining image and word embeddings. (`dot` or `concatenate`)
  * `--shuffle_options` - Shuffles the image options for every sample for each batch.
  * `--save` - Saves the best model seen during training in terms of validation accuracy.
  
### Evaluation

We can evaluate an existing model using the following command:
`main.py evaluate [--test_dir] [--step_size] [--train_size] [--dev_size] [--seed] model_dir data_dir`

Required arguments:
  * `model_dir` - The directory path where the model is stored. (Example: `../models_data/model_12345`)
  * `data_dir` - The directory path where the "data" directory is located. (Example: `'../data'`)

Optional arguments:
  * `--test_dir` - The directory path where the "test" directory is located. (Example: `'../data/test'`)
  * `--step_size` - The number of samples to use for each batch to run through the model.
  * `--train_size` - The number of samples from the training data to evaluate.
  * `--dev_size` - The number of samples from the validation data to evaluate.
  * `--seed` - The numpy random seed
  
The evaluation prints out the predictions, and outputs all the results to a .csv file file in the model's directory.
  



