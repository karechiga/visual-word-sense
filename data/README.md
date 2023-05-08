# Data directory

The Training Data and Test Data should be downloaded from [the SemEval-2023 Task 1 webpage](https://raganato.github.io/vwsd/).

## Setting up the directories

1. The "data" directory should contain three folders: "train", "test", and "embeddings".

2. The "train" folder should feature the downloaded training data from [the SemEval-2023 Task 1 website](https://raganato.github.io/vwsd/) - train.data.v1.txt, train.gold.v1.txt, and a folder containing all of the images that are part of the training.

3. Likewise, the "test" folder will include the downloaded test data from the SemEval-2023 webpage and the folder of test images.

4. The "embeddings" folder is only needed if using GloVe embeddings. The GloVe embeddings can be downloaded [here](https://nlp.stanford.edu/projects/glove/) and placed in the folder. It is recommended to use Sentence-Transformer instead of GloVe.

Once these directories are created, we are ready to run the code.


