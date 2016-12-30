# JigsawPuzzle-Mnist

Jigsaw Puzzle solver on MNIST dataset

Based on the paper: Unsupervised Learning of Visual Representions by solving Jigsaw Puzzles

This is a simple model to train and test Jigsaw Puzzle on MNIST data to validate the idea.
download a scratch model: https://drive.google.com/open?id=0B8ZQ8f5x7SYXeEpjVXFFSG5wbkk

Accuracy: ~99%

Data: 50K training and 10K validation

Each image is divided into 3X3 tile containing number between 0 and 9. All numbers in an image is unique.
The permutation to create the puzzle is set to 100.
Total permutation is huge since there can be 9! ways of creating the puzzle. A very small subset is chosen.

The data preprocessing is carried out exactly how it is done in the cited paper. It's just the data is different.

Siamese Network with 9 branches. Each branch is a LeNet w/o fully connected layer.
SGD optim.
