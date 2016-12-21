# JigsawPuzzle-Mnist

Jigsaw Puzzle solver on MNIST dataset

Based on the paper: Unsupervised Learning of Visual Representions by solving Jigsaw Puzzles

This is a simple model to train and test Jigsaw Puzzle on MNIST dataset to test the idea.

Accuracy: 98% after 5 epochs

Data: 50K training and 10K validation

Each image is 3X3 tile containing unique number between 0 and 9. The permutation to create the puzzle is set to 100 . Total permutation is huge since there can be 9! ways of creating the puzzle. A very small subset is chosen.

The data preprocessing is carried out exactly how it is done in the cited paper. It's just the data is different.

Siamese Network with 9 branches. Each branch is a LeNet. SGD optim.
