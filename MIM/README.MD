## This is the code used for the Mutual Information Maximization (MIM) objective.

To produce the accuracy reported on CIFAR-10, first learn the representation network using the Neural Bayes MIM objective:

`python main.py --lr 0.001 --dataset cifar10 --bs 4000 --mbs 500 --gpu 0 --epoch 100 --alpha 4 --beta 2 --multiscale 2 --all`

Running the above code will generate a save dir (<save_dir>). Then train the classifier using these features:

`python train_pred.py --hid 200  --lr 0.0001 --dataset cifar10 --gpu 0 --save_dir <save_dir>`
