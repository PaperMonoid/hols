# hols
Ordinary least squares in haskell using gradient descent with momentum on MNIST.

# Getting started
As a requiremnt you should dump the MNIST dataset into a CSV file within the `datasets` directory. This file will be around 200 MB.

```
$ cabal run
```

Currently this code is able to handle 1000 records in 20-ish seconds. Further work is required to optimize for speed. Example truncated output of `time cabal run`


```
...
3.0000  ->  0.5886
0.0000  ->  -18.4240
5.0000  ->  5.3485
6.0000  ->  3.0815
4.0000  ->  4.3783
4.0000  ->  -34.0654
2.0000  ->  3.9396
4.0000  ->  0.3503
4.0000  ->  -4.5552
3.0000  ->  -9.4794
1.0000  ->  2.5806
7.0000  ->  14.3306
7.0000  ->  6.8384
6.0000  ->  6.7874
0.0000  ->  -2.1624
3.0000  ->  -10.5516

real	0m23.529s
user	0m22.749s
sys	0m0.840s
```
