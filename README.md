# How to use

chainer1.15_fizzbuzz_test.pyを使うときには
基本epochぐらいを指定すれば良い
python chainer1.15_fizzbuzz_test.py -e 1000
1000ぐらいやればaccuracyも1ぐらいになる。

これをやって生成されたmy.modelとmy.stateを使って
python predict_fizzbuzz_to_use_model.py -m my.model -o my.state
で実行してあげれば100までの数字の予測をしてくれて、その数字がどの答えであるかの確率がでてくる。
