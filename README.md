# NP_MathNet

Python系のサンプルを元にニューラルネットワークの学習を行った際に作成したコード。
スムーズにPythonのコードで学習するためにNumpyに似たAPIを必要に応じて作成しています。
行列計算にはMath.Netを利用しています。

sin関数の学習サンプルコード
`
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NP_MathNet;
using NP_MathNet.NeuralNetwork;
using NP_MathNet.NeuralNetwork.ActivationFunction;


const int n_mid = 3;  //  中間層のニューロン数
const int n_out = 1;  //  出力層のニューロン数
const double wb_width = 0.01; //  重みとバイアスの広がり具合

const double eta = 0.1;      // 学習係数
const int interval = 100; // 経過の表示間隔

// -- 入力と正解の用意 --
Vector<double> input_data = NP_MathNet.NP.Linespace(-Math.PI, Math.PI);   // 入力
int n_data = input_data.Count;  // データ数
Vector<double> correct_data = Vector<double>.Build.Dense(n_data, (r) => { return Math.Sin(input_data[r]); });  // 正解

// -- 各層の初期化 --
NeuronOneInputLayer input_layer = new NeuronOneInputLayer(n_mid);
NeuronLayer hidden_layer = new NeuronLayer(n_mid, n_mid, wb_width, new SigmoidFunction());
NeuronLayer output_layer = new NeuronLayer(n_mid, n_out, wb_width, new IdentityFunction());

// -- 学習 --
int epoch = 2001;
for (int i = 0; i < epoch; ++i) {
    // インデックスをシャッフル
    // sinはindex通りに学習させると値が連続だが、連続でない方が学習効率が良い？
    int[] index_random = Enumerable.Range(0, n_data).OrderBy(i => Guid.NewGuid()).ToArray();

    // 結果の表示用
    double total_error = 0.0;
    foreach (int idx in index_random) {
        Vector<double> t = Vector<double>.Build.Dense(1, correct_data[idx]); // 正解を行列に変換
        Vector<double> vec_t = Vector<double>.Build.Dense(1, correct_data[idx]); // 正解を行列に変換

        // 順伝播
        Vector<double> y_inp = input_layer.Forward(input_data[idx]);
        Vector<double> y_mid = hidden_layer.Forward(y_inp);
        Vector<double> y = output_layer.Forward(y_mid);

        // 逆伝播
        Vector<double> grad_x = output_layer.Backward(t);
        hidden_layer.Backward(grad_x);

        // 重みとバイアスの更新
        hidden_layer.Update(eta);
        output_layer.Update(eta);

        if ((i % interval) == 0) {
            // 誤差の計算
            total_error += (y - vec_t).Sum(x => x * x) / 2.0;    // 二乗和誤差
        }
    }
    if ((i % interval) == 0)
    {
        // エポック数と誤差の表示
        Console.WriteLine("Epoch:" + i.ToString() + "/" + epoch.ToString()
            + "Error:" + (total_error / n_data).ToString());
    }
}
`
