# NP_MathNet

Python系のサンプルを元にニューラルネットワークの学習を行った際に作成したコード。

スムーズにPythonのコードで学習するためにNumpyに似たAPIを必要に応じて作成しています。

行列計算にはMath.Netを利用しています。

学習の主体をKelpNetに移行する予定なので、おそらく更新は行いません。

Code created to train a neural network based on Python-based samples.

Numpy-like APIs are created as needed to train in Python code smoothly.

Math.Net is used for matrix computation.

I will be shifting our learning focus to KelpNet, I will probably not update it.



sin関数の学習サンプルコード(Sample code for learning sin function)


    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;
    using NP_MathNet;
    using NP_MathNet.NeuralNetwork;
    using NP_MathNet.NeuralNetwork.ActivationFunction;


    const int n_mid = 3;  //  中間層のニューロン数(Number of neurons in the middle layer)
    const int n_out = 1;  //  出力層のニューロン数(Number of neurons in the output layer)
    const double wb_width = 0.01; //  重みとバイアスの広がり具合(Weight and bias spread)

    const double eta = 0.1;      // 学習係数(Learning coefficients)
    const int interval = 100; // 経過の表示間隔(Elapsed display interval)

    // -- 入力と正解の用意(Input and correct answer) --
    Vector<double> input_data = NP_MathNet.NP.Linespace(-Math.PI, Math.PI);   // 入力(Input)
    int n_data = input_data.Count;  // データ数(Count of data)
    Vector<double> correct_data = Vector<double>.Build.Dense(n_data, (r) => { return Math.Sin(input_data[r]); });  // 正解(Correct)

    // -- 各層の初期化(Initialization of each layer) --
    NeuronOneInputLayer input_layer = new NeuronOneInputLayer(n_mid);
    NeuronLayer hidden_layer = new NeuronLayer(n_mid, n_mid, wb_width, new SigmoidFunction());
    NeuronLayer output_layer = new NeuronLayer(n_mid, n_out, wb_width, new IdentityFunction());

    // -- 学習(Learning) --
    int epoch = 2001;
    for (int i = 0; i < epoch; ++i) {
        // インデックスをシャッフル(Shuffle the index)
        // sinはindex通りに学習させると値が連続だが、連続でない方が学習効率が良い？
        // When sin is trained as per index, the values are continuous, but is it more efficient to train it non-continuously?
        int[] index_random = Enumerable.Range(0, n_data).OrderBy(i => Guid.NewGuid()).ToArray();

        // 結果の表示用(For displaying results)
        double total_error = 0.0;
        foreach (int idx in index_random) {
            Vector<double> t = Vector<double>.Build.Dense(1, correct_data[idx]); // 正解を行列に変換(Convert the correct answer to a vector)
            Vector<double> vec_t = Vector<double>.Build.Dense(1, correct_data[idx]);

            // 順伝播(forward propagation)
            Vector<double> y_inp = input_layer.Forward(input_data[idx]);
            Vector<double> y_mid = hidden_layer.Forward(y_inp);
            Vector<double> y = output_layer.Forward(y_mid);

            // 逆伝播(back propagation)
            Vector<double> grad_x = output_layer.Backward(t);
            hidden_layer.Backward(grad_x);

            // 重みとバイアスの更新(Updating Weights and Biases)
            hidden_layer.Update(eta);
            output_layer.Update(eta);

            if ((i % interval) == 0) {
                // 誤差の計算(Calculating the error)
                total_error += (y - vec_t).Sum(x => x * x) / 2.0;    // 二乗和誤差(sum-of-squares error)
            }
        }
        if ((i % interval) == 0)
        {
            // エポック数と誤差の表示(Epoch number and error display)
            Console.WriteLine("Epoch:" + i.ToString() + "/" + epoch.ToString()
                + "Error:" + (total_error / n_data).ToString());
        }
    }

