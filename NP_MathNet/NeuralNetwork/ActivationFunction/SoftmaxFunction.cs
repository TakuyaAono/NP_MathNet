using MathNet.Numerics.LinearAlgebra;

namespace NP_MathNet.NeuralNetwork.ActivationFunction
{
    /// <summary>
    /// ソフトマックス関数(softmax = softargmax:正規化指数関数)
    /// The softmax function, also known as softargmax or normalized exponential function
    /// </summary>
    /// <see href="https://en.wikipedia.org/wiki/Softmax_function"/>
    public class SoftmaxFunction : IActivationFunction
    {
        /// <summary>
        /// 順伝播
        /// </summary>
        /// <param name="input">入力値</param>
        /// <returns>出力値</returns>
        public Vector<double> Forward(Vector<double> input)
        {
            // ベクトルの要素ごとの演算にも Map を使う
            Vector<double> exp_input = input.Map(x => Math.Exp(x));
            double sum_exp_input = exp_input.Sum();
            // 行列の要素ごとの演算には Map を使う
            return exp_input.Map(x => x / sum_exp_input);
        }

        /// <summary>
        /// 逆伝播
        /// </summary>
        /// <param name="input">上位層からの入力値</param>
        /// <param name="output">順伝播時の出力値</param>
        /// <param name="t">正解値</param>
        /// <returns>逆伝播する値</returns>
        public Vector<double> Backward(Vector<double> input, Vector<double> output, Vector<double> t)
        {
            // ソフトマックス関数の微分
            return output - t;
        }

    }
}
