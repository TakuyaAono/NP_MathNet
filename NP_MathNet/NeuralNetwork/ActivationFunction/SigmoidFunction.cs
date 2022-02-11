using MathNet.Numerics.LinearAlgebra;

namespace NP_MathNet.NeuralNetwork.ActivationFunction
{
    /// <summary>
    /// シグモイド関数(ロジスティック関数)
    /// </summary>
    /// <see href="https://ja.wikipedia.org/wiki/%E3%82%B7%E3%82%B0%E3%83%A2%E3%82%A4%E3%83%89%E9%96%A2%E6%95%B0"/>
    public class SigmoidFunction : IActivationFunction
    {
        /// <summary>
        /// 順伝播
        /// </summary>
        /// <param name="input">入力値</param>
        /// <returns>出力値</returns>
        public Vector<double> Forward(Vector<double> input)
        {
            // 行列の要素ごとの演算には Map を使う
            return input.Map(x => 1 / (1 + Math.Exp(-x)));
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
            // シグモイド関数の微分
            return t.PointwiseMultiply(1 - output).PointwiseMultiply(output);
        }
    }
}
