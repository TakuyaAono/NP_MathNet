using MathNet.Numerics.LinearAlgebra;

namespace NP_MathNet.NeuralNetwork.ActivationFunction
{
    /// <summary>
    /// 恒等関数
    /// </summary>
    public class IdentityFunction : IActivationFunction
    {
        /// <summary>
        /// 順伝播
        /// </summary>
        /// <param name="input">入力値</param>
        /// <returns>出力値</returns>
        public Vector<double> Forward(Vector<double> input)
        {
            // 恒等関数
            return input;
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
            // 恒等関数の微分
            return output - t;
        }

    }
}
