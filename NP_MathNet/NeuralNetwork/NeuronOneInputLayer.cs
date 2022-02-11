using MathNet.Numerics.LinearAlgebra;

namespace NP_MathNet.NeuralNetwork
{
    public class NeuronOneInputLayer
    {
        private int n_output;

        /// <summary>
        /// 回帰用出力層
        /// </summary>
        /// <param name="n_output">下位層のニューロン数</param>
        public NeuronOneInputLayer(int n_output)
        {
            this.n_output = n_output;
        }

        /// <summary>
        /// 順伝播
        /// </summary>
        /// <param name="input">上位層からの入力</param>
        /// <returns>下層への出力値</returns>
        public Vector<double> Forward(double input)
        {
            return Vector<double>.Build.Dense(n_output, input);
        }

        /// <summary>
        /// 逆伝播
        /// </summary>
        /// <param name="delta"></param>
        /// <returns></returns>
        public Vector<double> Backward(Vector<double> t)
        {
            // 恒等関数
            return NP_MathNet.NP.EMPTY_VECTOR;
        }

        /// <summary>
        /// 重みとバイアスの更新
        /// </summary>
        /// <param name="eta">学習係数</param>
        public void Update(double eta)
        {
        }

    }
}
