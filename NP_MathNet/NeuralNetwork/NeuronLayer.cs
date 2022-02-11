using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NP_MathNet.NeuralNetwork.ActivationFunction;

namespace NP_MathNet.NeuralNetwork
{
    public class NeuronLayer
    {
        private IActivationFunction function;
        public Matrix<double> Weight { get; set; }
        public Vector<double> Bias { get; set; }

        /// <summary>
        /// 回帰用出力層
        /// </summary>
        /// <param name="n_upper">上層のニューロ数</param>
        /// <param name="n">自層のニューロン数</param>
        /// <param name="wb_width">重みとバイアスの広がり具合</param>
        /// <see href="http://ja.wikipedia.org/wiki/ボックス＝ミュラー法"/>
        public NeuronLayer(int n_upper, int n, double wb_width, IActivationFunction function)
        {
            this.function = function;

            // 平均0標準偏差1の乱数を格納して重みとバイアスを初期化する。
            // ボックス＝ミュラー法 を使用している？
            // http://ja.wikipedia.org/wiki/ボックス＝ミュラー法
            // 重み（行列）
            // バイアス（ベクトル）
            //this.wb_width = wb_width;
            Weight = wb_width * DenseMatrix.Build.Random(n_upper, n);
            Bias = wb_width * DenseVector.Build.Random(n);
        }

        private Vector<double> input    = NP.EMPTY_VECTOR;
        private Vector<double> y        = NP.EMPTY_VECTOR;
        private Vector<double> delta    = NP.EMPTY_VECTOR;

        /// <summary>
        /// 順伝播
        /// </summary>
        /// <param name="input">上位層からの入力</param>
        /// <returns>下層への出力値</returns>
        public Vector<double> Forward(Vector<double> input)
        {
            this.input = input;

            y = function.Forward(input * Weight + Bias);
            return y;
        }

        private Matrix<double> grad_w       = NP.EMPTY_MATRIX;
        private Vector<double> grad_b       = NP.EMPTY_VECTOR;
        private Vector<double> grad_upper   = NP.EMPTY_VECTOR;

        /// <summary>
        /// 逆伝播
        /// </summary>
        /// <param name="delta"></param>
        /// <returns></returns>
        public Vector<double> Backward(Vector<double> t)
        {
            // 恒等関数
            this.delta = function.Backward(input, y, t);

            // deltaの列ごとの合計を求める
            grad_w = input.ToColumnMatrix() * delta.ToRowMatrix();
            grad_b = delta;
            grad_upper = delta * Weight.Transpose();

            return grad_upper;
        }

        /// <summary>
        /// 重みとバイアスの更新
        /// </summary>
        /// <param name="eta">学習係数</param>
        public void Update(double eta)
        {
            Weight -= (eta * grad_w);
            Bias -= (eta * grad_b);
        }

    }
}
