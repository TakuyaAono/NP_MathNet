using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NP_MathNet
{
    public class NP
    {
        public static Vector<double> EMPTY_VECTOR = Vector<double>.Build.Dense(1, 0.0);
        public static Matrix<double> EMPTY_MATRIX = Matrix.Build.Dense(1, 1, 0);

        public static Vector<double> Linespace(double start, double stop, int count = 50)
        {
            double step = (stop - start) / (count - 1);
            return Vector<double>.Build.DenseOfArray(
                    Enumerable.Range(0, count).Select(i => start + step * i).ToArray());
        }
        public static Vector<double> Arange(double start, double stop, double step)
        {
            int count = (int)((stop - start) / step) + 1;
            return Vector<double>.Build.DenseOfArray(
                    Enumerable.Range(0, count).Select(i => start + i * step).ToArray());
        }
        public static Matrix<double> Zeros(int rows, int cols)
        {
            return DenseMatrix.Build.Dense(rows, cols, 0);
        }
        public static Vector<double> Array(double[] vac)
        {
            return Vector<double>.Build.DenseOfArray(vac);
        }
        public static Matrix<double> Array(double[,] mat)
        {
            return DenseMatrix.Build.DenseOfArray(mat);
        }
    }
}
