using Microsoft.ML.Data;
using Microsoft.ML.SearchSpace;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AutoMLTest
{
    public class IrisInput
    {
        [ColumnName("sepal_length"), LoadColumn(0)]
        public float sepal_length { get; set; }

        [ColumnName("sepal_width"), LoadColumn(1)]
        public float sepal_width { get; set; }

        [ColumnName("petal_length"), LoadColumn(2)]
        public float petal_length { get; set; }

        [ColumnName("petal_width"), LoadColumn(3)]
        public float petal_width { get; set; }

        [ColumnName("Label"), LoadColumn(4)]
        public string? Label { get; set; }
    }

    public class IrisOutput
    {
        [ColumnName("PredictedLabel")]
        public string? PredictedLabel { get; set; }

        public float[]? Score { get; set; }
    }

    public class IrisSearchSpace
    {
        [Range(0, 1.0)]
        public float L1Regularization { get; set; } = 0.5f;

        [Range(0.0, 1.0)]
        public float L2Regularization { get; set; } = 1;

        [Range(1, 1000)]
        public int MaximumNumberOfIterations { get; set; } = 1;

        [Range(0, 1)]
        public float InitialWeightsDiameter { get; set; } = 0;
    }

}
