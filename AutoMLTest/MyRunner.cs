using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System.Data;
using System.Diagnostics;

namespace AutoMLTest
{
    public class MyRunner : ITrialRunner
    {
        private MLContext _context;
        private IDataView _trainDataset;
        private IDataView _evaluateDataset;

        private static bool _debug = true;
        private void debug(string msg)
        {
            if (_debug)
            {
                Console.WriteLine(msg);
            }
        }

        public MyRunner(MLContext context, IDataView trainDataset, IDataView evaluateDataset)
        {
            this._context = context;
            this._trainDataset = trainDataset;
            this._evaluateDataset = evaluateDataset;
        }

        public TrialResult Run(TrialSettings settings, IServiceProvider provider)
        {
            try
            {
                var trainDataset = this._trainDataset;
                var testDataset = this._evaluateDataset;

                var stopWatch = new Stopwatch();
                stopWatch.Start();
                var pipeline = settings.Pipeline.BuildTrainingPipeline(this._context, settings.Parameter);

                var model = pipeline.Fit(trainDataset);

                var predictEngine = _context.Model.CreatePredictionEngine<IrisInput, IrisOutput>(model);

                var predictedLabel1H = new List<IrisOutput>();

                var scoredData = model.Transform(trainDataset);

                foreach (var features in scoredData.GetColumn<Single[]>("Features"))
                {
                    var predict = predictEngine.Predict(new IrisInput
                    {
                        sepal_length = features[0],
                        sepal_width = features[1],
                        petal_length = features[2],
                        petal_width = features[3],
                    });

                    predictedLabel1H.Add(new IrisOutput
                    {
                        PredictedLabel = predict.PredictedLabel,
                        Score = predict.Score,
                    });

                    predictEngine.Predict(new IrisInput
                    {
                        sepal_length = features[0],
                        sepal_width = features[1],
                        petal_length = features[2],
                        petal_width = features[3],
                    });
                }

                var rmse = Enumerable.Zip(scoredData.GetColumn<Single[]>("Features"), predictedLabel1H)
                                .Select(x => Math.Pow(x.Second.Score[0] + x.Second.Score[1] + x.Second.Score[2], 2))
                                .Average();

                rmse = Math.Sqrt(rmse);

                return new TrialResult()
                {
                    Metric = rmse,
                    Model = model,
                    TrialSettings = settings,
                    DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
                };
            }
            catch (Exception Ex)
            {
                return new TrialResult()
                {
                    Metric = double.MaxValue,
                    Model = null,
                    TrialSettings = settings,
                    DurationInMilliseconds = 0,
                };
            }
        }
    }
}
