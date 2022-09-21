//using static Microsoft.DotNet.Interactive.Formatting.PocketViewTags;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
//using Plotly.NET;
//using Microsoft.DotNet.Interactive.Formatting;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.Trainers;
using System.Data;

namespace AutoMLTest
{
    class Program
    {
        static int Main(string[] args)
        {
            Console.WriteLine("Preparing AutoML Experiment... Please wait.");

            // Create a new MLContext (the starting point for all ML.NET operations)
            var mlContext = new MLContext();

            // Load data from a text file to an IDataView (a flexible, efficient way of describing tabular data)
            Console.WriteLine("Reading data.");
            IDataView trainValidateData = mlContext.Data.LoadFromTextFile<IrisInput>(
                path: "iris.csv",
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                allowSparse: false);

            IDataView testData = mlContext.Data.LoadFromTextFile<IrisInput>(
                path: "iris.csv",
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                allowSparse: false);

            var searchSpace = new SearchSpace<IrisSearchSpace>();

            var pipeline = mlContext.Transforms.Concatenate("Features", "sepal_length", "sepal_width", "petal_length", "petal_width")
                             .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "Label"))
                             .Append(mlContext.Auto().CreateSweepableEstimator((context, ss) =>
                             {
                                 LbfgsMaximumEntropyMulticlassTrainer.Options options = new LbfgsMaximumEntropyMulticlassTrainer.Options
                                 {
                                     MaximumNumberOfIterations = ss.MaximumNumberOfIterations,
                                     InitialWeightsDiameter = ss.InitialWeightsDiameter,
                                     L2Regularization = ss.L2Regularization,
                                     L1Regularization = ss.L1Regularization,
                                     FeatureColumnName = "Features",
                                     LabelColumnName = "Label",
                                 };

                                 return mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(options);
                             }, searchSpace))
                             .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));


            var dataSplit = mlContext.Data.TrainTestSplit(trainValidateData, testFraction: 0.2);

            // setup logger
            mlContext.Log += (e, o) =>
            {
                if (o.RawMessage.Contains("Trial"))
                {
                    Console.WriteLine(o.RawMessage);
                }
            };

            // NotebookMonitor plots trials and show best run nicely in notebook output cell.
            //var monitor = new NotebookMonitor();

            var experiment = mlContext.Auto().CreateExperiment()
                            .SetPipeline(pipeline)
                            .SetTrainingTimeInSeconds(30)
                            .SetDataset(dataSplit.TrainSet, dataSplit.TestSet)
                            .SetMulticlassClassificationMetric(MulticlassClassificationMetric.MacroAccuracy, "Label", "PredictedLabel");
                            //.SetMonitor(monitor);

            // Configure Visualizer			
            //monitor.SetUpdate(monitor.Display());

            // Start Experiment
            Console.WriteLine("Running Experiment...");
            //var res = await experiment.RunAsync();
            var res = experiment.Run();

            var model = res.Model;

            if(model == null)
            {
                Console.WriteLine("Experiment returned null due to a time-out.");
                return -1;
            }

            Console.WriteLine("Experiment completed.");

            // evaluate
            var predictEngine = mlContext.Model.CreatePredictionEngine<IrisInput, IrisOutput>(model);

            var predictedLabel1H = new List<IrisOutput>();

            var scoredData = model.Transform(dataSplit.TrainSet);

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

            var mse = predictedLabel1H.Select(x => Math.Pow(x.Score[0] + x.Score[1] + x.Score[2], 2)).Average();
            var rmse = Math.Sqrt(mse);
            //rmse

            // Evaluate the model using the cross validation method
            // Learn more about cross validation at https://aka.ms/mlnet-cross-validation
            var testDataPredictions = model.Transform(testData);
            MulticlassClassificationMetrics trainedModelMetrics = mlContext.MulticlassClassification.Evaluate(testDataPredictions);
           
            Console.WriteLine("\nLogLoss,LogLossReduction,MacroAccuracy,MicroAccuracy,TopKAccuracy,TopKPredictionCount,TopKAccuracyForAllK,PerClassLogLoss,ConfusionMatrix");
            Console.Write($"{trainedModelMetrics.LogLoss},{trainedModelMetrics.LogLossReduction},{trainedModelMetrics.MacroAccuracy},{trainedModelMetrics.MicroAccuracy}," +
                $"{trainedModelMetrics.TopKAccuracy},{trainedModelMetrics.TopKPredictionCount},{(trainedModelMetrics.TopKAccuracyForAllK == null ? "null" : trainedModelMetrics.TopKAccuracyForAllK)}" +
                $",[");

            int count = 0;
            foreach (var value in trainedModelMetrics.PerClassLogLoss)
            {
                Console.Write($"{(count++ == 0 ? "" : ", ")}{value}");
            }

            Console.Write("],PerClassPrecision: [");

            count = 0;
            foreach (var value in trainedModelMetrics.ConfusionMatrix.PerClassPrecision)
            {
                Console.Write($"{(count++ == 0 ? "" : ", ")}{value}");
            }

            Console.Write("], PerClassRecall: [");

            count = 0;
            foreach (var value in trainedModelMetrics.ConfusionMatrix.PerClassRecall)
            {
                Console.Write($"{(count++ == 0 ? "" : ", ")}{value}");
            }

            Console.Write("], Counts: [");

            foreach (var values in trainedModelMetrics.ConfusionMatrix.Counts)
            {
                Console.Write(" [");

                count = 0;
                foreach (var value in values)
                {
                    Console.Write($"{(count++ == 0 ? "" : ", ")}{value}");
                }

                Console.Write("],");

            }

            Console.Write($"\b ], NumberOfClasses: {trainedModelMetrics.ConfusionMatrix.NumberOfClasses}\n\n");

            // Define sample model input
            var SetosaData = new IrisInput()
            {
                sepal_length = 4.8F,
                sepal_width = 3.4F,
                petal_length = 1.6F,
                petal_width = 0.2F,
            };

            // Define sample model input
            var VersicolorData = new IrisInput()
            {
                sepal_length = 7F,
                sepal_width = 3.2F,
                petal_length = 4.7F,
                petal_width = 1.4F,
            };

            // Define sample model input
            var VirginicaData = new IrisInput()
            {
                sepal_length = 6.3F,
                sepal_width = 3.4F,
                petal_length = 5.6F,
                petal_width = 2.4F,
            };

            // Create a Prediction Engine (used to make single predictions)
            var predEngine = mlContext.Model.CreatePredictionEngine<IrisInput, IrisOutput>(model);

            var IrisList = new List<IrisInput>();
            IrisList.Add(SetosaData);
            IrisList.Add(VersicolorData);
            IrisList.Add(VirginicaData);

            // Use the model and Prediction Engine to predict on new sample data
            Console.WriteLine("Using model to make single prediction -- Comparing actual Label with predicted Label from sample data...\n");

            foreach (var iris in IrisList)
            {
                var predictionResult = predEngine.Predict(iris);

                Console.WriteLine($"Sepal_length: {iris.sepal_length}");
                Console.WriteLine($"Sepal_width: {iris.sepal_width}");
                Console.WriteLine($"Petal_length: {iris.petal_length}");
                Console.WriteLine($"Petal_width: {iris.petal_width}");
                Console.WriteLine($"Predicted Label: {predictionResult.PredictedLabel}\n");
            }

            return 0;
        }
    }
}