package ml.dmlc.xgboost4j.scala.example.spark

import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector => MLDenseVector}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

/*
 * author: Vesper Huang
 */

object SparkDFRegression2 {
  def main(args: Array[String]): Unit = {
    if (args.length != 7) {
      println(
        "usage: program num_of_rounds max_depth num_workers training_path test_path model_path boost_path")
      sys.exit(1)
    }
    val sparkConf = new SparkConf().setAppName("XGBoost-spark-example")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sparkConf.registerKryoClasses(Array(classOf[Booster]))
    implicit val sc = new SparkContext(sparkConf)

    val sparkSession = SparkSession.builder().config(sparkConf).getOrCreate()

    val inputTrainPath = args(3)  // 训练数据路径
    val inputTestPath = args(4)   // 测试数据路径
    val outputModelPath = args(5)  // 模型输出路径
    val boosterPath = args(6)
    // number of iterations
    val numRound = args(0).toInt  // 训练轮数
    val max_depth = args(1).toInt  // 决策树最大深度
    val numWorker = args(2).toInt

    //    val trainRDD = MLUtils.loadLibSVMFile(sc, inputTrainPath).map(lp =>
    //      MLLabeledPoint(lp.label, new MLDenseVector(lp.features.toArray)))

    val trainDF = sparkSession.sqlContext.read.format("libsvm").load(inputTrainPath)

    //    val testSet = MLUtils.loadLibSVMFile(sc, inputTestPath)
    //      .map(lp => new MLDenseVector(lp.features.toArray))

    //    val trainSet = MLUtils.loadLibSVMFile(sc, inputTrainPath)
    //      .map(lp => new MLDenseVector(lp.features.toArray))

    // training parameters
    val paramMap = List(
      "eta" -> 0.1f,
      "max_depth" -> max_depth,
      "eval_metric" -> "rmse",
      "gamma" -> 1,
      "objective" -> "reg:gamma").toMap

    // train
    val xgboostModel = XGBoost.trainWithDataFrame(trainDF, paramMap, numRound, nWorkers = numWorker,
      useExternalMemory = true)

    // predict
    //    xgboostModel.predict(testSet, missingValue = Float.NaN)

    // save test
    val testDF = sparkSession.sqlContext.read.format("libsvm").load(inputTestPath)
    val predict = xgboostModel.transform(testDF)
    predict.limit(20).show()
    val scoreAndLabels = predict.select(xgboostModel.getPredictionCol, xgboostModel.getLabelCol)
      .rdd.map{case Row(score: Float, label:Double) => (score.toDouble, label)}
    val metric = new RegressionMetrics(scoreAndLabels)
    val rmse = metric.rootMeanSquaredError
    println("test_set: rmse: " + rmse + ", r2: " + metric.r2)

    // train rmse
    //    val trainDF = sparkSession.sqlContext.read.format("libsvm").load(inputTrainPath)
    val train_value = xgboostModel.transform(trainDF)
    train_value.limit(20).show()
    val scoreAndLabels1 = train_value.select(xgboostModel.getPredictionCol, xgboostModel.getLabelCol)
      .rdd.map{case Row(score: Float, label:Double) => (score.toDouble, label)}
    val metric1 = new RegressionMetrics(scoreAndLabels1)
    val rmse1 = metric1.rootMeanSquaredError
    println("train_set: rmse: " + rmse1 + ", r2: " + metric1.r2)

    // save model to HDFS path
    xgboostModel.saveModelAsHadoopFile(outputModelPath)

    // save common format
    xgboostModel.booster.saveModel(boosterPath)
  }

}
