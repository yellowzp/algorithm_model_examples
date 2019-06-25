package ml.dmlc.xgboost4j.scala.example.spark

import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SparkSession}

/*
 * author: Vesper Huang
 */

object SparkDFPredict {
  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      println(
        "usage: program model_path input_data_path output_path")
      sys.exit(1)
    }

    val sparkConf = new SparkConf().setAppName("XGBoost-spark-example")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sparkConf.registerKryoClasses(Array(classOf[Booster]))
    implicit val sc = new SparkContext(sparkConf)

    val sparkSession = SparkSession.builder().config(sparkConf).getOrCreate()

    val model_path = args(0)
    val input_data_path = args(1)
    val output_path = args(2)

    val model = XGBoost.loadModelFromHadoopFile(model_path)(sc)

    val dataDF = sparkSession.sqlContext.read.format("libsvm").load(input_data_path)

    val result = model.transform(dataDF)

    val preRDD = result.select(model.getPredictionCol).rdd.map{case Row(score: Float) => score.toDouble}
    println("predict line num: " + preRDD.count())

    preRDD.saveAsTextFile(output_path)
  }
}
