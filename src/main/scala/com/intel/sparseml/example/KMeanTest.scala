package com.intel.sparseml.example

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.{KMeans, SparseKMeans}
import org.apache.spark.mllib.linalg.{SparseVector, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

/**
 * Created by yuhao on 1/23/16.
 */
object KMeanTest {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val conf = new SparkConf().setAppName(s"kmeans: ${args.mkString(",")}")
    val sc = new SparkContext(conf)

    val k = args(0).toInt
    val dimension = args(1).toDouble.toInt
    val recordNum = args(2).toDouble.toInt
    val sparsity = args(3).toDouble
    val iterations = args(4).toInt
    val means = args(5)

    val data: RDD[Vector] = sc.parallelize(1 to recordNum).map(i => {
      val indexArr = (1 to (dimension * sparsity).toInt).map(in => Random.nextInt(dimension)).sorted.toArray
      val valueArr = (1 to (dimension * sparsity).toInt).map(in => Random.nextDouble()).toArray
      val vec: Vector = new SparseVector(dimension, indexArr, valueArr)
      vec
    })
    data.setName("dataRDD")
    data.cache()
    println(s"${data.getNumPartitions} partitions ${data.count()} records generated")

    val st = System.nanoTime()

    if(means.toLowerCase() == "sparsekmeans") {
      println("running sparse kmeans")
      val model = new SparseKMeans()
        .setK(k)
        .setInitializationMode("random")
        .setMaxIterations(iterations)
        .run(data)

      println((System.nanoTime() - st) / 1e9 + " seconds cost")
      println(s"final clusters average density: ${model.clusterCenters.map(v => v.numNonzeros).reduce(_+_).toDouble/k/dimension}")
    } else {
      println("running mllib kmeans")
      val model = new KMeans()
        .setK(k)
        .setInitializationMode("random")
        .setMaxIterations(iterations)
        .run(data)

      println((System.nanoTime() - st) / 1e9 + " seconds cost")
      println(s"final clusters average density: ${model.clusterCenters.map(v => v.numNonzeros).reduce(_+_).toDouble/k/dimension}")
    }

    sc.stop()
  }

}
