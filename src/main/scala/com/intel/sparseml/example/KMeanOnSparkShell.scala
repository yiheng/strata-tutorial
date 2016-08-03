import org.apache.spark.mllib.linalg.{SparseVector, Vector}
import org.apache.spark.mllib.clustering.KMeans
import scala.util.Random
import org.apache.spark.rdd.RDD

//val k = 200
//val dimension = 1e7.toInt
//val recordNum = 2e7.toInt
//val sparsity = 1e-6
//val iterations = 10
//
//val parallel = 4
//val data: RDD[Vector] = sc.parallelize(1 to recordNum, parallel).map(i => {
//      val ran = new Random()
//      val indexArr = (1 to (dimension * sparsity).toInt).map(in => ran.nextInt(dimension)).sorted.toArray
//      val valueArr = (1 to (dimension * sparsity).toInt).map(in => ran.nextDouble()).sorted.toArray
//      val vec: Vector = new SparseVector(dimension, indexArr, valueArr)
//      vec
//    }).cache()
//
//val model = new KMeans().setK(k).setInitializationMode("random").setMaxIterations(iterations).run(data)
