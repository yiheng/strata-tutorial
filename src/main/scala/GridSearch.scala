import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object GridSearch {
  def main(args : Array[String]) : Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val trainDataFile = args(0)
    val testDataFile = args(1)

    val conf = new SparkConf().setAppName("Adult_Pipeline_GridSearch").setMaster("local[1]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val trainData = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(trainDataFile)

    val testData = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(testDataFile)

    trainData.printSchema()
    val pipeline = new Pipeline().setStages(Array(
      new StringIndexer().setInputCol("workclass").setOutputCol("workclass-idx"),
      new OneHotEncoder().setInputCol("workclass-idx").setOutputCol("workclass-encode"),
      new StringIndexer().setInputCol("education").setOutputCol("education-idx"),
      new OneHotEncoder().setInputCol("education-idx").setOutputCol("education-encode"),
      new StringIndexer().setInputCol("marital-status").setOutputCol("marital-status-idx"),
      new OneHotEncoder().setInputCol("marital-status-idx").setOutputCol("marital-status-encode"),
      new StringIndexer().setInputCol("occupation").setOutputCol("occupation-idx"),
      new OneHotEncoder().setInputCol("occupation-idx").setOutputCol("occupation-encode"),
      new StringIndexer().setInputCol("relationship").setOutputCol("relationship-idx"),
      new OneHotEncoder().setInputCol("relationship-idx").setOutputCol("relationship-encode"),
      new StringIndexer().setInputCol("race").setOutputCol("race-idx"),
      new OneHotEncoder().setInputCol("race-idx").setOutputCol("race-encode"),
      new StringIndexer().setInputCol("sex").setOutputCol("sex-idx"),
      new OneHotEncoder().setInputCol("sex-idx").setOutputCol("sex-encode"),
      new StringIndexer().setInputCol("native-country").setOutputCol("native-country-idx"),
      new OneHotEncoder().setInputCol("native-country-idx").setOutputCol("native-country-encode"),
      new StringIndexer().setInputCol("label").setOutputCol("label_idx"),
      new VectorAssembler().setInputCols(Array("age", "fnlwgt", "education-num", "education-num", "capital-loss",
        "hours-per-week", "workclass-encode", "education-encode", "marital-status-encode",
        "occupation-encode", "relationship-encode", "race-encode", "native-country-encode")).setOutputCol("feature")
    ))

    val featurePipeline = pipeline.fit(trainData)

    val lr = new LogisticRegression()
      .setFeaturesCol("feature")
      .setPredictionCol("predict")
      .setLabelCol("label_idx")
      .setRawPredictionCol("raw-predict")
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.01, 0.1, 1.0))
      .addGrid(lr.maxIter, Array(100, 150, 200))
      .build()
    val cv = new CrossValidator()
      .setNumFolds(3)
      .setEstimator(lr)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(new BinaryClassificationEvaluator()
        .setLabelCol("label_idx")
        .setRawPredictionCol("raw-predict"))

    val model = cv.fit(featurePipeline.transform(trainData))
    val predicts = model.transform(featurePipeline.transform(testData))
    
    println(s"Error rate is ${predicts.where("label_idx<>predict").count().toDouble / predicts.count()}")
  }
}