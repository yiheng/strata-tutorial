import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, SQLTransformer, StringIndexer, VectorAssembler}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.log4j.{Level, Logger}

object Adult {
  def main(args : Array[String]) : Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val trainDataFile = "adult.data"
    val testDataFile = "adult.test"

    val conf = new SparkConf().setAppName("Adult").setMaster("local[1]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val trainData = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(this.getClass.getResource(trainDataFile).getPath)

    val testData = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(this.getClass.getResource(testDataFile).getPath)

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
        "occupation-encode", "relationship-encode", "race-encode", "native-country-encode")).setOutputCol("feature"),
      new LogisticRegression().setFeaturesCol("feature").setLabelCol("label_idx")
        .setRawPredictionCol("raw-predict").setPredictionCol("predict")
    ))
    val model = pipeline.fit(trainData)
    val predicts = model.transform(testData)
    val correct = predicts.where("label_idx<>predict").count()
    val total = predicts.count()
    println(s"Error rate is ${correct.toDouble / total}")
  }
}