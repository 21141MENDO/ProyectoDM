
  import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
  import org.apache.spark.ml.feature.LabeledPoint
  import org.apache.spark.rdd.RDD
  import org.apache.spark.mllib.linalg.Vectors
  import org.apache.spark.ml.linalg.Vectors
  import org.apache.spark.sql.SparkSession
  import org.apache.spark.ml.classification.LogisticRegression

  import org.apache.log4j._
  Logger.getLogger("org").setLevel(Level.ERROR)

/*carga de datos, limpieza y conversion de columna a tipo int o double*/
val spark = SparkSession.builder().getOrCreate()
val data = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

val c1 = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val c2 = c1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val  c3= c2.withColumn("y",'y.cast("Int"))

val featureCols = Array("age","balance")
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
val df = assembler.transform(c3)
df.show(5)



/* k=10 Folding cross Validation */
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
  val lr = new LinearRegression().setMaxIter(10)
  val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.01)).addGrid(lr.fitIntercept).addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0)).build()
  val trainValidationSplit = new TrainValidationSplit().setEstimator(lr).setEvaluator(new RegressionEvaluator).setEstimatorParamMaps(paramGrid).setTrainRatio(0.8)
  val assembler = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))
  val assembler1 = (new VectorAssembler().setInputCols(Array("y")).setOutputCol("label"))

val df = assembler.transform(c3)
df.show(5)
val df1 = assembler1.transform(df)
df1.show(5)
val  c55= df1.withColumn("label",'y.cast("Double"))
val Array(trainingData, testData) = c55.randomSplit(Array(0.7,0.3))
val model22 = trainValidationSplit.fit(trainingData)
 model22.transform(testData).select("features", "label").show()


//KMEANS 
import org.apache.spark.ml.clustering.KMeans
val kmeans = new KMeans().setK(7).setSeed(1L).setPredictionCol("prediction")
val model = kmeans.fit(df)
val WSSE = model.computeCost(df)
println("Cluster Centers: ")
k_model.clusterCenters.foreach(println)
val categories = model.transform(testData)
val mostrandoresult =categories.select($"age",$"balance",$"prediction").groupBy("age","balance","prediction").agg(count("prediction")).orderBy("age","balance","prediction")
mostrandoresult.show(5)
categories.show(5)


  /* MULTILAYER PERCEPTRON */
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val splits = categories.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = splits(0)
val test = splits(1)
val layers = Array[Int](2, 1, 3, 3)
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
val  c4= categories.withColumn("prediction",'prediction.cast("Double"))
val  c5= c4.withColumn("label",'prediction.cast("Double"))
val predictionAndLabels = c5 .select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")




//Decisi.Tree
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor


val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(c55)
val Array(trainingData, testData) = c55.randomSplit(Array(0.7, 0.3))
val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))
val model43 = pipeline.fit(trainingData)
val predictions = model.transform(testData)
predictions.select("prediction", "label", "features").show(5)

//LR
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val e4= predictions.withColumn("prediction",'prediction.cast("Double"))
val rmse = evaluator.evaluate(e4)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

//SVM
import org.apache.spark.ml.classification.LinearSVC
val change = df.withColumnRenamed("y", "label")
val ft = change.select("label","features")
ft.show()
val cs1 = ft.withColumn("1abel",when(col("label").equalTo("1"),0).otherwise(col("label")))
val cs2 = cs1.withColumn("label",when(col("label").equalTo("2"),1).otherwise(col("label")))
val cs3 = cs2.withColumn("label",'label.cast("Int"))
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
val lsvcModel = lsvc.fit(cs3)
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")


mostrandoresult.show(5)
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
predictions.select("prediction", "label", "features").show(5)
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")