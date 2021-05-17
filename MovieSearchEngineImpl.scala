import spark.implicits._
import org.apache.spark.sql.functions.split
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.functions._
import scala.math

val plotSum = sc.textFile("/FileStore/tables/plot_summaries-1.txt")

// RDD to DataFrame
val plotSumDF = plotSum.toDF() 

// Split the Dataframe coulmn value into 2 columns Movie ID and Summary
val plotSumDF1 = plotSumDF.withColumn("_tmp", split($"value", "\\t")).select($"_tmp".getItem(0).as("MovieId"), $"_tmp".getItem(1).as("Summary")).drop("_tmp")

// Update the Summary column text into List of words
val tokenizer = new Tokenizer().setInputCol("Summary").setOutputCol("Words") 
val wordsData = tokenizer.transform(plotSumDF1)
val plotSumDF2 = wordsData.drop("Summary") 

// Remove stopwords from the Words Column in Dataframe
val stopWords = sc.textFile("/FileStore/tables/stopwords.txt").flatMap(line => line.split("\\W"))
val remover = new StopWordsRemover()
  .setInputCol("Words")
  .setOutputCol("WithoutStopWords")
  .setStopWords(stopWords.collect())
val plotSumDF3 = remover.transform(plotSumDF2).drop("Words")

val splitFile = plotSumDF3.select($"MovieId",explode($"WithoutStopWords"))

val newDF = splitFile.withColumn("col", regexp_replace(col("col"), ",", "")).withColumn("col", regexp_replace(col("col"), ":", "")).withColumn("col", regexp_replace(col("col"), ";", "")).withColumn("col", regexp_replace(col("col"), "!", ""))

// Mapping key/value pairs to a new key/value pairs and Reducing key/value pairs
val newPairs = newDF.rdd.map{case x => ((x(0).toString , x(1).toString),1)}

val reduce = newPairs.reduceByKey((x,y)=> x+y)
val tf = reduce.map{case (((x,y),z)) => (y,(x,z))}
val map1 =  reduce.map{case (((x,y),z)) => (y,(x,z,1))}
val map2 = map1.map{case (y,(x,z,1)) => (y,1)}
val reduce2 = map2.reduceByKey((x,y)=> x+y)

// Computing IDF
val totalDoc = plotSum.count()
val idf = reduce2.map{ case (x,y) => (x, math.log(totalDoc / y))}.map{case (x,y) => (x, y)}

// Computing TD-IDF for every term and every document 
val tf_idf = tf.join(idf)
val tf_idf1 = tf_idf.map{ case (x,((y,z),v)) => (y,x,z,v, (z * v))}

// Convert RDD to Dataframe
val tf_idfDF= tf_idf1.toDF("MovieId","Terms","TF","IDF","TF_IDF").sort("MovieId")

// Loading movie name file to get the movie names for the movie IDs
val movieFile = sc.textFile("/FileStore/tables/movie_metadata.tsv").toDF().withColumn("_tmp", split($"value", "\\t")).select($"_tmp".getItem(0).as("MovieId"), $"_tmp".getItem(1).as("1"),$"_tmp".getItem(2).as("MovieName"), $"_tmp".getItem(3).as("data")).drop("_tmp").drop("data").drop("1").sort("MovieId")

// Create a new Dataframe with MovieId, Terms, TF-IDF and MovieName and data sorted by TF-IDF in decreasing order
tf_idfDF.createOrReplaceTempView("tf_idfDF");
movieFile.createOrReplaceTempView("movieFile")
val newFile = sqlContext.sql("""SELECT tf_idfDF.MovieId, movieFile.MovieName, tf_idfDF.Terms, tf_idfDF.TF, tf_idfDF.IDF, tf_idfDF.TF_IDF 
                  FROM  tf_idfDF INNER JOIN  movieFile
                  ON tf_idfDF.MovieId == movieFile.MovieId""")

// User enters a single term: Output the top 10 movie name with the highest tf-idf values for the term queried
newFile.createOrReplaceTempView("data")
sqlContext.sql("""SELECT data.MovieName, data.TF_IDF, data.Terms 
                  FROM  data WHERE data.Terms = 'funny' """).sort(col("TF_IDF").desc).limit(10).show()

// Evaluating cosine similarity between the query and all the documents
val cosSim = udf { (tf: Array[Double], tfidf: Array[Double]) => 
    val x1 = tf
    val y1 = tfidf
    val Q = math.sqrt(x1.map(x => x*x).sum)
    val D = math.sqrt(y1.map(x => x*x).sum)
    val dotProd = x1.zip(y1).map(x => x._1*x._2).sum
    val cosine = dotProd/(Q*D)
    cosine
    }

// User enters a query consisting of multiple terms: Output the top 10 movie name with the highest cosine similarity values. 
val queryTerm = List("funny movie with action scenes").flatMap(line => line.split("\\W"))

val filteredDF = newFile.filter(col("Terms").isin(queryTerm: _*)).select('MovieId.alias("MovieId"), 'MovieName.alias("MovieName"), 'Terms.alias("Term"), 'IDF.alias("IDF"), 'TF_IDF.alias("TFIDF")).sort("MovieId")

val listDF = filteredDF.groupBy(col("MovieName")).agg(collect_list("IDF"), collect_list("TFIDF"))
val resultDF = listDF.withColumn("cosine_sim", cosSim(col("collect_list(IDF)"), col("collect_list(TFIDF)"))).drop("collect_list(IDF)").drop("collect_list(TFIDF)").sort(col("cosine_sim").desc).limit(10)
resultDF.withColumn("cosine_sim", col("cosine_sim").cast("float")).show()
